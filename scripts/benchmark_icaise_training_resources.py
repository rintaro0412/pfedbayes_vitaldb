from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class MethodSpec:
    method: str
    key: str
    config_path: str
    entrypoint: str


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec("Centralized", "centralized", "configs/centralized.yaml", "centralized/train.py"),
    MethodSpec("Local", "local", "configs/local.yaml", "scripts/train_local.py"),
    MethodSpec("FedAvg", "fedavg", "configs/fedavg.yaml", "federated/server.py"),
    MethodSpec("FedProx", "fedprox", "configs/fedprox.yaml", "federated/server.py"),
    MethodSpec("SCAFFOLD", "scaffold", "configs/scaffold.yaml", "federated/server.py"),
    MethodSpec("FedNova", "fednova", "configs/fednova.yaml", "federated/server.py"),
    MethodSpec("Per-FedAvg", "perfedavg", "configs/perfedavg.yaml", "federated/server.py"),
    MethodSpec("pFedMe", "pfedme", "configs/pfedme.yaml", "federated/server.py"),
    MethodSpec("pFedBayes", "pfedbayes", "configs/pfedbayes.yaml", "bayes_federated/pfedbayes_server.py"),
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # type: ignore

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
    import yaml  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _manifest_methods(path: Path) -> List[MethodSpec]:
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.suffix.lower() == ".json" else _load_yaml(path)
    rows = manifest.get("methods") if isinstance(manifest, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("Experiment manifest must contain a nonempty methods list.")
    methods: List[MethodSpec] = []
    seen_keys: set[str] = set()
    seen_labels: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(field), str) or not row[field].strip() for field in ("key", "label", "config", "entrypoint")):
            raise ValueError("Each method requires nonempty key, label, config, and entrypoint strings.")
        key, label = row["key"].strip(), row["label"].strip()
        if key.lower() in seen_keys or label.lower() in seen_labels:
            raise ValueError(f"Duplicate method key or label: {key}, {label}")
        seen_keys.add(key.lower())
        seen_labels.add(label.lower())
        methods.append(MethodSpec(label, key, row["config"], row["entrypoint"]))
    return methods


def _selected_methods(raw: str, methods: Sequence[MethodSpec] = METHODS) -> List[MethodSpec]:
    if not raw.strip():
        return list(methods)
    by_name = {m.method.lower(): m for m in methods}
    by_key = {m.key.lower(): m for m in methods}
    out: List[MethodSpec] = []
    missing: List[str] = []
    seen: set[str] = set()
    for item in [x.strip().lower() for x in raw.split(",") if x.strip()]:
        if item in seen:
            continue
        seen.add(item)
        if item in by_name:
            method = by_name[item]
        elif item in by_key:
            method = by_key[item]
        else:
            missing.append(item)
            continue
        if method not in out:
            out.append(method)
    if missing:
        raise SystemExit(f"unknown method(s): {', '.join(sorted(missing))}")
    return out


def _prepare_config(spec: MethodSpec, *, repo_root: Path, out_root: Path, tag: str, no_progress: bool, write: bool = True) -> Path:
    cfg = _load_yaml(repo_root / spec.config_path)
    cfg.setdefault("run", {})
    cfg["run"]["out_dir"] = str(out_root / spec.key)
    cfg["run"]["run_name"] = str(tag)
    cfg["run"]["resume"] = False
    cfg.setdefault("train", {})
    if no_progress:
        cfg["train"]["no_progress_bar"] = True
        cfg["train"]["client_progress_bar"] = False
    config_out = out_root / "configs" / f"{spec.key}_{tag}.yaml"
    if write:
        _write_yaml(config_out, cfg)
    else:
        print(json.dumps({"method": spec.method, "resolved_config": cfg}, ensure_ascii=False, indent=2))
    return config_out


def _read_proc_table() -> tuple[Dict[int, int], Dict[int, float]]:
    try:
        out = subprocess.check_output(["ps", "-eo", "pid=,ppid=,rss="], text=True)
    except Exception:
        return {}, {}
    ppid_by_pid: Dict[int, int] = {}
    rss_by_pid: Dict[int, float] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            rss_kb = float(parts[2])
        except Exception:
            continue
        ppid_by_pid[pid] = ppid
        rss_by_pid[pid] = rss_kb / 1024.0
    return ppid_by_pid, rss_by_pid


def _process_tree(root_pid: int) -> set[int]:
    ppid_by_pid, _ = _read_proc_table()
    children: Dict[int, List[int]] = {}
    for pid, ppid in ppid_by_pid.items():
        children.setdefault(ppid, []).append(pid)
    seen = {int(root_pid)}
    stack = [int(root_pid)]
    while stack:
        cur = stack.pop()
        for child in children.get(cur, []):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def _tree_rss_mb(root_pid: int) -> float:
    _, rss_by_pid = _read_proc_table()
    pids = _process_tree(root_pid)
    return float(sum(rss_by_pid.get(pid, 0.0) for pid in pids))


def _gpu_memory_by_pid_mb() -> Dict[int, float]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {}
    result: Dict[int, float] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            mem = float(parts[1])
        except Exception:
            continue
        result[pid] = mem
    return result


def _tree_gpu_mb(root_pid: int) -> float:
    pids = _process_tree(root_pid)
    mem_by_pid = _gpu_memory_by_pid_mb()
    return float(sum(mem_by_pid.get(pid, 0.0) for pid in pids))


def _gpu_info() -> List[Dict[str, Any]]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    rows = []
    for idx, line in enumerate(out.splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            rows.append({"index": idx, "name": parts[0], "memory_total_mb": float(parts[1])})
    return rows


def _build_command(python_bin: str, spec: MethodSpec, config_path: Path, no_progress: bool) -> List[str]:
    cmd = [python_bin, spec.entrypoint, "--config", str(config_path)]
    if no_progress:
        cmd.append("--no-progress-bar")
    return cmd


def _run_one(
    *,
    spec: MethodSpec,
    repo_root: Path,
    out_root: Path,
    tag: str,
    python_bin: str,
    interval_s: float,
    no_progress: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    config_path = _prepare_config(spec, repo_root=repo_root, out_root=out_root, tag=tag, no_progress=no_progress, write=not dry_run)
    method_dir = out_root / spec.key / tag
    log_dir = out_root / "logs"
    stdout_path = log_dir / f"{spec.key}_{tag}.stdout.log"
    stderr_path = log_dir / f"{spec.key}_{tag}.stderr.log"
    samples_path = log_dir / f"{spec.key}_{tag}.resource_samples.csv"
    summary_path = out_root / "summaries" / f"{spec.key}_{tag}.json"
    cmd = _build_command(python_bin, spec, config_path, no_progress=no_progress)
    row: Dict[str, Any] = {
        "method": spec.method,
        "key": spec.key,
        "tag": tag,
        "run_dir": str(method_dir),
        "config_path": str(config_path),
        "command": " ".join(shlex.quote(x) for x in cmd),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "resource_samples_csv": str(samples_path),
        "summary_json": str(summary_path),
        "started_utc": _now_iso(),
        "gpu_info": [] if dry_run else _gpu_info(),
    }
    if dry_run:
        row.update({"status": "dry_run", "returncode": None})
        print(f"[PLAN] {row['command']}")
        return row

    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    peak_cpu_rss_mb = 0.0
    peak_gpu_memory_mb = 0.0
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open("w", encoding="utf-8") as stderr_f, samples_path.open(
        "w", newline="", encoding="utf-8"
    ) as samples_f:
        writer = csv.DictWriter(samples_f, fieldnames=["elapsed_s", "cpu_rss_tree_mb", "gpu_memory_tree_mb"])
        writer.writeheader()
        proc = subprocess.Popen(cmd, cwd=str(repo_root), stdout=stdout_f, stderr=stderr_f, text=True)
        print(f"[START] {spec.method}: pid={proc.pid} run_dir={method_dir}")
        while True:
            rc = proc.poll()
            elapsed = time.perf_counter() - started
            cpu_rss = _tree_rss_mb(proc.pid) if rc is None else 0.0
            gpu_mem = _tree_gpu_mb(proc.pid) if rc is None else 0.0
            peak_cpu_rss_mb = max(peak_cpu_rss_mb, cpu_rss)
            peak_gpu_memory_mb = max(peak_gpu_memory_mb, gpu_mem)
            writer.writerow(
                {
                    "elapsed_s": f"{elapsed:.3f}",
                    "cpu_rss_tree_mb": f"{cpu_rss:.3f}",
                    "gpu_memory_tree_mb": f"{gpu_mem:.3f}",
                }
            )
            samples_f.flush()
            if rc is not None:
                break
            time.sleep(max(float(interval_s), 0.2))
    wall_time_s = time.perf_counter() - started
    row.update(
        {
            "finished_utc": _now_iso(),
            "status": "ok" if rc == 0 else "failed",
            "returncode": int(rc),
            "wall_time_s": float(wall_time_s),
            "wall_time_min": float(wall_time_s / 60.0),
            "peak_cpu_rss_tree_mb": float(peak_cpu_rss_mb),
            "peak_gpu_memory_tree_mb": float(peak_gpu_memory_mb),
        }
    )
    summary_path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[DONE] {spec.method}: status={row['status']} "
        f"wall={row['wall_time_min']:.2f} min "
        f"cpu_peak={peak_cpu_rss_mb:.1f} MB gpu_peak={peak_gpu_memory_mb:.1f} MB"
    )
    return row


def _write_outputs(out_root: Path, rows: Sequence[Dict[str, Any]], tag: str) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    json_path = out_root / f"training_resource_benchmark_{tag}.json"
    csv_path = out_root / f"training_resource_benchmark_{tag}.csv"
    latest_json = out_root / "training_resource_benchmark_latest.json"
    latest_csv = out_root / "training_resource_benchmark_latest.csv"
    payload = {"tag": tag, "rows": list(rows)}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    latest_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = [
        "method",
        "key",
        "tag",
        "status",
        "returncode",
        "wall_time_s",
        "wall_time_min",
        "peak_cpu_rss_tree_mb",
        "peak_gpu_memory_tree_mb",
        "run_dir",
        "config_path",
        "stdout_log",
        "stderr_log",
        "resource_samples_csv",
        "summary_json",
        "command",
    ]
    for path in [csv_path, latest_csv]:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    print(f"Saved summary CSV: {csv_path}")
    print(f"Saved latest CSV: {latest_csv}")
    print(f"Saved summary JSON: {json_path}")
    print(f"Saved latest JSON: {latest_json}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run ICAISE training resource benchmark with wall-time, CPU RSS, and GPU memory logging.")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--manifest", default=None, help="Experiment JSON/YAML with methods entries: key, label, config, entrypoint.")
    ap.add_argument("--out-root", default="runs/icaise_training_resource_benchmark")
    ap.add_argument("--methods", default="", help="Comma-separated methods/keys. Default: all.")
    ap.add_argument("--tag", default="", help="Run tag. Default: UTC timestamp.")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--interval-s", type=float, default=5.0, help="Resource sampling interval.")
    ap.add_argument("--no-progress-bar", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dry-run", action="store_true", help="Print resolved configs and commands without training or writing files.")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_root = (repo_root / args.out_root).resolve()
    tag = str(args.tag).strip() or f"resource_{_utc_stamp()}"
    registry = _manifest_methods(repo_root / args.manifest) if args.manifest else METHODS
    methods = _selected_methods(str(args.methods), registry)
    rows: List[Dict[str, Any]] = []

    print(f"Benchmark tag: {tag}")
    print(f"Output root: {out_root}")
    print("Methods:", ", ".join(m.method for m in methods))
    for spec in methods:
        row = _run_one(
            spec=spec,
            repo_root=repo_root,
            out_root=out_root,
            tag=tag,
            python_bin=str(args.python),
            interval_s=float(args.interval_s),
            no_progress=bool(args.no_progress_bar),
            dry_run=bool(args.dry_run),
        )
        rows.append(row)
        if not args.dry_run:
            _write_outputs(out_root, rows, tag)
        if row.get("status") == "failed":
            raise SystemExit(f"{spec.method} failed; see {row.get('stderr_log')}")


if __name__ == "__main__":
    main()
