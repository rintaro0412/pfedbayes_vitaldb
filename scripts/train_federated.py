#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from common.eval_summary import METRIC_KEYS, build_mode_report, write_eval_outputs
from common.io import ensure_dir, read_json


def _parse_seeds(text: str) -> List[int]:
    if not text:
        return []
    return [int(s.strip()) for s in str(text).split(",") if s.strip()]


def _parse_algos(text: str) -> List[str]:
    if not text:
        return []
    tokens = [t.strip().lower() for t in str(text).split(",") if t.strip()]
    if not tokens:
        return []
    if any(t in ("both", "all") for t in tokens):
        return ["fedavg", "feduab"]
    return tokens


def _strip_args(args: List[str], remove: Iterable[str]) -> List[str]:
    remove_set = set(remove)
    out: List[str] = []
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        key = a.split("=", 1)[0]
        if key in remove_set:
            if "=" not in a:
                skip_next = True
            continue
        out.append(a)
    return out


def _make_run_dir(out_root: Path, algo: str, seed: int, tag: str | None) -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    suffix = f"{ts}_{tag}" if tag else ts
    return out_root / algo / f"seed{seed}" / suffix


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _ensure_eval_modes_fedavg(run_dir: Path) -> None:
    out_path = run_dir / "eval_modes.json"
    if out_path.exists():
        return
    test_report = run_dir / "test_report.json"
    per_client_csv = run_dir / "test_report_per_client.csv"
    if not test_report.exists() or not per_client_csv.exists():
        raise FileNotFoundError("FedAvg outputs not found for eval_modes conversion.")

    rep = read_json(test_report)
    metrics_pre = rep.get("metrics_pre", {}) if isinstance(rep, dict) else {}
    overall = {k: metrics_pre.get(k, float("nan")) for k in ("n", "n_pos", "n_neg", *METRIC_KEYS)}

    per_client: Dict[str, Dict[str, Any]] = {}
    with per_client_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") not in (None, "", "ok"):
                continue
            cid = str(row.get("client_id", ""))
            if not cid:
                continue
            per_client[cid] = {
                "n": _safe_float(row.get("n", float("nan"))),
                "n_pos": _safe_float(row.get("n_pos", float("nan"))),
                "n_neg": _safe_float(row.get("n_neg", float("nan"))),
                "auroc": _safe_float(row.get("auroc_pre", float("nan"))),
                "auprc": _safe_float(row.get("auprc_pre", float("nan"))),
                "ece": _safe_float(row.get("ece_pre", float("nan"))),
                "nll": _safe_float(row.get("nll_pre", float("nan"))),
                "brier": _safe_float(row.get("brier_pre", float("nan"))),
            }

    # FedAvg has a single global model; reuse for all modes for format compatibility.
    modes = {
        "global_idless": build_mode_report(overall=overall, per_client=per_client),
        "personalized_oracle": build_mode_report(overall=overall, per_client=per_client, note="FedAvg uses global model"),
        "ensemble_idless": build_mode_report(overall=overall, per_client=per_client, note="FedAvg uses global model"),
    }
    write_eval_outputs(run_dir=run_dir, algo="fedavg", modes=modes)


def _load_eval_modes(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "eval_modes.json"
    if not path.exists():
        raise FileNotFoundError(f"eval_modes.json not found: {path}")
    return read_json(path)


def _aggregate_summary(run_dirs: List[Path], *, out_path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        payload = _load_eval_modes(run_dir)
        algo = str(payload.get("algo", run_dir.parent.name))
        modes = payload.get("modes", {}) if isinstance(payload, dict) else {}
        for mode, rep in modes.items():
            overall = rep.get("overall", {})
            stats = rep.get("client_stats", {})
            for stat_name, data in (
                ("overall", overall),
                ("macro", stats.get("macro", {})),
                ("min", stats.get("min", {})),
                ("std", stats.get("std", {})),
            ):
                for metric in METRIC_KEYS:
                    rows.append(
                        {
                            "algo": algo,
                            "mode": mode,
                            "stat": stat_name,
                            "metric": metric,
                            "value": _safe_float(data.get(metric, float("nan"))),
                            "run_dir": str(run_dir),
                        }
                    )

    grouped: Dict[tuple[str, str, str, str], List[float]] = {}
    for r in rows:
        key = (r["algo"], r["mode"], r["stat"], r["metric"])
        grouped.setdefault(key, []).append(float(r["value"]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algo", "mode", "stat", "metric", "mean", "std", "min", "max", "n_seeds"])
        for (algo, mode, stat, metric), vals in sorted(grouped.items()):
            arr = [v for v in vals if v == v]
            if not arr:
                writer.writerow([algo, mode, stat, metric, "nan", "nan", "nan", "nan", 0])
                continue
            mean = sum(arr) / len(arr)
            var = sum((v - mean) ** 2 for v in arr) / len(arr)
            std = var ** 0.5
            writer.writerow([algo, mode, stat, metric, mean, std, min(arr), max(arr), len(arr)])


def main() -> None:
    ap = argparse.ArgumentParser(description="Unified federated trainer (FedAvg/FedUAB) with multi-seed support.")
    ap.add_argument("--algo", default="fedavg", help="fedavg|feduab|both or comma-separated list")
    ap.add_argument("--seeds", default="0", help="Comma-separated seeds")
    ap.add_argument("--out-root", default="runs", help="Root directory for runs")
    ap.add_argument("--run-tag", default=None, help="Optional tag appended to run directory name")
    ap.add_argument("--summary-path", default="runs/summary.csv")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if run_dir already exists")
    args, unknown = ap.parse_known_args()

    algos = _parse_algos(args.algo)
    seeds = _parse_seeds(args.seeds)
    if not algos:
        raise SystemExit("No algo specified.")
    if not seeds:
        raise SystemExit("No seeds specified.")

    remove_flags = {"--seed", "--seeds", "--out-dir", "--run-name", "--run-dir"}
    pass_through = _strip_args(list(unknown), remove_flags)

    out_root = Path(args.out_root)
    fedavg_only_flags = {
        "--model-selection",
        "--selection-source",
        "--selection-metric",
        "--save-test-pred-npz",
        "--per-client-every-round",
        "--log-client-sim",
        "--cache-in-memory",
        "--max-cache-files",
        "--cache-dtype",
        "--use-lstm",
        "--lstm-hidden",
    }
    pass_through_feduab = _strip_args(pass_through, fedavg_only_flags)
    run_dirs: List[Path] = []
    for algo in algos:
        for seed in seeds:
            run_dir = _make_run_dir(out_root, algo, seed, args.run_tag)
            if args.skip_existing and run_dir.exists() and (run_dir / "meta.json").exists():
                print(f"[SKIP] {algo} seed{seed}: {run_dir}")
                run_dirs.append(run_dir)
                continue
            ensure_dir(run_dir)
            if algo == "fedavg":
                env = os.environ.copy()
                env["LEGACY_RUN_DIR"] = str(run_dir)
                cmd = [sys.executable, "federated/server.py"] + pass_through + ["--seed", str(seed)]
                print("[RUN]", " ".join(cmd))
                subprocess.run(cmd, check=True, env=env)
                _ensure_eval_modes_fedavg(run_dir)
            elif algo == "feduab":
                cmd = [sys.executable, "bayes_federated/feduab_server.py"] + pass_through_feduab + ["--seed", str(seed), "--run-dir", str(run_dir)]
                print("[RUN]", " ".join(cmd))
                subprocess.run(cmd, check=True)
            else:
                raise SystemExit(f"Unknown algo: {algo}")

            run_dirs.append(run_dir)

    _aggregate_summary(run_dirs, out_path=Path(args.summary_path))
    print(f"Summary: {args.summary_path}")


if __name__ == "__main__":
    main()
