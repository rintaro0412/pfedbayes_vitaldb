#!/usr/bin/env python3
"""Run a declared paper protocol through the existing training entrypoints.

Planning imports no training framework. --dry-run only reads configuration and
prints commands; it creates no directories and starts no child processes.
"""
from __future__ import annotations

import argparse
import copy
import fnmatch
import glob
import hashlib
import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGES = ("train", "evaluate", "aggregate", "figures", "resources")
ALL_STAGES = STAGES[:-1]  # Resource benchmarks are explicitly requested separately.
ENTRYPOINTS = {
    "centralized/train.py": "centralized",
    "scripts/train_local.py": "local",
    "federated/server.py": "federated",
    "bayes_federated/pfedbayes_server.py": "pfedbayes",
}
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


class ProtocolError(ValueError):
    pass


def read_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ProtocolError(f"Configuration not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise ProtocolError("YAML requires the existing PyYAML dependency; JSON is also supported.") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ProtocolError(f"Expected a mapping in {path}")
    return value


def merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def require_name(value: Any, where: str) -> str:
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise ProtocolError(f"{where} must use letters, digits, dots, underscores or hyphens.")
    return value


def parse_seeds(values: Any, where: str) -> list[int]:
    if not isinstance(values, list) or not values:
        raise ProtocolError(f"{where} must be a nonempty list of integer seeds.")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in values):
        raise ProtocolError(f"{where} seeds must be integers in [0, 2**32).")
    if len(set(values)) != len(values):
        raise ProtocolError(f"Duplicate seeds in {where}.")
    return values


def expand(value: str, context: dict[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in context:
            raise ProtocolError(f"Unknown or unavailable placeholder {{{key}}} in {value!r}")
        return context[key]
    return PLACEHOLDER.sub(replace, value)


def has_output(pattern: str) -> bool:
    return bool(glob.glob(pattern)) if glob.has_magic(pattern) else Path(pattern).exists()


@dataclass
class Run:
    method: dict[str, Any]
    seed: int
    config: dict[str, Any]
    run_dir: Path
    fingerprint: str
    context: dict[str, str]


@dataclass
class Task:
    stage: str
    name: str
    argv: list[str]
    outputs: list[str]
    fingerprint: str
    marker_dir: Path
    run: Run | None = None
    methods_path: Path | None = None
    methods_rows: list[dict[str, Any]] | None = None
    inputs: list[str] | None = None
    run_inputs: list[str] | None = None

    @property
    def complete_path(self) -> Path:
        return self.marker_dir / f"{self.stage}.{self.name}.complete.json"

    @property
    def started_path(self) -> Path:
        return self.marker_dir / f"{self.stage}.{self.name}.started.json"


def plan(manifest_path: Path, args: argparse.Namespace, root: Path = PROJECT_ROOT) -> list[Task]:
    if args.use_source_runs and args.stage in {"train", "evaluate", "all"}:
        raise ProtocolError("--use-source-runs is supported only for aggregate, figures and resources; historical runs stay read-only.")
    manifest = read_mapping(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ProtocolError("Only schema_version: 1 is supported.")
    protocol_id = require_name(manifest.get("protocol_id"), "protocol_id")
    methods = manifest.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ProtocolError("methods must be a nonempty list.")
    keys = [require_name(method.get("key"), "method.key") if isinstance(method, dict) else None for method in methods]
    if None in keys or len(set(keys)) != len(keys):
        raise ProtocolError("Methods must be mappings with unique keys.")
    chosen = {key.strip() for key in args.methods.split(",")} if args.methods else set(keys)
    if not chosen <= set(keys):
        raise ProtocolError(f"Unknown methods: {', '.join(sorted(chosen - set(keys)))}")
    seeds_override = None
    if args.seeds:
        try:
            seeds_override = parse_seeds([int(seed) for seed in args.seeds.split(",")], "--seeds")
        except ValueError as exc:
            raise ProtocolError("--seeds must be comma-separated nonnegative integers.") from exc
    data = manifest.get("data", {})
    if not isinstance(data, dict) or not isinstance(data.get("data_dir"), str):
        raise ProtocolError("data.data_dir must be a path string.")
    data_dir = resolve_path(args.data_dir or data["data_dir"], root)
    output_root = resolve_path(args.output_root or manifest.get("output_root", "runs/icaise2026"), root)
    protocol_dir = output_root / protocol_id
    common_overrides = manifest.get("overrides", {})
    if not isinstance(common_overrides, dict):
        raise ProtocolError("overrides must be a mapping.")
    context = {
        "python": sys.executable, "project_root": str(root), "protocol_dir": str(protocol_dir),
        "manifest_dir": str(manifest_path.parent),
        "data_dir": str(data_dir), "output_root": str(output_root), "protocol_id": protocol_id,
    }
    runs = []
    for method in methods:
        if method["key"] not in chosen:
            continue
        entrypoint = method.get("entrypoint")
        if entrypoint not in ENTRYPOINTS or not (root / entrypoint).is_file():
            raise ProtocolError(f"Unsupported or missing training entrypoint: {entrypoint}")
        if method.get("model_scope") not in {"global", "client_specific"}:
            raise ProtocolError(f"{method['key']}: model_scope must be global or client_specific.")
        config_path = method.get("config")
        if not isinstance(config_path, str):
            raise ProtocolError(f"{method['key']}: config must be a path string.")
        overrides = method.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ProtocolError(f"{method['key']}: overrides must be a mapping.")
        source_config = read_mapping(resolve_path(config_path, root))
        for seed in seeds_override or parse_seeds(method.get("seeds"), method["key"]):
            config = merge(merge(source_config, common_overrides), overrides)
            for section in ("run", "data", "train", "eval"):
                if not isinstance(config.setdefault(section, {}), dict):
                    raise ProtocolError(f"{method['key']}: config.{section} must be a mapping.")
            run_dir = protocol_dir / method["key"] / f"seed{seed}"
            if args.use_source_runs:
                source_run_dir = method.get("source_run_dir")
                if not isinstance(source_run_dir, str) or not source_run_dir:
                    raise ProtocolError(f"{method['key']}: --use-source-runs requires source_run_dir.")
                if seeds_override and seeds_override != method.get("seeds"):
                    raise ProtocolError("--use-source-runs cannot relabel historical seeds with --seeds.")
                if len(method.get("seeds", [])) != 1:
                    raise ProtocolError("source_run_dir currently represents one declared seed per method.")
                run_dir = resolve_path(expand(source_run_dir, dict(context, seed=str(seed))), root)
            config["run"].update(out_dir=str(run_dir.parent), run_name=run_dir.name, resume=False)
            if ENTRYPOINTS[entrypoint] == "federated":
                config["run"]["run_dir"] = str(run_dir)  # Also defeats LEGACY_RUN_DIR and enables resume.
            config["data"]["data_dir"] = str(data_dir)
            config["train"]["seed"] = seed
            run_context = dict(context, run_dir=str(run_dir), method=method["key"],
                               label=str(method.get("label", method["key"])), seed=str(seed))
            if method.get("source_run_dir"):
                run_context["source_run_dir"] = str(resolve_path(expand(method["source_run_dir"], dict(context, seed=str(seed))), root))
            fingerprint = digest({
                "protocol_id": protocol_id, "method": method["key"], "model_scope": method["model_scope"],
                "config": config, "entrypoint_sha256": file_digest(root / entrypoint),
                "dataset_summary_sha256": file_digest(data_dir / "summary.json"),
                "shared_code": {str(path.relative_to(root)): file_digest(path)
                                for directory in ("common", Path(entrypoint).parent.as_posix())
                                for path in sorted((root / directory).glob("*.py"))},
            })
            runs.append(Run(method, seed, config, run_dir, fingerprint, run_context))
    methods_rows = []
    for run in runs:
        kind = ENTRYPOINTS[run.method["entrypoint"]]
        methods_rows.append({
            "key": run.method["key"], "label": run.context["label"], "method": run.context["label"],
            "seed": run.seed, "run_dir": str(run.run_dir),
            "source_run_dir": run.context.get("source_run_dir"),
            "config": str(resolve_path(run.method["config"], root)) if args.use_source_runs
                      else str(run.run_dir / ".runner" / "resolved_config.json"),
            "model_scope": run.method["model_scope"],
            "curve_kind": run.method.get("curve_kind", "local_round_client" if kind == "local" else "history"),
            "inference_kind": run.method.get("inference_kind", {"local": "local_point", "pfedbayes": "bayes"}.get(kind, "point")),
            "mc_eval": int(run.method.get("mc_eval", run.config["train"].get("mc_eval", 25))),
        })
    plan_hash = digest({"protocol_id": protocol_id, "rows": methods_rows, "runs": [run.fingerprint for run in runs]})[:16]
    methods_path = protocol_dir / ".runner" / plan_hash / "methods.json"
    context["methods_json"] = str(methods_path)
    for run in runs:
        run.context["methods_json"] = str(methods_path)
    tasks = []
    selected_stages = ALL_STAGES if args.stage == "all" else (args.stage,)
    if "train" in selected_stages:
        for run in runs:
            kind = ENTRYPOINTS[run.method["entrypoint"]]
            required = {
                "centralized": ["checkpoints/model_last.pt", "history.csv", "run_config.json"],
                "local": ["checkpoints/client_*_last.pt", "selection_summary.json", "test_report_per_client.json"],
                "federated": ["checkpoints/model_last.pt", "test_report.json", "meta.json"],
                "pfedbayes": ["checkpoints/model_last.pt", "checkpoints/server_state.pt", "test_report.json", "summary.json"],
            }[kind]
            argv = [sys.executable, str(root / run.method["entrypoint"]), "--config",
                    str(run.run_dir / ".runner" / "resolved_config.json"), "--run-name", run.run_dir.name]
            if args.resume:
                argv.append("--resume")
            tasks.append(Task("train", run.method["key"], argv, [str(run.run_dir / path) for path in required],
                              run.fingerprint, run.run_dir / ".runner", run))
    stages = manifest.get("stages", {})
    if not isinstance(stages, dict):
        raise ProtocolError("stages must be a mapping of stage names to command lists.")
    for stage in selected_stages:
        if stage == "train":
            continue
        specs = stages.get(stage, [])
        if not isinstance(specs, list):
            raise ProtocolError(f"stages.{stage} must be a list.")
        if not specs:
            raise ProtocolError(f"No commands declared for stage {stage!r}.")
        for spec in specs:
            if not isinstance(spec, dict):
                raise ProtocolError(f"stages.{stage}: each command must be a mapping.")
            name = require_name(spec.get("name"), "command.name")
            argv = spec.get("argv")
            outputs = spec.get("outputs")
            if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) for arg in argv):
                raise ProtocolError(f"{stage}.{name}: argv must be a nonempty string list.")
            if not isinstance(outputs, list) or not outputs or any(not isinstance(path, str) for path in outputs):
                raise ProtocolError(f"{stage}.{name}: outputs must be a nonempty string list.")
            selected = spec.get("methods", keys)
            if not isinstance(selected, list) or any(key not in keys for key in selected):
                raise ProtocolError(f"{stage}.{name}: methods must list declared method keys.")
            scoped_runs = [run for run in runs if run.method["key"] in selected]
            if not scoped_runs:
                continue
            if stage in {"aggregate", "figures", "resources"} and not spec.get("for_each_run", False):
                run_counts = {key: sum(run.method["key"] == key for run in runs) for key in chosen}
                if any(count > 1 for count in run_counts.values()):
                    raise ProtocolError(
                        "Multiple seeds per method are not supported by combined downstream commands; "
                        "run downstream stages separately with --seeds <one>."
                    )
            for run in scoped_runs if spec.get("for_each_run", False) else [None]:
                task_context = run.context if run else dict(context, run_dirs=",".join(str(r.run_dir) for r in scoped_runs))
                command = [expand(arg, task_context) for arg in argv]
                output_paths = [str(resolve_path(expand(path, task_context), root)) for path in outputs]
                if any(not Path(path).is_relative_to(output_root) for path in output_paths):
                    raise ProtocolError(f"{stage}.{name}: outputs must stay under output_root ({output_root}).")
                if args.use_source_runs and any(Path(path).is_relative_to(r.run_dir) for path in output_paths for r in runs):
                    raise ProtocolError(f"{stage}.{name}: outputs cannot overwrite historical source runs.")
                input_paths = spec.get("inputs", [])
                if not isinstance(input_paths, list) or any(not isinstance(path, str) for path in input_paths):
                    raise ProtocolError(f"{stage}.{name}: inputs must be a string list.")
                inputs = [resolve_path(expand(path, task_context), root) for path in input_paths]
                task_name = f"{name}.{run.method['key']}.seed{run.seed}" if run else name
                fingerprint = digest({
                    "protocol_id": protocol_id, "argv": command, "outputs": output_paths,
                    "run_fingerprints": [r.fingerprint for r in ([run] if run else scoped_runs)],
                    "inputs": [str(path) for path in inputs],
                    "command_files": {arg: file_digest(resolve_path(arg, root)) for arg in command if Path(arg).suffix == ".py"},
                })
                run_inputs = [str(r.run_dir / filename) for r in ([run] if run else scoped_runs)
                              for filename in ("config.json", "run_config.json", "meta.json", "summary.json",
                                               "history.csv", "round_client_metrics.csv", "test_report.json",
                                               "test_report_per_client.csv", "global_reassessment.json",
                                               "test_report_personalized.json", ".runner/resolved_config.json")]
                # An evaluation's own reports are outputs, not prior inputs.
                run_inputs = [path for path in run_inputs
                              if not any(fnmatch.fnmatch(path, output) for output in output_paths)]
                tasks.append(Task(stage, task_name, command, output_paths, fingerprint,
                                  protocol_dir / ".runner", run,
                                  inputs=[str(path) for path in inputs], run_inputs=run_inputs))
    if len({str(task.complete_path) for task in tasks}) != len(tasks):
        raise ProtocolError("Duplicate stage command names produce the same completion marker.")
    for task in tasks:
        task.methods_path = methods_path
        task.methods_rows = methods_rows
    return tasks


def preflight_train(run: Run) -> None:
    config = run.config
    data_dir = Path(config["data"]["data_dir"])
    if not data_dir.is_dir():
        raise ProtocolError(f"Training data directory not found: {data_dir}")
    kind = ENTRYPOINTS[run.method["entrypoint"]]
    if config["eval"].get("selection_source", "val") != "val":
        raise ProtocolError("The paper protocol requires validation-based model selection.")
    selection = config["eval"].get("model_selection", "best")
    val_flag = "val_every_epoch" if kind == "centralized" else "val_every_round"
    if selection == "best" and not config["train"].get(val_flag, True):
        raise ProtocolError(f"{run.method['key']}: best selection requires train.{val_flag}=true.")
    split_dirs = {}
    for split in ("train", "val", "test"):
        split_name = str(config["data"].get(f"{split}_split", split))
        dirs = [path for path in data_dir.glob(f"*/{split_name}") if path.is_dir()]
        pooled = data_dir / split_name
        if pooled.is_dir():
            dirs.append(pooled)
        valid = [path for path in dirs if any(path.glob("*.npz")) or (path / "contiguous/manifest.json").is_file()]
        if not valid:
            raise ProtocolError(f"{run.method['key']}: no {split} data found for split {split_name!r}.")
        split_dirs[split] = valid
    if kind == "local":
        train_clients = {path.parent.name for path in split_dirs["train"] if path.parent != data_dir}
        val_clients = {path.parent.name for path in split_dirs["val"] if path.parent != data_dir}
        missing = train_clients - val_clients
        if missing:
            raise ProtocolError(f"Local clients lack validation data: {', '.join(sorted(missing))}")


def read_marker(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def write_record(path: Path, record: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def is_complete(task: Task) -> bool:
    return (read_marker(task.complete_path).get("fingerprint") == task.fingerprint
            and all(has_output(path) for path in task.outputs))


def preflight_train_state(task: Task, resume: bool) -> None:
    if is_complete(task):
        return
    run = task.run
    previous = read_marker(task.started_path)
    if run.run_dir.exists() and any(run.run_dir.iterdir()):
        if previous.get("fingerprint") != task.fingerprint:
            raise ProtocolError(f"Existing run has no matching runner record: {run.run_dir}. Use a new --output-root.")
        if not resume:
            raise ProtocolError(f"Incomplete run requires explicit --resume: {run.run_dir}")
    if resume:
        kind = ENTRYPOINTS[run.method["entrypoint"]]
        required = "server_state.pt" if kind == "pfedbayes" else "model_last.pt"
        checkpoint = run.run_dir / "checkpoints" / required
        if kind == "local":
            train_split = run.config["data"].get("train_split", "train")
            clients = [p.parent.name for p in Path(run.config["data"]["data_dir"]).glob(f"*/{train_split}") if p.is_dir()]
            paths = [run.run_dir / "checkpoints" / f"client_{cid}_last.pt" for cid in clients]
            if not paths or any(not path.is_file() for path in paths):
                raise ProtocolError(f"Local resume requires every client's last checkpoint: {run.run_dir}")
        elif not checkpoint.is_file():
            raise ProtocolError(f"Resume checkpoint not found: {checkpoint}")


def execute(tasks: list[Task], args: argparse.Namespace, root: Path = PROJECT_ROOT) -> None:
    if not tasks:
        print("[SKIP] No commands declared for the selected methods and stage.")
        return
    if args.dry_run:
        for task in tasks:
            print(f"[PLAN] {task.stage}/{task.name}: {shlex.join(task.argv)}")
        return
    # Validate every selected training run before writing or starting any of them.
    for task in tasks:
        if task.stage == "train":
            preflight_train(task.run)
            preflight_train_state(task, args.resume)
    if args.use_source_runs:
        for row in tasks[0].methods_rows if tasks else []:
            if not Path(row["run_dir"]).is_dir():
                raise ProtocolError(f"Historical source run not found: {row['run_dir']}")
    if tasks:
        write_record(tasks[0].methods_path, tasks[0].methods_rows)
    for task in tasks:
        fingerprint_inputs = (task.inputs or []) + (task.run_inputs or [])
        if fingerprint_inputs:
            task.fingerprint = digest({"plan_fingerprint": task.fingerprint,
                                       "input_sha256": {path: file_digest(Path(path)) for path in fingerprint_inputs}})
        if is_complete(task):
            print(f"[SKIP] {task.stage}/{task.name}: matching completed run")
            continue
        if task.stage == "train":
            run = task.run
            write_record(run.run_dir / ".runner" / "resolved_config.json", run.config)
        missing_inputs = [path for path in task.inputs or [] if not has_output(path)]
        if missing_inputs:
            raise ProtocolError(f"Required stage inputs are absent: {', '.join(missing_inputs)}")
        record = {"stage": task.stage, "name": task.name, "fingerprint": task.fingerprint,
                  "argv": task.argv, "outputs": task.outputs, "started_utc": datetime.now(timezone.utc).isoformat()}
        write_record(task.started_path, record)
        print(f"[RUN] {task.stage}/{task.name}: {shlex.join(task.argv)}", flush=True)
        subprocess.run(task.argv, cwd=root, check=True)
        missing = [path for path in task.outputs if not has_output(path)]
        if missing:
            raise ProtocolError(f"Command exited successfully but required outputs are absent: {', '.join(missing)}")
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_record(task.complete_path, record)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="configs/icaise2026/experiments.yaml")
    ap.add_argument("--methods", help="Comma-separated declared method keys; default: all.")
    ap.add_argument("--seeds", help="Comma-separated seed override for every selected method.")
    ap.add_argument("--stage", choices=(*STAGES, "all"), default="train")
    ap.add_argument("--data-dir", help="Override manifest data.data_dir.")
    ap.add_argument("--output-root", help="Override manifest output_root; use a new root for changed settings.")
    ap.add_argument("--resume", action="store_true", help="Resume matching incomplete training runs with their existing checkpoints.")
    ap.add_argument("--use-source-runs", action="store_true", help="Read historical source_run_dir for aggregate/figures/resources; never train or evaluate into those directories.")
    ap.add_argument("--dry-run", action="store_true", help="Print planned commands without writing or running subprocesses.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        manifest_path = resolve_path(args.manifest, PROJECT_ROOT)
        tasks = plan(manifest_path, args)
        execute(tasks, args)
    except (ProtocolError, OSError, json.JSONDecodeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
