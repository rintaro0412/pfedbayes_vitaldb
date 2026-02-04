#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except Exception:
        return json.loads(text) or {}


def _dump_config(cfg: Dict[str, Any], path: Path) -> None:
    try:
        import yaml  # type: ignore

        text = yaml.safe_dump(cfg, sort_keys=False)
    except Exception:
        text = json.dumps(cfg, indent=2)
    path.write_text(text, encoding="utf-8")


def _set_by_path(cfg: Dict[str, Any], path: str, value: Any) -> None:
    keys = [k for k in str(path).split(".") if k]
    if not keys:
        raise ValueError("Empty path")
    cur: Dict[str, Any] = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _parse_value(text: str) -> Any:
    t = text.strip()
    if t.lower() in ("true", "false"):
        return t.lower() == "true"
    try:
        if re.match(r"^-?\\d+$", t):
            return int(t)
    except Exception:
        pass
    try:
        return float(t)
    except Exception:
        pass
    try:
        return json.loads(t)
    except Exception:
        return t


def _parse_grid(text: str) -> List[Tuple[str, List[Any]]]:
    """
    Example: "train.lr=1e-4,1e-3;train.kl_coeff=1e-5,1e-4;model.var_reduction_h=1,2"
    """
    out: List[Tuple[str, List[Any]]] = []
    if not text:
        return out
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid grid part (missing '='): {part}")
        key, vals = part.split("=", 1)
        val_list = [_parse_value(v) for v in vals.split(",") if v.strip()]
        if not val_list:
            raise ValueError(f"No values for grid key: {key}")
        out.append((key.strip(), val_list))
    return out


def _load_grid_from_json(path: Path) -> List[Tuple[str, List[Any]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("grid JSON must be an object of key -> list")
    out = []
    for k, v in obj.items():
        if not isinstance(v, list):
            raise ValueError(f"grid[{k}] must be a list")
        out.append((str(k), v))
    return out


def _parse_seeds(text: str) -> List[int]:
    if not text:
        return []
    return [int(s.strip()) for s in text.split(",") if s.strip()]


def _fmt_value(v: Any) -> str:
    if isinstance(v, float):
        av = abs(v)
        if (av != 0.0) and (av < 1e-2 or av >= 1e2):
            return f"{v:.0e}".replace("+", "")
        return f"{v:g}"
    if isinstance(v, bool):
        return "t" if v else "f"
    return str(v)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "", text.replace("/", "_"))


def _safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        f = float(v)
    except Exception:
        return default
    if math.isnan(f):
        return default
    return f


def _metric_sign(metric: str) -> int:
    metric = str(metric).lower()
    if metric in {"nll", "brier", "ece"}:
        return -1
    return 1


def _load_eval_value(run_dir: Path, *, mode: str, stat: str, metric: str) -> float:
    path = run_dir / "eval_modes.json"
    if not path.exists():
        return float("nan")
    data = json.loads(path.read_text(encoding="utf-8"))
    modes = data.get("modes", {}) if isinstance(data, dict) else {}
    rep = modes.get(mode, {})
    if stat == "overall":
        src = rep.get("overall", {})
    else:
        stats = rep.get("client_stats", {})
        src = stats.get(stat, {})
    return _safe_float(src.get(metric, float("nan")))


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Grid sweep for FedUAB.")
    ap.add_argument("--config", default="configs/feduab.yaml", help="Base config.")
    ap.add_argument("--grid", default=None, help="Grid spec 'a=1,2;b=0.1,0.3'.")
    ap.add_argument("--grid-json", default=None, help="Path to JSON with key -> list.")
    ap.add_argument("--set", action="append", default=[], help="Override as key=value (repeatable).")
    ap.add_argument("--rounds", type=int, default=None, help="Override train.rounds for all runs.")
    ap.add_argument("--seeds", default="0", help="Comma-separated seeds (default: 0).")
    ap.add_argument("--prefix", default="sweep", help="Run name prefix.")
    ap.add_argument("--out-root", default="runs/feduab_sweep", help="Root directory for sweep runs.")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if run_dir already has eval_modes.json.")
    ap.add_argument("--dry-run", action="store_true", help="Print planned runs without executing.")
    ap.add_argument("--no-progress-bar", action="store_true", help="Disable progress bars for sweep runs.")
    ap.add_argument("--mode", default="personalized_oracle", choices=["personalized_oracle", "global_idless", "ensemble_idless"])
    ap.add_argument("--stat", default="overall", choices=["overall", "macro", "min", "std"])
    ap.add_argument("--metric", default="auprc", help="Metric key in eval_modes.json")
    ap.add_argument("--out", default="runs/feduab_sweep/results.csv", help="Output CSV summary.")
    ap.add_argument("--out-best", default="runs/feduab_sweep/best.csv", help="Output CSV for best run per seed.")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    base_cfg = _load_config(cfg_path)

    grid: List[Tuple[str, List[Any]]] = []
    if args.grid_json:
        grid = _load_grid_from_json(Path(args.grid_json))
    elif args.grid:
        grid = _parse_grid(args.grid)

    if not grid:
        grid = [
            ("train.kl_coeff", [1e-5, 1e-4, 1e-3]),
        ]

    overrides: List[Tuple[str, Any]] = []
    for item in args.set:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got: {item}")
        k, v = item.split("=", 1)
        overrides.append((k.strip(), _parse_value(v)))

    seeds = _parse_seeds(args.seeds)
    if not seeds:
        raise SystemExit("No seeds specified.")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    best_rows: Dict[int, Dict[str, Any]] = {}

    keys = [k for k, _ in grid]
    values = [v for _, v in grid]

    for combo in product(*values):
        kv = dict(zip(keys, combo))
        tag = ",".join([f"{k.split('.')[-1]}={_fmt_value(v)}" for k, v in kv.items()])
        slug = _slug(tag)
        for seed in seeds:
            run_dir = out_root / slug / f"seed{seed}"
            if args.skip_existing and (run_dir / "eval_modes.json").exists():
                value = _load_eval_value(run_dir, mode=args.mode, stat=args.stat, metric=args.metric)
                score = _metric_sign(args.metric) * value
                row = {
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "mode": args.mode,
                    "stat": args.stat,
                    "metric": args.metric,
                    "value": value,
                    "score": score,
                    **kv,
                }
                rows.append(row)
                cur_best = best_rows.get(seed)
                if cur_best is None or float(row["score"]) > float(cur_best["score"]):
                    best_rows[seed] = row
                continue

            cfg = json.loads(json.dumps(base_cfg))
            for k, v in overrides:
                _set_by_path(cfg, k, v)
            for k, v in kv.items():
                _set_by_path(cfg, k, v)
            _set_by_path(cfg, "train.seed", seed)
            if args.rounds is not None:
                _set_by_path(cfg, "train.rounds", int(args.rounds))
            _set_by_path(cfg, "run.run_name", f"{args.prefix}_{slug}_seed{seed}")
            _set_by_path(cfg, "run.out_dir", str(out_root))

            run_dir.mkdir(parents=True, exist_ok=True)
            cfg_path_run = run_dir / "config_used.yaml"
            _dump_config(cfg, cfg_path_run)

            cmd = [
                "python",
                "bayes_federated/feduab_server.py",
                "--config",
                str(cfg_path_run),
                "--run-dir",
                str(run_dir),
            ]
            if args.no_progress_bar:
                cmd.append("--no-progress-bar")

            print(f"[RUN] seed={seed} {tag}")
            if args.dry_run:
                print(" ".join(cmd))
                continue

            t0 = time.time()
            subprocess.run(cmd, check=True)
            dt = time.time() - t0
            value = _load_eval_value(run_dir, mode=args.mode, stat=args.stat, metric=args.metric)
            score = _metric_sign(args.metric) * value
            row = {
                "seed": seed,
                "run_dir": str(run_dir),
                "mode": args.mode,
                "stat": args.stat,
                "metric": args.metric,
                "value": value,
                "score": score,
                "elapsed_sec": dt,
                **kv,
            }
            rows.append(row)
            cur_best = best_rows.get(seed)
            if cur_best is None or float(row["score"]) > float(cur_best["score"]):
                best_rows[seed] = row

    if args.dry_run:
        return

    _ensure_parent(Path(args.out))
    with Path(args.out).open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    if best_rows:
        out_best = Path(args.out_best)
        _ensure_parent(out_best)
        with out_best.open("w", encoding="utf-8", newline="") as f:
            fieldnames = sorted({k for r in best_rows.values() for k in r.keys()})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for seed in sorted(best_rows.keys()):
                writer.writerow(best_rows[seed])

    print(f"[DONE] results -> {args.out}")
    if best_rows:
        print(f"[DONE] best per seed -> {args.out_best}")


if __name__ == "__main__":
    main()
