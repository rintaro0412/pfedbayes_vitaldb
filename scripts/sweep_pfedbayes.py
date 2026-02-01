#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
import math
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)


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
    Example: "pfedbayes.zeta=1e-6,1e-5;pfedbayes.server_beta=0.3,0.5;train.lr_w=1e-4,1e-3"
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


def _history_best(history_path: Path) -> Dict[str, Any]:
    if not history_path.exists():
        return {}
    with history_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    best = max(rows, key=lambda r: float(r.get("val_auprc", "-inf")))
    out = {k: best.get(k) for k in best.keys()}
    return out


def _read_val_post(run_dir: Path, best_round: int) -> Dict[str, Any]:
    path = run_dir / f"round_{best_round:03d}_val_post.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _safe_float(v: Any, default: float) -> float:
    try:
        f = float(v)
    except Exception:
        return default
    if math.isnan(f):
        return default
    return f


def _rank_key(row: Dict[str, Any]) -> Tuple[float, float, float, float]:
    # Primary: AUROC (desc), Secondary: ECE (asc), Tertiary: NLL (asc), Quaternary: AUPRC (desc)
    auroc = _safe_float(row.get("auroc_post"), float("-inf"))
    ece = _safe_float(row.get("ece_post"), float("inf"))
    nll = _safe_float(row.get("nll_post"), float("inf"))
    auprc = _safe_float(row.get("auprc_post"), float("-inf"))
    return (-auroc, ece, nll, -auprc)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Grid sweep for pFedBayes.")
    ap.add_argument("--config", default="configs/pfedbayes.yaml", help="Base config.")
    ap.add_argument("--grid", default=None, help="Grid spec 'a=1,2;b=0.1,0.3'.")
    ap.add_argument("--grid-json", default=None, help="Path to JSON with key -> list.")
    ap.add_argument("--set", action="append", default=[], help="Override as key=value (repeatable).")
    ap.add_argument("--rounds", type=int, default=None, help="Override train.rounds for all runs.")
    ap.add_argument("--device", default=None, help="Override device for all runs.")
    ap.add_argument("--seeds", default="0", help="Comma-separated seeds (default: 0).")
    ap.add_argument("--prefix", default="sweep_", help="Run name prefix.")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if run_dir already has summary.json.")
    ap.add_argument("--dry-run", action="store_true", help="Print planned runs without executing.")
    ap.add_argument("--cmd", default=None, help="Override command template.")
    ap.add_argument("--out", default="runs/pfedbayes/sweep_results.csv", help="Output CSV summary.")
    ap.add_argument("--out-best", default="runs/pfedbayes/sweep_best.csv", help="Output CSV for best run per seed.")
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
            ("pfedbayes.zeta", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
            ("pfedbayes.server_beta", [0.3, 0.5, 0.7, 1.0]),
            ("train.lr_w", [1e-4, 3e-4, 1e-3]),
        ]

    overrides: List[Tuple[str, Any]] = []
    for s in args.set:
        if "=" not in s:
            raise ValueError(f"--set expects key=value, got: {s}")
        k, v = s.split("=", 1)
        overrides.append((k.strip(), _parse_value(v)))

    seeds = _parse_seeds(args.seeds)
    if not seeds:
        seeds = [0]

    grid_keys = [k for k, _ in grid]
    grid_values = [v for _, v in grid]
    combos = list(product(*grid_values))

    out_csv = Path(args.out)
    _ensure_parent(out_csv)
    write_header = not out_csv.exists()

    cmd_tpl = args.cmd or "python bayes_federated/pfedbayes_server.py --config {config}"

    results: List[Dict[str, Any]] = []
    with out_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "run_name",
                    "seed",
                    "best_round",
                    "val_auprc",
                    "auprc_post",
                    "auroc_post",
                    "ece_post",
                    "nll_post",
                    "brier_post",
                    "temperature",
                ]
                + grid_keys
            )

        for idx, combo in enumerate(combos, start=1):
            params = dict(zip(grid_keys, combo))
            for seed in seeds:
                cfg = json.loads(json.dumps(base_cfg))
                for k, v in overrides:
                    _set_by_path(cfg, k, v)
                for k, v in params.items():
                    _set_by_path(cfg, k, v)
                if args.rounds is not None:
                    _set_by_path(cfg, "train.rounds", int(args.rounds))
                if args.device is not None:
                    _set_by_path(cfg, "device", str(args.device))
                _set_by_path(cfg, "seed", int(seed))

                short_parts = []
                for k, v in params.items():
                    short_key = k.split(".")[-1]
                    short_parts.append(f"{short_key}{_fmt_value(v)}")
                name = f"{args.prefix}{idx:03d}_" + "_".join(short_parts) + f"_seed{seed}"
                name = _slug(name)
                _set_by_path(cfg, "run.run_name", name)

                out_dir = Path(cfg["run"]["out_dir"])
                run_dir = out_dir / name
                if args.skip_existing and (run_dir / "summary.json").exists():
                    print(f"[skip] {run_dir} exists")
                    if args.dry_run:
                        continue
                    best = _history_best(run_dir / "history.csv")
                    best_round = int(float(best.get("round", 0) or 0))
                    val_auprc = float(best.get("val_auprc", "nan"))
                    temp = float(best.get("temperature", "nan"))
                    val_post = _read_val_post(run_dir, best_round)
                    metrics = val_post.get("metrics_post") or val_post.get("metrics_pre") or {}
                    auprc_post = float(metrics.get("auprc", "nan"))
                    auroc_post = float(metrics.get("auroc", "nan"))
                    ece_post = float(metrics.get("ece", "nan"))
                    nll_post = float(metrics.get("nll", "nan"))
                    brier_post = float(metrics.get("brier", "nan"))
                    row = {
                        "run_name": name,
                        "seed": int(seed),
                        "best_round": int(best_round),
                        "val_auprc": val_auprc,
                        "auprc_post": auprc_post,
                        "auroc_post": auroc_post,
                        "ece_post": ece_post,
                        "nll_post": nll_post,
                        "brier_post": brier_post,
                        "temperature": temp,
                        **params,
                    }
                    results.append(row)
                    writer.writerow(
                        [
                            name,
                            int(seed),
                            int(best_round),
                            val_auprc,
                            auprc_post,
                            auroc_post,
                            ece_post,
                            nll_post,
                            brier_post,
                            temp,
                        ]
                        + [params[k] for k in grid_keys]
                    )
                    f.flush()
                    continue

                tmp_cfg_dir = out_dir / "_sweep_configs"
                tmp_cfg_dir.mkdir(parents=True, exist_ok=True)
                tmp_cfg_path = tmp_cfg_dir / f"{name}.yaml"
                _dump_config(cfg, tmp_cfg_path)

                print(f"[run] {name}")
                if args.dry_run:
                    continue

                cmd = cmd_tpl.format(config=str(tmp_cfg_path))
                start = time.time()
                subprocess.run(cmd, shell=True, check=True)
                elapsed = time.time() - start

                best = _history_best(run_dir / "history.csv")
                best_round = int(float(best.get("round", 0) or 0))
                val_auprc = float(best.get("val_auprc", "nan"))
                temp = float(best.get("temperature", "nan"))
                val_post = _read_val_post(run_dir, best_round)
                metrics = val_post.get("metrics_post") or val_post.get("metrics_pre") or {}
                auprc_post = float(metrics.get("auprc", "nan"))
                auroc_post = float(metrics.get("auroc", "nan"))
                ece_post = float(metrics.get("ece", "nan"))
                nll_post = float(metrics.get("nll", "nan"))
                brier_post = float(metrics.get("brier", "nan"))

                row = {
                    "run_name": name,
                    "seed": int(seed),
                    "best_round": int(best_round),
                    "val_auprc": val_auprc,
                    "auprc_post": auprc_post,
                    "auroc_post": auroc_post,
                    "ece_post": ece_post,
                    "nll_post": nll_post,
                    "brier_post": brier_post,
                    "temperature": temp,
                    **params,
                }
                results.append(row)
                writer.writerow(
                    [
                        name,
                        int(seed),
                        int(best_round),
                        val_auprc,
                        auprc_post,
                        auroc_post,
                        ece_post,
                        nll_post,
                        brier_post,
                        temp,
                    ]
                    + [params[k] for k in grid_keys]
                )
                f.flush()
                print(f"[done] {name} elapsed={elapsed:.1f}s val_auprc={val_auprc:.4f}")

    if results and args.out_best:
        best_by_seed: Dict[int, Dict[str, Any]] = {}
        for row in results:
            seed = int(row.get("seed", 0))
            if seed not in best_by_seed or _rank_key(row) < _rank_key(best_by_seed[seed]):
                best_by_seed[seed] = row
        out_best = Path(args.out_best)
        _ensure_parent(out_best)
        with out_best.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "run_name",
                    "seed",
                    "best_round",
                    "auroc_post",
                    "ece_post",
                    "nll_post",
                    "auprc_post",
                    "temperature",
                ]
                + grid_keys
            )
            for seed in sorted(best_by_seed.keys()):
                row = best_by_seed[seed]
                writer.writerow(
                    [
                        row.get("run_name"),
                        row.get("seed"),
                        row.get("best_round"),
                        row.get("auroc_post"),
                        row.get("ece_post"),
                        row.get("nll_post"),
                        row.get("auprc_post"),
                        row.get("temperature"),
                    ]
                    + [row.get(k) for k in grid_keys]
                )
        for seed in sorted(best_by_seed.keys()):
            row = best_by_seed[seed]
            print(
                f"[best] seed={seed} run={row.get('run_name')} auroc={row.get('auroc_post')} ece={row.get('ece_post')} nll={row.get('nll_post')}"
            )


if __name__ == "__main__":
    main()
