#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)


def _dump_yaml(cfg: Dict[str, Any], path: Path) -> None:
    try:
        import yaml  # type: ignore

        text = yaml.safe_dump(cfg, sort_keys=False)
    except Exception:
        text = json.dumps(cfg, indent=2)
    path.write_text(text, encoding="utf-8")


def _set_by_path(cfg: Dict[str, Any], path: str, value: Any) -> None:
    keys = [k for k in str(path).split(".") if k]
    cur: Dict[str, Any] = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _run(cmd: str) -> None:
    print(f"[cmd] {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    import csv

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    if math.isnan(v):
        return default
    return v


def _best_row_by_metric(rows: List[Dict[str, str]], key: str) -> Optional[Dict[str, str]]:
    if not rows:
        return None
    return max(rows, key=lambda r: _to_float(r.get(key), float("-inf")))


def _central_metrics(run_dir: Path) -> Dict[str, float]:
    rows = _read_csv_rows(run_dir / "history.csv")
    best = _best_row_by_metric(rows, "val_auroc")
    if not best:
        return {}
    return {
        "val_auroc": _to_float(best.get("val_auroc")),
        "val_auprc": _to_float(best.get("val_auprc")),
        "val_ece": _to_float(best.get("val_ece")),
        "val_nll": _to_float(best.get("val_nll")),
    }


def _fedavg_metrics(run_dir: Path) -> Dict[str, float]:
    rows = _read_csv_rows(run_dir / "history.csv")
    best = _best_row_by_metric(rows, "val_auroc")
    if not best:
        return {}
    return {
        "val_auroc": _to_float(best.get("val_auroc")),
        "val_auprc": _to_float(best.get("val_auprc")),
        "val_ece": _to_float(best.get("val_ece")),
        "val_nll": _to_float(best.get("val_nll")),
    }


def _pfedbayes_metrics(run_dir: Path) -> Dict[str, float]:
    rows = _read_csv_rows(run_dir / "history.csv")
    best = _best_row_by_metric(rows, "val_auprc")
    if not best:
        return {}
    try:
        best_round = int(float(best.get("round", 0) or 0))
    except Exception:
        best_round = 0
    val_post = run_dir / f"round_{best_round:03d}_val_post.json"
    if not val_post.exists():
        return {}
    try:
        rep = json.loads(val_post.read_text(encoding="utf-8"))
    except Exception:
        return {}
    metrics = rep.get("metrics_post") or rep.get("metrics_pre") or {}
    return {
        "val_auroc": _to_float(metrics.get("auroc")),
        "val_auprc": _to_float(metrics.get("auprc")),
        "val_ece": _to_float(metrics.get("ece")),
        "val_nll": _to_float(metrics.get("nll")),
    }


def _mean(vals: List[float]) -> float:
    good = [v for v in vals if not math.isnan(v)]
    if not good:
        return float("nan")
    return float(sum(good) / len(good))


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch size sweep for Central/FedAvg/pFedBayes.")
    ap.add_argument("--data-dir", default="federated_data")
    ap.add_argument("--config", default="configs/pfedbayes.yaml")
    ap.add_argument("--batch-sizes", default="64,128,256,512")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--python", default=None, help="Python executable (default: ./venv/bin/python if exists).")
    ap.add_argument("--out", default="outputs/batch_size_sweep.csv")
    args = ap.parse_args()

    data_dir = args.data_dir
    batch_sizes = [int(s.strip()) for s in str(args.batch_sizes).split(",") if s.strip()]
    seed = int(args.seed)
    py_exec = args.python
    if not py_exec:
        venv_py = Path("venv") / "bin" / "python"
        py_exec = str(venv_py) if venv_py.exists() else "python"

    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()

    base_cfg = _load_yaml(Path(args.config))

    rows_out: List[Dict[str, Any]] = []

    with out_csv.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                [
                    "batch_size",
                    "central_val_auroc",
                    "central_val_auprc",
                    "central_val_ece",
                    "central_val_nll",
                    "fedavg_val_auroc",
                    "fedavg_val_auprc",
                    "fedavg_val_ece",
                    "fedavg_val_nll",
                    "pfedbayes_val_auroc",
                    "pfedbayes_val_auprc",
                    "pfedbayes_val_ece",
                    "pfedbayes_val_nll",
                    "avg_val_auroc",
                    "avg_val_ece",
                    "avg_val_nll",
                ]
            )

        for bs in batch_sizes:
            print(f"=== batch_size={bs} seed={seed} ===")
            central_dir = Path(f"runs/centralized_bs{bs}") / f"seed{seed}"
            fedavg_dir = Path(f"runs/fedavg_bs{bs}") / f"seed{seed}"
            pfb_dir = Path(f"runs/pfedbayes_bs{bs}") / f"seed{seed}"

            start = time.time()
            failed = False
            # Central
            if not (central_dir / "history.csv").exists() or not args.skip_existing:
                cmd = (
                    f"{py_exec} centralized/train.py "
                    f"--data-dir {data_dir} --out-dir runs/centralized_bs{bs} --run-name seed{seed} "
                    f"--epochs {int(args.epochs)} --batch-size {bs} --seed {seed} "
                    f"--num-workers {int(args.num_workers)} --train-split train --val-split val --test-split test"
                )
                try:
                    _run(cmd)
                except subprocess.CalledProcessError:
                    print(f"[fail] central batch_size={bs}")
                    failed = True

            # FedAvg
            if not (fedavg_dir / "history.csv").exists() or not args.skip_existing:
                cmd = (
                    f"{py_exec} federated/server.py "
                    f"--data-dir {data_dir} --out-dir runs/fedavg_bs{bs} --run-name seed{seed} "
                    f"--rounds {int(args.rounds)} --batch-size {bs} --seed {seed} "
                    f"--num-workers {int(args.num_workers)} --train-split train --val-split val --test-split test "
                    "--test-every-round"
                )
                try:
                    _run(cmd)
                except subprocess.CalledProcessError:
                    print(f"[fail] fedavg batch_size={bs}")
                    failed = True

            # pFedBayes
            if not (pfb_dir / "history.csv").exists() or not args.skip_existing:
                cfg = json.loads(json.dumps(base_cfg))
                _set_by_path(cfg, "run.out_dir", f"runs/pfedbayes_bs{bs}")
                _set_by_path(cfg, "run.run_name", f"seed{seed}")
                _set_by_path(cfg, "seed", seed)
                _set_by_path(cfg, "train.batch_size", bs)
                _set_by_path(cfg, "train.rounds", int(args.rounds))
                _set_by_path(cfg, "train.num_workers", int(args.num_workers))
                tmp_cfg_dir = Path(f"runs/pfedbayes_bs{bs}") / "_sweep_configs"
                tmp_cfg_dir.mkdir(parents=True, exist_ok=True)
                tmp_cfg_path = tmp_cfg_dir / f"seed{seed}.yaml"
                _dump_yaml(cfg, tmp_cfg_path)
                cmd = f"{py_exec} bayes_federated/pfedbayes_server.py --config {tmp_cfg_path} --run-name seed{seed}"
                try:
                    _run(cmd)
                except subprocess.CalledProcessError:
                    print(f"[fail] pfedbayes batch_size={bs}")
                    failed = True

            central = _central_metrics(central_dir)
            fedavg = _fedavg_metrics(fedavg_dir)
            pfb = _pfedbayes_metrics(pfb_dir)

            avg_auroc = _mean([central.get("val_auroc", float("nan")), fedavg.get("val_auroc", float("nan")), pfb.get("val_auroc", float("nan"))])
            avg_ece = _mean([central.get("val_ece", float("nan")), fedavg.get("val_ece", float("nan")), pfb.get("val_ece", float("nan"))])
            avg_nll = _mean([central.get("val_nll", float("nan")), fedavg.get("val_nll", float("nan")), pfb.get("val_nll", float("nan"))])

            row = {
                "batch_size": bs,
                "central": central,
                "fedavg": fedavg,
                "pfedbayes": pfb,
                "avg_val_auroc": avg_auroc,
                "avg_val_ece": avg_ece,
                "avg_val_nll": avg_nll,
                "elapsed_sec": time.time() - start,
                "failed": failed,
            }
            rows_out.append(row)

            writer.writerow(
                [
                    bs,
                    central.get("val_auroc"),
                    central.get("val_auprc"),
                    central.get("val_ece"),
                    central.get("val_nll"),
                    fedavg.get("val_auroc"),
                    fedavg.get("val_auprc"),
                    fedavg.get("val_ece"),
                    fedavg.get("val_nll"),
                    pfb.get("val_auroc"),
                    pfb.get("val_auprc"),
                    pfb.get("val_ece"),
                    pfb.get("val_nll"),
                    avg_auroc,
                    avg_ece,
                    avg_nll,
                ]
            )
            f.flush()

    # pick best by AUROC desc, ECE asc, NLL asc
    def _rank(r: Dict[str, Any]) -> Tuple[float, float, float]:
        auroc = r.get("avg_val_auroc", float("-inf"))
        ece = r.get("avg_val_ece", float("inf"))
        nll = r.get("avg_val_nll", float("inf"))
        return (-auroc, ece, nll)

    valid_rows = [r for r in rows_out if not r.get("failed") and not math.isnan(r.get("avg_val_auroc", float("nan")))]
    best = min(valid_rows, key=_rank) if valid_rows else None
    if best:
        print(
            f"[best] batch_size={best['batch_size']} avg_auroc={best['avg_val_auroc']} avg_ece={best['avg_val_ece']} avg_nll={best['avg_val_nll']}"
        )


if __name__ == "__main__":
    main()
