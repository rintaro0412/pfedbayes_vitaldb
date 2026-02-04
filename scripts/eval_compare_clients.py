from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root in sys.path when executed as `python scripts/eval_compare_clients.py`
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bayes_federated.models import BFLModel
from bayes_federated.eval import mc_predict
from common.calibration import fit_temperature
from common.dataset import WindowedNPZDataset, list_client_ids, list_npz_files
from common.io import now_utc_iso, read_json, write_json
from common.checkpoint import load_checkpoint
from common.ioh_model import IOHModelConfig, IOHNet, normalize_model_cfg
from common.metrics import BinaryMetrics, compute_binary_metrics


def _load_summary_min_cases(data_dir: str, default: int) -> int:
    p = Path(data_dir) / "summary.json"
    if not p.exists():
        return int(default)
    try:
        js = read_json(p)
        return int(js.get("min_client_cases", default))
    except Exception:
        return int(default)


def _list_client_files(data_dir: str, split: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for cid in list_client_ids(data_dir):
        files = list_npz_files(data_dir, split, client_id=str(cid))
        if files:
            out[str(cid)] = files
    return out


def _load_fedavg_model(run_dir: Path, device: torch.device) -> IOHNet:
    ckpt_path = run_dir / "checkpoints" / "model_best.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "model_last.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"FedAvg checkpoint not found under {run_dir}/checkpoints")
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    model = IOHNet(normalize_model_cfg(ckpt.get("model_cfg", {})))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(device)


def _load_bfl_model(run_dir: Path, device: torch.device) -> BFLModel:
    ckpt_path = run_dir / "checkpoints" / "model_best.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "model_last.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"BFL checkpoint not found under {run_dir}/checkpoints")
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    cfg = ckpt["model_cfg"]
    prior_sigma = 0.1
    logvar_min = -12.0
    logvar_max = 6.0
    full_bayes = bool(ckpt.get("full_bayes", False))
    cfg_path = run_dir / "config.json"
    if cfg_path.exists():
        try:
            cfg_json = read_json(cfg_path)
            bayes = cfg_json.get("bayes", {}) if isinstance(cfg_json, dict) else {}
            prior_sigma = float(bayes.get("prior_sigma", prior_sigma))
            logvar_min = float(bayes.get("logvar_min", logvar_min))
            logvar_max = float(bayes.get("logvar_max", logvar_max))
            full_bayes = bool(bayes.get("full_bayes", full_bayes))
        except Exception:
            pass
    model = BFLModel(
        normalize_model_cfg(cfg),
        prior_sigma=float(prior_sigma),
        logvar_min=float(logvar_min),
        logvar_max=float(logvar_max),
        full_bayes=bool(full_bayes),
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(device)


def _dl(files: List[str], batch_size: int, num_workers: int) -> Tuple[WindowedNPZDataset, DataLoader]:
    ds = WindowedNPZDataset(files, use_clin="true", cache_in_memory=False, max_cache_files=32, cache_dtype="float32")
    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    return ds, dl


@torch.no_grad()
def _predict_fedavg(model: IOHNet, dl: DataLoader, *, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    for x, y in dl:
        if isinstance(x, (tuple, list)):
            x = tuple(t.to(device, non_blocking=True) for t in x)
        else:
            x = x.to(device, non_blocking=True)
        logits = model(x).detach().cpu().view(-1).numpy()
        logits_all.append(logits)
        y_all.append(y.detach().cpu().view(-1).numpy())
    return np.concatenate(logits_all, axis=0), np.concatenate(y_all, axis=0)


def _fit_temperature_from_val_fedavg(model: IOHNet, files: List[str], *, batch_size: int, num_workers: int, device: torch.device) -> float:
    ds, dl = _dl(files, batch_size=batch_size, num_workers=num_workers)
    logits, y = _predict_fedavg(model, dl, device=device)
    res = fit_temperature(logits, y, device=str(device))
    return float(res.temperature)


def _fit_temperature_from_val_bfl(model: BFLModel, files: List[str], *, batch_size: int, num_workers: int, mc_eval: int, device: torch.device) -> float:
    _, dl = _dl(files, batch_size=batch_size, num_workers=num_workers)
    pred = mc_predict(model, dl, mc_eval=int(mc_eval), device=device, return_y=True)
    logits_mean = pred["logits_mean"]
    y = pred["y_true"]
    res = fit_temperature(logits_mean, y, device=str(device))
    return float(res.temperature)


def _eval_clients_fedavg(
    model: IOHNet,
    client_files: Dict[str, List[str]],
    *,
    temperature: float,
    batch_size: int,
    num_workers: int,
    n_bins: int,
    device: torch.device,
    keep_arrays: bool,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cid, files in client_files.items():
        ds, dl = _dl(files, batch_size=batch_size, num_workers=num_workers)
        logits, y = _predict_fedavg(model, dl, device=device)
        prob = 1.0 / (1.0 + np.exp(-logits / float(temperature)))
        metrics = compute_binary_metrics(y, prob, n_bins=int(n_bins))
        entry: Dict[str, Any] = {"metrics": asdict(metrics)}
        if keep_arrays:
            entry["y_true"] = y.astype(np.int64, copy=False)
            entry["prob"] = prob.astype(np.float64, copy=False)
        out[str(cid)] = entry
    return out


def _eval_clients_bfl(
    model: BFLModel,
    client_files: Dict[str, List[str]],
    *,
    temperature: float,
    batch_size: int,
    num_workers: int,
    n_bins: int,
    mc_eval: int,
    device: torch.device,
    keep_arrays: bool,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cid, files in client_files.items():
        ds, dl = _dl(files, batch_size=batch_size, num_workers=num_workers)
        pred = mc_predict(model, dl, mc_eval=int(mc_eval), device=device, temperature=float(temperature), return_y=True)
        y = pred["y_true"]
        prob = pred.get("prob_mean_cal") if pred.get("prob_mean_cal") is not None else pred["prob_mean"]
        metrics = compute_binary_metrics(y, prob, n_bins=int(n_bins))
        entry: Dict[str, Any] = {"metrics": asdict(metrics)}
        if keep_arrays:
            entry["y_true"] = y.astype(np.int64, copy=False)
            entry["prob"] = prob.astype(np.float64, copy=False)
        out[str(cid)] = entry
    return out


def _macro_mean(metrics_by_client: Dict[str, Dict[str, Any]], eligible: set[str]) -> Dict[str, float]:
    keys = ["auprc", "auroc", "ece", "brier", "nll"]
    vals = {k: [] for k in keys}
    for cid, rec in metrics_by_client.items():
        if cid not in eligible:
            continue
        m = rec["metrics"]
        for k in keys:
            vals[k].append(float(m[k]))
    return {k: float(np.mean(v)) if v else float("nan") for k, v in vals.items()}


def _hier_bootstrap_diff(
    *,
    eligible: List[str],
    fed: Dict[str, Any],
    bfl: Dict[str, Any],
    n_boot: int,
    seed: int,
    n_bins: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(int(seed))
    diffs = []
    for _ in range(int(n_boot)):
        sampled = rng.choice(eligible, size=len(eligible), replace=True)
        per_client = []
        for cid in sampled:
            f = fed[cid]
            b = bfl[cid]
            y = f["y_true"]
            idx = rng.integers(0, len(y), size=len(y))
            y_s = y[idx]
            prob_f = f["prob"][idx]
            prob_b = b["prob"][idx]
            mf = compute_binary_metrics(y_s, prob_f, n_bins=int(n_bins))
            mb = compute_binary_metrics(y_s, prob_b, n_bins=int(n_bins))
            per_client.append((mf, mb))
        if not per_client:
            diffs.append(float("nan"))
            continue
        diff_mean = float(np.mean([mb.auprc - mf.auprc for mf, mb in per_client]))
        diffs.append(diff_mean)
    arr = np.asarray(diffs, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": float("nan"), "p2_5": float("nan"), "p97_5": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "p2_5": float(np.percentile(finite, 2.5)),
        "p97_5": float(np.percentile(finite, 97.5)),
    }


def _sign_flip_p(diffs: List[float], *, n_perm: int = 20000, seed: int = 42) -> float:
    arr = np.asarray(diffs, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    obs = float(np.mean(arr))
    rng = np.random.default_rng(int(seed))
    sims = []
    for _ in range(int(n_perm)):
        signs = rng.choice([-1.0, 1.0], size=arr.size)
        sims.append(float(np.mean(arr * signs)))
    sims = np.asarray(sims, dtype=np.float64)
    p = float(np.mean(np.abs(sims) >= abs(obs)))
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare FedAvg vs BFL with per-client macro averages (eligible clients only)")
    ap.add_argument("--data-dir", default="federated_data")
    ap.add_argument("--fedavg-runs", required=True, help="Comma-separated FedAvg run dirs (same order/length as bfl)")
    ap.add_argument("--bfl-runs", required=True, help="Comma-separated BFL run dirs (same order/length as fedavg)")
    ap.add_argument("--seeds", default=None, help="Optional comma-separated seed labels matching runs")
    ap.add_argument("--min-client-cases", type=int, default=None, help="Eligible client threshold; default=summary.json or 150")
    ap.add_argument("--mc-eval", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--hier-bootstrap-n", type=int, default=0, help="Hierarchical bootstrap iterations (0=skip)")
    ap.add_argument("--hier-bootstrap-seed", type=int, default=123)
    ap.add_argument("--out-json", default="compare_clients.json")
    ap.add_argument("--out-csv", default="compare_clients.csv")
    args = ap.parse_args()

    fed_runs = [Path(s) for s in str(args.fedavg_runs).split(",") if s.strip()]
    bfl_runs = [Path(s) for s in str(args.bfl_runs).split(",") if s.strip()]
    if len(fed_runs) != len(bfl_runs):
        raise SystemExit("fedavg-runs と bfl-runs の数が一致しません。")
    seeds = [s.strip() for s in str(args.seeds).split(",")] if args.seeds else [f"seed{i}" for i in range(len(fed_runs))]
    if len(seeds) != len(fed_runs):
        raise SystemExit("seeds の数が run ペアと一致しません。")

    data_dir = str(args.data_dir)
    val_files = list_npz_files(data_dir, "val")
    test_by_client = _list_client_files(data_dir, "test")
    if not val_files or not test_by_client:
        raise SystemExit("val/test の .npz が見つかりません。val分割が必須です。--data-dir を確認してください。")

    min_cases = int(args.min_client_cases) if args.min_client_cases is not None else _load_summary_min_cases(data_dir, 150)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {
        "started_utc": now_utc_iso(),
        "data_dir": data_dir,
        "min_client_cases": int(min_cases),
        "notes": "Macro averages use eligible clients only (n_cases >= min_client_cases). Small clients reported separately.",
        "seeds": seeds,
        "pairs": [],
    }

    rows_csv = []
    diffs_auprc: List[float] = []

    for idx, (seed_label, fed_run, bfl_run) in enumerate(zip(seeds, fed_runs, bfl_runs)):
        fed_model = _load_fedavg_model(fed_run, device=device)
        bfl_model = _load_bfl_model(bfl_run, device=device)

        t_fed = _fit_temperature_from_val_fedavg(fed_model, val_files, batch_size=int(args.batch_size), num_workers=int(args.num_workers), device=device)
        t_bfl = _fit_temperature_from_val_bfl(bfl_model, val_files, batch_size=int(args.batch_size), num_workers=int(args.num_workers), mc_eval=int(args.mc_eval), device=device)

        fed_metrics = _eval_clients_fedavg(
            fed_model,
            test_by_client,
            temperature=t_fed,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            n_bins=int(args.n_bins),
            device=device,
            keep_arrays=bool(int(args.hier_bootstrap_n) > 0),
        )
        bfl_metrics = _eval_clients_bfl(
            bfl_model,
            test_by_client,
            temperature=t_bfl,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            n_bins=int(args.n_bins),
            mc_eval=int(args.mc_eval),
            device=device,
            keep_arrays=bool(int(args.hier_bootstrap_n) > 0),
        )

        case_counts = {cid: int(len(test_by_client.get(cid, []))) for cid in fed_metrics.keys()}
        eligible = {cid for cid, n in case_counts.items() if n >= min_cases}
        excluded = {cid for cid in case_counts if cid not in eligible}

        macro_fed = _macro_mean(fed_metrics, eligible)
        macro_bfl = _macro_mean(bfl_metrics, eligible)
        diff_macro = {k: float(macro_bfl[k] - macro_fed[k]) for k in macro_fed}
        diffs_auprc.append(diff_macro["auprc"])

        pair = {
            "seed": seed_label,
            "fedavg_run": str(fed_run),
            "bfl_run": str(bfl_run),
            "temperature": {"fedavg": t_fed, "bfl": t_bfl},
            "eligible_clients": sorted(list(eligible)),
            "excluded_clients": sorted(list(excluded)),
            "macro": {"fedavg": macro_fed, "bfl": macro_bfl, "diff": diff_macro},
            "clients": {
                cid: {
                    "fedavg": fed_metrics[cid]["metrics"],
                    "bfl": bfl_metrics[cid]["metrics"],
                    "eligible": cid in eligible,
                }
                for cid in sorted(fed_metrics.keys())
            },
        }
        if int(args.hier_bootstrap_n) > 0:
            bs = _hier_bootstrap_diff(
                eligible=sorted(list(eligible)),
                fed=fed_metrics,
                bfl=bfl_metrics,
                n_boot=int(args.hier_bootstrap_n),
                seed=int(args.hier_bootstrap_seed),
                n_bins=int(args.n_bins),
            )
            pair["hier_bootstrap_diff_auprc"] = bs

        results["pairs"].append(pair)

        rows_csv.append(
            {
                "seed": seed_label,
                "method": "fedavg",
                **macro_fed,
            }
        )
        rows_csv.append(
            {
                "seed": seed_label,
                "method": "bfl",
                **macro_bfl,
            }
        )

    # Summary across runs (AUPRC primary)
    diffs_arr = np.asarray(diffs_auprc, dtype=np.float64)
    summary = {
        "diff_auprc_mean": float(np.nanmean(diffs_arr)),
        "diff_auprc_p2_5": float(np.nanpercentile(diffs_arr, 2.5)),
        "diff_auprc_p97_5": float(np.nanpercentile(diffs_arr, 97.5)),
        "diff_auprc_p_signflip": _sign_flip_p(diffs_auprc, n_perm=20000, seed=42),
    }
    results["summary"] = summary
    results["finished_utc"] = now_utc_iso()

    write_json(args.out_json, results)

    # CSV (seed,method,metrics)
    import pandas as pd

    pd.DataFrame(rows_csv).to_csv(args.out_csv, index=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved JSON: {args.out_json}")
    print(f"Saved CSV:  {args.out_csv}")


if __name__ == "__main__":
    main()
