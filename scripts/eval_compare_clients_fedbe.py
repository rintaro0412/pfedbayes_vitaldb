from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root in sys.path when executed as `python scripts/eval_compare_clients_fedbe.py`
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.calibration import fit_temperature
from common.checkpoint import load_checkpoint
from common.dataset import WindowedNPZDataset, list_client_ids, list_npz_files
from common.io import now_utc_iso, read_json, write_json
from common.ioh_model import IOHNet, normalize_model_cfg
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


def _resolve_ckpt(run_dir: Path) -> Path:
    for name in ["model_best.pt", "model_last.pt"]:
        ckpt = run_dir / "checkpoints" / name
        if ckpt.exists():
            return ckpt
    raise FileNotFoundError(f"checkpoint not found under {run_dir}/checkpoints")


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            val = ckpt.get(key)
            if isinstance(val, dict) and val and all(torch.is_tensor(v) for v in val.values()):
                return val
        if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt  # already a raw state_dict
    raise KeyError("state_dict")


def _load_model_cfg(run_dir: Path, ckpt: Any) -> Dict[str, Any]:
    if isinstance(ckpt, dict):
        for key in ("model_cfg", "model"):
            val = ckpt.get(key)
            if isinstance(val, dict) and val:
                return val
    meta = run_dir / "meta.json"
    if meta.exists():
        try:
            js = read_json(meta)
            for key in ("model", "model_cfg"):
                val = js.get(key)
                if isinstance(val, dict) and val:
                    return val
        except Exception:
            pass
    cfg_yaml = run_dir / "config_used.yaml"
    if cfg_yaml.exists():
        try:
            import yaml  # type: ignore

            cfg = yaml.safe_load(cfg_yaml.read_text(encoding="utf-8"))
            if isinstance(cfg, dict):
                val = cfg.get("model")
                if isinstance(val, dict) and val:
                    return val
        except Exception:
            pass
    raise KeyError("model_cfg")


def _load_ioh_model(run_dir: Path, device: torch.device) -> IOHNet:
    ckpt = load_checkpoint(_resolve_ckpt(run_dir), map_location="cpu")
    model_cfg = normalize_model_cfg(_load_model_cfg(run_dir, ckpt))
    state_dict = _extract_state_dict(ckpt)
    model = IOHNet(model_cfg)
    model.load_state_dict(state_dict, strict=True)
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
def _predict_logits(model: IOHNet, dl: DataLoader, *, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
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


def _fit_temperature_from_val(model: IOHNet, files: List[str], *, batch_size: int, num_workers: int, device: torch.device) -> float:
    _, dl = _dl(files, batch_size=batch_size, num_workers=num_workers)
    logits, y = _predict_logits(model, dl, device=device)
    res = fit_temperature(logits, y, device=str(device))
    return float(res.temperature)


def _eval_clients(
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
        _, dl = _dl(files, batch_size=batch_size, num_workers=num_workers)
        logits, y = _predict_logits(model, dl, device=device)
        prob = 1.0 / (1.0 + np.exp(-logits / float(temperature)))
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
    fedbe: Dict[str, Any],
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
            y = fed[cid]["y_true"]
            prob_fed = fed[cid]["prob"]
            prob_fedbe = fedbe[cid]["prob"]
            m_fed = compute_binary_metrics(y, prob_fed, n_bins=int(n_bins))
            m_fedbe = compute_binary_metrics(y, prob_fedbe, n_bins=int(n_bins))
            per_client.append(float(m_fedbe.auprc - m_fed.auprc))
        diffs.append(float(np.mean(per_client)))
    diffs = np.asarray(diffs, dtype=np.float64)
    return {
        "mean": float(np.mean(diffs)),
        "p2_5": float(np.percentile(diffs, 2.5)),
        "p97_5": float(np.percentile(diffs, 97.5)),
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
    ap = argparse.ArgumentParser(description="Compare FedAvg vs FedBE with per-client macro averages (eligible clients only)")
    ap.add_argument("--data-dir", default="federated_data")
    ap.add_argument("--fedavg-runs", required=True, help="Comma-separated FedAvg run dirs (same order/length as fedbe)")
    ap.add_argument("--fedbe-runs", required=True, help="Comma-separated FedBE run dirs (same order/length as fedavg)")
    ap.add_argument("--seeds", default=None, help="Optional comma-separated seed labels matching runs")
    ap.add_argument("--min-client-cases", type=int, default=None, help="Eligible client threshold; default=summary.json or 150")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--n-bins", type=int, default=15)
    ap.add_argument("--hier-bootstrap-n", type=int, default=0, help="Hierarchical bootstrap iterations (0=skip)")
    ap.add_argument("--hier-bootstrap-seed", type=int, default=123)
    ap.add_argument("--out-json", default="compare_clients.json")
    ap.add_argument("--out-csv", default="compare_clients.csv")
    args = ap.parse_args()

    fed_runs = [Path(s) for s in str(args.fedavg_runs).split(",") if s.strip()]
    fedbe_runs = [Path(s) for s in str(args.fedbe_runs).split(",") if s.strip()]
    if len(fed_runs) != len(fedbe_runs):
        raise SystemExit("fedavg-runs と fedbe-runs の数が一致しません。")
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

    for seed_label, fed_run, fedbe_run in zip(seeds, fed_runs, fedbe_runs):
        fed_model = _load_ioh_model(fed_run, device=device)
        fedbe_model = _load_ioh_model(fedbe_run, device=device)

        t_fed = _fit_temperature_from_val(fed_model, val_files, batch_size=int(args.batch_size), num_workers=int(args.num_workers), device=device)
        t_fedbe = _fit_temperature_from_val(fedbe_model, val_files, batch_size=int(args.batch_size), num_workers=int(args.num_workers), device=device)

        keep_arrays = bool(int(args.hier_bootstrap_n) > 0)
        fed_metrics = _eval_clients(
            fed_model,
            test_by_client,
            temperature=t_fed,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            n_bins=int(args.n_bins),
            device=device,
            keep_arrays=keep_arrays,
        )
        fedbe_metrics = _eval_clients(
            fedbe_model,
            test_by_client,
            temperature=t_fedbe,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            n_bins=int(args.n_bins),
            device=device,
            keep_arrays=keep_arrays,
        )

        case_counts = {cid: int(len(test_by_client.get(cid, []))) for cid in fed_metrics.keys()}
        eligible = {cid for cid, n in case_counts.items() if n >= min_cases}
        excluded = {cid for cid in case_counts if cid not in eligible}

        macro_fed = _macro_mean(fed_metrics, eligible)
        macro_fedbe = _macro_mean(fedbe_metrics, eligible)
        diff_macro = {k: float(macro_fedbe[k] - macro_fed[k]) for k in macro_fed}
        diffs_auprc.append(diff_macro["auprc"])

        pair = {
            "seed": seed_label,
            "fedavg_run": str(fed_run),
            "fedbe_run": str(fedbe_run),
            "temperature": {"fedavg": t_fed, "fedbe": t_fedbe},
            "eligible_clients": sorted(list(eligible)),
            "excluded_clients": sorted(list(excluded)),
            "macro": {"fedavg": macro_fed, "fedbe": macro_fedbe, "diff": diff_macro},
            "clients": {
                cid: {
                    "fedavg": fed_metrics[cid]["metrics"],
                    "fedbe": fedbe_metrics[cid]["metrics"],
                    "eligible": cid in eligible,
                }
                for cid in sorted(fed_metrics.keys())
            },
        }
        if int(args.hier_bootstrap_n) > 0:
            pair["hier_bootstrap_diff_auprc"] = _hier_bootstrap_diff(
                eligible=sorted(list(eligible)),
                fed=fed_metrics,
                fedbe=fedbe_metrics,
                n_boot=int(args.hier_bootstrap_n),
                seed=int(args.hier_bootstrap_seed),
                n_bins=int(args.n_bins),
            )

        results["pairs"].append(pair)

        rows_csv.append({"seed": seed_label, "method": "fedavg", **macro_fed})
        rows_csv.append({"seed": seed_label, "method": "fedbe", **macro_fedbe})

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

    import pandas as pd

    pd.DataFrame(rows_csv).to_csv(args.out_csv, index=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved JSON: {args.out_json}")
    print(f"Saved CSV:  {args.out_csv}")


if __name__ == "__main__":
    main()
