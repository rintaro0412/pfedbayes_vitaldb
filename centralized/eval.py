from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root in sys.path when executed as `python centralized/eval.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.io import now_utc_iso, write_json
from common.dataset import WindowedNPZDataset, list_client_ids, list_npz_files
from common.ioh_model import IOHModelConfig, IOHNet, normalize_model_cfg
from common.metrics import (
    auprc,
    auroc,
    bootstrap_group_ci,
    confusion_at_threshold,
    compute_binary_metrics,
    derived_from_confusion,
    sigmoid_np,
)


@torch.no_grad()
def _predict_logits(model, dl, *, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    for x, y in dl:
        if isinstance(x, (tuple, list)):
            x = tuple(t.to(device, non_blocking=True) for t in x)
        else:
            x = x.to(device, non_blocking=True)
        logits = model(x).detach().cpu().view(-1).numpy()
        logits_all.append(logits)
        y_all.append(y.detach().cpu().view(-1).numpy())
    return np.concatenate(logits_all, axis=0), np.concatenate(y_all, axis=0)


def _expand_case_ids(ds: WindowedNPZDataset) -> np.ndarray:
    return np.concatenate([np.full(n, cid, dtype=np.int64) for n, cid in zip(ds.file_sizes, ds.case_ids)])


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate IOH model on windowed NPZ splits")
    ap.add_argument("--data-dir", default="federated_data", help="Output of scripts/build_dataset.py")
    ap.add_argument("--run-dir", required=True, help="Run dir created by centralized/train.py")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--cache-in-memory", action="store_true")
    ap.add_argument("--max-cache-files", type=int, default=32)
    ap.add_argument("--cache-dtype", default="float32", choices=["float16", "float32"])

    ap.add_argument("--bootstrap", type=int, default=0, help="Number of group bootstraps (0=off)")
    ap.add_argument("--bootstrap-seed", type=int, default=42)
    ap.add_argument("--save-pred-npz", default=None, help="Optional .npz to save per-sample predictions.")
    ap.add_argument("--per-client", action="store_true", help="Save per-client report (test split only).")
    ap.add_argument("--threshold", type=float, default=0.5, help="Fixed threshold for confusion metrics.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "checkpoints" / "model_best.pt"
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "model_last.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found under {run_dir}/checkpoints")

    threshold = float(args.threshold)

    files = list_npz_files(args.data_dir, args.split)
    if not files:
        raise SystemExit("No files to evaluate (check --data-dir and --split).")

    ds = WindowedNPZDataset(
        files,
        use_clin="true",
        cache_in_memory=bool(args.cache_in_memory),
        max_cache_files=int(args.max_cache_files),
        cache_dtype=str(args.cache_dtype),
    )
    dl = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(int(args.num_workers) > 0),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = IOHNet(normalize_model_cfg(ckpt.get("model_cfg", {}))).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    logits, y_true = _predict_logits(model, dl, device=device)
    prob = sigmoid_np(logits)
    group = _expand_case_ids(ds)
    report: Dict[str, Any] = {
        "started_utc": now_utc_iso(),
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "data_dir": str(args.data_dir),
        "split": str(args.split),
        "n": int(len(y_true)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "metrics_pre": asdict(compute_binary_metrics(y_true, prob, n_bins=15)),
    }
    report["threshold_method"] = "fixed"
    report["threshold_selected"] = float(threshold)
    report["threshold"] = float(threshold)
    report["confusion_pre"] = confusion_at_threshold(y_true, prob, thr=float(threshold))

    if int(args.bootstrap) > 0:
        n_boot = int(args.bootstrap)
        report["bootstrap"] = {
            "n_boot": n_boot,
            "seed": int(args.bootstrap_seed),
            "group": "caseid",
            "auprc_pre": bootstrap_group_ci(
                group_ids=group,
                y_true=y_true,
                prob=prob,
                metric_fn=auprc,
                n_boot=n_boot,
                seed=int(args.bootstrap_seed),
            ),
            "auroc_pre": bootstrap_group_ci(
                group_ids=group,
                y_true=y_true,
                prob=prob,
                metric_fn=auroc,
                n_boot=n_boot,
                seed=int(args.bootstrap_seed),
            ),
        }

    report["finished_utc"] = now_utc_iso()

    out_path = run_dir / f"eval_{args.split}.json"
    write_json(out_path, report)

    if args.save_pred_npz:
        pred_path = Path(args.save_pred_npz)
        if not pred_path.is_absolute() and not str(pred_path).startswith(str(run_dir)):
            pred_path = run_dir / pred_path
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "y_true": y_true.astype(np.int64, copy=False),
            "prob_mean": prob.astype(np.float64, copy=False),
            "case_id": group.astype(np.int64, copy=False),
        }
        np.savez(pred_path, **payload)

    # Optional: per-group summary for audit
    group_rows = []
    offset = 0
    for caseid, n in zip(ds.case_ids, ds.file_sizes):
        n = int(n)
        if n <= 0:
            continue
        y_slice = y_true[offset : offset + n]
        p_slice = prob[offset : offset + n]
        row = {
            "caseid": int(caseid),
            "n": int(n),
            "n_pos": int((y_slice == 1).sum()),
            "prob_pre_mean": float(np.mean(p_slice)),
            "prob_pre_max": float(np.max(p_slice)),
        }
        group_rows.append(row)
        offset += n
    import pandas as pd

    pd.DataFrame(group_rows).to_csv(run_dir / f"eval_{args.split}_per_group.csv", index=False)

    if args.per_client and str(args.split) == "test":
        per_client_reports: Dict[str, Any] = {}
        per_client_rows = []
        for cid in list_client_ids(args.data_dir):
            files = list_npz_files(args.data_dir, args.split, client_id=str(cid))
            if not files:
                per_client_reports[str(cid)] = {"client_id": str(cid), "status": "no_files", "n": 0}
                per_client_rows.append({"client_id": str(cid), "status": "no_files", "n": 0})
                continue
            ds_c = WindowedNPZDataset(
                files,
                use_clin="true",
                cache_in_memory=bool(args.cache_in_memory),
                max_cache_files=int(args.max_cache_files),
                cache_dtype=str(args.cache_dtype),
            )
            dl_c = DataLoader(
                ds_c,
                batch_size=int(args.batch_size),
                shuffle=False,
                num_workers=int(args.num_workers),
                pin_memory=torch.cuda.is_available(),
                persistent_workers=(int(args.num_workers) > 0),
            )
            logits_c, y_c = _predict_logits(model, dl_c, device=device)
            prob_c = sigmoid_np(logits_c)
            metrics_pre = compute_binary_metrics(y_c, prob_c, n_bins=15)
            report_c: Dict[str, Any] = {
                "client_id": str(cid),
                "status": "ok",
                "n": int(metrics_pre.n),
                "threshold": float(threshold),
                "metrics_pre": asdict(metrics_pre),
            }
            report_c["confusion_pre"] = confusion_at_threshold(y_c, prob_c, thr=float(threshold))
            per_client_reports[str(cid)] = report_c

            conf_pre = report_c.get("confusion_pre") or {}
            d_pre = derived_from_confusion(conf_pre) if conf_pre else {}
            per_client_rows.append(
                {
                    "client_id": str(cid),
                    "status": "ok",
                    "n": int(metrics_pre.n),
                    "n_pos": int(metrics_pre.n_pos),
                    "n_neg": int(metrics_pre.n_neg),
                    "auprc_pre": float(metrics_pre.auprc),
                    "auroc_pre": float(metrics_pre.auroc),
                    "brier_pre": float(metrics_pre.brier),
                    "nll_pre": float(metrics_pre.nll),
                    "ece_pre": float(metrics_pre.ece),
                    "accuracy_pre": float(d_pre.get("accuracy", float("nan"))),
                    "f1_pre": float(d_pre.get("f1", float("nan"))),
                    "mcc_pre": float(d_pre.get("mcc", float("nan"))),
                    "threshold": float(threshold),
                }
            )

        write_json(
            run_dir / "test_report_per_client.json",
            {
                "threshold": float(threshold),
                "clients": per_client_reports,
            },
        )
        pd.DataFrame(per_client_rows).to_csv(run_dir / "test_report_per_client.csv", index=False)

    print(json.dumps(report["metrics_pre"], indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
