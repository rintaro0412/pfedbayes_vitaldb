from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root in sys.path when executed as `python bayes_federated/eval.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bayes_federated.models import BFLModel
from common.experiment import seed_worker
from common.io import write_json
from common.dataset import WindowedNPZDataset, list_npz_files
from common.metrics import (
    auprc,
    auroc,
    bootstrap_group_ci,
    confusion_at_threshold,
    compute_binary_metrics,
)
from common.ioh_model import IOHModelConfig, normalize_model_cfg


def _group_ids(ds: WindowedNPZDataset) -> np.ndarray:
    return np.concatenate([np.full(n, cid, dtype=np.int64) for n, cid in zip(ds.file_sizes, ds.case_ids)])


@torch.no_grad()
def mc_predict(
    model: BFLModel,
    dl: DataLoader,
    *,
    mc_eval: int,
    device: torch.device,
    temperature: float | None = None,
    return_y: bool = False,
) -> Dict[str, np.ndarray]:
    model.eval()
    prob_mean_list = []
    prob_var_list = []
    prob_alea_list = []
    prob_epi_list = []
    prob_total_var_list = []
    entropy_list = []
    logits_mean_list = []
    prob_mean_cal_list = []
    y_list = []

    for x, y in dl:
        if isinstance(x, (tuple, list)):
            x = tuple(t.to(device, non_blocking=True) for t in x)
        else:
            x = x.to(device, non_blocking=True)
        logits_mc = model(x, sample=True, n_samples=int(mc_eval))
        if logits_mc.dim() == 1:
            logits_mc = logits_mc.unsqueeze(0).unsqueeze(-1)
        elif logits_mc.dim() == 2 and logits_mc.shape[-1] == 1:
            logits_mc = logits_mc.unsqueeze(0)
        logits_mc = logits_mc.squeeze(-1)  # (MC, B)
        logits_mean = logits_mc.mean(dim=0)
        probs_mc = torch.sigmoid(logits_mc)
        prob_mean = probs_mc.mean(dim=0)
        prob_var = probs_mc.var(dim=0, unbiased=False)
        prob_alea = (probs_mc * (1.0 - probs_mc)).mean(dim=0)
        prob_epi = prob_var
        prob_total_var = prob_alea + prob_epi
        eps = 1e-12
        entropy = -prob_mean * torch.log(prob_mean + eps) - (1.0 - prob_mean) * torch.log(1.0 - prob_mean + eps)

        prob_mean_list.append(prob_mean.detach().cpu().numpy())
        prob_var_list.append(prob_var.detach().cpu().numpy())
        prob_alea_list.append(prob_alea.detach().cpu().numpy())
        prob_epi_list.append(prob_epi.detach().cpu().numpy())
        prob_total_var_list.append(prob_total_var.detach().cpu().numpy())
        entropy_list.append(entropy.detach().cpu().numpy())
        logits_mean_list.append(logits_mean.detach().cpu().numpy())

        if temperature is not None:
            probs_cal = torch.sigmoid(logits_mc / float(temperature))
            prob_mean_cal = probs_cal.mean(dim=0)
            prob_mean_cal_list.append(prob_mean_cal.detach().cpu().numpy())
        if return_y:
            y_list.append(y.detach().cpu().view(-1).numpy())

    out = {
        "prob_mean": np.concatenate(prob_mean_list, axis=0),
        "prob_var": np.concatenate(prob_var_list, axis=0),
        "prob_alea": np.concatenate(prob_alea_list, axis=0),
        "prob_epi": np.concatenate(prob_epi_list, axis=0),
        "prob_total_var": np.concatenate(prob_total_var_list, axis=0),
        "entropy": np.concatenate(entropy_list, axis=0),
        "logits_mean": np.concatenate(logits_mean_list, axis=0),
    }
    if temperature is not None:
        out["prob_mean_cal"] = np.concatenate(prob_mean_cal_list, axis=0)
    if return_y:
        out["y_true"] = np.concatenate(y_list, axis=0) if y_list else np.zeros((0,), dtype=np.int64)
    return out


def evaluate_split(
    *,
    model: BFLModel,
    files: list[str],
    mc_eval: int,
    device: torch.device,
    temperature: float | None = None,
    threshold: float | None = None,
    fixed_threshold: float = 0.5,
    bootstrap_n: int = 0,
    bootstrap_seed: int = 42,
    save_pred_path: str | None = None,
    batch_size: int = 128,
    num_workers: int = 0,
) -> Dict[str, Any]:
    if not files:
        raise ValueError("No files provided for evaluation.")

    ds = WindowedNPZDataset(files, use_clin="true", cache_in_memory=False, max_cache_files=32, cache_dtype="float32")
    num_workers = int(num_workers)
    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        persistent_workers=(num_workers > 0),
    )

    pred = mc_predict(model, dl, mc_eval=mc_eval, device=device, temperature=temperature, return_y=True)
    y_true = pred["y_true"].astype(int)
    group = _group_ids(ds)

    prob = pred["prob_mean"]
    metrics_pre = compute_binary_metrics(y_true, prob, n_bins=15)
    report: Dict[str, Any] = {
        "n": int(len(y_true)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "metrics_pre": asdict(metrics_pre),
        "uncertainty": {
            "prob_var_mean": float(np.mean(pred["prob_var"])),
            "prob_var_std": float(np.std(pred["prob_var"])),
            "aleatoric_mean": float(np.mean(pred["prob_alea"])),
            "aleatoric_std": float(np.std(pred["prob_alea"])),
            "epistemic_mean": float(np.mean(pred["prob_epi"])),
            "epistemic_std": float(np.std(pred["prob_epi"])),
            "total_var_mean": float(np.mean(pred["prob_total_var"])),
            "total_var_std": float(np.std(pred["prob_total_var"])),
            "entropy_mean": float(np.mean(pred["entropy"])),
            "entropy_std": float(np.std(pred["entropy"])),
        },
    }

    if temperature is not None and "prob_mean_cal" in pred:
        prob_cal = pred["prob_mean_cal"]
        metrics_post = compute_binary_metrics(y_true, prob_cal, n_bins=15)
        report["metrics_post"] = asdict(metrics_post)
    else:
        prob_cal = None

    if threshold is None:
        threshold = float(fixed_threshold)
    report["threshold_selected"] = float(threshold)

    report["confusion_pre"] = confusion_at_threshold(y_true, prob, thr=float(threshold))
    if prob_cal is not None:
        report["confusion_post"] = confusion_at_threshold(y_true, prob_cal, thr=float(threshold))

    if int(bootstrap_n) > 0:
        report["bootstrap"] = {
            "n_boot": int(bootstrap_n),
            "seed": int(bootstrap_seed),
            "group": "caseid",
            "auprc_pre": bootstrap_group_ci(group_ids=group, y_true=y_true, prob=prob, metric_fn=auprc, n_boot=bootstrap_n, seed=bootstrap_seed),
            "auroc_pre": bootstrap_group_ci(group_ids=group, y_true=y_true, prob=prob, metric_fn=auroc, n_boot=bootstrap_n, seed=bootstrap_seed),
        }
        if prob_cal is not None:
            report["bootstrap"]["auprc_post"] = bootstrap_group_ci(group_ids=group, y_true=y_true, prob=prob_cal, metric_fn=auprc, n_boot=bootstrap_n, seed=bootstrap_seed)
            report["bootstrap"]["auroc_post"] = bootstrap_group_ci(group_ids=group, y_true=y_true, prob=prob_cal, metric_fn=auroc, n_boot=bootstrap_n, seed=bootstrap_seed)

    if save_pred_path:
        out_path = Path(save_pred_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "y_true": y_true.astype(np.int64, copy=False),
            "prob_mean": prob.astype(np.float64, copy=False),
            "prob_var": pred["prob_var"].astype(np.float64, copy=False),
            "prob_alea": pred["prob_alea"].astype(np.float64, copy=False),
            "prob_epi": pred["prob_epi"].astype(np.float64, copy=False),
            "prob_total_var": pred["prob_total_var"].astype(np.float64, copy=False),
            "entropy": pred["entropy"].astype(np.float64, copy=False),
            "case_id": group.astype(np.int64, copy=False),
        }
        if prob_cal is not None:
            payload["prob_mean_cal"] = prob_cal.astype(np.float64, copy=False)
        np.savez(out_path, **payload)

    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="BFL evaluation (MC, calibration, CI)")
    ap.add_argument("--data-dir", default="federated_data", help="Output of scripts/build_dataset.py")
    ap.add_argument("--checkpoint", required=True, help="BFL best checkpoint (model state)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--mc-eval", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--bootstrap-n", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--output", default=None)
    ap.add_argument("--save-pred-npz", default=None, help="Optional .npz to save per-sample y/prob/uncertainty.")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["model_cfg"]
    full_bayes = bool(ckpt.get("full_bayes", False))
    model = BFLModel(normalize_model_cfg(cfg), full_bayes=bool(full_bayes))
    model.load_state_dict(ckpt["state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    files = list_npz_files(args.data_dir, args.split)
    report = evaluate_split(
        model=model,
        files=files,
        mc_eval=int(args.mc_eval),
        device=device,
        temperature=(float(args.temperature) if args.temperature is not None else None),
        threshold=float(args.threshold),
        fixed_threshold=float(args.threshold),
        bootstrap_n=int(args.bootstrap_n),
        bootstrap_seed=int(args.bootstrap_seed),
        save_pred_path=(str(args.save_pred_npz) if args.save_pred_npz else None),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
    )

    out_path = args.output or f"bfl_eval_{args.split}.json"
    write_json(out_path, report)
    print(json.dumps(report["metrics_pre"], indent=2))
    if "metrics_post" in report:
        print("post-calibration:")
        print(json.dumps(report["metrics_post"], indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
