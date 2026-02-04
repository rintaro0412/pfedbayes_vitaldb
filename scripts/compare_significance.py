from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Ensure project root in sys.path when executed as `python scripts/compare_significance.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bayes_federated.eval import mc_predict
from bayes_federated.models import BFLModel
from common.dataset import WindowedNPZDataset, list_npz_files
from common.io import now_utc_iso, read_json, write_json
from common.ioh_model import IOHModelConfig, IOHNet, normalize_model_cfg
from common.metrics import (
    auprc,
    auroc,
    bootstrap_group_diff_ci,
    brier_score,
    compute_binary_metrics,
    expected_calibration_error,
    nll_binary,
    sigmoid_np,
)


def _resolve_checkpoint(run_dir: Path) -> Path:
    for name in ["model_best.pt", "model_last.pt"]:
        p = run_dir / "checkpoints" / name
        if p.exists():
            return p
    raise FileNotFoundError(f"checkpoint not found under {run_dir}/checkpoints")


def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            val = ckpt.get(key)
            if isinstance(val, dict) and val and all(torch.is_tensor(v) for v in val.values()):
                return val
        if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
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


def _load_temperature_and_threshold(kind: str, run_dir: Path) -> Tuple[float | None, float | None]:
    kind = str(kind)
    if kind == "ioh":
        t = None
        thr = None
        t_path = run_dir / "temperature.json"
        if t_path.exists():
            t = float(read_json(t_path)["temperature"])
        thr_path = run_dir / "threshold.json"
        if thr_path.exists():
            thr = float(read_json(thr_path)["threshold"])
        return t, thr

    if kind == "bfl":
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            return None, None
        summary = read_json(summary_path)
        best = summary.get("best", {}) if isinstance(summary, dict) else {}
        t = best.get("temperature")
        thr = best.get("threshold")
        return (float(t) if t is not None else None), (float(thr) if thr is not None else None)

    raise ValueError(f"unknown kind: {kind}")


def _compute_reliability(y_true: np.ndarray, prob: np.ndarray, *, n_bins: int) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    rows: list[dict[str, float]] = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if i == len(bins) - 2:
            mask = (prob >= lo) & (prob <= hi)
        else:
            mask = (prob >= lo) & (prob < hi)
        if not np.any(mask):
            rows.append({"bin": i, "bin_start": lo, "bin_end": hi, "confidence": np.nan, "accuracy": np.nan, "weight": 0.0})
            continue
        conf = float(np.mean(prob[mask]))
        acc = float(np.mean(y_true[mask]))
        weight = float(np.mean(mask))
        rows.append({"bin": i, "bin_start": lo, "bin_end": hi, "confidence": conf, "accuracy": acc, "weight": weight})
    return pd.DataFrame(rows)


@torch.no_grad()
def _predict_logits(model: IOHNet, dl: DataLoader, *, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired bootstrap significance test (case-level) between two methods")
    ap.add_argument("--data-dir", default="federated_data", help="Output of scripts/build_dataset.py")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n-bins", type=int, default=15, help="ECE bins")

    ap.add_argument("--a-kind", default="ioh", choices=["ioh", "bfl"])
    ap.add_argument("--a-run-dir", required=True, help="Run dir (e.g., runs/fedavg/<run> or runs/centralized/<run> or runs/bfl/<run>)")
    ap.add_argument("--a-label", default="FedAvg", help="Label for method A (used in reliability plot).")
    ap.add_argument("--b-kind", default="bfl", choices=["ioh", "bfl"])
    ap.add_argument("--b-run-dir", required=True)
    ap.add_argument("--b-label", default="BFL", help="Label for method B (used in reliability plot).")

    ap.add_argument("--variant", default="post", choices=["pre", "post"], help="Compare on pre/post-calibration probabilities")
    ap.add_argument("--mc-eval", type=int, default=50, help="MC samples for BFL (when kind=bfl)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--cache-in-memory", action="store_true")
    ap.add_argument("--max-cache-files", type=int, default=32)
    ap.add_argument("--cache-dtype", default="float32", choices=["float16", "float32"])

    ap.add_argument("--bootstrap-n", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="Output JSON path")
    args = ap.parse_args()

    data_dir = str(args.data_dir)
    files = list_npz_files(data_dir, args.split)
    if not files:
        raise SystemExit("No files found. Check --data-dir and --split.")

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

    group_ids = np.concatenate([np.full(n, cid, dtype=np.int64) for n, cid in zip(ds.file_sizes, ds.case_ids)])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_method(kind: str, run_dir: Path) -> Dict[str, Any]:
        ckpt_path = _resolve_checkpoint(run_dir)
        temperature, threshold = _load_temperature_and_threshold(kind, run_dir)

        out: Dict[str, Any] = {
            "kind": str(kind),
            "run_dir": str(run_dir),
            "checkpoint": str(ckpt_path),
            "temperature": float(temperature) if temperature is not None else None,
            "threshold": float(threshold) if threshold is not None else None,
        }

        if str(kind) == "ioh":
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model_cfg = normalize_model_cfg(_load_model_cfg(run_dir, ckpt))
            state_dict = _extract_state_dict(ckpt)
            model = IOHNet(model_cfg).to(device)
            model.load_state_dict(state_dict, strict=True)
            logits, y_true = _predict_logits(model, dl, device=device)
            prob_pre = sigmoid_np(logits)
            out["y_true"] = y_true.astype(np.int64, copy=False)
            out["logits"] = logits.astype(np.float64, copy=False)
            out["prob_pre"] = prob_pre.astype(np.float64, copy=False)
            if temperature is not None:
                out["prob_post"] = sigmoid_np(logits / float(temperature))
            return out

        if str(kind) == "bfl":
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            prior_sigma = 0.1
            logvar_min = -12.0
            logvar_max = 6.0
            full_bayes = bool(ckpt.get("full_bayes", False))
            cfg_path = run_dir / "config.json"
            if cfg_path.exists():
                cfg = read_json(cfg_path)
                bayes = cfg.get("bayes", {}) if isinstance(cfg, dict) else {}
                prior_sigma = float(bayes.get("prior_sigma", prior_sigma))
                logvar_min = float(bayes.get("logvar_min", logvar_min))
                logvar_max = float(bayes.get("logvar_max", logvar_max))
                full_bayes = bool(bayes.get("full_bayes", full_bayes))

            model_cfg = normalize_model_cfg(_load_model_cfg(run_dir, ckpt))
            state_dict = _extract_state_dict(ckpt)
            model = BFLModel(
                model_cfg,
                prior_sigma=float(prior_sigma),
                logvar_min=float(logvar_min),
                logvar_max=float(logvar_max),
                full_bayes=bool(full_bayes),
            )
            model.load_state_dict(state_dict, strict=True)
            model = model.to(device)
            pred = mc_predict(
                model,
                dl,
                mc_eval=int(args.mc_eval),
                device=device,
                temperature=(float(temperature) if temperature is not None else None),
                return_y=True,
            )
            out["y_true"] = pred["y_true"].astype(np.int64, copy=False)
            out["prob_pre"] = pred["prob_mean"].astype(np.float64, copy=False)
            if temperature is not None and "prob_mean_cal" in pred:
                out["prob_post"] = pred["prob_mean_cal"].astype(np.float64, copy=False)
            out["uncertainty"] = {
                "prob_var_mean": float(np.mean(pred["prob_var"])),
                "prob_var_std": float(np.std(pred["prob_var"])),
                "entropy_mean": float(np.mean(pred["entropy"])),
                "entropy_std": float(np.std(pred["entropy"])),
            }
            return out

        raise ValueError(f"unknown kind: {kind}")

    a = run_method(str(args.a_kind), Path(args.a_run_dir))
    b = run_method(str(args.b_kind), Path(args.b_run_dir))

    y_a = np.asarray(a["y_true"], dtype=np.int64)
    y_b = np.asarray(b["y_true"], dtype=np.int64)
    if y_a.shape != y_b.shape or np.any(y_a != y_b):
        raise RuntimeError("y_true mismatch between methods (check dataset ordering / split / data-dir).")
    y_true = y_a

    prob_a_pre = np.asarray(a["prob_pre"], dtype=np.float64)
    prob_b_pre = np.asarray(b["prob_pre"], dtype=np.float64)

    prob_a_post = np.asarray(a.get("prob_post"), dtype=np.float64) if a.get("prob_post") is not None else None
    prob_b_post = np.asarray(b.get("prob_post"), dtype=np.float64) if b.get("prob_post") is not None else None

    if str(args.variant) == "post":
        if prob_a_post is None or prob_b_post is None:
            raise SystemExit("variant=post requires both methods to have temperature (and prob_post).")
        prob_a = prob_a_post
        prob_b = prob_b_post
    else:
        prob_a = prob_a_pre
        prob_b = prob_b_pre

    metrics = {
        "auprc": auprc,
        "auroc": auroc,
        "ece": lambda y, p, sample_weight=None: expected_calibration_error(y, p, n_bins=int(args.n_bins), sample_weight=sample_weight),
        "brier": brier_score,
        "nll": nll_binary,
    }

    report: Dict[str, Any] = {
        "started_utc": now_utc_iso(),
        "data_dir": str(args.data_dir),
        "split": str(args.split),
        "group": "caseid",
        "n": int(len(y_true)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "bootstrap": {"n_boot": int(args.bootstrap_n), "seed": int(args.bootstrap_seed)},
        "variant": str(args.variant),
        "method_a": {k: v for k, v in a.items() if k not in ("y_true", "logits", "prob_pre", "prob_post")},
        "method_b": {k: v for k, v in b.items() if k not in ("y_true", "logits", "prob_pre", "prob_post")},
        "metrics_a": {
            "pre": asdict(compute_binary_metrics(y_true, prob_a_pre, n_bins=int(args.n_bins))),
            "post": (asdict(compute_binary_metrics(y_true, prob_a_post, n_bins=int(args.n_bins))) if prob_a_post is not None else None),
        },
        "metrics_b": {
            "pre": asdict(compute_binary_metrics(y_true, prob_b_pre, n_bins=int(args.n_bins))),
            "post": (asdict(compute_binary_metrics(y_true, prob_b_post, n_bins=int(args.n_bins))) if prob_b_post is not None else None),
        },
        "comparison": {},
    }

    # Reliability curves (labels are configurable for reuse with Central vs FedAvg)
    report["reliability"] = {
        str(args.a_label): _compute_reliability(y_true, prob_a, n_bins=int(args.n_bins)).to_dict("records"),
        str(args.b_label): _compute_reliability(y_true, prob_b, n_bins=int(args.n_bins)).to_dict("records"),
    }

    comp = {}
    for name, fn in metrics.items():
        comp[name] = bootstrap_group_diff_ci(
            group_ids=group_ids,
            y_true=y_true,
            prob_a=prob_a,
            prob_b=prob_b,
            metric_fn=fn,
            n_boot=int(args.bootstrap_n),
            seed=int(args.bootstrap_seed),
        )
    report["comparison"] = comp
    report["finished_utc"] = now_utc_iso()

    out_path = args.out
    if out_path is None:
        out_path = f"compare_{args.a_kind}_vs_{args.b_kind}_{args.split}_{args.variant}.json"
    write_json(out_path, report)

    print(json.dumps(report["comparison"], indent=2, ensure_ascii=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
