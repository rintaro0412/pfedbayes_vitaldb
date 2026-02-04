from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root in sys.path when executed as `python centralized/train.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.io import ensure_dir, get_git_hash, now_utc_iso, write_json
from common.dataset import WindowedNPZDataset, list_client_ids, list_npz_files, scan_label_stats
from common.ioh_model import IOHModelConfig, IOHNet, normalize_model_cfg
from common.metrics import compute_binary_metrics, confusion_at_threshold, sigmoid_np
from common.experiment import save_env_snapshot
from common.utils import calc_comprehensive_metrics, set_seed


def _infer_one_epoch(
    model,
    dl,
    *,
    loss_fn,
    opt,
    scaler,
    device,
    log_interval: int,
    epoch: int,
    total_epochs: int,
    show_progress: bool,
) -> float:
    model.train()
    total = 0.0
    n = 0
    log_interval = int(log_interval)
    if log_interval < 1:
        log_interval = 0
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    iterator = tqdm(
        dl,
        total=len(dl),
        desc=f"train {epoch}/{total_epochs}",
        leave=False,
        disable=(not show_progress),
    )

    total_steps = len(dl)
    for step, (x, y) in enumerate(iterator, 1):
        if isinstance(x, (tuple, list)):
            x = tuple(t.to(device, non_blocking=True) for t in x)
        else:
            x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=autocast_device, enabled=(device.type == "cuda")):
            logits = model(x)
            loss = loss_fn(logits.view(-1), y.view(-1))  # mean over batch
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        total += float(loss.item()) * int(y.shape[0])
        n += int(y.shape[0])
        if log_interval and (step % log_interval == 0):
            avg_loss = total / max(n, 1)
            if show_progress:
                iterator.set_postfix(loss=f"{avg_loss:.4f}")
            else:
                print(f"[{epoch:03d}] step {step}/{total_steps} loss={avg_loss:.4f}")
    return float(total / max(n, 1))


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Centralized baseline training (IOH)")
    ap.add_argument("--data-dir", default="federated_data", help="Output directory from scripts/build_dataset.py")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--out-dir", default="runs/centralized")
    ap.add_argument("--run-name", default=None, help="Optional run directory name")

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)


    ap.add_argument("--model-base-channels", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use-gru", dest="use_gru", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use-lstm", dest="use_gru", action="store_true", help="Deprecated (maps to GRU).")
    ap.add_argument("--gru-hidden", dest="gru_hidden", type=int, default=64)
    ap.add_argument("--lstm-hidden", dest="gru_hidden", type=int, help="Deprecated (maps to GRU).")

    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor (workers>0).")
    ap.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--cache-in-memory", action="store_true")
    ap.add_argument("--max-cache-files", type=int, default=32)
    ap.add_argument("--cache-dtype", default="float32", choices=["float16", "float32"])
    ap.add_argument("--log-interval", type=int, default=50, help="Batch interval for progress updates.")
    ap.add_argument("--no-progress-bar", action="store_true", help="Disable per-epoch progress bar.")
    ap.add_argument("--test-every-epoch", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save-round-json", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--per-client-every-epoch", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--eval-threshold", type=float, default=0.5, help="Fixed threshold for round-by-round metrics.")
    ap.add_argument("--model-selection", default="last", choices=["last", "best"])
    ap.add_argument("--selection-metric", default="auroc", choices=["auroc", "auprc", "ece", "brier", "nll"])
    args = ap.parse_args()

    set_seed(int(args.seed))
    torch.set_num_threads(max(1, min(os.cpu_count() or 4, 8)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = args.run_name or f"run_{now_utc_iso().replace(':', '').replace('-', '')}"
    run_dir = ensure_dir(Path(args.out_dir) / run_name)
    ensure_dir(run_dir / "checkpoints")
    save_env_snapshot(run_dir, {"args": vars(args)})

    train_files = list_npz_files(args.data_dir, args.train_split)
    test_files = list_npz_files(args.data_dir, args.test_split)
    if not train_files:
        raise SystemExit("No train files found. Check --data-dir and split names.")
    train_pos, train_total = scan_label_stats(train_files)
    test_pos, test_total = scan_label_stats(test_files) if test_files else (0, 0)
    train_counts = {"n": int(train_total), "n_pos": int(train_pos), "n_neg": int(train_total - train_pos)}
    test_counts = {"n": int(test_total), "n_pos": int(test_pos), "n_neg": int(test_total - test_pos)}

    # Build datasets/dataloaders
    ds_train = WindowedNPZDataset(
        train_files,
        use_clin="true",
        cache_in_memory=bool(args.cache_in_memory),
        max_cache_files=int(args.max_cache_files),
        cache_dtype=str(args.cache_dtype),
    )
    ds_test = None
    if test_files and bool(args.test_every_epoch):
        ds_test = WindowedNPZDataset(
            test_files,
            use_clin="true",
            cache_in_memory=bool(args.cache_in_memory),
            max_cache_files=int(args.max_cache_files),
            cache_dtype=str(args.cache_dtype),
        )

    pin_memory = bool(args.pin_memory) and (device.type == "cuda")
    num_workers = int(args.num_workers)
    persistent_workers = bool(args.persistent_workers) and (num_workers > 0)
    dl_common = dict(
        batch_size=int(args.batch_size),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    if num_workers > 0:
        dl_train = DataLoader(
            ds_train,
            shuffle=True,
            prefetch_factor=int(args.prefetch_factor),
            **dl_common,
        )
        dl_test = None
        if ds_test is not None:
            dl_test = DataLoader(
                ds_test,
                shuffle=False,
                prefetch_factor=int(args.prefetch_factor),
                **dl_common,
            )
    else:
        dl_train = DataLoader(ds_train, shuffle=True, **dl_common)
        dl_test = DataLoader(ds_test, shuffle=False, **dl_common) if ds_test is not None else None

    model_cfg = IOHModelConfig(
        in_channels=int(getattr(ds_train, "wave_channels", 4) or 4),
        base_channels=int(args.model_base_channels),
        dropout=float(args.dropout),
        use_gru=bool(args.use_gru),
        gru_hidden=int(args.gru_hidden) if args.gru_hidden is not None else 64,
        clin_dim=int(getattr(ds_train, "clin_dim", 0) or 0),
    )
    model = IOHNet(model_cfg).to(device)

    # Loss (BCE, with optional pos_weight)
    n_pos = max(train_counts["n_pos"], 1)
    n_neg = max(train_counts["n_neg"], 1)
    pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    run_meta: Dict[str, Any] = {
        "started_utc": now_utc_iso(),
        "git_hash": get_git_hash(PROJECT_ROOT),
        "device": str(device),
        "data_dir": str(args.data_dir),
        "splits": {
            "train": str(args.train_split),
            "test": str(args.test_split),
        },
        "n_files": {
            "train": int(len(train_files)),
            "test": int(len(test_files)),
        },
        "counts": {"train": train_counts, "test": test_counts},
        "seed": int(args.seed),
        "hyper": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "loss": "bce",
            "pos_weight": float(pos_weight),
            "log_interval": int(args.log_interval),
            "progress_bar": bool(not args.no_progress_bar),
            "num_workers": int(args.num_workers),
            "prefetch_factor": int(args.prefetch_factor),
            "pin_memory": bool(args.pin_memory),
            "persistent_workers": bool(args.persistent_workers),
            "test_every_epoch": bool(args.test_every_epoch),
            "save_round_json": bool(args.save_round_json),
            "per_client_every_epoch": bool(args.per_client_every_epoch),
            "eval_threshold": float(args.eval_threshold),
            "model_selection": str(args.model_selection),
            "selection_metric": str(args.selection_metric),
        },
        "model": asdict(model_cfg),
    }
    write_json(run_dir / "run_config.json", run_meta)

    history_rows = []
    last_path = None
    best_path = None
    selection_source = "test"
    best = {"epoch": 0, "metric": None, "source": selection_source, "metric_name": str(args.selection_metric)}
    selection_enabled = str(args.model_selection).lower() == "best"
    per_client_rounds: list[dict[str, Any]] = []
    client_test_files: Dict[str, list[str]] = {}
    if bool(args.per_client_every_epoch) and test_files:
        client_test_files = {cid: list_npz_files(args.data_dir, args.test_split, client_id=str(cid)) for cid in list_client_ids(args.data_dir)}

    for epoch in range(1, int(args.epochs) + 1):
        tr_loss = _infer_one_epoch(
            model,
            dl_train,
            loss_fn=loss_fn,
            opt=opt,
            scaler=scaler,
            device=device,
            log_interval=int(args.log_interval),
            epoch=int(epoch),
            total_epochs=int(args.epochs),
            show_progress=not args.no_progress_bar,
        )

        row = {
            "epoch": int(epoch),
            "train_loss": float(tr_loss),
        }
        if dl_test is not None:
            test_logits, test_y = _predict_logits(model, dl_test, device=device)
            test_prob = sigmoid_np(test_logits)
            m_test = compute_binary_metrics(test_y, test_prob, n_bins=15)
            row.update(
                {
                    "test_auprc": float(m_test.auprc),
                    "test_auroc": float(m_test.auroc),
                    "test_brier": float(m_test.brier),
                    "test_nll": float(m_test.nll),
                    "test_ece": float(m_test.ece),
                }
            )
            if bool(args.save_round_json):
                thr = float(args.eval_threshold)
                metrics_thr = calc_comprehensive_metrics(test_y, test_prob, threshold=thr)
                write_json(
                    run_dir / f"round_{epoch:03d}_test.json",
                    {
                        "epoch": int(epoch),
                        "n": int(m_test.n),
                        "n_pos": int(m_test.n_pos),
                        "n_neg": int(m_test.n_neg),
                        "metrics_pre": asdict(m_test),
                        "threshold": float(thr),
                        "metrics_threshold": metrics_thr,
                        "confusion_pre": confusion_at_threshold(test_y, test_prob, thr=thr),
                    },
                )
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False)

        if selection_enabled:
            metric_name = str(args.selection_metric).lower()
            metrics = None
            if dl_test is not None:
                metrics = m_test
            if metrics is not None:
                score = float(getattr(metrics, metric_name))
                prev = best.get("metric")
                if prev is None or score > float(prev):
                    best = {"epoch": int(epoch), "metric": float(score), "source": selection_source, "metric_name": metric_name}
                    best_path = run_dir / "checkpoints" / "model_best.pt"
                    torch.save({"model_cfg": asdict(model_cfg), "state_dict": model.state_dict()}, best_path)

        if bool(args.per_client_every_epoch) and client_test_files:
            round_rows = []
            for cid, files in client_test_files.items():
                if not files:
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
                    pin_memory=pin_memory,
                    persistent_workers=persistent_workers,
                )
                logits_c, y_c = _predict_logits(model, dl_c, device=device)
                prob_c = sigmoid_np(logits_c)
                m_c = compute_binary_metrics(y_c, prob_c, n_bins=15)
                thr = float(args.eval_threshold)
                m_thr = calc_comprehensive_metrics(y_c, prob_c, threshold=thr)
                row_c = {
                    "round": int(epoch),
                    "client_id": str(cid),
                    "n": int(m_c.n),
                    "n_pos": int(m_c.n_pos),
                    "n_neg": int(m_c.n_neg),
                    "pos_rate": float(m_thr.get("pos_rate", float("nan"))),
                    "auprc": float(m_c.auprc),
                    "auroc": float(m_c.auroc),
                    "brier": float(m_c.brier),
                    "nll": float(m_c.nll),
                    "ece": float(m_c.ece),
                    "threshold": float(thr),
                    "accuracy": float(m_thr.get("accuracy", float("nan"))),
                    "f1": float(m_thr.get("f1", float("nan"))),
                    "sensitivity": float(m_thr.get("sensitivity", float("nan"))),
                    "specificity": float(m_thr.get("specificity", float("nan"))),
                    "ppv": float(m_thr.get("ppv", float("nan"))),
                    "npv": float(m_thr.get("npv", float("nan"))),
                }
                round_rows.append(row_c)
                per_client_rounds.append(row_c)
            if round_rows:
                pd.DataFrame(round_rows).to_csv(run_dir / f"round_{epoch:03d}_test_per_client.csv", index=False)
                pd.DataFrame(per_client_rounds).to_csv(run_dir / "round_client_metrics.csv", index=False)
                write_json(run_dir / "round_client_metrics.json", per_client_rounds)

        last_path = run_dir / "checkpoints" / "model_last.pt"
        torch.save({"model_cfg": asdict(model_cfg), "state_dict": model.state_dict()}, last_path)

        tqdm.write(f"[{epoch:03d}] loss={tr_loss:.4f}")

    if last_path is None:
        last_path = run_dir / "checkpoints" / "model_last.pt"
        torch.save({"model_cfg": asdict(model_cfg), "state_dict": model.state_dict()}, last_path)

    thr = float(args.eval_threshold)

    run_meta["finished_utc"] = now_utc_iso()
    run_meta["artifacts"] = {
        "last_checkpoint": str(last_path),
        "best_checkpoint": str(best_path) if best_path is not None else None,
        "history_csv": str(run_dir / "history.csv"),
        "best": best,
    }
    write_json(run_dir / "run_config.json", run_meta)

    print("Done.")
    print(f"Run dir: {run_dir}")
    print(f"Fixed threshold: {thr:.3f}")


if __name__ == "__main__":
    main()
