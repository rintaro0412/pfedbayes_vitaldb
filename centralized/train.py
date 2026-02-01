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

from common.calibration import fit_temperature
from common.io import ensure_dir, get_git_hash, now_utc_iso, write_json
from common.dataset import WindowedNPZDataset, list_client_ids, list_npz_files, scan_label_stats
from common.ioh_model import FocalLoss, IOHModelConfig, IOHNet, normalize_model_cfg
from common.metrics import best_threshold_youden, compute_binary_metrics, confusion_at_threshold, derived_from_confusion, sigmoid_np
from common.utils import set_seed


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
    ap.add_argument("--val-split", default="val")
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
    args = ap.parse_args()

    set_seed(int(args.seed))
    torch.set_num_threads(max(1, min(os.cpu_count() or 4, 8)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = args.run_name or f"run_{now_utc_iso().replace(':', '').replace('-', '')}"
    run_dir = ensure_dir(Path(args.out_dir) / run_name)
    ensure_dir(run_dir / "checkpoints")

    train_files = list_npz_files(args.data_dir, args.train_split)
    val_files = list_npz_files(args.data_dir, args.val_split)
    test_files = list_npz_files(args.data_dir, args.test_split)
    if not train_files:
        raise SystemExit("No train files found. Check --data-dir and split names.")
    if not val_files:
        val_files = train_files
        args.val_split = args.train_split

    train_pos, train_total = scan_label_stats(train_files)
    val_pos, val_total = scan_label_stats(val_files)
    test_pos, test_total = scan_label_stats(test_files) if test_files else (0, 0)
    train_counts = {"n": int(train_total), "n_pos": int(train_pos), "n_neg": int(train_total - train_pos)}
    val_counts = {"n": int(val_total), "n_pos": int(val_pos), "n_neg": int(val_total - val_pos)}
    test_counts = {"n": int(test_total), "n_pos": int(test_pos), "n_neg": int(test_total - test_pos)}

    # Build datasets/dataloaders
    ds_train = WindowedNPZDataset(
        train_files,
        use_clin="true",
        cache_in_memory=bool(args.cache_in_memory),
        max_cache_files=int(args.max_cache_files),
        cache_dtype=str(args.cache_dtype),
    )
    ds_val = WindowedNPZDataset(
        val_files,
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
        dl_val = DataLoader(
            ds_val,
            shuffle=False,
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
        dl_val = DataLoader(ds_val, shuffle=False, **dl_common)
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

    # Loss
    n_pos = max(train_counts["n_pos"], 1)
    n_neg = max(train_counts["n_neg"], 1)
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    run_meta: Dict[str, Any] = {
        "started_utc": now_utc_iso(),
        "git_hash": get_git_hash(PROJECT_ROOT),
        "device": str(device),
        "data_dir": str(args.data_dir),
        "splits": {
            "train": str(args.train_split),
            "val": str(args.val_split),
            "test": str(args.test_split),
        },
        "n_files": {
            "train": int(len(train_files)),
            "val": int(len(val_files)),
            "test": int(len(test_files)),
        },
        "counts": {"train": train_counts, "val": val_counts, "test": test_counts},
        "seed": int(args.seed),
        "hyper": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "use_focal": True,
            "log_interval": int(args.log_interval),
            "progress_bar": bool(not args.no_progress_bar),
            "num_workers": int(args.num_workers),
            "prefetch_factor": int(args.prefetch_factor),
            "pin_memory": bool(args.pin_memory),
            "persistent_workers": bool(args.persistent_workers),
            "test_every_epoch": bool(args.test_every_epoch),
            "save_round_json": bool(args.save_round_json),
            "per_client_every_epoch": bool(args.per_client_every_epoch),
        },
        "model": asdict(model_cfg),
    }
    write_json(run_dir / "run_config.json", run_meta)

    history_rows = []
    best_val_auprc = -1.0
    best_path = None
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

        val_logits, val_y = _predict_logits(model, dl_val, device=device)
        val_prob = sigmoid_np(val_logits)
        m_val = compute_binary_metrics(val_y, val_prob, n_bins=15)

        row = {
            "epoch": int(epoch),
            "train_loss": float(tr_loss),
            "val_auprc": float(m_val.auprc),
            "val_auroc": float(m_val.auroc),
            "val_brier": float(m_val.brier),
            "val_nll": float(m_val.nll),
            "val_ece": float(m_val.ece),
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
                write_json(
                    run_dir / f"round_{epoch:03d}_test.json",
                    {
                        "epoch": int(epoch),
                        "n": int(m_test.n),
                        "n_pos": int(m_test.n_pos),
                        "n_neg": int(m_test.n_neg),
                        "metrics_pre": asdict(m_test),
                    },
                )
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False)

        if bool(args.save_round_json):
            write_json(
                run_dir / f"round_{epoch:03d}_val.json",
                {
                    "epoch": int(epoch),
                    "train_loss": float(tr_loss),
                    "n": int(m_val.n),
                    "n_pos": int(m_val.n_pos),
                    "n_neg": int(m_val.n_neg),
                    "metrics_pre": asdict(m_val),
                },
            )
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
                row_c = {
                    "round": int(epoch),
                    "client_id": str(cid),
                    "n": int(m_c.n),
                    "n_pos": int(m_c.n_pos),
                    "n_neg": int(m_c.n_neg),
                    "auprc": float(m_c.auprc),
                    "auroc": float(m_c.auroc),
                    "brier": float(m_c.brier),
                    "nll": float(m_c.nll),
                    "ece": float(m_c.ece),
                }
                round_rows.append(row_c)
                per_client_rounds.append(row_c)
            if round_rows:
                pd.DataFrame(round_rows).to_csv(run_dir / f"round_{epoch:03d}_test_per_client.csv", index=False)
                pd.DataFrame(per_client_rounds).to_csv(run_dir / "round_client_metrics.csv", index=False)

        if np.isfinite(m_val.auprc) and float(m_val.auprc) > best_val_auprc:
            best_val_auprc = float(m_val.auprc)
            best_path = run_dir / "checkpoints" / "model_best.pt"
            torch.save({"model_cfg": asdict(model_cfg), "state_dict": model.state_dict()}, best_path)

        tqdm.write(
            f"[{epoch:03d}] loss={tr_loss:.4f} val_auprc={m_val.auprc:.4f} val_auroc={m_val.auroc:.4f}"
        )

    if best_path is None:
        best_path = run_dir / "checkpoints" / "model_last.pt"
        torch.save({"model_cfg": asdict(model_cfg), "state_dict": model.state_dict()}, best_path)

    # Fit temperature scaling on VAL (no test peeking)
    ckpt = torch.load(best_path, map_location="cpu")
    model = IOHNet(normalize_model_cfg(ckpt.get("model_cfg", {}))).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    val_logits, val_y = _predict_logits(model, dl_val, device=device)
    tfit = fit_temperature(val_logits, val_y, device=str(device), max_iter=100)
    write_json(run_dir / "temperature.json", {"temperature": float(tfit.temperature), **asdict(tfit)})

    # Choose threshold on raw (uncalibrated) VAL probs
    val_prob_raw = sigmoid_np(val_logits)
    val_prob_cal = sigmoid_np(val_logits / float(tfit.temperature))
    thr = best_threshold_youden(val_y, val_prob_raw, fallback=0.5)
    write_json(run_dir / "threshold.json", {"threshold": float(thr), "method": "val_youden_raw"})

    # Save final val report (pre/post calibration)
    m_pre = compute_binary_metrics(val_y, sigmoid_np(val_logits), n_bins=15)
    m_post = compute_binary_metrics(val_y, val_prob_cal, n_bins=15)
    write_json(
        run_dir / "val_report.json",
        {
            "n": int(len(val_y)),
            "temperature": float(tfit.temperature),
            "threshold": float(thr),
            "metrics_pre": asdict(m_pre),
            "metrics_post": asdict(m_post),
        },
    )

    run_meta["finished_utc"] = now_utc_iso()
    run_meta["artifacts"] = {
        "best_checkpoint": str(best_path),
        "history_csv": str(run_dir / "history.csv"),
        "temperature_json": str(run_dir / "temperature.json"),
        "threshold_json": str(run_dir / "threshold.json"),
        "val_report_json": str(run_dir / "val_report.json"),
    }
    write_json(run_dir / "run_config.json", run_meta)

    print("Done.")
    print(f"Run dir: {run_dir}")
    print(f"Best checkpoint: {best_path}")
    print(f"Val temperature: {tfit.temperature:.4f}, threshold: {thr:.3f}")


if __name__ == "__main__":
    main()
