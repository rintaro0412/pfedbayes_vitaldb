from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root in sys.path when executed as `python scripts/train_local.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.dataset import WindowedNPZDataset, list_client_ids, list_npz_files, scan_label_stats
from common.experiment import save_env_snapshot
from common.io import ensure_dir, get_git_hash, now_utc_iso, write_json
from common.ioh_model import IOHModelConfig, IOHNet
from common.metrics import best_threshold_youden, compute_binary_metrics, sigmoid_np
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
    ap = argparse.ArgumentParser(description="Local training baseline (per-client, no aggregation).")
    ap.add_argument("--data-dir", default="federated_data", help="Output directory from scripts/build_dataset.py")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--out-dir", default="runs/local")
    ap.add_argument("--run-name", default=None, help="Optional run directory name")

    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-threshold", type=float, default=0.5, help="Fixed threshold for round-by-round metrics.")
    ap.add_argument("--threshold-method", default="youden-val", choices=["fixed", "youden-val"])
    ap.add_argument("--val-split", default="val", help="Split name used to select threshold when --threshold-method=youden-val.")

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
    args = ap.parse_args()

    thr_method = str(args.threshold_method).lower()

    set_seed(int(args.seed))
    torch.set_num_threads(max(1, min(os.cpu_count() or 4, 8)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = args.run_name or f"run_{now_utc_iso().replace(':', '').replace('-', '')}"
    run_dir = ensure_dir(Path(args.out_dir) / run_name)
    save_env_snapshot(run_dir, {"args": vars(args)})

    client_ids = list_client_ids(args.data_dir)
    client_train_files: Dict[str, List[str]] = {}
    client_test_files: Dict[str, List[str]] = {}
    client_val_files: Dict[str, List[str]] = {}
    for cid in client_ids:
        train_files = list_npz_files(args.data_dir, args.train_split, client_id=str(cid))
        test_files = list_npz_files(args.data_dir, args.test_split, client_id=str(cid))
        val_files: List[str] = []
        if thr_method == "youden-val":
            val_files = list_npz_files(args.data_dir, args.val_split, client_id=str(cid))
        if train_files:
            client_train_files[str(cid)] = train_files
            client_test_files[str(cid)] = test_files
            if thr_method == "youden-val":
                client_val_files[str(cid)] = val_files
    client_ids = sorted(client_train_files.keys())
    if not client_ids:
        raise SystemExit("No client train files found under --data-dir.")
    if thr_method == "youden-val":
        has_val = any(client_val_files.get(cid) for cid in client_ids)
        if not has_val:
            print("[WARN] youden-val requested but no val files found; falling back to fixed threshold.")
            thr_method = "fixed"

    sample_file = next(iter(client_train_files.values()))[0]
    ds_sample = WindowedNPZDataset(
        [sample_file],
        use_clin="true",
        cache_in_memory=False,
        max_cache_files=int(args.max_cache_files),
        cache_dtype=str(args.cache_dtype),
    )
    model_cfg = IOHModelConfig(
        in_channels=int(getattr(ds_sample, "wave_channels", 4) or 4),
        base_channels=int(args.model_base_channels),
        dropout=float(args.dropout),
        use_gru=bool(args.use_gru),
        gru_hidden=int(args.gru_hidden) if args.gru_hidden is not None else 64,
        clin_dim=int(getattr(ds_sample, "clin_dim", 0) or 0),
    )

    # Build per-client datasets/dataloaders
    pin_memory = bool(args.pin_memory) and (device.type == "cuda")
    num_workers = int(args.num_workers)
    persistent_workers = bool(args.persistent_workers) and (num_workers > 0)
    dl_common = dict(
        batch_size=int(args.batch_size),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    client_train_dl: Dict[str, DataLoader] = {}
    client_test_dl: Dict[str, DataLoader] = {}
    client_val_dl: Dict[str, DataLoader] = {}
    client_counts: Dict[str, Dict[str, int]] = {}
    for cid, files in client_train_files.items():
        ds = WindowedNPZDataset(
            files,
            use_clin="true",
            cache_in_memory=bool(args.cache_in_memory),
            max_cache_files=int(args.max_cache_files),
            cache_dtype=str(args.cache_dtype),
        )
        if num_workers > 0:
            client_train_dl[cid] = DataLoader(
                ds,
                shuffle=True,
                prefetch_factor=int(args.prefetch_factor),
                **dl_common,
            )
        else:
            client_train_dl[cid] = DataLoader(ds, shuffle=True, **dl_common)
        pos, total = scan_label_stats(files)
        client_counts[cid] = {"n": int(total), "n_pos": int(pos), "n_neg": int(total - pos)}

        test_files = client_test_files.get(cid, [])
        if test_files:
            ds_t = WindowedNPZDataset(
                test_files,
                use_clin="true",
                cache_in_memory=bool(args.cache_in_memory),
                max_cache_files=int(args.max_cache_files),
                cache_dtype=str(args.cache_dtype),
            )
            if num_workers > 0:
                client_test_dl[cid] = DataLoader(
                    ds_t,
                    shuffle=False,
                    prefetch_factor=int(args.prefetch_factor),
                    **dl_common,
                )
            else:
                client_test_dl[cid] = DataLoader(ds_t, shuffle=False, **dl_common)
        if thr_method == "youden-val":
            val_files = client_val_files.get(cid, [])
            if val_files:
                ds_v = WindowedNPZDataset(
                    val_files,
                    use_clin="true",
                    cache_in_memory=bool(args.cache_in_memory),
                    max_cache_files=int(args.max_cache_files),
                    cache_dtype=str(args.cache_dtype),
                )
                if num_workers > 0:
                    client_val_dl[cid] = DataLoader(
                        ds_v,
                        shuffle=False,
                        prefetch_factor=int(args.prefetch_factor),
                        **dl_common,
                    )
                else:
                    client_val_dl[cid] = DataLoader(ds_v, shuffle=False, **dl_common)

    splits = {"train": str(args.train_split), "test": str(args.test_split)}
    if thr_method == "youden-val":
        splits["val"] = str(args.val_split)
    run_meta: Dict[str, Any] = {
        "started_utc": now_utc_iso(),
        "git_hash": get_git_hash(PROJECT_ROOT),
        "device": str(device),
        "data_dir": str(args.data_dir),
        "splits": splits,
        "seed": int(args.seed),
        "rounds": int(args.rounds),
        "clients": client_ids,
        "counts_train": client_counts,
        "hyper": {
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "eval_threshold": float(args.eval_threshold),
            "threshold_method": str(thr_method),
            "val_split": str(args.val_split),
            "log_interval": int(args.log_interval),
            "progress_bar": bool(not args.no_progress_bar),
            "num_workers": int(args.num_workers),
            "prefetch_factor": int(args.prefetch_factor),
            "pin_memory": bool(args.pin_memory),
            "persistent_workers": bool(args.persistent_workers),
        },
        "model": asdict(model_cfg),
    }
    write_json(run_dir / "run_config.json", run_meta)

    # Init per-client models/optimizers
    models: Dict[str, IOHNet] = {}
    opts: Dict[str, torch.optim.Optimizer] = {}
    scalers: Dict[str, torch.amp.GradScaler] = {}
    client_pos_weight: Dict[str, float] = {}
    for cid, files in client_train_files.items():
        pos, total = scan_label_stats(files)
        neg = int(total - pos)
        if pos <= 0:
            client_pos_weight[cid] = 1.0
        else:
            client_pos_weight[cid] = float(max(neg, 1) / max(pos, 1))
    for cid in client_ids:
        model = IOHNet(model_cfg).to(device)
        models[cid] = model
        opts[cid] = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
        scalers[cid] = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    per_client_rounds: List[Dict[str, Any]] = []
    for rnd in range(1, int(args.rounds) + 1):
        round_rows = []
        train_losses: Dict[str, float] = {}
        for cid in client_ids:
            dl_train = client_train_dl[cid]
            model = models[cid]
            opt = opts[cid]
            scaler = scalers[cid]

            loss_fn = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([client_pos_weight.get(cid, 1.0)], device=device)
            )
            tr_loss = _infer_one_epoch(
                model,
                dl_train,
                loss_fn=loss_fn,
                opt=opt,
                scaler=scaler,
                device=device,
                log_interval=int(args.log_interval),
                epoch=int(rnd),
                total_epochs=int(args.rounds),
                show_progress=not args.no_progress_bar,
            )
            train_losses[str(cid)] = float(tr_loss)

        thr = float(args.eval_threshold)
        if thr_method == "youden-val":
            val_y_all: List[np.ndarray] = []
            val_prob_all: List[np.ndarray] = []
            for cid, dl_val in client_val_dl.items():
                logits_val, y_val = _predict_logits(models[cid], dl_val, device=device)
                prob_val = sigmoid_np(logits_val)
                val_y_all.append(y_val)
                val_prob_all.append(prob_val)
            if val_y_all:
                y_val_cat = np.concatenate(val_y_all, axis=0)
                prob_val_cat = np.concatenate(val_prob_all, axis=0)
                thr = best_threshold_youden(y_val_cat, prob_val_cat, fallback=float(args.eval_threshold))

        for cid in client_ids:
            dl_test = client_test_dl.get(cid)
            tr_loss = float(train_losses.get(str(cid), float("nan")))
            if dl_test is None:
                row_c = {
                    "round": int(rnd),
                    "client_id": str(cid),
                    "status": "no_test_files",
                    "train_loss": float(tr_loss),
                    "n": 0,
                }
                round_rows.append(row_c)
                per_client_rounds.append(row_c)
                continue

            logits_c, y_c = _predict_logits(models[cid], dl_test, device=device)
            prob_c = sigmoid_np(logits_c)
            m_c = compute_binary_metrics(y_c, prob_c, n_bins=15)
            m_thr = calc_comprehensive_metrics(y_c, prob_c, threshold=float(thr))
            row_c = {
                "round": int(rnd),
                "client_id": str(cid),
                "status": "ok",
                "train_loss": float(tr_loss),
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
            pd.DataFrame(round_rows).to_csv(run_dir / f"round_{rnd:03d}_test_per_client.csv", index=False)
        pd.DataFrame(per_client_rounds).to_csv(run_dir / "round_client_metrics.csv", index=False)
        write_json(run_dir / "round_client_metrics.json", per_client_rounds)

    # Final per-client summary (last round)
    last_round = int(args.rounds)
    last_rows = [r for r in per_client_rounds if int(r.get("round", 0)) == last_round]
    per_client_reports: Dict[str, Any] = {}
    per_client_rows: List[Dict[str, Any]] = []
    for row in last_rows:
        cid = str(row.get("client_id"))
        if row.get("status") != "ok":
            per_client_reports[cid] = {"client_id": cid, "status": row.get("status"), "n": int(row.get("n", 0))}
            per_client_rows.append({"client_id": cid, "status": row.get("status"), "n": int(row.get("n", 0))})
            continue
        per_client_reports[cid] = row
        per_client_rows.append(row)
    write_json(run_dir / "test_report_per_client.json", {"round": last_round, "clients": per_client_reports})
    pd.DataFrame(per_client_rows).to_csv(run_dir / "test_report_per_client.csv", index=False)

    run_meta["finished_utc"] = now_utc_iso()
    run_meta["artifacts"] = {
        "round_client_metrics_csv": str(run_dir / "round_client_metrics.csv"),
        "round_client_metrics_json": str(run_dir / "round_client_metrics.json"),
        "test_report_per_client_json": str(run_dir / "test_report_per_client.json"),
        "test_report_per_client_csv": str(run_dir / "test_report_per_client.csv"),
    }
    write_json(run_dir / "run_config.json", run_meta)

    print("Done.")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
