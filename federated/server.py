from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root in sys.path when executed as `python federated/server.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.calibration import fit_temperature
from common.io import ensure_dir, get_git_hash, now_utc_iso, write_json
from common.dataset import WindowedNPZDataset, list_client_ids, list_npz_files, list_npz_files_by_client
from common.ioh_model import IOHModelConfig, IOHNet
from common.metrics import best_threshold_youden, compute_binary_metrics, confusion_at_threshold, sigmoid_np
from common.utils import set_seed
from federated.client import LocalTrainConfig, train_one_client


def fedavg(states: List[Dict[str, torch.Tensor]], weights: List[int]) -> Dict[str, torch.Tensor]:
    if not states:
        raise ValueError("no client states to aggregate")
    tot = float(sum(int(w) for w in weights))
    if tot <= 0:
        raise ValueError("sum(weights) must be >0")

    keys = list(states[0].keys())
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        acc = None
        for sd, w in zip(states, weights):
            t = sd[k].float()
            if acc is None:
                acc = t * float(w)
            else:
                acc = acc + t * float(w)
        out[k] = (acc / tot).type_as(states[0][k])
    return out


def _flatten_state_delta(updated: Dict[str, torch.Tensor], base: Dict[str, torch.Tensor]) -> np.ndarray:
    parts = []
    for k in sorted(updated.keys()):
        u = updated[k].detach().cpu().float().view(-1)
        b = base[k].detach().cpu().float().view(-1)
        parts.append((u - b).numpy())
    if not parts:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(parts, axis=0).astype(np.float64, copy=False)


def _cosine_sim_matrix(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normed = vectors / norms
    return (normed @ normed.T).astype(np.float64, copy=False)


@torch.no_grad()
def predict_logits(model, dl, *, device) -> tuple[np.ndarray, np.ndarray]:
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
    ap = argparse.ArgumentParser(description="Minimal FedAvg baseline (no Flower) for IOH")
    ap.add_argument("--data-dir", default="federated_data", help="Output of scripts/build_dataset.py")
    ap.add_argument("--out-dir", default="runs/fedavg")
    ap.add_argument("--run-name", default=None)

    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--cache-in-memory", action="store_true")
    ap.add_argument("--max-cache-files", type=int, default=32)
    ap.add_argument("--cache-dtype", default="float32", choices=["float16", "float32"])
    ap.add_argument("--min-client-examples", type=int, default=10)
    ap.add_argument("--client-progress-bar", action="store_true", help="Show per-client batch progress bar.")
    ap.add_argument("--no-progress-bar", action="store_true", help="Disable progress bars.")
    ap.add_argument("--log-client-sim", action="store_true", help="Save per-round client cosine similarity matrix.")
    ap.add_argument("--test-every-round", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save-round-json", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--per-client-every-round", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--save-test-pred-npz", default=None, help="Optional .npz to save per-sample test predictions.")

    ap.add_argument("--model-base-channels", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use-gru", dest="use_gru", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--use-lstm", dest="use_gru", action="store_true", help="Deprecated (maps to GRU).")
    ap.add_argument("--gru-hidden", dest="gru_hidden", type=int, default=64)
    ap.add_argument("--lstm-hidden", dest="gru_hidden", type=int, help="Deprecated (maps to GRU).")
    args = ap.parse_args()

    set_seed(int(args.seed))
    torch.set_num_threads(max(1, min(os.cpu_count() or 4, 8)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = args.run_name or f"run_{now_utc_iso().replace(':', '').replace('-', '')}"
    run_dir = ensure_dir(Path(args.out_dir) / run_name)
    ensure_dir(run_dir / "checkpoints")
    summary_path = Path(args.data_dir) / "summary.json"
    dataset_summary = None
    if summary_path.exists():
        try:
            dataset_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as e:
            dataset_summary = {"error": f"failed to read summary.json: {e}"}

    client_ids = list_client_ids(args.data_dir)
    client_train_files: Dict[str, List[str]] = {}
    for cid in client_ids:
        files = list_npz_files(args.data_dir, args.train_split, client_id=str(cid))
        if files:
            client_train_files[str(cid)] = files
    client_ids = sorted(client_train_files.keys())
    if not client_ids:
        raise SystemExit("No client train files found under --data-dir.")

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
    global_model = IOHNet(model_cfg).to(device)
    global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

    local_cfg = LocalTrainConfig(
        epochs=int(args.local_epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        use_focal=True,
        focal_alpha=0.25,
        focal_gamma=2.0,
        num_workers=int(args.num_workers),
        cache_in_memory=bool(args.cache_in_memory),
        max_cache_files=int(args.max_cache_files),
        cache_dtype=str(args.cache_dtype),
    )

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
        "seed": int(args.seed),
        "rounds": int(args.rounds),
        "local_cfg": asdict(local_cfg),
        "model": asdict(model_cfg),
        "clients": client_ids,
        "test_every_round": bool(args.test_every_round),
        "save_round_json": bool(args.save_round_json),
        "per_client_every_round": bool(args.per_client_every_round),
    }
    if dataset_summary:
        keys = ["client_scheme", "merge_strategy", "opname_threshold", "min_client_cases", "clients"]
        run_meta["dataset_summary"] = {k: dataset_summary.get(k) for k in keys if k in dataset_summary}
        write_json(run_dir / "dataset_summary.json", dataset_summary)
    write_json(run_dir / "run_config.json", run_meta)

    history = []
    per_client_rounds: List[Dict[str, Any]] = []
    best_val_auprc = -1.0
    best_state = None

    # Validation loader (global)
    val_files = list_npz_files(args.data_dir, args.val_split)
    if not val_files:
        val_files = list_npz_files(args.data_dir, args.train_split)
        run_meta["splits"]["val"] = str(args.train_split)
    ds_val = WindowedNPZDataset(
        val_files,
        use_clin="true",
        cache_in_memory=bool(args.cache_in_memory),
        max_cache_files=int(args.max_cache_files),
        cache_dtype=str(args.cache_dtype),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(args.num_workers) > 0),
    )
    test_files = list_npz_files(args.data_dir, args.test_split)
    ds_test = None
    if test_files and bool(args.test_every_round):
        ds_test = WindowedNPZDataset(
            test_files,
            use_clin="true",
            cache_in_memory=bool(args.cache_in_memory),
            max_cache_files=int(args.max_cache_files),
            cache_dtype=str(args.cache_dtype),
        )
    dl_test = None
    if ds_test is not None:
        dl_test = DataLoader(
            ds_test,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
            persistent_workers=(int(args.num_workers) > 0),
        )
    client_test_files = {}
    if bool(args.per_client_every_round):
        client_test_files = list_npz_files_by_client(args.data_dir, args.test_split)

    for rnd in range(1, int(args.rounds) + 1):
        global_state_ref = {k: v.detach().cpu() for k, v in global_state.items()} if args.log_client_sim else {}
        client_states: List[Dict[str, torch.Tensor]] = []
        client_weights: List[int] = []
        client_metrics = []
        client_vecs: Dict[str, np.ndarray] = {}
        client_n: Dict[str, int] = {}
        train_loss_sum = 0.0
        train_loss_weight = 0

        client_iter = tqdm(
            client_ids,
            total=len(client_ids),
            desc=f"round {rnd}/{int(args.rounds)} clients",
            leave=False,
            disable=bool(args.no_progress_bar),
        )
        for cid in client_iter:
            train_files = client_train_files.get(str(cid), [])
            updated, n_ex, m = train_one_client(
                client_id=str(cid),
                train_files=train_files,
                model_cfg=model_cfg,
                global_state=global_state,
                cfg=local_cfg,
                device=device,
                show_progress=bool(args.client_progress_bar) and (not bool(args.no_progress_bar)),
            )
            client_metrics.append(m)
            if m.get("status") == "ok":
                client_iter.set_postfix(
                    client=str(cid),
                    n=int(n_ex),
                    loss=f"{float(m.get('avg_loss', 0.0)):.4f}",
                )
            else:
                client_iter.set_postfix(client=str(cid), status=str(m.get("status")))
            if int(n_ex) >= int(args.min_client_examples):
                client_states.append(updated)
                client_weights.append(int(n_ex))
                if args.log_client_sim and m.get("status") == "ok":
                    client_vecs[str(cid)] = _flatten_state_delta(updated, global_state_ref)
                    client_n[str(cid)] = int(n_ex)
                if m.get("status") == "ok" and "avg_loss" in m:
                    train_loss_sum += float(m["avg_loss"]) * float(n_ex)
                    train_loss_weight += int(n_ex)

        if not client_states:
            raise SystemExit("All clients empty after filtering/min_client_examples.")

        global_state = fedavg(client_states, client_weights)
        global_model = IOHNet(model_cfg).to(device)
        global_model.load_state_dict(global_state, strict=True)

        val_logits, val_y = predict_logits(global_model, dl_val, device=device)
        val_prob = sigmoid_np(val_logits)
        m_val = compute_binary_metrics(val_y, val_prob, n_bins=15)

        train_loss = float(train_loss_sum / max(train_loss_weight, 1))
        row = {
            "round": int(rnd),
            "train_loss": float(train_loss),
            "val_auprc": float(m_val.auprc),
            "val_auroc": float(m_val.auroc),
            "val_brier": float(m_val.brier),
            "val_nll": float(m_val.nll),
            "val_ece": float(m_val.ece),
            "n_clients_used": int(len(client_states)),
            "sum_examples": int(sum(client_weights)),
        }
        if dl_test is not None:
            test_logits, test_y = predict_logits(global_model, dl_test, device=device)
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
                    run_dir / f"round_{rnd:03d}_test.json",
                    {
                        "round": int(rnd),
                        "n": int(m_test.n),
                        "n_pos": int(m_test.n_pos),
                        "n_neg": int(m_test.n_neg),
                        "metrics_pre": asdict(m_test),
                    },
                )
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if bool(args.save_round_json):
            write_json(
                run_dir / f"round_{rnd:03d}_val.json",
                {
                    "round": int(rnd),
                    "train_loss": float(train_loss),
                    "n": int(m_val.n),
                    "n_pos": int(m_val.n_pos),
                    "n_neg": int(m_val.n_neg),
                    "metrics_pre": asdict(m_val),
                },
            )
        if bool(args.per_client_every_round) and client_test_files:
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
                    pin_memory=(device.type == "cuda"),
                    persistent_workers=(int(args.num_workers) > 0),
                )
                logits_c, y_c = predict_logits(global_model, dl_c, device=device)
                prob_c = sigmoid_np(logits_c)
                m_c = compute_binary_metrics(y_c, prob_c, n_bins=15)
                row_c = {
                    "round": int(rnd),
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
                pd.DataFrame(round_rows).to_csv(run_dir / f"round_{rnd:03d}_test_per_client.csv", index=False)
                pd.DataFrame(per_client_rounds).to_csv(run_dir / "round_client_metrics.csv", index=False)
        write_json(run_dir / f"round_{rnd:03d}_clients.json", {"clients": client_metrics})
        if args.log_client_sim and client_vecs:
            client_ids = list(client_vecs.keys())
            vecs = np.stack([client_vecs[cid] for cid in client_ids], axis=0)
            sim = _cosine_sim_matrix(vecs)
            write_json(
                run_dir / f"round_{rnd:03d}_client_similarity.json",
                {
                    "metric": "cosine_update_delta",
                    "client_ids": client_ids,
                    "n_examples": client_n,
                    "matrix": sim.tolist(),
                },
            )

        tqdm.write(
            f"[round {rnd:03d}] train_loss={train_loss:.4f} val_auprc={m_val.auprc:.4f} used={len(client_states)}"
        )

        if np.isfinite(m_val.auprc) and float(m_val.auprc) > best_val_auprc:
            best_val_auprc = float(m_val.auprc)
            best_state = {k: v.clone() for k, v in global_state.items()}
            torch.save({"model_cfg": asdict(model_cfg), "state_dict": best_state}, run_dir / "checkpoints" / "model_best.pt")

    if best_state is None:
        best_state = global_state
        torch.save({"model_cfg": asdict(model_cfg), "state_dict": best_state}, run_dir / "checkpoints" / "model_last.pt")

    # Fit temperature & threshold on VAL (global)
    best_model = IOHNet(model_cfg).to(device)
    best_model.load_state_dict(best_state, strict=True)
    val_logits, val_y = predict_logits(best_model, dl_val, device=device)
    tfit = fit_temperature(val_logits, val_y, device=str(device), max_iter=100)
    write_json(run_dir / "temperature.json", {"temperature": float(tfit.temperature), **asdict(tfit)})

    val_prob_raw = sigmoid_np(val_logits)
    val_prob_cal = sigmoid_np(val_logits / float(tfit.temperature))
    thr = best_threshold_youden(val_y, val_prob_raw, fallback=0.5)
    write_json(run_dir / "threshold.json", {"threshold": float(thr), "method": "val_youden_raw"})
    write_json(
        run_dir / "val_report.json",
        {
            "n": int(len(val_y)),
            "temperature": float(tfit.temperature),
            "threshold": float(thr),
            "metrics_pre": asdict(compute_binary_metrics(val_y, sigmoid_np(val_logits), n_bins=15)),
            "metrics_post": asdict(compute_binary_metrics(val_y, val_prob_cal, n_bins=15)),
        },
    )

    # Evaluate on TEST (global) with fixed threshold/temperature
    ds_test = WindowedNPZDataset(
        test_files,
        use_clin="true",
        cache_in_memory=bool(args.cache_in_memory),
        max_cache_files=int(args.max_cache_files),
        cache_dtype=str(args.cache_dtype),
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(args.num_workers) > 0),
    )
    test_logits, test_y = predict_logits(best_model, dl_test, device=device)
    test_prob = sigmoid_np(test_logits)
    test_prob_cal = sigmoid_np(test_logits / float(tfit.temperature))
    if args.save_test_pred_npz:
        pred_path = Path(args.save_test_pred_npz)
        if not pred_path.is_absolute() and not str(pred_path).startswith(str(run_dir)):
            pred_path = run_dir / pred_path
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        case_id = np.concatenate([np.full(n, cid, dtype=np.int64) for n, cid in zip(ds_test.file_sizes, ds_test.case_ids)])
        np.savez(
            pred_path,
            y_true=test_y.astype(np.int64, copy=False),
            prob_mean=test_prob.astype(np.float64, copy=False),
            prob_mean_cal=test_prob_cal.astype(np.float64, copy=False),
            case_id=case_id.astype(np.int64, copy=False),
        )
    write_json(
        run_dir / "test_report.json",
        {
            "n": int(len(test_y)),
            "metrics_pre": asdict(compute_binary_metrics(test_y, test_prob, n_bins=15)),
            "metrics_post": asdict(compute_binary_metrics(test_y, test_prob_cal, n_bins=15)),
            "threshold": float(thr),
        },
    )

    # Per-client TEST report (uses global temperature/threshold)
    per_client_reports: Dict[str, Any] = {}
    per_client_rows: List[Dict[str, Any]] = []
    for cid in client_ids:
        files = list_npz_files(args.data_dir, args.test_split, client_id=str(cid))
        if not files:
            per_client_reports[str(cid)] = {"client_id": str(cid), "status": "no_files", "n": 0}
            per_client_rows.append({"client_id": str(cid), "status": "no_files", "n": 0})
            continue

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
            pin_memory=(device.type == "cuda"),
            persistent_workers=(int(args.num_workers) > 0),
        )
        logits, y_true = predict_logits(best_model, dl, device=device)
        prob = sigmoid_np(logits)
        prob_cal = sigmoid_np(logits / float(tfit.temperature))

        metrics_pre = compute_binary_metrics(y_true, prob, n_bins=15)
        metrics_post = compute_binary_metrics(y_true, prob_cal, n_bins=15)
        report = {
            "client_id": str(cid),
            "status": "ok",
            "n": int(metrics_pre.n),
            "temperature": float(tfit.temperature),
            "threshold": float(thr),
            "metrics_pre": asdict(metrics_pre),
            "metrics_post": asdict(metrics_post),
            "confusion_pre": confusion_at_threshold(y_true, prob, thr=float(thr)),
            "confusion_post": confusion_at_threshold(y_true, prob_cal, thr=float(thr)),
        }
        per_client_reports[str(cid)] = report
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
                "auprc_post": float(metrics_post.auprc),
                "auroc_post": float(metrics_post.auroc),
                "brier_post": float(metrics_post.brier),
                "nll_post": float(metrics_post.nll),
                "ece_post": float(metrics_post.ece),
                "temperature": float(tfit.temperature),
                "threshold": float(thr),
            }
        )

    write_json(
        run_dir / "test_report_per_client.json",
        {"temperature": float(tfit.temperature), "threshold": float(thr), "clients": per_client_reports},
    )
    pd.DataFrame(per_client_rows).to_csv(run_dir / "test_report_per_client.csv", index=False)

    run_meta["finished_utc"] = now_utc_iso()
    run_meta["artifacts"] = {
        "history_csv": str(run_dir / "history.csv"),
        "checkpoint_best": str(run_dir / "checkpoints" / "model_best.pt"),
        "temperature_json": str(run_dir / "temperature.json"),
        "threshold_json": str(run_dir / "threshold.json"),
        "val_report_json": str(run_dir / "val_report.json"),
        "test_report_json": str(run_dir / "test_report.json"),
        "test_report_per_client_json": str(run_dir / "test_report_per_client.json"),
        "test_report_per_client_csv": str(run_dir / "test_report_per_client.csv"),
    }
    if args.save_test_pred_npz:
        pred_path = Path(args.save_test_pred_npz)
        if not pred_path.is_absolute():
            pred_path = run_dir / pred_path
        run_meta["artifacts"]["test_predictions_npz"] = str(pred_path)
    write_json(run_dir / "run_config.json", run_meta)

    print("Done.")
    print(f"Run dir: {run_dir}")
    print(f"Best val AUPRC: {best_val_auprc:.4f}")
    print(f"Val temperature: {tfit.temperature:.4f}, threshold: {thr:.3f}")


if __name__ == "__main__":
    main()
