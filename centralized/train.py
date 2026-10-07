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


_LOWER_IS_BETTER = {"ece", "brier", "nll"}


def _is_better_score(metric_name: str, score: float, prev: float | None) -> bool:
    if prev is None:
        return True
    if str(metric_name).lower() in _LOWER_IS_BETTER:
        return float(score) < float(prev)
    return float(score) > float(prev)


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


def _load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise SystemExit(f"--config not found: {cfg_path}")
    text = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() in (".json",):
        return json.loads(text) or {}
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except Exception as exc:
        raise SystemExit("PyYAML required to parse centralized config") from exc


def _cfg_get(cfg: Dict[str, Any], path: str, default: Any) -> Any:
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur if cur is not None else default


def _read_history_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if df.empty:
        return []
    return df.to_dict(orient="records")


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    cfg = _load_config(pre_args.config)

    ap = argparse.ArgumentParser(description="Centralized baseline training (hypoxemia prediction)")
    ap.add_argument("--config", default=pre_args.config, help="Optional YAML/JSON config path.")
    ap.add_argument("--data-dir", default=_cfg_get(cfg, "data.data_dir", "federated_data"), help="Output directory from scripts/build_dataset.py")
    ap.add_argument("--train-split", default=_cfg_get(cfg, "data.train_split", "train"))
    ap.add_argument("--val-split", default=_cfg_get(cfg, "data.val_split", "val"))
    ap.add_argument("--test-split", default=_cfg_get(cfg, "data.test_split", "test"))
    ap.add_argument("--out-dir", default=_cfg_get(cfg, "run.out_dir", "runs/centralized"))
    ap.add_argument("--run-name", default=_cfg_get(cfg, "run.run_name", None), help="Optional run directory name")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "run.resume", False)), help="Resume from checkpoints/model_last.pt.")

    ap.add_argument("--epochs", type=int, default=_cfg_get(cfg, "train.epochs", 100))
    ap.add_argument("--batch-size", type=int, default=_cfg_get(cfg, "train.batch_size", 64))
    ap.add_argument("--lr", type=float, default=_cfg_get(cfg, "train.lr", 1e-3))
    ap.add_argument("--weight-decay", type=float, default=_cfg_get(cfg, "train.weight_decay", 1e-4))
    ap.add_argument("--seed", type=int, default=_cfg_get(cfg, "train.seed", 42))


    ap.add_argument("--model-base-channels", type=int, default=_cfg_get(cfg, "model.base_channels", 32))
    ap.add_argument("--dropout", type=float, default=_cfg_get(cfg, "model.dropout", 0.1))
    ap.add_argument("--use-gru", dest="use_gru", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "model.use_gru", False)), help="Deprecated (no-op).")
    ap.add_argument("--use-lstm", dest="use_gru", action="store_true", help="Deprecated (no-op).")
    ap.add_argument("--gru-hidden", dest="gru_hidden", type=int, default=_cfg_get(cfg, "model.gru_hidden", 64), help="Deprecated (no-op).")
    ap.add_argument("--lstm-hidden", dest="gru_hidden", type=int, help="Deprecated (no-op).")

    ap.add_argument("--num-workers", type=int, default=_cfg_get(cfg, "train.num_workers", 0))
    ap.add_argument("--prefetch-factor", type=int, default=_cfg_get(cfg, "train.prefetch_factor", 2), help="DataLoader prefetch factor (workers>0).")
    ap.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "train.pin_memory", True)))
    ap.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "train.persistent_workers", True)))
    ap.add_argument("--cache-in-memory", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "train.cache_in_memory", False)))
    ap.add_argument("--max-cache-files", type=int, default=_cfg_get(cfg, "train.max_cache_files", 32))
    ap.add_argument("--cache-dtype", default=_cfg_get(cfg, "train.cache_dtype", "float32"), choices=["float16", "float32"])
    ap.add_argument("--log-interval", type=int, default=_cfg_get(cfg, "train.log_interval", 50), help="Batch interval for progress updates.")
    ap.add_argument("--no-progress-bar", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "train.no_progress_bar", False)), help="Disable per-epoch progress bar.")
    ap.add_argument("--val-every-epoch", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "train.val_every_epoch", True)))
    ap.add_argument("--test-every-epoch", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "train.test_every_epoch", False)))
    ap.add_argument("--save-round-json", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "train.save_round_json", True)))
    ap.add_argument("--per-client-every-epoch", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "train.per_client_every_epoch", False)))
    ap.add_argument("--eval-threshold", type=float, default=_cfg_get(cfg, "eval.eval_threshold", 0.5), help="Fixed threshold for round-by-round metrics.")
    ap.add_argument("--model-selection", default=_cfg_get(cfg, "eval.model_selection", "best"), choices=["last", "best"])
    ap.add_argument("--selection-metric", default=_cfg_get(cfg, "eval.selection_metric", "nll"), choices=["auroc", "auprc", "ece", "brier", "nll"])
    args = ap.parse_args()

    set_seed(int(args.seed))
    torch.set_num_threads(max(1, min(os.cpu_count() or 4, 8)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = args.run_name or f"run_{now_utc_iso().replace(':', '').replace('-', '')}"
    run_dir = Path(args.out_dir) / run_name
    if bool(args.resume) and not run_dir.exists():
        raise SystemExit(f"--resume requested but run_dir not found: {run_dir}")
    run_dir = ensure_dir(run_dir)
    ensure_dir(run_dir / "checkpoints")
    save_env_snapshot(run_dir, {"args": vars(args)})

    train_files = list_npz_files(args.data_dir, args.train_split)
    val_files = list_npz_files(args.data_dir, args.val_split)
    test_files = list_npz_files(args.data_dir, args.test_split)
    if not train_files:
        raise SystemExit("No train files found. Check --data-dir and split names.")
    train_pos, train_total = scan_label_stats(train_files)
    val_pos, val_total = scan_label_stats(val_files) if val_files else (0, 0)
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
    ds_val = None
    if val_files and bool(args.val_every_epoch):
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
        dl_val = None
        if ds_val is not None:
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
        dl_val = DataLoader(ds_val, shuffle=False, **dl_common) if ds_val is not None else None
        dl_test = DataLoader(ds_test, shuffle=False, **dl_common) if ds_test is not None else None

    model_cfg = IOHModelConfig(
        in_channels=int(getattr(ds_train, "wave_channels", 4) or 4),
        base_channels=int(args.model_base_channels),
        dropout=float(args.dropout),
        use_gru=False,
        gru_hidden=64,
        clin_dim=int(getattr(ds_train, "clin_dim", 0) or 0),
    )
    model = IOHNet(model_cfg).to(device)

    selection_source = "val" if dl_val is not None else ("test" if dl_test is not None else "train")
    selection_enabled = str(args.model_selection).lower() == "best"
    selection_metric = str(args.selection_metric).lower()
    best = {"epoch": 0, "metric": None, "source": selection_source, "metric_name": selection_metric}
    history_rows: List[Dict[str, Any]] = []
    start_epoch = 1
    last_path: Path | None = None
    best_path: Path | None = run_dir / "checkpoints" / "model_best.pt"
    if not best_path.exists():
        best_path = None
    resume_ckpt: Dict[str, Any] | None = None
    if bool(args.resume):
        ckpt_path = run_dir / "checkpoints" / "model_last.pt"
        if not ckpt_path.exists():
            raise SystemExit(f"--resume requested but checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        loaded_model_cfg = ckpt.get("model_cfg", None)
        if loaded_model_cfg is not None:
            model_cfg = normalize_model_cfg(loaded_model_cfg)
            model = IOHNet(model_cfg).to(device)
        loaded_state = ckpt.get("state_dict", None)
        if not isinstance(loaded_state, dict):
            raise SystemExit(f"Invalid checkpoint (missing state_dict): {ckpt_path}")
        model.load_state_dict(loaded_state, strict=True)
        resume_ckpt = ckpt

        history_rows = _read_history_csv(run_dir / "history.csv")
        last_epoch_history = 0
        for row in history_rows:
            try:
                last_epoch_history = max(last_epoch_history, int(float(row.get("epoch", 0))))
            except Exception:
                continue
        last_epoch_ckpt = int(ckpt.get("epoch", 0) or 0)
        last_epoch = max(last_epoch_history, last_epoch_ckpt)
        start_epoch = int(last_epoch) + 1

        ckpt_best = ckpt.get("best")
        if isinstance(ckpt_best, dict) and ckpt_best.get("metric") is not None:
            best = {
                "epoch": int(ckpt_best.get("epoch", 0) or 0),
                "metric": float(ckpt_best.get("metric")),
                "source": str(ckpt_best.get("source", selection_source)),
                "metric_name": str(ckpt_best.get("metric_name", selection_metric)),
            }
        elif selection_enabled:
            metric_key = f"{selection_source}_{selection_metric}"
            for row in history_rows:
                val = row.get(metric_key, None)
                if val is None:
                    continue
                try:
                    score = float(val)
                except Exception:
                    continue
                prev = best.get("metric")
                if _is_better_score(selection_metric, score, None if prev is None else float(prev)):
                    try:
                        epoch_idx = int(float(row.get("epoch", 0)))
                    except Exception:
                        epoch_idx = 0
                    best = {"epoch": epoch_idx, "metric": score, "source": selection_source, "metric_name": selection_metric}
        print(f"[INFO] resume mode: run_dir={run_dir} start_epoch={start_epoch} epochs={int(args.epochs)}")

    # Loss (BCE, with optional pos_weight)
    n_pos = max(train_counts["n_pos"], 1)
    n_neg = max(train_counts["n_neg"], 1)
    pos_weight = float(n_neg) / float(n_pos) if n_pos > 0 else 1.0
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    if resume_ckpt is not None:
        opt_state = resume_ckpt.get("optimizer_state", None)
        if isinstance(opt_state, dict):
            try:
                opt.load_state_dict(opt_state)
            except Exception as exc:
                print(f"[WARN] failed to load optimizer_state; continuing with a fresh optimizer: {exc}")
        scaler_state = resume_ckpt.get("scaler_state", None)
        if isinstance(scaler_state, dict):
            try:
                scaler.load_state_dict(scaler_state)
            except Exception as exc:
                print(f"[WARN] failed to load scaler_state; continuing with a fresh scaler: {exc}")

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
            "loss": "bce",
            "pos_weight": float(pos_weight),
            "log_interval": int(args.log_interval),
            "progress_bar": bool(not args.no_progress_bar),
            "num_workers": int(args.num_workers),
            "prefetch_factor": int(args.prefetch_factor),
            "pin_memory": bool(args.pin_memory),
            "persistent_workers": bool(args.persistent_workers),
            "val_every_epoch": bool(args.val_every_epoch),
            "test_every_epoch": bool(args.test_every_epoch),
            "save_round_json": bool(args.save_round_json),
            "per_client_every_epoch": bool(args.per_client_every_epoch),
            "eval_threshold": float(args.eval_threshold),
            "model_selection": str(args.model_selection),
            "selection_metric": str(args.selection_metric),
        },
        "model": asdict(model_cfg),
        "resume": bool(args.resume),
        "start_epoch": int(start_epoch),
        "resumed_from_epoch": int(start_epoch - 1) if bool(args.resume) else 0,
    }
    if bool(args.resume):
        run_meta["resumed_utc"] = now_utc_iso()
    write_json(run_dir / "run_config.json", run_meta)

    per_client_rounds: list[dict[str, Any]] = []
    eval_split = str(args.val_split) if val_files else str(args.test_split)
    client_eval_files: Dict[str, list[str]] = {}
    if bool(args.per_client_every_epoch) and eval_split:
        client_eval_files = {cid: list_npz_files(args.data_dir, eval_split, client_id=str(cid)) for cid in list_client_ids(args.data_dir)}

    last_completed_epoch = int(start_epoch - 1)
    for epoch in range(int(start_epoch), int(args.epochs) + 1):
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
        m_val = None
        if dl_val is not None:
            val_logits, val_y = _predict_logits(model, dl_val, device=device)
            val_prob = sigmoid_np(val_logits)
            m_val = compute_binary_metrics(val_y, val_prob, n_bins=15)
            row.update(
                {
                    "val_auprc": float(m_val.auprc),
                    "val_auroc": float(m_val.auroc),
                    "val_brier": float(m_val.brier),
                    "val_nll": float(m_val.nll),
                    "val_ece": float(m_val.ece),
                }
            )
            if bool(args.save_round_json):
                thr = float(args.eval_threshold)
                metrics_thr = calc_comprehensive_metrics(val_y, val_prob, threshold=thr)
                write_json(
                    run_dir / f"round_{epoch:03d}_val.json",
                    {
                        "epoch": int(epoch),
                        "n": int(m_val.n),
                        "n_pos": int(m_val.n_pos),
                        "n_neg": int(m_val.n_neg),
                        "metrics_pre": asdict(m_val),
                        "threshold": float(thr),
                        "metrics_threshold": metrics_thr,
                        "confusion_pre": confusion_at_threshold(val_y, val_prob, thr=thr),
                    },
                )
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
            if dl_val is not None:
                metrics = m_val
            elif dl_test is not None:
                metrics = m_test
            if metrics is not None:
                score = float(getattr(metrics, metric_name))
                prev = best.get("metric")
                if _is_better_score(metric_name, score, None if prev is None else float(prev)):
                    best = {"epoch": int(epoch), "metric": float(score), "source": selection_source, "metric_name": metric_name}
                    best_path = run_dir / "checkpoints" / "model_best.pt"
                    torch.save(
                        {
                            "epoch": int(epoch),
                            "selection": "best",
                            "best": best,
                            "model_cfg": asdict(model_cfg),
                            "state_dict": model.state_dict(),
                        },
                        best_path,
                    )

        if bool(args.per_client_every_epoch) and client_eval_files:
            round_rows = []
            for cid, files in client_eval_files.items():
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
                    "split": str(eval_split),
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
                pd.DataFrame(round_rows).to_csv(run_dir / f"round_{epoch:03d}_{eval_split}_per_client.csv", index=False)
                pd.DataFrame(per_client_rounds).to_csv(run_dir / "round_client_metrics.csv", index=False)
                write_json(run_dir / "round_client_metrics.json", per_client_rounds)

        last_path = run_dir / "checkpoints" / "model_last.pt"
        last_completed_epoch = int(epoch)
        torch.save(
            {
                "epoch": int(epoch),
                "selection": "last",
                "best": best,
                "model_cfg": asdict(model_cfg),
                "state_dict": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "scaler_state": scaler.state_dict(),
            },
            last_path,
        )

        tqdm.write(f"[{epoch:03d}] loss={tr_loss:.4f}")

    if last_path is None:
        last_path = run_dir / "checkpoints" / "model_last.pt"
        torch.save(
            {
                "epoch": int(last_completed_epoch),
                "selection": "last",
                "best": best,
                "model_cfg": asdict(model_cfg),
                "state_dict": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "scaler_state": scaler.state_dict(),
            },
            last_path,
        )

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
    print(f"Selection source: {selection_source}")


if __name__ == "__main__":
    main()
