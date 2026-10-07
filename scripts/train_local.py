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
from common.ioh_model import IOHModelConfig, IOHNet, normalize_model_cfg
from common.metrics import compute_binary_metrics, sigmoid_np
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
        raise SystemExit("PyYAML required to parse local config") from exc


def _cfg_get(cfg: Dict[str, Any], path: str, default: Any) -> Any:
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur if cur is not None else default


def _read_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    return []


def _state_dict_to_cpu(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _eval_client_split_row(
    *,
    model: IOHNet,
    dl: DataLoader | None,
    client_id: str,
    split: str,
    rnd: int | None,
    train_loss: float | None,
    thr: float,
    device: torch.device,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "client_id": str(client_id),
        "split": str(split),
        "threshold": float(thr),
        "train_loss": float(train_loss) if train_loss is not None else float("nan"),
    }
    if rnd is not None:
        row["round"] = int(rnd)
    if dl is None:
        row["status"] = f"no_{split}_files"
        row["n"] = 0
        return row

    logits_c, y_c = _predict_logits(model, dl, device=device)
    prob_c = sigmoid_np(logits_c)
    m_c = compute_binary_metrics(y_c, prob_c, n_bins=15)
    m_thr = calc_comprehensive_metrics(y_c, prob_c, threshold=float(thr))
    row.update(
        {
            "status": "ok",
            "n": int(m_c.n),
            "n_pos": int(m_c.n_pos),
            "n_neg": int(m_c.n_neg),
            "pos_rate": float(m_thr.get("pos_rate", float("nan"))),
            "auprc": float(m_c.auprc),
            "auroc": float(m_c.auroc),
            "brier": float(m_c.brier),
            "nll": float(m_c.nll),
            "ece": float(m_c.ece),
            "accuracy": float(m_thr.get("accuracy", float("nan"))),
            "f1": float(m_thr.get("f1", float("nan"))),
            "sensitivity": float(m_thr.get("sensitivity", float("nan"))),
            "specificity": float(m_thr.get("specificity", float("nan"))),
            "ppv": float(m_thr.get("ppv", float("nan"))),
            "npv": float(m_thr.get("npv", float("nan"))),
        }
    )
    return row


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    cfg = _load_config(pre_args.config)

    ap = argparse.ArgumentParser(description="Local training baseline (per-client, no aggregation).")
    ap.add_argument("--config", default=pre_args.config, help="Optional YAML/JSON config path.")
    ap.add_argument("--data-dir", default=_cfg_get(cfg, "data.data_dir", "federated_data"), help="Output directory from scripts/build_dataset.py")
    ap.add_argument("--train-split", default=_cfg_get(cfg, "data.train_split", "train"))
    ap.add_argument("--val-split", default=_cfg_get(cfg, "data.val_split", "val"))
    ap.add_argument("--test-split", default=_cfg_get(cfg, "data.test_split", "test"))
    ap.add_argument("--out-dir", default=_cfg_get(cfg, "run.out_dir", "runs/local"))
    ap.add_argument("--run-name", default=_cfg_get(cfg, "run.run_name", None), help="Optional run directory name")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "run.resume", False)), help="Resume from checkpoints/client_*_last.pt.")

    ap.add_argument("--rounds", type=int, default=_cfg_get(cfg, "train.rounds", 100))
    ap.add_argument("--batch-size", type=int, default=_cfg_get(cfg, "train.batch_size", 64))
    ap.add_argument("--lr", type=float, default=_cfg_get(cfg, "train.lr", 1e-3))
    ap.add_argument("--weight-decay", type=float, default=_cfg_get(cfg, "train.weight_decay", 1e-4))
    ap.add_argument("--seed", type=int, default=_cfg_get(cfg, "train.seed", 42))
    ap.add_argument("--eval-threshold", type=float, default=_cfg_get(cfg, "eval.eval_threshold", 0.5), help="Fixed threshold for round-by-round metrics.")
    ap.add_argument("--val-every-round", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "train.val_every_round", True)))
    ap.add_argument("--test-every-round", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "train.test_every_round", False)))
    ap.add_argument("--model-selection", default=_cfg_get(cfg, "eval.model_selection", "best"), choices=["last", "best"])
    ap.add_argument("--selection-metric", default=_cfg_get(cfg, "eval.selection_metric", "nll"), choices=["auroc", "auprc", "ece", "brier", "nll"])

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

    client_ids = list_client_ids(args.data_dir)
    client_train_files: Dict[str, List[str]] = {}
    client_val_files: Dict[str, List[str]] = {}
    client_test_files: Dict[str, List[str]] = {}
    for cid in client_ids:
        train_files = list_npz_files(args.data_dir, args.train_split, client_id=str(cid))
        val_files = list_npz_files(args.data_dir, args.val_split, client_id=str(cid))
        test_files = list_npz_files(args.data_dir, args.test_split, client_id=str(cid))
        if train_files:
            client_train_files[str(cid)] = train_files
            client_val_files[str(cid)] = val_files
            client_test_files[str(cid)] = test_files
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
        use_gru=False,
        gru_hidden=64,
        clin_dim=int(getattr(ds_sample, "clin_dim", 0) or 0),
    )
    if bool(args.resume):
        sample_ckpt_path = run_dir / "checkpoints" / f"client_{client_ids[0]}_last.pt"
        if not sample_ckpt_path.exists():
            raise SystemExit(f"--resume requested but checkpoint not found: {sample_ckpt_path}")
        sample_ckpt = torch.load(sample_ckpt_path, map_location="cpu")
        loaded_model_cfg = sample_ckpt.get("model_cfg", None)
        if loaded_model_cfg is not None:
            model_cfg = normalize_model_cfg(loaded_model_cfg)

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
    client_val_dl: Dict[str, DataLoader] = {}
    client_test_dl: Dict[str, DataLoader] = {}
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

    splits = {"train": str(args.train_split), "val": str(args.val_split), "test": str(args.test_split)}
    run_meta: Dict[str, Any] = {
        "started_utc": now_utc_iso(),
        "git_hash": get_git_hash(PROJECT_ROOT),
        "device": str(device),
        "data_dir": str(args.data_dir),
        "splits": splits,
        "seed": int(args.seed),
        "rounds": int(args.rounds),
        "resume": bool(args.resume),
        "clients": client_ids,
        "counts_train": client_counts,
        "hyper": {
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "eval_threshold": float(args.eval_threshold),
            "val_every_round": bool(args.val_every_round),
            "test_every_round": bool(args.test_every_round),
            "model_selection": str(args.model_selection),
            "selection_metric": str(args.selection_metric),
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

    selection_metric = str(args.selection_metric).lower()
    selection_requested = str(args.model_selection).lower() == "best"
    has_val_split = any(client_val_dl.get(cid) is not None for cid in client_ids)
    selection_enabled = bool(selection_requested and has_val_split and bool(args.val_every_round))
    best_tracker: Dict[str, Dict[str, Any]] = {}
    best_states: Dict[str, Dict[str, torch.Tensor]] = {}

    per_client_rounds: List[Dict[str, Any]] = []
    start_round = 1
    if bool(args.resume):
        per_client_rounds = _read_json_list(run_dir / "round_client_metrics.json")
        loaded_rounds: List[int] = []
        for cid in client_ids:
            ckpt_path = run_dir / "checkpoints" / f"client_{cid}_last.pt"
            if not ckpt_path.exists():
                raise SystemExit(f"--resume requested but checkpoint not found: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            loaded_state = ckpt.get("state_dict", None)
            if not isinstance(loaded_state, dict):
                raise SystemExit(f"Invalid checkpoint (missing state_dict): {ckpt_path}")
            models[cid].load_state_dict(loaded_state, strict=True)
            loaded_rounds.append(int(ckpt.get("round", 0) or 0))
            opt_state = ckpt.get("optimizer_state", None)
            if isinstance(opt_state, dict):
                try:
                    opts[cid].load_state_dict(opt_state)
                except Exception as exc:
                    print(f"[WARN] failed to load optimizer_state for client {cid}; continuing with a fresh optimizer: {exc}")
            scaler_state = ckpt.get("scaler_state", None)
            if isinstance(scaler_state, dict):
                try:
                    scalers[cid].load_state_dict(scaler_state)
                except Exception as exc:
                    print(f"[WARN] failed to load scaler_state for client {cid}; continuing with a fresh scaler: {exc}")

            best_path = run_dir / "checkpoints" / f"client_{cid}_best.pt"
            if best_path.exists():
                best_ckpt = torch.load(best_path, map_location="cpu")
                best_state = best_ckpt.get("state_dict", None)
                if isinstance(best_state, dict):
                    best_states[cid] = {k: v.detach().cpu().clone() for k, v in best_state.items()}
                    if best_ckpt.get("metric") is not None:
                        best_tracker[cid] = {
                            "round": int(best_ckpt.get("round", 0) or 0),
                            "metric": float(best_ckpt.get("metric")),
                            "metric_name": str(best_ckpt.get("metric_name", selection_metric)),
                            "source": str(best_ckpt.get("source", "val")),
                        }

        last_round_ckpt = min(loaded_rounds) if loaded_rounds else 0
        start_round = int(last_round_ckpt) + 1

        if selection_enabled:
            for row in per_client_rounds:
                if row.get("split") != "val" or row.get("status") != "ok":
                    continue
                cid = str(row.get("client_id", ""))
                if cid not in client_ids or cid in best_states:
                    continue
                val = row.get(selection_metric, None)
                if val is None:
                    continue
                try:
                    score = float(val)
                    round_idx = int(float(row.get("round", 0)))
                except Exception:
                    continue
                prev = best_tracker.get(cid, {}).get("metric")
                if _is_better_score(selection_metric, score, None if prev is None else float(prev)):
                    best_tracker[cid] = {"round": round_idx, "metric": score, "metric_name": str(selection_metric), "source": "val"}

        print(f"[INFO] resume mode: run_dir={run_dir} start_round={start_round} rounds={int(args.rounds)}")

    run_meta["start_round"] = int(start_round)
    run_meta["resumed_from_round"] = int(start_round - 1) if bool(args.resume) else 0
    if bool(args.resume):
        run_meta["resumed_utc"] = now_utc_iso()
    write_json(run_dir / "run_config.json", run_meta)

    last_completed_round = int(start_round - 1)
    for rnd in range(int(start_round), int(args.rounds) + 1):
        round_val_rows = []
        round_test_rows = []
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
        for cid in client_ids:
            tr_loss = float(train_losses.get(str(cid), float("nan")))
            if bool(args.val_every_round):
                row_val = _eval_client_split_row(
                    model=models[cid],
                    dl=client_val_dl.get(cid),
                    client_id=str(cid),
                    split="val",
                    rnd=int(rnd),
                    train_loss=float(tr_loss),
                    thr=float(thr),
                    device=device,
                )
                round_val_rows.append(row_val)
                per_client_rounds.append(row_val)
                if selection_enabled and row_val.get("status") == "ok":
                    score = float(row_val.get(selection_metric, float("nan")))
                    prev = best_tracker.get(cid, {}).get("metric")
                    if _is_better_score(selection_metric, score, None if prev is None else float(prev)):
                        best_tracker[cid] = {"round": int(rnd), "metric": float(score), "metric_name": str(selection_metric), "source": "val"}
                        best_states[cid] = _state_dict_to_cpu(models[cid])
                        torch.save(
                            {
                                "client_id": str(cid),
                                "round": int(rnd),
                                "selection": "best",
                                "source": "val",
                                "metric_name": str(selection_metric),
                                "metric": float(score),
                                "model_cfg": asdict(model_cfg),
                                "state_dict": best_states[cid],
                            },
                            run_dir / "checkpoints" / f"client_{cid}_best.pt",
                        )

            if bool(args.test_every_round):
                row_test = _eval_client_split_row(
                    model=models[cid],
                    dl=client_test_dl.get(cid),
                    client_id=str(cid),
                    split="test",
                    rnd=int(rnd),
                    train_loss=float(tr_loss),
                    thr=float(thr),
                    device=device,
                )
                round_test_rows.append(row_test)
                per_client_rounds.append(row_test)

        if round_val_rows:
            pd.DataFrame(round_val_rows).to_csv(run_dir / f"round_{rnd:03d}_val_per_client.csv", index=False)
        if round_test_rows:
            pd.DataFrame(round_test_rows).to_csv(run_dir / f"round_{rnd:03d}_test_per_client.csv", index=False)
        pd.DataFrame(per_client_rounds).to_csv(run_dir / "round_client_metrics.csv", index=False)
        write_json(run_dir / "round_client_metrics.json", per_client_rounds)
        last_completed_round = int(rnd)
        for cid in client_ids:
            torch.save(
                {
                    "client_id": str(cid),
                    "round": int(rnd),
                    "selection": "last",
                    "model_cfg": asdict(model_cfg),
                    "state_dict": _state_dict_to_cpu(models[cid]),
                    "optimizer_state": opts[cid].state_dict(),
                    "scaler_state": scalers[cid].state_dict(),
                },
                run_dir / "checkpoints" / f"client_{cid}_last.pt",
            )

    final_round = int(last_completed_round)
    for cid in client_ids:
        last_state = _state_dict_to_cpu(models[cid])
        torch.save(
            {
                "client_id": str(cid),
                "round": int(final_round),
                "selection": "last",
                "model_cfg": asdict(model_cfg),
                "state_dict": last_state,
                "optimizer_state": opts[cid].state_dict(),
                "scaler_state": scalers[cid].state_dict(),
            },
            run_dir / "checkpoints" / f"client_{cid}_last.pt",
        )
        if cid in best_states:
            torch.save(
                {
                    "client_id": str(cid),
                    "round": int(best_tracker[cid]["round"]),
                    "selection": "best",
                    "source": "val",
                    "metric_name": str(best_tracker[cid]["metric_name"]),
                    "metric": float(best_tracker[cid]["metric"]),
                    "model_cfg": asdict(model_cfg),
                    "state_dict": best_states[cid],
                },
                run_dir / "checkpoints" / f"client_{cid}_best.pt",
            )

    # Final per-client summary (evaluate selected model on test split)
    per_client_reports: Dict[str, Any] = {}
    per_client_rows: List[Dict[str, Any]] = []
    selection_summary: Dict[str, Any] = {}
    for cid in client_ids:
        model_eval = IOHNet(model_cfg).to(device)
        selected = "last"
        selected_round = int(final_round)
        if selection_enabled and cid in best_states:
            model_eval.load_state_dict(best_states[cid], strict=True)
            selected = "best"
            selected_round = int(best_tracker[cid]["round"])
            selection_summary[cid] = {
                "client_id": str(cid),
                "selection": "best",
                "source": "val",
                "round": int(best_tracker[cid]["round"]),
                "metric_name": str(best_tracker[cid]["metric_name"]),
                "metric": float(best_tracker[cid]["metric"]),
            }
        else:
            model_eval.load_state_dict(_state_dict_to_cpu(models[cid]), strict=True)
            selection_summary[cid] = {"client_id": str(cid), "selection": "last", "round": int(final_round)}

        row = _eval_client_split_row(
            model=model_eval,
            dl=client_test_dl.get(cid),
            client_id=str(cid),
            split="test",
            rnd=None,
            train_loss=None,
            thr=float(args.eval_threshold),
            device=device,
        )
        row["model_selection"] = str(selected)
        row["selected_round"] = int(selected_round)
        per_client_reports[cid] = row
        per_client_rows.append(row)

    write_json(
        run_dir / "test_report_per_client.json",
        {"round": int(final_round), "selection_enabled": bool(selection_enabled), "clients": per_client_reports},
    )
    pd.DataFrame(per_client_rows).to_csv(run_dir / "test_report_per_client.csv", index=False)
    write_json(run_dir / "selection_summary.json", selection_summary)

    run_meta["finished_utc"] = now_utc_iso()
    run_meta["artifacts"] = {
        "round_client_metrics_csv": str(run_dir / "round_client_metrics.csv"),
        "round_client_metrics_json": str(run_dir / "round_client_metrics.json"),
        "selection_summary_json": str(run_dir / "selection_summary.json"),
        "test_report_per_client_json": str(run_dir / "test_report_per_client.json"),
        "test_report_per_client_csv": str(run_dir / "test_report_per_client.csv"),
    }
    write_json(run_dir / "run_config.json", run_meta)

    print("Done.")
    print(f"Run dir: {run_dir}")
    print(f"Model selection: {'best(val)' if selection_enabled else 'last'}")


if __name__ == "__main__":
    main()
