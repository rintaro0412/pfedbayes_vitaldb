from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
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

from common.io import ensure_dir, get_git_hash, now_utc_iso, write_json
from common.dataset import WindowedNPZDataset, list_client_ids, list_npz_files, list_npz_files_by_client
from common.ioh_model import IOHModelConfig, IOHNet, normalize_model_cfg
from common.metrics import compute_binary_metrics, confusion_at_threshold, sigmoid_np
from common.experiment import save_env_snapshot
from common.trace import hash_state_dict, l2_diff_state_dict
from common.utils import calc_comprehensive_metrics, set_seed
from federated.client import LocalTrainConfig, infer_pos_weight, train_one_client

LAST_META_PATH: Path | None = None
_LOWER_IS_BETTER = {"ece", "brier", "nll"}
_FL_ALGOS = ("fedavg", "fedprox", "scaffold", "fednova", "perfedavg", "pfedme")


def _is_better_score(metric_name: str, score: float, prev: float | None) -> bool:
    if prev is None:
        return True
    if str(metric_name).lower() in _LOWER_IS_BETTER:
        return float(score) < float(prev)
    return float(score) > float(prev)


def _copy_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def _zeros_like_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: torch.zeros_like(v.detach().cpu()) for k, v in state.items()}


def _coerce_state_like(ref_state: Dict[str, torch.Tensor], raw_state: Any) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    raw = raw_state if isinstance(raw_state, dict) else {}
    for k, ref in ref_state.items():
        rv = raw.get(k, None)
        if torch.is_tensor(rv):
            out[k] = rv.detach().cpu().to(dtype=ref.dtype)
        else:
            out[k] = torch.zeros_like(ref.detach().cpu())
    return out


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


def fednova(
    global_state: Dict[str, torch.Tensor],
    states: List[Dict[str, torch.Tensor]],
    weights: List[int],
    local_normalizers: List[float],
) -> tuple[Dict[str, torch.Tensor], float]:
    if not states:
        raise ValueError("no client states to aggregate")
    if len(states) != len(weights) or len(states) != len(local_normalizers):
        raise ValueError("states/weights/local_normalizers length mismatch")

    total_w = float(sum(int(w) for w in weights))
    if total_w <= 0:
        raise ValueError("sum(weights) must be >0")
    probs = [float(int(w)) / total_w for w in weights]
    a_vals = [max(float(a), 1e-12) for a in local_normalizers]
    tau_eff = float(sum(p * a for p, a in zip(probs, a_vals)))

    out: Dict[str, torch.Tensor] = {}
    for k in global_state.keys():
        if not torch.is_floating_point(global_state[k]):
            acc = None
            for sd, w in zip(states, weights):
                t = sd[k].float()
                if acc is None:
                    acc = t * float(w)
                else:
                    acc = acc + t * float(w)
            assert acc is not None
            out[k] = (acc / total_w).type_as(global_state[k])
            continue
        g = global_state[k].float()
        d = torch.zeros_like(g)
        for sd, p, a_i in zip(states, probs, a_vals):
            d = d + float(p) * (g - sd[k].float()) / float(a_i)
        out[k] = (g - float(tau_eff) * d).type_as(global_state[k])
    return out, float(tau_eff)


def _update_scaffold_server_control(
    *,
    server_control: Dict[str, torch.Tensor],
    client_control_deltas: List[Dict[str, torch.Tensor]],
    n_total_clients: int,
    weighting: str = "uniform",
    client_weights: List[int] | None = None,
) -> Dict[str, torch.Tensor]:
    if not client_control_deltas:
        return _copy_state_dict(server_control)
    out = _copy_state_dict(server_control)
    factor = float(len(client_control_deltas)) / float(max(int(n_total_clients), 1))
    mode = str(weighting).strip().lower()
    if mode not in {"uniform", "sample"}:
        raise ValueError(f"unknown scaffold weighting: {weighting}")
    probs: List[float] | None = None
    if mode == "sample":
        if client_weights is None or len(client_weights) != len(client_control_deltas):
            raise ValueError("scaffold sample weighting requires client_weights aligned with client_control_deltas")
        tot = float(sum(int(w) for w in client_weights))
        if tot <= 0:
            raise ValueError("sum(client_weights) must be >0 for scaffold sample weighting")
        probs = [float(int(w)) / tot for w in client_weights]
    for k in out.keys():
        acc = torch.zeros_like(out[k], dtype=torch.float32)
        if mode == "sample":
            assert probs is not None
            for dc, p in zip(client_control_deltas, probs):
                acc = acc + float(p) * dc[k].detach().cpu().float()
            delta = acc
        else:
            for dc in client_control_deltas:
                acc = acc + dc[k].detach().cpu().float()
            delta = acc / float(len(client_control_deltas))
        out[k] = (out[k].float() + float(factor) * delta).to(dtype=out[k].dtype)
    return out


def _blend_states(
    prev_state: Dict[str, torch.Tensor],
    new_state: Dict[str, torch.Tensor],
    *,
    beta: float,
) -> Dict[str, torch.Tensor]:
    b = float(beta)
    if b < 0.0:
        raise ValueError("beta must be >= 0")
    out: Dict[str, torch.Tensor] = {}
    for k, prev in prev_state.items():
        new = new_state[k]
        if torch.is_floating_point(prev):
            out[k] = ((1.0 - b) * prev.float() + b * new.float()).to(dtype=prev.dtype)
        else:
            out[k] = new.detach().cpu().clone().to(dtype=prev.dtype)
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
        raise SystemExit("PyYAML required to parse fedavg config") from exc


def _cfg_get(cfg: Dict[str, Any], path: str, default: Any) -> Any:
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur if cur is not None else default


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    cfg = _load_config(pre_args.config)

    ap = argparse.ArgumentParser(description="Federated baseline trainer (FedAvg/FedProx/SCAFFOLD/FedNova/Per-FedAvg/pFedMe) for hypoxemia prediction")
    ap.add_argument("--config", default=pre_args.config, help="Optional YAML/JSON config path.")
    ap.add_argument("--data-dir", default=_cfg_get(cfg, "data.data_dir", "federated_data"), help="Output of scripts/build_dataset.py")
    ap.add_argument("--out-dir", default=_cfg_get(cfg, "run.out_dir", "runs/fedavg"))
    ap.add_argument("--run-name", default=_cfg_get(cfg, "run.run_name", None))
    ap.add_argument("--run-dir", default=_cfg_get(cfg, "run.run_dir", None), help="Explicit run directory (overrides LEGACY_RUN_DIR/default).")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "run.resume", False)), help="Resume from checkpoints/model_last.pt.")
    ap.add_argument(
        "--fl-algo",
        default=str(_cfg_get(cfg, "federated.algo", _cfg_get(cfg, "run.fl_algo", "fedavg"))).lower(),
        choices=list(_FL_ALGOS),
        help="Federated optimization algorithm.",
    )
    ap.add_argument(
        "--fedprox-mu",
        type=float,
        default=float(_cfg_get(cfg, "federated.fedprox_mu", 0.1)),
        help="FedProx proximal coefficient (used when --fl-algo fedprox).",
    )
    ap.add_argument(
        "--scaffold-weighting",
        default=str(_cfg_get(cfg, "federated.scaffold_weighting", "uniform")).lower(),
        choices=["uniform", "sample"],
        help="SCAFFOLD aggregation weighting for both x and c updates: uniform (paper-style) or sample (p_i-weighted).",
    )
    ap.add_argument(
        "--perfedavg-alpha",
        type=float,
        default=float(_cfg_get(cfg, "federated.perfedavg_alpha", _cfg_get(cfg, "train.lr", 1e-3))),
        help="Per-FedAvg(FO) inner adaptation learning rate alpha.",
    )
    ap.add_argument(
        "--perfedavg-beta",
        type=float,
        default=float(_cfg_get(cfg, "federated.perfedavg_beta", _cfg_get(cfg, "train.lr", 1e-3))),
        help="Per-FedAvg(FO) meta update learning rate beta.",
    )
    ap.add_argument(
        "--perfedavg-weighting",
        default=str(_cfg_get(cfg, "federated.perfedavg_weighting", "uniform")).lower(),
        choices=["uniform", "sample"],
        help="Per-FedAvg server averaging: uniform (paper) or sample (n_i-weighted).",
    )
    ap.add_argument(
        "--pfedme-lambda",
        type=float,
        default=float(_cfg_get(cfg, "federated.pfedme_lambda", 15.0)),
        help="pFedMe regularization coefficient lambda.",
    )
    ap.add_argument(
        "--pfedme-beta",
        type=float,
        default=float(_cfg_get(cfg, "federated.pfedme_beta", 1.0)),
        help="pFedMe server moving-average coefficient beta (paper default 1.0).",
    )
    ap.add_argument(
        "--pfedme-weighting",
        default=str(_cfg_get(cfg, "federated.pfedme_weighting", "uniform")).lower(),
        choices=["uniform", "sample"],
        help="pFedMe server averaging of client w_i: uniform (paper) or sample (repo-style).",
    )
    ap.add_argument(
        "--pfedme-k",
        type=int,
        default=int(_cfg_get(cfg, "federated.pfedme_k", 5)),
        help="pFedMe inner personalized gradient steps K per mini-batch.",
    )
    ap.add_argument(
        "--pfedme-personal-lr",
        type=float,
        default=float(_cfg_get(cfg, "federated.pfedme_personal_lr", 5e-3)),
        help="pFedMe personalized inner-loop learning rate.",
    )
    ap.add_argument(
        "--pfedme-mu",
        type=float,
        default=float(_cfg_get(cfg, "federated.pfedme_mu", 0.0)),
        help="pFedMe optional inner L2 coefficient (official implementation uses 1e-3).",
    )

    ap.add_argument("--train-split", default=_cfg_get(cfg, "data.train_split", "train"))
    ap.add_argument("--val-split", default=_cfg_get(cfg, "data.val_split", "val"))
    ap.add_argument("--test-split", default=_cfg_get(cfg, "data.test_split", "test"))
    ap.add_argument("--rounds", type=int, default=_cfg_get(cfg, "train.rounds", 100))
    ap.add_argument("--local-epochs", type=int, default=_cfg_get(cfg, "train.local_epochs", 1))
    ap.add_argument("--batch-size", type=int, default=_cfg_get(cfg, "train.batch_size", 64))
    ap.add_argument("--lr", type=float, default=_cfg_get(cfg, "train.lr", 1e-3))
    ap.add_argument("--weight-decay", type=float, default=_cfg_get(cfg, "train.weight_decay", 1e-4))
    ap.add_argument("--seed", type=int, default=_cfg_get(cfg, "train.seed", 42))
    ap.add_argument("--num-workers", type=int, default=_cfg_get(cfg, "train.num_workers", 0))
    ap.add_argument("--cache-in-memory", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "train.cache_in_memory", False)))
    ap.add_argument("--max-cache-files", type=int, default=_cfg_get(cfg, "train.max_cache_files", 32))
    ap.add_argument("--cache-dtype", default=_cfg_get(cfg, "train.cache_dtype", "float32"), choices=["float16", "float32"])
    ap.add_argument("--min-client-examples", type=int, default=_cfg_get(cfg, "train.min_client_examples", 10))
    ap.add_argument("--client-progress-bar", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "train.client_progress_bar", False)), help="Show per-client batch progress bar.")
    ap.add_argument("--no-progress-bar", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "train.no_progress_bar", False)), help="Disable progress bars.")
    ap.add_argument("--log-client-sim", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "run.log_client_sim", False)), help="Save per-round client cosine similarity matrix.")
    ap.add_argument("--val-every-round", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "train.val_every_round", True)))
    ap.add_argument("--test-every-round", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "train.test_every_round", False)))
    ap.add_argument("--save-round-json", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "train.save_round_json", True)))
    ap.add_argument("--per-client-every-round", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "train.per_client_every_round", False)))
    ap.add_argument("--save-test-pred-npz", default=_cfg_get(cfg, "eval.save_test_pred_npz", None), help="Optional .npz to save per-sample test predictions.")
    ap.add_argument("--eval-threshold", type=float, default=_cfg_get(cfg, "eval.eval_threshold", 0.5), help="Fixed threshold for round-by-round metrics.")
    ap.add_argument("--model-selection", default=_cfg_get(cfg, "eval.model_selection", "best"), choices=["last", "best"])
    ap.add_argument("--selection-metric", default=_cfg_get(cfg, "eval.selection_metric", "nll"), choices=["auroc", "auprc", "ece", "brier", "nll"])

    ap.add_argument("--model-base-channels", type=int, default=_cfg_get(cfg, "model.base_channels", 32))
    ap.add_argument("--dropout", type=float, default=_cfg_get(cfg, "model.dropout", 0.1))
    ap.add_argument("--use-gru", dest="use_gru", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "model.use_gru", False)), help="Deprecated (no-op).")
    ap.add_argument("--use-lstm", dest="use_gru", action="store_true", help="Deprecated (no-op).")
    ap.add_argument("--gru-hidden", dest="gru_hidden", type=int, default=_cfg_get(cfg, "model.gru_hidden", 64), help="Deprecated (no-op).")
    ap.add_argument("--lstm-hidden", dest="gru_hidden", type=int, help="Deprecated (no-op).")
    args = ap.parse_args()

    set_seed(int(args.seed))
    torch.set_num_threads(max(1, min(os.cpu_count() or 4, 8)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    algo_name = str(args.fl_algo).strip().lower()
    if algo_name == "fedprox" and float(args.fedprox_mu) < 0.0:
        raise SystemExit("--fedprox-mu must be >= 0.")
    scaffold_weighting = str(args.scaffold_weighting).strip().lower()
    perfedavg_weighting = str(args.perfedavg_weighting).strip().lower()
    pfedme_weighting = str(args.pfedme_weighting).strip().lower()
    if algo_name == "perfedavg":
        if float(args.perfedavg_alpha) <= 0.0 or float(args.perfedavg_beta) <= 0.0:
            raise SystemExit("--perfedavg-alpha and --perfedavg-beta must be > 0.")
    if algo_name == "pfedme":
        if float(args.pfedme_lambda) < 0.0:
            raise SystemExit("--pfedme-lambda must be >= 0.")
        if float(args.pfedme_beta) < 0.0:
            raise SystemExit("--pfedme-beta must be >= 0.")
        if int(args.pfedme_k) <= 0:
            raise SystemExit("--pfedme-k must be >= 1.")
        if float(args.pfedme_personal_lr) <= 0.0:
            raise SystemExit("--pfedme-personal-lr must be > 0.")
        if float(args.pfedme_mu) < 0.0:
            raise SystemExit("--pfedme-mu must be >= 0.")
    tag = args.run_name or "default"
    run_id = str(tag)
    legacy_dir = os.environ.get("LEGACY_RUN_DIR")
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif legacy_dir:
        run_dir = Path(legacy_dir)
    elif bool(args.resume):
        raise SystemExit("--resume requires --run-dir or LEGACY_RUN_DIR.")
    else:
        run_dir = Path(args.out_dir) / run_id
    if bool(args.resume) and not run_dir.exists():
        raise SystemExit(f"--resume requested but run_dir not found: {run_dir}")
    run_dir = ensure_dir(run_dir)
    ensure_dir(run_dir / "checkpoints")
    save_env_snapshot(run_dir, {"args": vars(args)})
    summary_path = Path(args.data_dir) / "summary.json"
    dataset_summary = None
    if summary_path.exists():
        try:
            dataset_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as e:
            dataset_summary = {"error": f"failed to read summary.json: {e}"}

    client_ids = list_client_ids(args.data_dir)
    client_train_assets: Dict[str, Dict[str, Any]] = {}
    # Always rebuild client train dataset/loader per round (no in-memory reuse).
    keep_train_assets_in_memory = False
    sample_wave_channels: int | None = None
    sample_clin_dim: int | None = None
    for cid in client_ids:
        files = list_npz_files(args.data_dir, args.train_split, client_id=str(cid))
        if not files:
            continue
        ds_client = WindowedNPZDataset(
            files,
            use_clin="true",
            cache_in_memory=bool(args.cache_in_memory),
            max_cache_files=int(args.max_cache_files),
            cache_dtype=str(args.cache_dtype),
        )
        n_examples = int(len(ds_client))
        if n_examples <= 0:
            continue
        if sample_wave_channels is None:
            sample_wave_channels = int(getattr(ds_client, "wave_channels", 4) or 4)
            sample_clin_dim = int(getattr(ds_client, "clin_dim", 0) or 0)
        asset: Dict[str, Any] = {
            "files": files,
            "n_examples": int(n_examples),
            "pos_weight": float(infer_pos_weight(files)),
        }
        if keep_train_assets_in_memory:
            dl_client = DataLoader(
                ds_client,
                batch_size=int(args.batch_size),
                shuffle=True,
                num_workers=int(args.num_workers),
                pin_memory=(device.type == "cuda"),
                persistent_workers=(int(args.num_workers) > 0),
            )
            asset["dataset"] = ds_client
            asset["loader"] = dl_client
        client_train_assets[str(cid)] = asset
    client_ids = sorted(client_train_assets.keys())
    if not client_ids:
        raise SystemExit("No client train files found under --data-dir.")

    model_cfg = IOHModelConfig(
        in_channels=int(sample_wave_channels if sample_wave_channels is not None else 4),
        base_channels=int(args.model_base_channels),
        dropout=float(args.dropout),
        use_gru=False,
        gru_hidden=64,
        clin_dim=int(sample_clin_dim if sample_clin_dim is not None else 0),
    )
    global_model = IOHNet(model_cfg).to(device)
    global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

    local_cfg = LocalTrainConfig(
        epochs=int(args.local_epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        num_workers=int(args.num_workers),
        cache_in_memory=bool(args.cache_in_memory),
        max_cache_files=int(args.max_cache_files),
        cache_dtype=str(args.cache_dtype),
        fl_algo=str(algo_name),
        fedprox_mu=float(args.fedprox_mu),
        perfedavg_alpha=float(args.perfedavg_alpha),
        perfedavg_beta=float(args.perfedavg_beta),
        pfedme_lambda=float(args.pfedme_lambda),
        pfedme_k=int(args.pfedme_k),
        pfedme_personal_lr=float(args.pfedme_personal_lr),
        pfedme_mu=float(args.pfedme_mu),
    )
    scaffold_server_control: Dict[str, torch.Tensor] | None = None
    scaffold_client_controls: Dict[str, Dict[str, torch.Tensor]] = {}

    started_utc = now_utc_iso()
    if bool(args.resume):
        prev_meta_path = run_dir / "meta.json"
        if prev_meta_path.exists():
            try:
                prev_meta_obj = json.loads(prev_meta_path.read_text(encoding="utf-8"))
                if isinstance(prev_meta_obj, dict) and prev_meta_obj.get("started_utc"):
                    started_utc = str(prev_meta_obj["started_utc"])
            except Exception:
                pass

    run_meta: Dict[str, Any] = {
        "started_utc": started_utc,
        "git_hash": get_git_hash(PROJECT_ROOT),
        "device": str(device),
        "algo": str(algo_name),
        "federated_algo": str(algo_name),
        "run_id": str(run_id),
        "run_dir": str(run_dir),
        "legacy_run_dir": str(legacy_dir) if legacy_dir else None,
        "resume": bool(args.resume),
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
        "val_every_round": bool(args.val_every_round),
        "test_every_round": bool(args.test_every_round),
        "save_round_json": bool(args.save_round_json),
        "per_client_every_round": bool(args.per_client_every_round),
        "eval_threshold": float(args.eval_threshold),
        "compute_steps_def": "client_steps=sum(local_epochs * n_batches_per_client), distill_steps=0",
        "runtime_optimizations": {
            "reuse_client_train_dataset": False,
            "reuse_client_train_dataloader": False,
            "precompute_client_pos_weight": True,
        },
        "federated": {
            "algo": str(algo_name),
            "fedprox_mu": float(args.fedprox_mu),
            "scaffold_weighting": str(scaffold_weighting),
            "perfedavg_alpha": float(args.perfedavg_alpha),
            "perfedavg_beta": float(args.perfedavg_beta),
            "perfedavg_weighting": str(perfedavg_weighting),
            "pfedme_lambda": float(args.pfedme_lambda),
            "pfedme_beta": float(args.pfedme_beta),
            "pfedme_weighting": str(pfedme_weighting),
            "pfedme_k": int(args.pfedme_k),
            "pfedme_personal_lr": float(args.pfedme_personal_lr),
            "pfedme_mu": float(args.pfedme_mu),
        },
    }
    if dataset_summary:
        keys = ["client_scheme", "client_column", "merge_strategy", "opname_threshold", "min_client_cases", "clients"]
        run_meta["dataset_summary"] = {k: dataset_summary.get(k) for k in keys if k in dataset_summary}
        write_json(run_dir / "dataset_summary.json", dataset_summary)
    meta_path = run_dir / "meta.json"
    write_json(meta_path, run_meta)
    global LAST_META_PATH
    LAST_META_PATH = meta_path

    config_used_path = run_dir / "config_used.yaml"
    try:
        import yaml  # type: ignore

        config_used_path.write_text(yaml.safe_dump({"args": vars(args)}, sort_keys=False), encoding="utf-8")
    except Exception:
        config_used_path.write_text(json.dumps({"args": vars(args)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    history: List[Dict[str, Any]] = []
    compute_log: List[Dict[str, Any]] = []
    model_trace_path = run_dir / "model_trace.jsonl"
    metrics_csv_path = run_dir / "metrics_round.csv"
    if not metrics_csv_path.exists():
        metrics_csv_path.write_text(
            "round,algo,auroc,auprc,ece,nll,brier,threshold,acc,f1,precision,recall\n",
            encoding="utf-8",
        )
    per_client_rounds: List[Dict[str, Any]] = []
    last_state: Dict[str, torch.Tensor] | None = None

    val_files = list_npz_files(args.data_dir, args.val_split)
    test_files = list_npz_files(args.data_dir, args.test_split)
    selection_enabled = str(args.model_selection).lower() == "best"
    selection_source = "val" if val_files else "test"
    selection_metric = str(args.selection_metric).lower()
    best = {"round": 0, "metric": None, "source": selection_source, "metric_name": selection_metric}

    start_round = 1
    if bool(args.resume):
        ckpt_path = run_dir / "checkpoints" / "model_last.pt"
        if not ckpt_path.exists():
            raise SystemExit(f"--resume requested but checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt_algo = str(ckpt.get("algo", algo_name)).lower()
        if ckpt_algo != str(algo_name):
            raise SystemExit(f"Checkpoint algo mismatch: ckpt={ckpt_algo} requested={algo_name}")
        loaded_model_cfg = ckpt.get("model_cfg", None)
        if loaded_model_cfg is not None:
            model_cfg = normalize_model_cfg(loaded_model_cfg)
        global_model = IOHNet(model_cfg).to(device)
        loaded_state = ckpt.get("state_dict", None)
        if not isinstance(loaded_state, dict):
            raise SystemExit(f"Invalid checkpoint (missing state_dict): {ckpt_path}")
        global_model.load_state_dict(loaded_state, strict=True)
        global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        last_state = {k: v.clone() for k, v in global_state.items()}

        history = _read_history_csv(run_dir / "history.csv")
        compute_log = _read_json_list(run_dir / "compute_log.json")
        per_client_rounds = _read_json_list(run_dir / "round_client_metrics.json")

        last_round_history = 0
        for row in history:
            try:
                last_round_history = max(last_round_history, int(row.get("round", 0)))
            except Exception:
                continue
        last_round_ckpt = int(ckpt.get("round", 0) or 0)
        last_round = max(last_round_history, last_round_ckpt)
        start_round = int(last_round) + 1

        ckpt_best = ckpt.get("best")
        if isinstance(ckpt_best, dict) and ckpt_best.get("metric") is not None:
            best = {
                "round": int(ckpt_best.get("round", 0) or 0),
                "metric": float(ckpt_best.get("metric")),
                "source": str(ckpt_best.get("source", selection_source)),
                "metric_name": str(ckpt_best.get("metric_name", selection_metric)),
            }
        elif selection_enabled:
            metric_key = f"{selection_source}_{selection_metric}"
            for row in history:
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
                        round_idx = int(float(row.get("round", 0)))
                    except Exception:
                        round_idx = 0
                    best = {"round": round_idx, "metric": score, "source": selection_source, "metric_name": selection_metric}

        if algo_name == "scaffold":
            scaff = ckpt.get("scaffold", {})
            scaffold_server_control = _coerce_state_like(global_state, scaff.get("server_control", {}))
            loaded_client_controls = scaff.get("client_control", {})
            if isinstance(loaded_client_controls, dict):
                for cid in client_ids:
                    raw = loaded_client_controls.get(str(cid), {})
                    scaffold_client_controls[str(cid)] = _coerce_state_like(global_state, raw)
            else:
                for cid in client_ids:
                    scaffold_client_controls[str(cid)] = _zeros_like_state(global_state)

        print(f"[INFO] resume mode: run_dir={run_dir} start_round={start_round} rounds={int(args.rounds)}")

    if algo_name == "scaffold" and scaffold_server_control is None:
        scaffold_server_control = _zeros_like_state(global_state)
        scaffold_client_controls = {str(cid): _zeros_like_state(global_state) for cid in client_ids}

    run_meta["start_round"] = int(start_round)
    run_meta["resumed_from_round"] = int(start_round - 1) if bool(args.resume) else 0
    run_meta["model"] = asdict(model_cfg)
    if bool(args.resume):
        run_meta["resumed_utc"] = now_utc_iso()
    write_json(meta_path, run_meta)
    ds_val = None
    if val_files and bool(args.val_every_round):
        ds_val = WindowedNPZDataset(
            val_files,
            use_clin="true",
            cache_in_memory=bool(args.cache_in_memory),
            max_cache_files=int(args.max_cache_files),
            cache_dtype=str(args.cache_dtype),
        )
    ds_test = None
    if test_files and bool(args.test_every_round):
        ds_test = WindowedNPZDataset(
            test_files,
            use_clin="true",
            cache_in_memory=bool(args.cache_in_memory),
            max_cache_files=int(args.max_cache_files),
            cache_dtype=str(args.cache_dtype),
        )
    dl_val = None
    if ds_val is not None:
        dl_val = DataLoader(
            ds_val,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
            persistent_workers=(int(args.num_workers) > 0),
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
    per_client_eval_split = str(args.val_split) if val_files else str(args.test_split)
    client_eval_files = {}
    if bool(args.per_client_every_round):
        client_eval_files = list_npz_files_by_client(args.data_dir, per_client_eval_split)

    prev_after_hash: str | None = hash_state_dict(global_state) if bool(args.resume) else None
    test_y: np.ndarray | None = None
    test_prob: np.ndarray | None = None
    last_completed_round = int(start_round - 1)
    for rnd in range(int(start_round), int(args.rounds) + 1):
        global_in_hash = hash_state_dict(global_state)
        if prev_after_hash is not None and global_in_hash != prev_after_hash:
            raise RuntimeError("global_state_in_hash does not match previous global_state_after_aggregate_hash")
        with model_trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "round_start", "round": rnd, "global_state_in_hash": global_in_hash}) + "\n")
        global_state_ref = {k: v.detach().cpu() for k, v in global_state.items()} if args.log_client_sim else {}
        client_states: List[Dict[str, torch.Tensor]] = []
        client_weights: List[int] = []
        fednova_local_normalizers: List[float] = []
        client_metrics = []
        client_vecs: Dict[str, np.ndarray] = {}
        client_n: Dict[str, int] = {}
        scaffold_client_deltas: List[Dict[str, torch.Tensor]] = []
        scaffold_client_delta_weights: List[int] = []
        scaffold_client_updates: Dict[str, Dict[str, torch.Tensor]] = {}
        train_loss_sum = 0.0
        train_loss_weight = 0
        match_count = 0
        trace_clients: List[Dict[str, Any]] = []

        client_iter = tqdm(
            client_ids,
            total=len(client_ids),
            desc=f"round {rnd}/{int(args.rounds)} clients",
            leave=False,
            disable=bool(args.no_progress_bar),
        )
        for cid in client_iter:
            train_asset = client_train_assets.get(str(cid))
            if train_asset is None:
                continue
            train_files = train_asset["files"]
            client_init_hash = hash_state_dict(global_state)
            if client_init_hash != global_in_hash:
                raise RuntimeError("client_init_hash != global_state_in_hash")
            match_count += 1
            if len(trace_clients) < 5:
                trace_clients.append({"client_id": str(cid), "client_init_hash": client_init_hash})
            updated, n_ex, m, aux = train_one_client(
                client_id=str(cid),
                train_files=train_files,
                model_cfg=model_cfg,
                global_state=global_state,
                cfg=local_cfg,
                device=device,
                show_progress=bool(args.client_progress_bar) and (not bool(args.no_progress_bar)),
                train_dataset=(train_asset.get("dataset") if keep_train_assets_in_memory else None),
                train_loader=(train_asset.get("loader") if keep_train_assets_in_memory else None),
                n_examples=int(train_asset["n_examples"]),
                pos_weight=float(train_asset["pos_weight"]),
                server_control=(scaffold_server_control if algo_name == "scaffold" else None),
                client_control=(scaffold_client_controls.get(str(cid)) if algo_name == "scaffold" else None),
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
                if algo_name == "fednova":
                    raw_norm = aux.get("fednova_local_normalizer", m.get("n_steps", 1))
                    try:
                        a_i = float(raw_norm)
                    except Exception:
                        a_i = float(max(1, int(m.get("n_steps", 1))))
                    fednova_local_normalizers.append(float(max(a_i, 1e-12)))
                if args.log_client_sim and m.get("status") == "ok":
                    client_vecs[str(cid)] = _flatten_state_delta(updated, global_state_ref)
                    client_n[str(cid)] = int(n_ex)
                if m.get("status") == "ok" and "avg_loss" in m:
                    train_loss_sum += float(m["avg_loss"]) * float(n_ex)
                    train_loss_weight += int(n_ex)
                if algo_name == "scaffold":
                    c_new = aux.get("client_control_new", None)
                    c_delta = aux.get("client_control_delta", None)
                    if not isinstance(c_new, dict) or not isinstance(c_delta, dict):
                        raise RuntimeError("invalid scaffold client control payload")
                    scaffold_client_updates[str(cid)] = c_new
                    scaffold_client_deltas.append(c_delta)
                    scaffold_client_delta_weights.append(int(n_ex))

        with model_trace_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "event": "client_init",
                        "round": rnd,
                        "client_init_hash": global_in_hash,
                        "match_count": int(match_count),
                        "total_clients": int(len(client_ids)),
                        "clients": trace_clients,
                    }
                )
                + "\n"
            )

        if not client_states:
            raise SystemExit("All clients empty after filtering/min_client_examples.")

        prev_state = global_state
        fednova_tau_eff: float | None = None
        if algo_name == "fednova":
            global_state, fednova_tau_eff = fednova(
                prev_state,
                client_states,
                client_weights,
                fednova_local_normalizers,
            )
        elif algo_name == "pfedme":
            pfedme_w = ([1] * len(client_states)) if pfedme_weighting == "uniform" else client_weights
            pfedme_avg = fedavg(client_states, pfedme_w)
            global_state = _blend_states(prev_state, pfedme_avg, beta=float(args.pfedme_beta))
        elif algo_name == "scaffold":
            if scaffold_weighting == "uniform":
                global_state = fedavg(client_states, [1] * len(client_states))
            else:
                global_state = fedavg(client_states, client_weights)
        elif algo_name == "perfedavg":
            perfedavg_w = ([1] * len(client_states)) if perfedavg_weighting == "uniform" else client_weights
            global_state = fedavg(client_states, perfedavg_w)
        else:
            global_state = fedavg(client_states, client_weights)
        if algo_name == "scaffold":
            assert scaffold_server_control is not None
            for cid, c_new in scaffold_client_updates.items():
                scaffold_client_controls[str(cid)] = _coerce_state_like(global_state, c_new)
            scaffold_server_control = _update_scaffold_server_control(
                server_control=scaffold_server_control,
                client_control_deltas=scaffold_client_deltas,
                n_total_clients=len(client_ids),
                weighting=scaffold_weighting,
                client_weights=scaffold_client_delta_weights,
            )
        global_after_hash = hash_state_dict(global_state)
        prev_after_hash = global_after_hash
        global_update_l2 = l2_diff_state_dict(global_state, prev_state)
        with model_trace_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "event": "round_end",
                        "round": rnd,
                        "global_state_after_aggregate_hash": global_after_hash,
                        "global_update_l2": float(global_update_l2),
                        "global_state_in_hash": global_in_hash,
                        "hash_changed": bool(global_after_hash != global_in_hash),
                    }
                )
                + "\n"
            )
        global_model = IOHNet(model_cfg).to(device)
        global_model.load_state_dict(global_state, strict=True)

        train_loss = float(train_loss_sum / max(train_loss_weight, 1))
        row = {
            "round": int(rnd),
            "algo": str(algo_name),
            "train_loss": float(train_loss),
            "n_clients_used": int(len(client_states)),
            "sum_examples": int(sum(client_weights)),
        }
        if fednova_tau_eff is not None:
            row["fednova_tau_eff"] = float(fednova_tau_eff)
        client_steps = 0
        for m in client_metrics:
            client_steps += int(m.get("n_steps", 0))
        compute_row = {
            "round": int(rnd),
            "algo": str(algo_name),
            "client_steps": int(client_steps),
            "distill_steps": 0,
            "total_steps": int(client_steps),
        }
        if fednova_tau_eff is not None:
            compute_row["fednova_tau_eff"] = float(fednova_tau_eff)
        compute_log.append(compute_row)
        write_json(run_dir / "compute_log.json", compute_log)
        m_val = None
        val_y = None
        val_prob = None
        if dl_val is not None:
            val_logits, val_y = predict_logits(global_model, dl_val, device=device)
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
            thr = float(args.eval_threshold)
            metrics_thr = calc_comprehensive_metrics(val_y, val_prob, threshold=thr)
            if bool(args.save_round_json):
                write_json(
                    run_dir / f"round_{rnd:03d}_val.json",
                    {
                        "round": int(rnd),
                        "n": int(m_val.n),
                        "n_pos": int(m_val.n_pos),
                        "n_neg": int(m_val.n_neg),
                        "metrics_pre": asdict(m_val),
                        "threshold": float(thr),
                        "threshold_method": "fixed",
                        "metrics_threshold": metrics_thr,
                        "confusion_pre": confusion_at_threshold(val_y, val_prob, thr=thr),
                    },
                )
        m_test = None
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
            thr = float(args.eval_threshold)
            metrics_thr = calc_comprehensive_metrics(test_y, test_prob, threshold=thr)
            if bool(args.save_round_json):
                write_json(
                    run_dir / f"round_{rnd:03d}_test.json",
                    {
                        "round": int(rnd),
                        "n": int(m_test.n),
                        "n_pos": int(m_test.n_pos),
                        "n_neg": int(m_test.n_neg),
                        "metrics_pre": asdict(m_test),
                        "threshold": float(thr),
                        "threshold_method": "fixed",
                        "metrics_threshold": metrics_thr,
                        "confusion_pre": confusion_at_threshold(test_y, test_prob, thr=thr),
                    },
                )
        metrics_obj = m_val if m_val is not None else m_test
        metrics_y = val_y if m_val is not None else test_y
        metrics_prob = val_prob if m_val is not None else test_prob
        if metrics_obj is not None and metrics_y is not None and metrics_prob is not None:
            thr = float(args.eval_threshold)
            metrics_thr = calc_comprehensive_metrics(metrics_y, metrics_prob, threshold=thr)
            metrics_csv_path.write_text(
                metrics_csv_path.read_text(encoding="utf-8")
                + f"{rnd},{algo_name},{metrics_obj.auroc},{metrics_obj.auprc},{metrics_obj.ece},{metrics_obj.nll},{metrics_obj.brier},{thr},{metrics_thr.get('accuracy', '')},{metrics_thr.get('f1', '')},{metrics_thr.get('ppv', '')},{metrics_thr.get('sensitivity', '')}\n",
                encoding="utf-8",
            )
        if selection_enabled:
            metrics_sel = m_val if m_val is not None else (m_test if dl_test is not None else None)
            if metrics_sel is not None:
                score = float(getattr(metrics_sel, selection_metric))
                prev = best.get("metric")
                if _is_better_score(selection_metric, score, None if prev is None else float(prev)):
                    best = {"round": int(rnd), "metric": float(score), "source": selection_source, "metric_name": selection_metric}
                    best_payload: Dict[str, Any] = {
                        "model_cfg": asdict(model_cfg),
                        "state_dict": global_state,
                        "algo": str(algo_name),
                    }
                    if algo_name == "scaffold" and scaffold_server_control is not None:
                        best_payload["scaffold"] = {"server_control": scaffold_server_control}
                    torch.save(best_payload, run_dir / "checkpoints" / "model_best.pt")
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if bool(args.per_client_every_round) and client_eval_files:
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
                    pin_memory=(device.type == "cuda"),
                    persistent_workers=(int(args.num_workers) > 0),
                )
                logits_c, y_c = predict_logits(global_model, dl_c, device=device)
                prob_c = sigmoid_np(logits_c)
                m_c = compute_binary_metrics(y_c, prob_c, n_bins=15)
                thr = float(args.eval_threshold)
                m_thr = calc_comprehensive_metrics(y_c, prob_c, threshold=thr)
                row_c = {
                    "round": int(rnd),
                    "split": str(per_client_eval_split),
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
                    "threshold_method": "fixed",
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
                pd.DataFrame(round_rows).to_csv(run_dir / f"round_{rnd:03d}_{per_client_eval_split}_per_client.csv", index=False)
                pd.DataFrame(per_client_rounds).to_csv(run_dir / "round_client_metrics.csv", index=False)
                write_json(run_dir / "round_client_metrics.json", per_client_rounds)
        write_json(run_dir / f"round_{rnd:03d}_clients.json", {"clients": client_metrics})
        if args.log_client_sim and client_vecs:
            sim_client_ids = list(client_vecs.keys())
            vecs = np.stack([client_vecs[cid] for cid in sim_client_ids], axis=0)
            sim = _cosine_sim_matrix(vecs)
            write_json(
                run_dir / f"round_{rnd:03d}_client_similarity.json",
                {
                    "metric": "cosine_update_delta",
                    "client_ids": sim_client_ids,
                    "n_examples": client_n,
                    "matrix": sim.tolist(),
                },
            )

        tqdm.write(f"[round {rnd:03d}] train_loss={train_loss:.4f} used={len(client_states)}")

        last_state = {k: v.clone() for k, v in global_state.items()}
        last_payload: Dict[str, Any] = {
            "model_cfg": asdict(model_cfg),
            "state_dict": last_state,
            "round": int(rnd),
            "best": best,
            "algo": str(algo_name),
        }
        if algo_name == "scaffold" and scaffold_server_control is not None:
            last_payload["scaffold"] = {
                "server_control": scaffold_server_control,
                "client_control": scaffold_client_controls,
            }
        torch.save(last_payload, run_dir / "checkpoints" / "model_last.pt")
        last_completed_round = int(rnd)

    if last_state is None:
        last_state = {k: v.clone() for k, v in global_state.items()}
    if not (run_dir / "checkpoints" / "model_last.pt").exists():
        last_payload = {
            "model_cfg": asdict(model_cfg),
            "state_dict": last_state,
            "round": int(last_completed_round),
            "best": best,
            "algo": str(algo_name),
        }
        if algo_name == "scaffold" and scaffold_server_control is not None:
            last_payload["scaffold"] = {
                "server_control": scaffold_server_control,
                "client_control": scaffold_client_controls,
            }
        torch.save(last_payload, run_dir / "checkpoints" / "model_last.pt")

    # Fixed threshold; best checkpoint is selected on val when available.
    model_sel = "last"
    ckpt_path = run_dir / "checkpoints" / "model_last.pt"
    if selection_enabled and (run_dir / "checkpoints" / "model_best.pt").exists():
        model_sel = "best"
        ckpt_path = run_dir / "checkpoints" / "model_best.pt"
    best_model = IOHNet(model_cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    best_model.load_state_dict(ckpt["state_dict"], strict=True)
    thr = float(args.eval_threshold)

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
            case_id=case_id.astype(np.int64, copy=False),
        )
    write_json(
        run_dir / "test_report.json",
        {
            "n": int(len(test_y)),
            "metrics_pre": asdict(compute_binary_metrics(test_y, test_prob, n_bins=15)),
            "threshold": float(thr),
            "threshold_method": "fixed",
            "model_selected": str(model_sel),
            "best": best,
            "confusion_pre": confusion_at_threshold(test_y, test_prob, thr=float(thr)),
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
        metrics_pre = compute_binary_metrics(y_true, prob, n_bins=15)
        report = {
            "client_id": str(cid),
            "status": "ok",
            "n": int(metrics_pre.n),
                "threshold": float(thr),
                "threshold_method": "fixed",
                "metrics_pre": asdict(metrics_pre),
                "confusion_pre": confusion_at_threshold(y_true, prob, thr=float(thr)),
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
                "threshold": float(thr),
                "threshold_method": "fixed",
            }
        )

    write_json(
        run_dir / "test_report_per_client.json",
        {"threshold": float(thr), "threshold_method": "fixed", "clients": per_client_reports},
    )
    pd.DataFrame(per_client_rows).to_csv(run_dir / "test_report_per_client.csv", index=False)

    run_meta["finished_utc"] = now_utc_iso()
    run_meta["artifacts"] = {
        "history_csv": str(run_dir / "history.csv"),
        "checkpoint_last": str(run_dir / "checkpoints" / "model_last.pt"),
        "checkpoint_best": str(run_dir / "checkpoints" / "model_best.pt") if selection_enabled and (run_dir / "checkpoints" / "model_best.pt").exists() else None,
        "test_report_json": str(run_dir / "test_report.json"),
        "test_report_per_client_json": str(run_dir / "test_report_per_client.json"),
        "test_report_per_client_csv": str(run_dir / "test_report_per_client.csv"),
    }
    run_meta["best"] = best
    if args.save_test_pred_npz:
        pred_path = Path(args.save_test_pred_npz)
        if not pred_path.is_absolute():
            pred_path = run_dir / pred_path
        run_meta["artifacts"]["test_predictions_npz"] = str(pred_path)
    write_json(run_dir / "meta.json", run_meta)

    print("Done.")
    print(f"Run dir: {run_dir}")
    print(f"Fixed threshold: {thr:.3f}")
    print(f"Selection source: {selection_source}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if LAST_META_PATH is not None:
            try:
                err = {"error": "exception", "traceback": traceback.format_exc()}
                existing = {}
                try:
                    existing = json.loads(LAST_META_PATH.read_text(encoding="utf-8"))
                except Exception:
                    existing = {}
                existing.update(err)
                LAST_META_PATH.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except Exception:
                pass
        raise
