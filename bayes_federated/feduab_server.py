from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root in sys.path when executed as `python bayes_federated/feduab_server.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bayes_federated.agg import AggConfig, feduab_aggregate_dict, poe_prior_corrected_dict
from bayes_federated.bayes_layers import BayesParams
from bayes_federated.feduab_client import FedUABClientConfig, train_client_feduab
from bayes_federated.models import BFLModel, build_bfl_model_from_point_checkpoint
from common.checkpoint import load_checkpoint, save_checkpoint
from common.dataset import WindowedNPZDataset, list_client_ids, list_npz_files, list_npz_files_by_client
from common.eval_summary import build_mode_report, metrics_from_binary, write_eval_outputs
from common.experiment import save_env_snapshot, seed_everything, seed_worker
from common.io import ensure_dir, get_git_hash, now_utc_iso, write_json
from common.ioh_model import IOHModelConfig, normalize_model_cfg
from common.metrics import best_threshold_youden
from common.utils import calc_comprehensive_metrics


def _dataset_sample_cfg(
    data_dir: str,
    split: str,
    *,
    base_channels: int,
    dropout: float,
    use_gru: bool,
    gru_hidden: int,
) -> IOHModelConfig:
    files = list_npz_files(data_dir, split)
    if not files:
        raise SystemExit(f"No files found under --data-dir {data_dir} split {split}.")
    ds = WindowedNPZDataset(
        [files[0]],
        use_clin="true",
        cache_in_memory=False,
        max_cache_files=32,
        cache_dtype="float32",
    )
    return IOHModelConfig(
        in_channels=int(getattr(ds, "wave_channels", 4) or 4),
        base_channels=int(base_channels),
        dropout=float(dropout),
        use_gru=bool(use_gru),
        gru_hidden=int(gru_hidden),
        clin_dim=int(getattr(ds, "clin_dim", 0) or 0),
    )


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
    except Exception as exc:  # pragma: no cover - best effort without yaml
        raise SystemExit("PyYAML required to parse feduab config") from exc


def _cfg_get(cfg: Dict[str, Any], path: str, default: Any) -> Any:
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur if cur is not None else default


def _cfg_get_first(cfg: Dict[str, Any], paths: List[str], default: Any) -> Any:
    for path in paths:
        val = _cfg_get(cfg, path, None)
        if val is not None:
            return val
    return default


def _make_loader(
    files: list[str],
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    ds = WindowedNPZDataset(files, use_clin="true", cache_in_memory=False, max_cache_files=32, cache_dtype="float32")
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(num_workers) > 0),
        worker_init_fn=seed_worker,
    )


def _select_clients(
    client_ids: list[str],
    *,
    rnd: int,
    seed: int,
    clients_per_round: int,
    client_fraction: float,
) -> list[str]:
    if not client_ids:
        return []
    total = int(len(client_ids))
    if int(clients_per_round) > 0:
        m = min(total, int(clients_per_round))
    else:
        frac = float(client_fraction)
        if frac >= 1.0:
            m = total
        elif frac <= 0.0:
            m = total
        else:
            m = int(max(1, math.ceil(total * frac)))
    if m >= total:
        return list(client_ids)
    rng = random.Random(int(seed) + int(rnd))
    chosen = rng.sample(list(client_ids), k=int(m))
    return sorted(chosen)


def _save_history(path: Path, rows: List[Dict[str, Any]]) -> None:
    import pandas as pd

    pd.DataFrame(rows).to_csv(path, index=False)


def _read_history_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        import pandas as pd

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


def _load_client_posteriors(ckpt_dir: Path) -> Dict[str, Dict[str, BayesParams]]:
    out: Dict[str, Dict[str, BayesParams]] = {}
    if not ckpt_dir.exists():
        return out
    for path in sorted(ckpt_dir.glob("client_*.pt")):
        try:
            ckpt = load_checkpoint(path, map_location="cpu")
        except Exception:
            continue
        posterior = ckpt.get("posterior", None)
        if not isinstance(posterior, dict):
            continue
        cid = str(ckpt.get("client_id", path.stem.replace("client_", "")))
        out[cid] = posterior
    return out


@torch.no_grad()
def _predict_probs(
    model: BFLModel,
    dl: DataLoader,
    *,
    mc_samples: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    model.eval()
    prob_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    prob_var_all: list[np.ndarray] = []
    prob_alea_all: list[np.ndarray] = []
    prob_epi_all: list[np.ndarray] = []
    prob_total_var_all: list[np.ndarray] = []
    entropy_all: list[np.ndarray] = []
    for x, y in dl:
        if isinstance(x, (tuple, list)):
            x = tuple(t.to(device, non_blocking=True) for t in x)
        else:
            x = x.to(device, non_blocking=True)
        logits_mc = model(x, sample=True, n_samples=int(mc_samples))
        if logits_mc.dim() == 2:
            logits_mc = logits_mc.unsqueeze(0)
        logits_mc = logits_mc.squeeze(-1)
        probs = torch.sigmoid(logits_mc)
        prob_mean = probs.mean(dim=0)
        prob_var = probs.var(dim=0, unbiased=False)
        prob_alea = (probs * (1.0 - probs)).mean(dim=0)
        prob_epi = prob_var
        prob_total_var = prob_alea + prob_epi
        eps = 1e-12
        entropy = -prob_mean * torch.log(prob_mean + eps) - (1.0 - prob_mean) * torch.log(1.0 - prob_mean + eps)
        prob_all.append(prob_mean.detach().cpu().numpy())
        prob_var_all.append(prob_var.detach().cpu().numpy())
        prob_alea_all.append(prob_alea.detach().cpu().numpy())
        prob_epi_all.append(prob_epi.detach().cpu().numpy())
        prob_total_var_all.append(prob_total_var.detach().cpu().numpy())
        entropy_all.append(entropy.detach().cpu().numpy())
        y_all.append(y.detach().cpu().view(-1).numpy())
    prob = np.concatenate(prob_all, axis=0)
    y_true = np.concatenate(y_all, axis=0)
    unc = {
        "prob_var": np.concatenate(prob_var_all, axis=0),
        "prob_alea": np.concatenate(prob_alea_all, axis=0),
        "prob_epi": np.concatenate(prob_epi_all, axis=0),
        "prob_total_var": np.concatenate(prob_total_var_all, axis=0),
        "entropy": np.concatenate(entropy_all, axis=0),
    }
    return prob, y_true, unc


def _eval_model(
    *,
    model: BFLModel,
    files: list[str],
    mc_samples: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    dl = _make_loader(files, batch_size=batch_size, num_workers=num_workers, device=device)
    prob, y_true, unc = _predict_probs(model, dl, mc_samples=mc_samples, device=device)
    metrics = metrics_from_binary(y_true, prob, n_bins=15)
    metrics["uncertainty"] = {
        "prob_var_mean": float(np.mean(unc["prob_var"])),
        "prob_var_std": float(np.std(unc["prob_var"])),
        "aleatoric_mean": float(np.mean(unc["prob_alea"])),
        "aleatoric_std": float(np.std(unc["prob_alea"])),
        "epistemic_mean": float(np.mean(unc["prob_epi"])),
        "epistemic_std": float(np.std(unc["prob_epi"])),
        "total_var_mean": float(np.mean(unc["prob_total_var"])),
        "total_var_std": float(np.std(unc["prob_total_var"])),
        "entropy_mean": float(np.mean(unc["entropy"])),
        "entropy_std": float(np.std(unc["entropy"])),
    }
    return metrics, y_true, prob, unc


def _make_model(
    *,
    model_cfg: IOHModelConfig,
    base_state: Dict[str, torch.Tensor],
    params: Dict[str, BayesParams],
    device: torch.device,
    prior_sigma: float,
    logvar_min: float,
    logvar_max: float,
    full_bayes: bool,
    param_type: str,
    mu_init: str,
    init_rho: float | None,
    var_reduction_h: float,
) -> BFLModel:
    model = BFLModel(
        model_cfg,
        prior_sigma=float(prior_sigma),
        logvar_min=float(logvar_min),
        logvar_max=float(logvar_max),
        full_bayes=bool(full_bayes),
        param_type=str(param_type),
        mu_init=str(mu_init),
        init_rho=init_rho,
        var_reduction_h=float(var_reduction_h),
    )
    model.load_state_dict(base_state, strict=False)
    model.set_posterior(params)
    model.set_prior(params)
    return model.to(device)


def _eval_modes(
    *,
    model_cfg: IOHModelConfig,
    base_state: Dict[str, torch.Tensor],
    global_params: Dict[str, BayesParams],
    client_posteriors: Dict[str, Dict[str, BayesParams]],
    client_test_files: Dict[str, list[str]],
    test_files: list[str],
    mc_samples: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    prior_sigma: float,
    logvar_min: float,
    logvar_max: float,
    full_bayes: bool,
    param_type: str,
    mu_init: str,
    init_rho: float | None,
    var_reduction_h: float,
    eval_personalized: bool,
    eval_global: bool,
    eval_ensemble: bool,
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    modes: Dict[str, Dict[str, Any]] = {}

    def _set_eval_seed(tag: str) -> None:
        stable = sum(bytearray(tag.encode("utf-8"))) % 10000
        seed_everything(int(seed) + int(stable), deterministic=True)

    if eval_global:
        _set_eval_seed("global_idless")
        overall_metrics, _, _, _ = _eval_model(
            model=_make_model(
                model_cfg=model_cfg,
                base_state=base_state,
                params=global_params,
                device=device,
                prior_sigma=prior_sigma,
                logvar_min=logvar_min,
                logvar_max=logvar_max,
                full_bayes=full_bayes,
                param_type=param_type,
                mu_init=mu_init,
                init_rho=init_rho,
                var_reduction_h=var_reduction_h,
            ),
            files=test_files,
            mc_samples=mc_samples,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        per_client: Dict[str, Dict[str, Any]] = {}
        for cid, files in client_test_files.items():
            if not files:
                continue
            metrics_c, _, _, _ = _eval_model(
                model=_make_model(
                    model_cfg=model_cfg,
                    base_state=base_state,
                    params=global_params,
                    device=device,
                    prior_sigma=prior_sigma,
                    logvar_min=logvar_min,
                    logvar_max=logvar_max,
                    full_bayes=full_bayes,
                    param_type=param_type,
                    mu_init=mu_init,
                    init_rho=init_rho,
                    var_reduction_h=var_reduction_h,
                ),
                files=files,
                mc_samples=mc_samples,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            per_client[str(cid)] = metrics_c
        modes["global_idless"] = build_mode_report(overall=overall_metrics, per_client=per_client)

    if eval_personalized:
        _set_eval_seed("personalized_oracle")
        y_list = []
        p_list = []
        unc_list = []
        per_client = {}
        for cid, files in client_test_files.items():
            if not files:
                continue
            params = client_posteriors.get(str(cid))
            if params is None:
                continue
            metrics_c, y_c, p_c, unc_c = _eval_model(
                model=_make_model(
                    model_cfg=model_cfg,
                    base_state=base_state,
                    params=params,
                    device=device,
                    prior_sigma=prior_sigma,
                    logvar_min=logvar_min,
                    logvar_max=logvar_max,
                    full_bayes=full_bayes,
                    param_type=param_type,
                    mu_init=mu_init,
                    init_rho=init_rho,
                    var_reduction_h=var_reduction_h,
                ),
                files=files,
                mc_samples=mc_samples,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            per_client[str(cid)] = metrics_c
            y_list.append(y_c)
            p_list.append(p_c)
            unc_list.append(unc_c)
        if y_list:
            overall = metrics_from_binary(np.concatenate(y_list, axis=0), np.concatenate(p_list, axis=0), n_bins=15)
            # Aggregate uncertainty across all client predictions.
            if unc_list:
                def _cat(key: str) -> np.ndarray:
                    return np.concatenate([u[key] for u in unc_list], axis=0)

                overall["uncertainty"] = {
                    "prob_var_mean": float(np.mean(_cat("prob_var"))),
                    "prob_var_std": float(np.std(_cat("prob_var"))),
                    "aleatoric_mean": float(np.mean(_cat("prob_alea"))),
                    "aleatoric_std": float(np.std(_cat("prob_alea"))),
                    "epistemic_mean": float(np.mean(_cat("prob_epi"))),
                    "epistemic_std": float(np.std(_cat("prob_epi"))),
                    "total_var_mean": float(np.mean(_cat("prob_total_var"))),
                    "total_var_std": float(np.std(_cat("prob_total_var"))),
                    "entropy_mean": float(np.mean(_cat("entropy"))),
                    "entropy_std": float(np.std(_cat("entropy"))),
                }
        else:
            overall = {
                "n": 0,
                "n_pos": 0,
                "n_neg": 0,
                "auroc": float("nan"),
                "auprc": float("nan"),
                "ece": float("nan"),
                "nll": float("nan"),
                "brier": float("nan"),
            }
        modes["personalized_oracle"] = build_mode_report(overall=overall, per_client=per_client)

    if eval_ensemble:
        _set_eval_seed("ensemble_idless")
        try:
            params_list = [p for _, p in sorted(client_posteriors.items(), key=lambda kv: kv[0])]
            if not params_list:
                modes["ensemble_idless"] = build_mode_report(
                    overall={"n": 0, "n_pos": 0, "n_neg": 0, "auroc": float("nan"), "auprc": float("nan"), "ece": float("nan"), "nll": float("nan"), "brier": float("nan")},
                    per_client={},
                    note="no client posteriors available for ensemble",
                )
            else:
                def _ensemble_predict(files: list[str]) -> Tuple[np.ndarray, np.ndarray]:
                    dl = _make_loader(files, batch_size=batch_size, num_workers=num_workers, device=device)
                    prob_all: list[np.ndarray] = []
                    y_all: list[np.ndarray] = []
                    model = _make_model(
                        model_cfg=model_cfg,
                        base_state=base_state,
                        params=params_list[0],
                        device=device,
                        prior_sigma=prior_sigma,
                        logvar_min=logvar_min,
                        logvar_max=logvar_max,
                        full_bayes=full_bayes,
                        param_type=param_type,
                        mu_init=mu_init,
                        init_rho=init_rho,
                        var_reduction_h=var_reduction_h,
                    )
                    model.eval()
                    for x, y in dl:
                        if isinstance(x, (tuple, list)):
                            x = tuple(t.to(device, non_blocking=True) for t in x)
                        else:
                            x = x.to(device, non_blocking=True)
                        probs_sum = None
                        for params in params_list:
                            model.set_posterior(params)
                            logits_mc = model(x, sample=True, n_samples=int(mc_samples))
                            if logits_mc.dim() == 2:
                                logits_mc = logits_mc.unsqueeze(0)
                            logits_mc = logits_mc.squeeze(-1)
                            probs = torch.sigmoid(logits_mc).mean(dim=0)
                            if probs_sum is None:
                                probs_sum = probs
                            else:
                                probs_sum = probs_sum + probs
                        prob_mean = (probs_sum / float(len(params_list))).detach().cpu().numpy()
                        prob_all.append(prob_mean)
                        y_all.append(y.detach().cpu().view(-1).numpy())
                    return np.concatenate(prob_all, axis=0), np.concatenate(y_all, axis=0)

                per_client: Dict[str, Dict[str, Any]] = {}
                y_list = []
                p_list = []
                for cid, files in client_test_files.items():
                    if not files:
                        continue
                    p_c, y_c = _ensemble_predict(files)
                    metrics_c = metrics_from_binary(y_c, p_c, n_bins=15)
                    per_client[str(cid)] = metrics_c
                    y_list.append(y_c)
                    p_list.append(p_c)
                if y_list:
                    overall = metrics_from_binary(np.concatenate(y_list, axis=0), np.concatenate(p_list, axis=0), n_bins=15)
                else:
                    overall = {"n": 0, "n_pos": 0, "n_neg": 0, "auroc": float("nan"), "auprc": float("nan"), "ece": float("nan"), "nll": float("nan"), "brier": float("nan")}
                modes["ensemble_idless"] = build_mode_report(overall=overall, per_client=per_client)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "out of memory" not in msg:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            modes["ensemble_idless"] = build_mode_report(
                overall={"n": 0, "n_pos": 0, "n_neg": 0, "auroc": float("nan"), "auprc": float("nan"), "ece": float("nan"), "nll": float("nan"), "brier": float("nan")},
                per_client={},
                note="skipped ensemble due to OOM",
            )

    return modes


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()
    cfg = _load_config(pre_args.config)

    ap = argparse.ArgumentParser(description="FedUAB server (Bayesian federated learning with posterior aggregation).")
    ap.add_argument("--config", default=pre_args.config, help="Optional YAML/JSON config path.")
    ap.add_argument("--data-dir", default=_cfg_get_first(cfg, ["data.data_dir", "data.federated_dir"], "federated_data"), help="Output of scripts/build_dataset.py")
    ap.add_argument("--out-dir", default=_cfg_get(cfg, "run.out_dir", "runs/feduab"))
    ap.add_argument("--run-name", default=_cfg_get(cfg, "run.run_name", None))
    ap.add_argument("--run-dir", default=_cfg_get(cfg, "run.run_dir", None), help="Explicit run dir (overrides out-dir/run-name).")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "run.resume", False)))

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
    ap.add_argument("--min-client-examples", type=int, default=_cfg_get(cfg, "train.min_client_examples", 10))
    ap.add_argument("--per-client-every-round", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "train.per_client_every_round", False)))
    ap.add_argument("--client-progress-bar", action="store_true", default=bool(_cfg_get(cfg, "train.client_progress_bar", False)), help="Show per-client batch progress bar.")
    ap.add_argument("--no-progress-bar", action="store_true", default=bool(_cfg_get(cfg, "train.no_progress_bar", False)), help="Disable progress bars.")
    ap.add_argument("--test-every-round", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "train.test_every_round", False)))

    ap.add_argument("--mc-train", type=int, default=_cfg_get(cfg, "train.mc_train", 3), help="MC samples for local BBB training.")
    ap.add_argument("--mc-samples", type=int, default=_cfg_get(cfg, "train.mc_samples", 20), help="MC samples for evaluation.")
    ap.add_argument("--kl-coeff", type=float, default=_cfg_get(cfg, "train.kl_coeff", 1e-4), help="KL coefficient for BBB loss.")
    ap.add_argument("--loss-type", default=_cfg_get(cfg, "train.loss_type", "bce"), choices=["bce", "weighted_bce"])
    ap.add_argument("--pos-weight", default=_cfg_get(cfg, "train.pos_weight", "auto"), help="Positive class weight (auto|float).")
    ap.add_argument("--grad-clip", type=float, default=_cfg_get(cfg, "train.grad_clip", 0.0))
    ap.add_argument("--max-steps", type=int, default=_cfg_get(cfg, "train.max_steps", 0))
    ap.add_argument("--clients-per-round", type=int, default=_cfg_get(cfg, "train.clients_per_round", 0), help="Clients sampled per round (0=all).")
    ap.add_argument("--client-fraction", type=float, default=_cfg_get(cfg, "train.client_fraction", 1.0), help="Fraction of clients per round when clients-per-round=0.")

    ap.add_argument("--full-bayes", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "model.full_bayes", True)))
    ap.add_argument("--train-deterministic", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "train.train_deterministic", False)), help="Train non-Bayes params (default: false).")
    ap.add_argument("--prior-sigma", type=float, default=_cfg_get(cfg, "model.prior_sigma", 0.1))
    ap.add_argument("--var-reduction-h", type=float, default=_cfg_get(cfg, "model.var_reduction_h", 2.0), help="Initial variance reduction factor H (per layer variance /= H).")
    ap.add_argument("--logvar-min", type=float, default=_cfg_get(cfg, "model.logvar_min", -12.0))
    ap.add_argument("--logvar-max", type=float, default=_cfg_get(cfg, "model.logvar_max", 6.0))
    ap.add_argument("--param-type", default=_cfg_get(cfg, "model.param_type", "rho"), choices=["logvar", "rho"])
    ap.add_argument("--mu-init", default=_cfg_get(cfg, "model.mu_init", "zeros"), choices=["zeros", "pytorch", "kaiming"])
    ap.add_argument("--init-rho", type=float, default=_cfg_get(cfg, "model.init_rho", None))
    ap.add_argument("--backbone-checkpoint", default=_cfg_get(cfg, "model.backbone_checkpoint", None), help="Point-estimate checkpoint for initialization.")
    ap.add_argument("--model-base-channels", type=int, default=_cfg_get(cfg, "model.base_channels", 32))
    ap.add_argument("--dropout", type=float, default=_cfg_get(cfg, "model.dropout", 0.1))
    ap.add_argument("--use-gru", dest="use_gru", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "model.use_gru", True)))
    ap.add_argument("--gru-hidden", type=int, default=_cfg_get(cfg, "model.gru_hidden", 64))

    ap.add_argument("--agg-mode", default=_cfg_get(cfg, "agg.mode", "feduab"), choices=["feduab", "poe_prior_corrected"])
    ap.add_argument("--agg-beta-mode", default=_cfg_get(cfg, "agg.beta_mode", "normalized"), choices=["fixed", "normalized", "num_clients", "raw"])
    ap.add_argument("--agg-beta-value", type=float, default=_cfg_get(cfg, "agg.beta_value", 1.0))
    ap.add_argument("--agg-precision-clamp", type=float, default=_cfg_get(cfg, "agg.precision_clamp", 1e-6))
    ap.add_argument("--client-weight-mode", default=_cfg_get(cfg, "agg.client_weight_mode", "samples"), choices=["samples", "uniform"])

    ap.add_argument("--eval-threshold", type=float, default=_cfg_get(cfg, "eval.eval_threshold", 0.5), help="Fixed threshold for round-by-round metrics.")
    ap.add_argument("--threshold-method", default=_cfg_get(cfg, "eval.threshold_method", "fixed"), choices=["fixed", "youden", "youden-val"])

    ap.add_argument("--eval-personalized", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "eval.eval_personalized", True)))
    ap.add_argument("--eval-global", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "eval.eval_global", True)))
    ap.add_argument("--eval-ensemble", action=argparse.BooleanOptionalAction, default=bool(_cfg_get(cfg, "eval.eval_ensemble", True)))
    args = ap.parse_args()

    if str(args.agg_mode).lower() == "feduab":
        if str(args.client_weight_mode).lower() != "samples":
            raise SystemExit("FedUAB Eq.(8) requires client_weight_mode=samples (n_k/n).")
        if str(args.agg_beta_mode).lower() != "normalized":
            raise SystemExit("FedUAB Eq.(8) requires agg_beta_mode=normalized.")

    seed_everything(int(args.seed), deterministic=True)
    torch.set_num_threads(max(1, min(os.cpu_count() or 4, 8)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    algo_name = "feduab"
    tag = args.run_name or "default"
    ts = now_utc_iso().replace("-", "").replace(":", "").replace("T", "_").replace("Z", "")
    run_id = f"{ts}_{algo_name}_{tag}"
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
    ensure_dir(run_dir / "checkpoints" / "clients")
    save_env_snapshot(run_dir, {"args": vars(args)})

    client_ids = list_client_ids(args.data_dir)
    client_train_files: Dict[str, List[str]] = {}
    for cid in client_ids:
        files = list_npz_files(args.data_dir, args.train_split, client_id=str(cid))
        if files:
            client_train_files[str(cid)] = files
    client_ids = sorted(client_train_files.keys())
    if not client_ids:
        raise SystemExit("No client train files found under --data-dir.")

    model_cfg = _dataset_sample_cfg(
        args.data_dir,
        args.train_split,
        base_channels=int(args.model_base_channels),
        dropout=float(args.dropout),
        use_gru=bool(args.use_gru),
        gru_hidden=int(args.gru_hidden),
    )
    model, init_prior, used_point = build_bfl_model_from_point_checkpoint(
        args.backbone_checkpoint,
        prior_sigma=float(args.prior_sigma),
        logvar_min=float(args.logvar_min),
        logvar_max=float(args.logvar_max),
        full_bayes=bool(args.full_bayes),
        fallback_cfg=model_cfg,
        param_type=str(args.param_type),
        mu_init=str(args.mu_init),
        init_rho=(float(args.init_rho) if args.init_rho is not None else None),
        var_reduction_h=float(args.var_reduction_h),
    )
    model_cfg = model.cfg
    base_state = model.state_dict()
    global_params: Dict[str, BayesParams] = init_prior

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
        "run_id": str(run_id),
        "run_dir": str(run_dir),
        "resume": bool(args.resume),
        "config_path": str(args.config) if args.config else None,
        "data_dir": str(args.data_dir),
        "splits": {"train": str(args.train_split), "test": str(args.test_split), "val": str(args.val_split)},
        "seed": int(args.seed),
        "rounds": int(args.rounds),
        "full_bayes": bool(args.full_bayes),
        "param_type": str(args.param_type),
        "kl_coeff": float(args.kl_coeff),
        "var_reduction_h": float(args.var_reduction_h),
        "used_point_init": bool(used_point),
        "model": asdict(model_cfg),
        "clients": client_ids,
        "client_sampling": {
            "clients_per_round": int(args.clients_per_round),
            "client_fraction": float(args.client_fraction),
        },
        "agg": {
            "mode": str(args.agg_mode),
            "beta_mode": str(args.agg_beta_mode),
            "beta_value": float(args.agg_beta_value),
            "precision_clamp": float(args.agg_precision_clamp),
            "weight_mode": str(args.client_weight_mode),
        },
        "eval_modes": {
            "personalized_oracle": bool(args.eval_personalized),
            "global_idless": bool(args.eval_global),
            "ensemble_idless": bool(args.eval_ensemble),
        },
        "notes": "Aggregation defaults to product-of-Gaussians over client posteriors (FedUAB-style).",
    }
    write_json(run_dir / "meta.json", run_meta)

    local_cfg = FedUABClientConfig(
        local_epochs=int(args.local_epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        num_workers=int(args.num_workers),
        mc_train=int(args.mc_train),
        grad_clip=float(args.grad_clip),
        seed=int(args.seed),
        loss_type=str(args.loss_type),
        pos_weight=None if str(args.pos_weight).lower() in ("auto", "client") else float(args.pos_weight),
        kl_coeff=float(args.kl_coeff),
        max_steps=int(args.max_steps),
        param_type=str(args.param_type),
        train_deterministic=bool(args.train_deterministic),
    )

    agg_cfg = AggConfig(
        precision_clamp=float(args.agg_precision_clamp),
        beta_mode=str(args.agg_beta_mode),
        beta_value=float(args.agg_beta_value),
    )

    history: List[Dict[str, Any]] = []
    compute_log: List[Dict[str, Any]] = []
    per_client_rounds: List[Dict[str, Any]] = []
    start_round = 1
    if bool(args.resume):
        ckpt_path = run_dir / "checkpoints" / "global_last.pt"
        if not ckpt_path.exists():
            raise SystemExit(f"--resume requested but checkpoint not found: {ckpt_path}")
        ckpt = load_checkpoint(ckpt_path, map_location="cpu")
        loaded_model_cfg = ckpt.get("model_cfg", None)
        if loaded_model_cfg is not None:
            model_cfg = normalize_model_cfg(loaded_model_cfg)

        loaded_base_state = ckpt.get("base_state", None)
        if not isinstance(loaded_base_state, dict):
            raise SystemExit(f"Invalid checkpoint (missing base_state): {ckpt_path}")
        base_state = {k: v.detach().cpu().clone() for k, v in loaded_base_state.items()}

        loaded_global_params = ckpt.get("global_params", None)
        if not isinstance(loaded_global_params, dict):
            raise SystemExit(f"Invalid checkpoint (missing global_params): {ckpt_path}")
        global_params = loaded_global_params

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
        print(f"[INFO] resume mode: run_dir={run_dir} start_round={start_round} rounds={int(args.rounds)}")

    run_meta["start_round"] = int(start_round)
    run_meta["resumed_from_round"] = int(start_round - 1) if bool(args.resume) else 0
    run_meta["model"] = asdict(model_cfg)
    if bool(args.resume):
        run_meta["resumed_utc"] = now_utc_iso()
    write_json(run_dir / "meta.json", run_meta)

    metrics_csv_path = run_dir / "metrics_round.csv"
    if not metrics_csv_path.exists():
        metrics_csv_path.write_text(
            "round,algo,auroc,auprc,ece,nll,brier,threshold,acc,f1,precision,recall\n",
            encoding="utf-8",
        )

    test_files = list_npz_files(args.data_dir, args.test_split)
    thr_method = str(args.threshold_method).lower()
    val_files: list[str] = []
    if thr_method == "youden-val":
        val_files = list_npz_files(args.data_dir, args.val_split)
        if not val_files:
            print("[WARN] youden-val requested but no val files found; falling back to fixed threshold.")
            thr_method = "fixed"
    client_test_files = list_npz_files_by_client(args.data_dir, args.test_split)

    final_client_posteriors: Dict[str, Dict[str, BayesParams]] = {}
    show_progress = not bool(args.no_progress_bar)
    n_rounds_left = max(0, int(args.rounds) - int(start_round) + 1)
    round_iter: Any = range(int(start_round), int(args.rounds) + 1)
    if show_progress:
        round_iter = tqdm(round_iter, total=n_rounds_left, desc="rounds", leave=True, position=0)
    for rnd in round_iter:
        selected_client_ids = _select_clients(
            client_ids,
            rnd=int(rnd),
            seed=int(args.seed),
            clients_per_round=int(args.clients_per_round),
            client_fraction=float(args.client_fraction),
        )
        client_posteriors: List[Dict[str, BayesParams]] = []
        client_posteriors_by_id: Dict[str, Dict[str, BayesParams]] = {}
        client_weights: List[int] = []
        client_metrics: List[Dict[str, Any]] = []
        train_loss_sum = 0.0
        train_loss_weight = 0
        total_steps = 0

        iterator = tqdm(
            selected_client_ids,
            total=len(selected_client_ids),
            desc=f"round {rnd}/{int(args.rounds)} clients",
            leave=False,
            disable=(not show_progress),
            position=1 if show_progress else 0,
        )
        for cid in iterator:
            train_files = client_train_files.get(str(cid), [])
            if not train_files:
                continue
            model_i = BFLModel(
                model_cfg,
                prior_sigma=float(args.prior_sigma),
                logvar_min=float(args.logvar_min),
                logvar_max=float(args.logvar_max),
                full_bayes=bool(args.full_bayes),
                param_type=str(args.param_type),
                mu_init=str(args.mu_init),
                init_rho=(float(args.init_rho) if args.init_rho is not None else None),
                var_reduction_h=float(args.var_reduction_h),
            )
            model_i.load_state_dict(base_state, strict=False)
            posterior, n_examples, metrics = train_client_feduab(
                client_id=str(cid),
                train_files=train_files,
                model=model_i,
                prior_params=global_params,
                cfg=local_cfg,
                device=device,
                logvar_min=float(args.logvar_min) if args.logvar_min is not None else None,
                logvar_max=float(args.logvar_max) if args.logvar_max is not None else None,
                show_progress=bool(args.client_progress_bar) and (not bool(args.no_progress_bar)),
            )
            client_metrics.append(metrics)
            if int(n_examples) > 0 and metrics.get("status") == "ok":
                client_posteriors_by_id[str(cid)] = posterior
            if int(n_examples) >= int(args.min_client_examples):
                client_posteriors.append(posterior)
                client_weights.append(int(n_examples))
                if metrics.get("status") == "ok":
                    train_loss_sum += float(metrics.get("avg_loss", 0.0)) * float(n_examples)
                    train_loss_weight += int(n_examples)
                    total_steps += int(metrics.get("n_steps", 0))

        if not client_posteriors:
            raise SystemExit("All clients empty after filtering/min_client_examples.")

        if str(args.client_weight_mode).lower() == "uniform":
            weights = [1.0] * len(client_posteriors)
        else:
            weights = [float(w) for w in client_weights]

        if str(args.agg_mode).lower() == "poe_prior_corrected":
            global_params, _ = poe_prior_corrected_dict(
                prior=global_params,
                posteriors=client_posteriors,
                weights=weights,
                cfg=agg_cfg,
            )
        else:
            global_params, _ = feduab_aggregate_dict(
                posteriors=client_posteriors,
                weights=weights,
                cfg=agg_cfg,
            )

        train_loss = float(train_loss_sum / max(train_loss_weight, 1))
        row = {
            "round": int(rnd),
            "train_loss": float(train_loss),
            "n_clients_selected": int(len(selected_client_ids)),
            "n_clients_used": int(len(client_posteriors)),
            "sum_examples": int(sum(client_weights)),
        }
        history.append(row)
        compute_log.append({"round": int(rnd), "client_steps": int(total_steps), "total_steps": int(total_steps)})
        write_json(run_dir / "compute_log.json", compute_log)
        write_json(
            run_dir / f"round_{rnd:03d}_clients.json",
            {"clients": client_metrics, "selected_clients": selected_client_ids},
        )

        if bool(args.test_every_round) and test_files:
            eval_model = _make_model(
                model_cfg=model_cfg,
                base_state=base_state,
                params=global_params,
                device=device,
                prior_sigma=float(args.prior_sigma),
                logvar_min=float(args.logvar_min),
                logvar_max=float(args.logvar_max),
                full_bayes=bool(args.full_bayes),
                param_type=str(args.param_type),
                mu_init=str(args.mu_init),
                init_rho=(float(args.init_rho) if args.init_rho is not None else None),
                var_reduction_h=float(args.var_reduction_h),
            )
            metrics, y_true, prob, _ = _eval_model(
                model=eval_model,
                files=test_files,
                mc_samples=int(args.mc_samples),
                device=device,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
            )
            if thr_method == "youden":
                thr = best_threshold_youden(y_true, prob, fallback=float(args.eval_threshold))
            elif thr_method == "youden-val" and val_files:
                _, y_val, p_val, _ = _eval_model(
                    model=eval_model,
                    files=val_files,
                    mc_samples=int(args.mc_samples),
                    device=device,
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                )
                thr = best_threshold_youden(y_val, p_val, fallback=float(args.eval_threshold))
            else:
                thr = float(args.eval_threshold)
            metrics_thr = calc_comprehensive_metrics(y_true, prob, threshold=thr, from_logits=False, n_bins=15)
            metrics_csv_path.write_text(
                metrics_csv_path.read_text(encoding="utf-8")
                + f"{rnd},{algo_name},{metrics.get('auroc','')},{metrics.get('auprc','')},{metrics.get('ece','')},{metrics.get('nll','')},{metrics.get('brier','')},{thr},{metrics_thr.get('accuracy','')},{metrics_thr.get('f1','')},{metrics_thr.get('ppv','')},{metrics_thr.get('sensitivity','')}\n",
                encoding="utf-8",
            )
            row.update(
                {
                    "test_auprc": float(metrics.get("auprc", float("nan"))),
                    "test_auroc": float(metrics.get("auroc", float("nan"))),
                    "test_brier": float(metrics.get("brier", float("nan"))),
                    "test_nll": float(metrics.get("nll", float("nan"))),
                    "test_ece": float(metrics.get("ece", float("nan"))),
                }
            )

        if bool(args.per_client_every_round) and client_test_files:
            eval_model = _make_model(
                model_cfg=model_cfg,
                base_state=base_state,
                params=global_params,
                device=device,
                prior_sigma=float(args.prior_sigma),
                logvar_min=float(args.logvar_min),
                logvar_max=float(args.logvar_max),
                full_bayes=bool(args.full_bayes),
                param_type=str(args.param_type),
                mu_init=str(args.mu_init),
                init_rho=(float(args.init_rho) if args.init_rho is not None else None),
                var_reduction_h=float(args.var_reduction_h),
            )
            val_y = None
            val_prob = None
            if thr_method == "youden-val" and val_files:
                _, val_y, val_prob, _ = _eval_model(
                    model=eval_model,
                    files=val_files,
                    mc_samples=int(args.mc_samples),
                    device=device,
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                )
            round_rows = []
            for cid, files in client_test_files.items():
                if not files:
                    continue
                metrics_c, y_c, prob_c, _ = _eval_model(
                    model=eval_model,
                    files=files,
                    mc_samples=int(args.mc_samples),
                    device=device,
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                )
                if thr_method == "youden":
                    thr = best_threshold_youden(y_c, prob_c, fallback=float(args.eval_threshold))
                elif thr_method == "youden-val" and val_y is not None and val_prob is not None:
                    thr = best_threshold_youden(val_y, val_prob, fallback=float(args.eval_threshold))
                else:
                    thr = float(args.eval_threshold)
                metrics_thr = calc_comprehensive_metrics(y_c, prob_c, threshold=thr, from_logits=False, n_bins=15)
                row_c = {
                    "round": int(rnd),
                    "client_id": str(cid),
                    "n": int(metrics_c.get("n", 0)),
                    "n_pos": int(metrics_c.get("n_pos", 0)),
                    "n_neg": int(metrics_c.get("n_neg", 0)),
                    "pos_rate": float(metrics_thr.get("pos_rate", float("nan"))),
                    "auprc": float(metrics_c.get("auprc", float("nan"))),
                    "auroc": float(metrics_c.get("auroc", float("nan"))),
                    "brier": float(metrics_c.get("brier", float("nan"))),
                    "nll": float(metrics_c.get("nll", float("nan"))),
                    "ece": float(metrics_c.get("ece", float("nan"))),
                    "threshold": float(thr),
                    "threshold_method": str(thr_method),
                    "accuracy": float(metrics_thr.get("accuracy", float("nan"))),
                    "f1": float(metrics_thr.get("f1", float("nan"))),
                    "sensitivity": float(metrics_thr.get("sensitivity", float("nan"))),
                    "specificity": float(metrics_thr.get("specificity", float("nan"))),
                    "ppv": float(metrics_thr.get("ppv", float("nan"))),
                    "npv": float(metrics_thr.get("npv", float("nan"))),
                }
                round_rows.append(row_c)
                per_client_rounds.append(row_c)
            if round_rows:
                import pandas as pd

                pd.DataFrame(round_rows).to_csv(run_dir / f"round_{rnd:03d}_test_per_client.csv", index=False)
                pd.DataFrame(per_client_rounds).to_csv(run_dir / "round_client_metrics.csv", index=False)
                write_json(run_dir / "round_client_metrics.json", per_client_rounds)

        # Save last global checkpoint each round
        save_checkpoint(
            run_dir / "checkpoints" / "global_last.pt",
            {
                "round": int(rnd),
                "model_cfg": asdict(model_cfg),
                "full_bayes": bool(args.full_bayes),
                "param_type": str(args.param_type),
                "prior_sigma": float(args.prior_sigma),
                "var_reduction_h": float(args.var_reduction_h),
                "logvar_min": float(args.logvar_min),
                "logvar_max": float(args.logvar_max),
                "mu_init": str(args.mu_init),
                "init_rho": float(args.init_rho) if args.init_rho is not None else None,
                "base_state": base_state,
                "global_params": global_params,
            },
        )
        _save_history(run_dir / "history.csv", history)

        eval_total = int(args.rounds) if (bool(args.test_every_round) and test_files) else 0
        eval_done = int(rnd - 1) if eval_total else 0
        msg = (
            f"[round {rnd:03d}] train_loss={train_loss:.4f} "
            f"used={len(client_posteriors)}/{len(selected_client_ids)} "
            f"progress total={rnd}/{int(args.rounds)} eval={eval_done}/{eval_total}"
        )
        if show_progress:
            tqdm.write(msg)
        else:
            print(msg)

        if rnd == int(args.rounds):
            final_client_posteriors = dict(client_posteriors_by_id)
            for cid, posterior in final_client_posteriors.items():
                save_checkpoint(
                    run_dir / "checkpoints" / "clients" / f"client_{cid}.pt",
                    {
                        "client_id": str(cid),
                        "posterior": posterior,
                    },
                )

    if not final_client_posteriors:
        final_client_posteriors = _load_client_posteriors(run_dir / "checkpoints" / "clients")

    # Evaluation modes
    if test_files:
        modes = _eval_modes(
            model_cfg=model_cfg,
            base_state=base_state,
            global_params=global_params,
            client_posteriors=final_client_posteriors,
            client_test_files=client_test_files,
            test_files=test_files,
            mc_samples=int(args.mc_samples),
            device=device,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            prior_sigma=float(args.prior_sigma),
            logvar_min=float(args.logvar_min),
            logvar_max=float(args.logvar_max),
            full_bayes=bool(args.full_bayes),
            param_type=str(args.param_type),
            mu_init=str(args.mu_init),
            init_rho=(float(args.init_rho) if args.init_rho is not None else None),
            var_reduction_h=float(args.var_reduction_h),
            eval_personalized=bool(args.eval_personalized),
            eval_global=bool(args.eval_global),
            eval_ensemble=bool(args.eval_ensemble),
            seed=int(args.seed),
        )
        write_eval_outputs(run_dir=run_dir, algo=algo_name, modes=modes)
        # Write a simple global test report for compatibility
        if "global_idless" in modes:
            write_json(
                run_dir / "test_report.json",
                {"metrics_pre": modes["global_idless"].get("overall", {})},
            )
            write_json(
                run_dir / "test_report_per_client.json",
                {"clients": modes["global_idless"].get("per_client", {})},
            )

    run_meta["finished_utc"] = now_utc_iso()
    run_meta["artifacts"] = {
        "history_csv": str(run_dir / "history.csv"),
        "metrics_round_csv": str(run_dir / "metrics_round.csv"),
        "global_checkpoint": str(run_dir / "checkpoints" / "global_last.pt"),
        "client_checkpoints": str(run_dir / "checkpoints" / "clients"),
        "eval_modes_json": str(run_dir / "eval_modes.json"),
    }
    write_json(run_dir / "meta.json", run_meta)

    print("Done.")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
