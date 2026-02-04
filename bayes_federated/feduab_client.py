from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bayes_federated.bayes_layers import BayesParams
from bayes_federated.models import BFLModel
from bayes_federated.pfedbayes_utils import bayes_dict_to_cpu, clamp_bayes_logvar_
from bayes_federated.vi import LossConfig, elbo_loss, kl_bayes_dict
from common.dataset import WindowedNPZDataset, scan_label_stats
from common.experiment import seed_worker


@dataclass(frozen=True)
class FedUABClientConfig:
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    mc_train: int = 3
    grad_clip: float = 0.0
    seed: int = 42
    loss_type: str = "bce"
    pos_weight: float | None = None
    kl_coeff: float = 1e-4
    max_steps: int = 0
    param_type: str = "rho"
    train_deterministic: bool = False


def _infer_pos_weight(train_files: list[str]) -> float:
    n_pos, n_total = scan_label_stats(train_files)
    n_neg = int(n_total - n_pos)
    if n_pos <= 0:
        return 1.0
    return float(max(n_neg, 1) / max(n_pos, 1))


def _make_loader(
    ds: WindowedNPZDataset,
    *,
    cfg: FedUABClientConfig,
    device: torch.device,
    seed: int,
) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(int(seed))
    return DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(cfg.num_workers) > 0),
        worker_init_fn=seed_worker,
        generator=g,
    )


def _configure_trainable_params(model: BFLModel, *, train_deterministic: bool) -> None:
    if train_deterministic:
        for p in model.parameters():
            p.requires_grad = True
        return
    for name, p in model.named_parameters():
        if any(tok in name for tok in ("weight_mu", "weight_logvar", "bias_mu", "bias_logvar")):
            p.requires_grad = True
        else:
            p.requires_grad = False


def _prior_to_device(params: Dict[str, BayesParams], *, device: torch.device) -> Dict[str, BayesParams]:
    out: Dict[str, BayesParams] = {}
    for k, p in params.items():
        out[k] = BayesParams(
            weight_mu=p.weight_mu.detach().to(device),
            weight_logvar=p.weight_logvar.detach().to(device),
            bias_mu=p.bias_mu.detach().to(device),
            bias_logvar=p.bias_logvar.detach().to(device),
        )
    return out


def train_client_feduab(
    *,
    client_id: str,
    train_files: list[str],
    model: BFLModel,
    prior_params: Dict[str, BayesParams],
    cfg: FedUABClientConfig,
    device: torch.device,
    logvar_min: float | None,
    logvar_max: float | None,
    show_progress: bool = False,
) -> tuple[Dict[str, BayesParams], int, Dict[str, Any]]:
    if not train_files:
        return prior_params, 0, {"client_id": str(client_id), "status": "empty"}

    ds = WindowedNPZDataset(
        train_files,
        use_clin="true",
        cache_in_memory=False,
        max_cache_files=32,
        cache_dtype="float32",
    )
    n_examples = int(len(ds))
    if n_examples == 0:
        return prior_params, 0, {"client_id": str(client_id), "status": "empty"}

    model = model.to(device)
    prior_dev = _prior_to_device(prior_params, device=device)
    model.set_posterior(prior_dev)
    model.set_prior(prior_dev)
    _configure_trainable_params(model, train_deterministic=bool(cfg.train_deterministic))

    stable = sum(bytearray(str(client_id).encode("utf-8"))) % 10000
    dl = _make_loader(ds, cfg=cfg, device=device, seed=cfg.seed + int(stable))

    pos_weight = float(cfg.pos_weight) if cfg.pos_weight is not None else _infer_pos_weight(train_files)
    loss_cfg = LossConfig(loss_type=str(cfg.loss_type), pos_weight=pos_weight)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    total_loss = 0.0
    total_nll = 0.0
    total_kl = 0.0
    total_steps = 0

    for ep in range(int(cfg.local_epochs)):
        model.train()
        iterator = tqdm(
            dl,
            total=len(dl),
            desc=f"client {client_id} ep {ep + 1}/{int(cfg.local_epochs)}",
            leave=False,
            disable=(not bool(show_progress)),
        )
        for x, y in iterator:
            if isinstance(x, (tuple, list)):
                x = tuple(t.to(device, non_blocking=True) for t in x)
            else:
                x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=autocast_device, enabled=(device.type == "cuda")):
                logits_mc = model(x, sample=True, n_samples=int(cfg.mc_train))
                if logits_mc.dim() == 2:
                    logits_mc = logits_mc.unsqueeze(0)
                elif logits_mc.dim() == 1:
                    logits_mc = logits_mc.view(1, -1, 1)
                q_params = model.get_posterior_params()
                kl_qp = kl_bayes_dict(
                    q_params,
                    prior_dev,
                    logvar_min=logvar_min,
                    logvar_max=logvar_max,
                    param_type=cfg.param_type,
                )
                batch_scale = float(n_examples) / float(max(int(y.shape[0]), 1))
                loss, nll, kl_det = elbo_loss(
                    logits_mc,
                    y,
                    kl=kl_qp,
                    beta=float(cfg.kl_coeff),
                    loss_cfg=loss_cfg,
                    nll_scale=batch_scale,
                )

            scaler.scale(loss).backward()
            if cfg.grad_clip and float(cfg.grad_clip) > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            scaler.step(opt)
            scaler.update()

            clamp_bayes_logvar_(
                model.get_posterior_params(),
                logvar_min=logvar_min,
                logvar_max=logvar_max,
                param_type=cfg.param_type,
            )

            total_loss += float(loss.item())
            total_nll += float(nll.item())
            total_kl += float(kl_det.item())
            total_steps += 1
            if show_progress:
                iterator.set_postfix(
                    loss=f"{total_loss / max(total_steps, 1):.4f}",
                    nll=f"{total_nll / max(total_steps, 1):.4f}",
                    kl=f"{total_kl / max(total_steps, 1):.2f}",
                )

            if int(cfg.max_steps) > 0 and total_steps >= int(cfg.max_steps):
                break
        if int(cfg.max_steps) > 0 and total_steps >= int(cfg.max_steps):
            break

    avg_loss = total_loss / max(total_steps, 1)
    avg_nll = total_nll / max(total_steps, 1)
    avg_kl = total_kl / max(total_steps, 1)

    q_params = bayes_dict_to_cpu(model.get_posterior())
    metrics = {
        "client_id": str(client_id),
        "status": "ok",
        "n_examples": int(n_examples),
        "n_steps": int(total_steps),
        "avg_loss": float(avg_loss),
        "avg_nll": float(avg_nll),
        "avg_kl": float(avg_kl),
        "pos_weight": float(pos_weight),
        "kl_coeff": float(cfg.kl_coeff),
    }
    return q_params, int(n_examples), metrics
