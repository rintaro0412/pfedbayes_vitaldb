from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bayes_federated.bayes_layers import BayesParams
from bayes_federated.models import BFLModel
from bayes_federated.pfedbayes_utils import (
    bayes_dict_to_cpu,
    bayes_param_list,
    clamp_bayes_logvar_,
    clone_bayes_dict,
    detach_bayes_dict,
)
from bayes_federated.vi import LossConfig, elbo_loss, kl_bayes_dict
from common.checkpoint import load_checkpoint, save_checkpoint
from common.dataset import WindowedNPZDataset
from common.experiment import seed_worker


@dataclass(frozen=True)
class PFBayesClientConfig:
    local_epochs: int = 1
    batch_size: int = 64
    lr_q: float = 1e-3
    lr_w: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    mc_train: int = 3
    grad_clip: float = 0.0
    grad_clip_w: float = 0.0
    train_backbone: bool = False
    seed: int = 42
    loss_type: str = "bce"
    pos_weight: float | None = None
    zeta: float = 1.0
    max_steps: int = 0
    q_optim: str = "sgd"  # sgd | adam | adamw
    w_optim: str = "sgd"  # sgd | adam | adamw
    param_type: str = "logvar"


@dataclass(frozen=True)
class ClientResult:
    client_id: str
    n_examples: int
    avg_loss: float
    avg_nll: float
    avg_kl: float
    avg_zeta: float
    avg_w_kl: float
    steps: int


def _make_loader(
    ds: WindowedNPZDataset,
    *,
    cfg: PFBayesClientConfig,
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


def _make_w_optimizer(params: Dict[str, BayesParams], *, cfg: PFBayesClientConfig) -> torch.optim.Optimizer:
    w_params = bayes_param_list(params)
    mode = str(cfg.w_optim).lower()
    if mode == "adam":
        return torch.optim.Adam(w_params, lr=float(cfg.lr_w))
    if mode == "adamw":
        return torch.optim.AdamW(w_params, lr=float(cfg.lr_w))
    return torch.optim.SGD(w_params, lr=float(cfg.lr_w))


def _make_q_optimizer(params: torch.nn.Module, *, cfg: PFBayesClientConfig) -> torch.optim.Optimizer:
    mode = str(cfg.q_optim).lower()
    if mode == "adam":
        return torch.optim.Adam(params.parameters(), lr=float(cfg.lr_q), weight_decay=float(cfg.weight_decay))
    if mode == "adamw":
        return torch.optim.AdamW(params.parameters(), lr=float(cfg.lr_q), weight_decay=float(cfg.weight_decay))
    return torch.optim.SGD(params.parameters(), lr=float(cfg.lr_q), weight_decay=float(cfg.weight_decay))


def train_client_pfedbayes(
    *,
    client_id: str,
    train_files: list[str],
    model: BFLModel,
    global_params: Dict[str, BayesParams],
    cfg: PFBayesClientConfig,
    device: torch.device,
    logvar_min: float | None,
    logvar_max: float | None,
    show_progress: bool = False,
    resume_path: str | None = None,
    save_path: str | None = None,
) -> tuple[Dict[str, BayesParams], Dict[str, BayesParams], ClientResult, Dict[str, Any]]:
    if not train_files:
        empty = ClientResult(
            client_id=str(client_id),
            n_examples=0,
            avg_loss=0.0,
            avg_nll=0.0,
            avg_kl=0.0,
            avg_zeta=float(cfg.zeta),
            avg_w_kl=0.0,
            steps=0,
        )
        return global_params, global_params, empty, {"status": "empty"}

    ds = WindowedNPZDataset(
        train_files,
        use_clin="true",
        cache_in_memory=False,
        max_cache_files=32,
        cache_dtype="float32",
    )
    n_examples = int(len(ds))
    if n_examples == 0:
        empty = ClientResult(
            client_id=str(client_id),
            n_examples=0,
            avg_loss=0.0,
            avg_nll=0.0,
            avg_kl=0.0,
            avg_zeta=float(cfg.zeta),
            avg_w_kl=0.0,
            steps=0,
        )
        return global_params, global_params, empty, {"status": "empty"}

    if resume_path and Path(resume_path).exists():
        ckpt = load_checkpoint(resume_path, map_location="cpu")
        if ckpt.get("completed") is True and "result" in ckpt:
            res = ClientResult(**ckpt["result"])
            meta = ckpt.get("meta", {"status": "ok"})
            q_params = ckpt.get("posterior", global_params)
            w_params = ckpt.get("localized_global", global_params)
            return q_params, w_params, res, meta

    model = model.to(device)
    model.set_posterior(global_params)
    if not cfg.train_backbone:
        model.freeze_backbone()

    w_params = clone_bayes_dict(global_params, device=device, requires_grad=True)
    clamp_bayes_logvar_(w_params, logvar_min=logvar_min, logvar_max=logvar_max, param_type=cfg.param_type)

    opt_q = _make_q_optimizer(model, cfg=cfg)
    opt_w = _make_w_optimizer(w_params, cfg=cfg)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    stable = sum(bytearray(str(client_id).encode("utf-8"))) % 10000
    dl = _make_loader(ds, cfg=cfg, device=device, seed=cfg.seed + int(stable))

    loss_cfg = LossConfig(
        loss_type=str(cfg.loss_type),
        pos_weight=cfg.pos_weight,
    )

    total_loss = 0.0
    total_nll = 0.0
    total_kl = 0.0
    total_w_kl = 0.0
    total_steps = 0

    for epoch in range(int(cfg.local_epochs)):
        model.train()
        iterator = tqdm(
            dl,
            total=len(dl),
            desc=f"client {client_id} ep {epoch + 1}/{int(cfg.local_epochs)}",
            leave=False,
            disable=(not bool(show_progress)),
        )
        for x, y in iterator:
            if isinstance(x, (tuple, list)):
                x = tuple(t.to(device, non_blocking=True) for t in x)
            else:
                x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Update personalized posterior q_i
            opt_q.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=autocast_device, enabled=(device.type == "cuda")):
                logits_mc = model(x, sample=True, n_samples=int(cfg.mc_train))
                q_params = model.get_posterior_params()
                kl_qw = kl_bayes_dict(q_params, w_params, logvar_min=logvar_min, logvar_max=logvar_max, param_type=cfg.param_type)
                batch_scale = float(n_examples) / float(max(int(y.shape[0]), 1))
                loss, nll, kl_det = elbo_loss(
                    logits_mc,
                    y,
                    kl=kl_qw,
                    beta=float(cfg.zeta),
                    loss_cfg=loss_cfg,
                    nll_scale=batch_scale,
                )

            scaler.scale(loss).backward()
            if cfg.grad_clip and float(cfg.grad_clip) > 0:
                scaler.unscale_(opt_q)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            scaler.step(opt_q)
            scaler.update()

            # Update localized global parameters w_i
            opt_w.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=autocast_device, enabled=False):
                q_det = detach_bayes_dict(model.get_posterior_params())
                w_kl = kl_bayes_dict(q_det, w_params, logvar_min=logvar_min, logvar_max=logvar_max, param_type=cfg.param_type)
            w_kl.backward()
            if cfg.grad_clip_w and float(cfg.grad_clip_w) > 0:
                torch.nn.utils.clip_grad_norm_(bayes_param_list(w_params), float(cfg.grad_clip_w))
            opt_w.step()
            clamp_bayes_logvar_(w_params, logvar_min=logvar_min, logvar_max=logvar_max, param_type=cfg.param_type)

            total_loss += float(loss.item())
            total_nll += float(nll.item())
            total_kl += float(kl_det.item())
            total_w_kl += float(w_kl.item())
            total_steps += 1
            if show_progress:
                iterator.set_postfix(
                    loss=f"{total_loss / max(total_steps, 1):.4f}",
                    nll=f"{total_nll / max(total_steps, 1):.4f}",
                    kl=f"{total_kl / max(total_steps, 1):.2f}",
                    w_kl=f"{total_w_kl / max(total_steps, 1):.2f}",
                )

            if int(cfg.max_steps) > 0 and total_steps >= int(cfg.max_steps):
                break
        if int(cfg.max_steps) > 0 and total_steps >= int(cfg.max_steps):
            break

    avg_loss = total_loss / max(total_steps, 1)
    avg_nll = total_nll / max(total_steps, 1)
    avg_kl = total_kl / max(total_steps, 1)
    avg_w_kl = total_w_kl / max(total_steps, 1)

    result = ClientResult(
        client_id=str(client_id),
        n_examples=n_examples,
        avg_loss=float(avg_loss),
        avg_nll=float(avg_nll),
        avg_kl=float(avg_kl),
        avg_zeta=float(cfg.zeta),
        avg_w_kl=float(avg_w_kl),
        steps=int(total_steps),
    )
    meta = {
        "status": "ok",
        "train_backbone": bool(cfg.train_backbone),
        "zeta": float(cfg.zeta),
    }

    q_params = model.get_posterior()
    w_params_out = bayes_dict_to_cpu(w_params)

    if save_path:
        save_checkpoint(
            save_path,
            {
                "client_id": str(client_id),
                "completed": True,
                "result": asdict(result),
                "meta": meta,
                "posterior": q_params,
                "localized_global": w_params_out,
            },
        )

    return q_params, w_params_out, result, meta
