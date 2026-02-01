from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from bayes_federated.bayes_layers import BayesParams
from bayes_federated.models import BFLModel
from bayes_federated.vi import BetaConfig, LossConfig, compute_beta, elbo_loss
from common.checkpoint import capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint
from common.experiment import seed_worker
from common.dataset import WindowedNPZDataset


@dataclass(frozen=True)
class ClientTrainConfig:
    local_epochs: int = 1
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    mc_train: int = 3
    grad_clip: float = 0.0
    train_backbone: bool = False
    seed: int = 42
    loss_type: str = "bce"
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    pos_weight: float | None = None


@dataclass(frozen=True)
class ClientResult:
    client_id: str
    n_examples: int
    avg_loss: float
    avg_nll: float
    avg_kl: float
    avg_beta: float
    steps: int


def _make_loader(
    ds: WindowedNPZDataset,
    *,
    cfg: ClientTrainConfig,
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


def train_client_round(
    *,
    client_id: str,
    train_files: list[str],
    model: BFLModel,
    prior: Dict[str, BayesParams],
    cfg: ClientTrainConfig,
    beta_cfg: BetaConfig,
    device: torch.device,
    show_progress: bool = False,
    resume_path: str | None = None,
    save_path: str | None = None,
) -> tuple[BayesParams, ClientResult, Dict[str, Any]]:
    if not train_files:
        empty = ClientResult(client_id=str(client_id), n_examples=0, avg_loss=0.0, avg_nll=0.0, avg_kl=0.0, avg_beta=0.0, steps=0)
        return prior, empty, {"status": "empty"}

    ds = WindowedNPZDataset(
        train_files,
        use_clin="true",
        cache_in_memory=False,
        max_cache_files=32,
        cache_dtype="float32",
    )
    n_examples = int(len(ds))
    if n_examples == 0:
        empty = ClientResult(client_id=str(client_id), n_examples=0, avg_loss=0.0, avg_nll=0.0, avg_kl=0.0, avg_beta=0.0, steps=0)
        return prior, empty, {"status": "empty"}

    model = model.to(device)
    model.set_prior(prior)
    model.set_posterior(prior)

    if not cfg.train_backbone:
        model.freeze_backbone()

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    start_epoch = 0
    global_step = 0
    if resume_path and Path(resume_path).exists():
        ckpt = load_checkpoint(resume_path, map_location="cpu")
        if ckpt.get("completed") is True and "result" in ckpt:
            model.load_state_dict(ckpt["model_state"], strict=True)
            res = ClientResult(**ckpt["result"])
            meta = ckpt.get("meta", {"status": "ok"})
            return model.get_posterior(), res, meta
        model.load_state_dict(ckpt["model_state"], strict=True)
        opt.load_state_dict(ckpt["opt_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        if "rng_state" in ckpt:
            restore_rng_state(ckpt["rng_state"])

    stable = sum(bytearray(str(client_id).encode("utf-8"))) % 10000
    dl = _make_loader(ds, cfg=cfg, device=device, seed=cfg.seed + int(stable))

    total_loss = 0.0
    total_nll = 0.0
    total_kl = 0.0
    total_beta = 0.0
    total_steps = 0
    last_beta = 0.0

    for epoch in range(start_epoch, int(cfg.local_epochs)):
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
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=autocast_device, enabled=(device.type == "cuda")):
                logits_mc = model(x, sample=True, n_samples=int(cfg.mc_train))
                kl = model.kl_divergence()
                beta = compute_beta(step=global_step, dataset_size=n_examples, cfg=beta_cfg)
                loss_cfg = LossConfig(
                    loss_type=str(cfg.loss_type),
                    focal_gamma=float(cfg.focal_gamma),
                    focal_alpha=float(cfg.focal_alpha),
                    pos_weight=cfg.pos_weight,
                )
                loss, nll, kl_det = elbo_loss(logits_mc, y, kl=kl, beta=beta, loss_cfg=loss_cfg)

            scaler.scale(loss).backward()
            if cfg.grad_clip and float(cfg.grad_clip) > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            scaler.step(opt)
            scaler.update()

            total_loss += float(loss.item())
            total_nll += float(nll.item())
            total_kl += float(kl_det.item())
            total_beta += float(beta)
            last_beta = float(beta)
            total_steps += 1
            global_step += 1
            if show_progress:
                iterator.set_postfix(
                    loss=f"{total_loss / max(total_steps, 1):.4f}",
                    nll=f"{total_nll / max(total_steps, 1):.4f}",
                    kl=f"{total_kl / max(total_steps, 1):.2f}",
                    beta=f"{last_beta:.2e}",
                )

        if save_path:
            save_checkpoint(
                save_path,
                {
                    "client_id": str(client_id),
                    "epoch": int(epoch + 1),
                    "global_step": int(global_step),
                    "model_state": model.state_dict(),
                    "opt_state": opt.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "rng_state": capture_rng_state(),
                },
            )

    avg_loss = total_loss / max(total_steps, 1)
    avg_nll = total_nll / max(total_steps, 1)
    avg_kl = total_kl / max(total_steps, 1)
    avg_beta = total_beta / max(total_steps, 1)

    result = ClientResult(
        client_id=str(client_id),
        n_examples=n_examples,
        avg_loss=float(avg_loss),
        avg_nll=float(avg_nll),
        avg_kl=float(avg_kl),
        avg_beta=float(avg_beta),
        steps=int(total_steps),
    )
    meta = {"status": "ok", "beta_last": float(last_beta), "train_backbone": bool(cfg.train_backbone)}
    if save_path:
        save_checkpoint(
            save_path,
            {
                "client_id": str(client_id),
                "completed": True,
                "result": asdict(result),
                "meta": meta,
                "model_state": model.state_dict(),
            },
        )
    return model.get_posterior(), result, meta
