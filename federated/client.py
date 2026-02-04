from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.dataset import WindowedNPZDataset, scan_label_stats
from common.ioh_model import IOHModelConfig, IOHNet


@dataclass(frozen=True)
class LocalTrainConfig:
    epochs: int = 1
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    pos_weight: float | None = None
    num_workers: int = 0
    cache_in_memory: bool = False
    max_cache_files: int = 32
    cache_dtype: str = "float32"


def _infer_pos_weight(train_files: list[str]) -> float:
    n_pos, n_total = scan_label_stats(train_files)
    n_neg = int(n_total - n_pos)
    if n_pos <= 0:
        return 1.0
    return float(max(n_neg, 1) / max(n_pos, 1))


def train_one_client(
    *,
    client_id: str,
    train_files: list[str],
    model_cfg: IOHModelConfig,
    global_state: Dict[str, torch.Tensor],
    cfg: LocalTrainConfig = LocalTrainConfig(),
    device: torch.device,
    show_progress: bool = False,
) -> Tuple[Dict[str, torch.Tensor], int, Dict[str, Any]]:
    """
    Train a single client locally starting from `global_state`.
    Returns (updated_state_dict, n_examples, metrics_dict).
    """
    if not train_files:
        return global_state, 0, {"client_id": str(client_id), "status": "empty"}

    ds = WindowedNPZDataset(
        train_files,
        use_clin="true",
        cache_in_memory=bool(cfg.cache_in_memory),
        max_cache_files=int(cfg.max_cache_files),
        cache_dtype=str(cfg.cache_dtype),
    )
    if len(ds) == 0:
        return global_state, 0, {"client_id": str(client_id), "status": "empty"}

    dl = DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(cfg.num_workers) > 0),
    )

    model = IOHNet(model_cfg).to(device)
    model.load_state_dict(global_state, strict=True)
    model.train()

    pos_weight = float(cfg.pos_weight) if cfg.pos_weight is not None else _infer_pos_weight(train_files)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    total_loss = 0.0
    total_n = 0
    total_steps = 0
    for ep in range(int(cfg.epochs)):
        iterator = tqdm(
            dl,
            total=len(dl),
            desc=f"client {client_id} ep {ep + 1}/{int(cfg.epochs)}",
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
                logits = model(x).view(-1)
                loss = loss_fn(logits, y.view(-1))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_loss += float(loss.item()) * int(y.shape[0])
            total_n += int(y.shape[0])
            total_steps += 1
            if show_progress:
                iterator.set_postfix(loss=f"{total_loss / max(total_n, 1):.4f}")

    updated = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    metrics = {
        "client_id": str(client_id),
        "status": "ok",
        "n_examples": int(len(ds)),
        "n_steps": int(total_steps),
        "avg_loss": float(total_loss / max(total_n, 1)),
        "pos_weight": float(pos_weight),
    }
    return updated, int(len(ds)), metrics
