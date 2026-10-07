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
    fl_algo: str = "fedavg"
    fedprox_mu: float = 0.1
    perfedavg_alpha: float = 1e-3
    perfedavg_beta: float = 1e-3
    pfedme_lambda: float = 15.0
    pfedme_k: int = 5
    pfedme_personal_lr: float = 5e-3
    pfedme_mu: float = 0.0


def _build_local_optimizer(
    *,
    algo: str,
    params,
    lr: float,
    weight_decay: float,
) -> tuple[torch.optim.Optimizer, str]:
    # Paper-faithful default: FedProx/SCAFFOLD/FedNova/pFedMe/Per-FedAvg use SGD local updates.
    if algo in {"fedprox", "scaffold", "fednova", "pfedme", "perfedavg"}:
        return (
            torch.optim.SGD(
                params,
                lr=float(lr),
                momentum=0.0,
                weight_decay=float(weight_decay),
            ),
            "sgd",
        )
    return (
        torch.optim.AdamW(
            params,
            lr=float(lr),
            weight_decay=float(weight_decay),
        ),
        "adamw",
    )


def infer_pos_weight(train_files: list[str]) -> float:
    n_pos, n_total = scan_label_stats(train_files)
    n_neg = int(n_total - n_pos)
    if n_pos <= 0:
        return 1.0
    return float(max(n_neg, 1) / max(n_pos, 1))


def _state_to_device(state: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in state.items()}


def _move_batch_to_device(
    batch: Tuple[Any, Any],
    *,
    device: torch.device,
) -> Tuple[Any, torch.Tensor]:
    x, y = batch
    if isinstance(x, (tuple, list)):
        x = tuple(t.to(device, non_blocking=True) for t in x)
    else:
        x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    return x, y


def _copy_module_state_(dst: IOHNet, src: IOHNet) -> None:
    # Reuse an already-allocated module to avoid per-step model re-construction.
    with torch.no_grad():
        for p_dst, p_src in zip(dst.parameters(), src.parameters()):
            p_dst.copy_(p_src)
        for b_dst, b_src in zip(dst.buffers(), src.buffers()):
            b_dst.copy_(b_src)


def _train_perfedavg_fo(
    *,
    model: IOHNet,
    model_cfg: IOHModelConfig,
    dl: DataLoader,
    loss_fn: torch.nn.Module,
    cfg: LocalTrainConfig,
    device: torch.device,
    show_progress: bool,
    client_id: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    alpha = float(cfg.perfedavg_alpha)
    beta = float(cfg.perfedavg_beta)
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("perfedavg requires perfedavg_alpha>0 and perfedavg_beta>0")
    if len(dl) <= 0:
        raise ValueError("perfedavg requires non-empty dataloader")

    total_task_loss = 0.0
    total_obj_loss = 0.0
    total_n = 0
    total_steps = 0
    amp_enabled = device.type == "cuda"
    autocast_device = "cuda" if amp_enabled else "cpu"
    temp = IOHNet(model_cfg).to(device)
    temp.train()
    temp_params = tuple(temp.parameters())
    model_params = tuple(model.parameters())

    stream_iter = iter(dl)

    def next_batch() -> Tuple[Any, torch.Tensor]:
        nonlocal stream_iter
        try:
            raw = next(stream_iter)
        except StopIteration:
            stream_iter = iter(dl)
            raw = next(stream_iter)
        return _move_batch_to_device(raw, device=device)

    n_outer_steps = int(cfg.epochs) * max(int(len(dl)), 1)
    iterator = tqdm(
        range(n_outer_steps),
        total=n_outer_steps,
        desc=f"client {client_id} perfedavg",
        leave=False,
        disable=(not bool(show_progress)),
    )
    for _ in iterator:
        # Per-FedAvg(FO): temp one-step adaptation with alpha, then meta update with beta.
        _copy_module_state_(temp, model)

        x1, y1 = next_batch()
        with torch.amp.autocast(device_type=autocast_device, enabled=amp_enabled):
            logits1 = temp(x1).view(-1)
            loss1 = loss_fn(logits1, y1.view(-1))
        grads1 = torch.autograd.grad(loss1, temp_params, create_graph=False)
        with torch.no_grad():
            for p, g in zip(temp_params, grads1):
                p.sub_(alpha * g)

        x2, y2 = next_batch()
        with torch.amp.autocast(device_type=autocast_device, enabled=amp_enabled):
            logits2 = temp(x2).view(-1)
            loss2 = loss_fn(logits2, y2.view(-1))
        grads2 = torch.autograd.grad(loss2, temp_params, create_graph=False)
        with torch.no_grad():
            for p, g in zip(model_params, grads2):
                p.sub_(beta * g)

        batch_n = int(y2.shape[0])
        total_task_loss += float(loss2.item()) * batch_n
        total_obj_loss += float(loss2.item()) * batch_n
        total_n += batch_n
        total_steps += 1
        if show_progress:
            iterator.set_postfix(loss=f"{total_task_loss / max(total_n, 1):.4f}")

    updated = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    metrics = {
        "optimizer": "manual_perfedavg_fo_amp" if amp_enabled else "manual_perfedavg_fo",
        "n_steps": int(total_steps),
        "avg_loss": float(total_task_loss / max(total_n, 1)),
        "avg_objective": float(total_obj_loss / max(total_n, 1)),
    }
    return updated, metrics


def _train_pfedme(
    *,
    model: IOHNet,
    dl: DataLoader,
    loss_fn: torch.nn.Module,
    cfg: LocalTrainConfig,
    device: torch.device,
    show_progress: bool,
    client_id: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], Dict[str, Any]]:
    lamda = float(cfg.pfedme_lambda)
    personal_lr = float(cfg.pfedme_personal_lr)
    k_inner = int(cfg.pfedme_k)
    mu = float(cfg.pfedme_mu)
    if lamda < 0.0:
        raise ValueError("pfedme requires pfedme_lambda>=0")
    if personal_lr <= 0.0:
        raise ValueError("pfedme requires pfedme_personal_lr>0")
    if k_inner <= 0:
        raise ValueError("pfedme requires pfedme_k>=1")

    local_anchor: Dict[str, torch.Tensor] = {
        name: p.detach().clone() for name, p in model.named_parameters()
    }
    personal_opt = torch.optim.SGD(
        model.parameters(),
        lr=personal_lr,
        momentum=0.0,
        weight_decay=0.0,
    )
    amp_enabled = device.type == "cuda"
    autocast_device = "cuda" if amp_enabled else "cpu"
    scaler = torch.amp.GradScaler(enabled=amp_enabled)

    total_task_loss = 0.0
    total_obj_loss = 0.0
    total_n = 0
    total_steps = 0
    for _ in range(int(cfg.epochs)):
        iterator = tqdm(
            dl,
            total=len(dl),
            desc=f"client {client_id} pfedme",
            leave=False,
            disable=(not bool(show_progress)),
        )
        for raw in iterator:
            x, y = _move_batch_to_device(raw, device=device)
            batch_n = int(y.shape[0])
            for _k in range(k_inner):
                personal_opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=autocast_device, enabled=amp_enabled):
                    logits = model(x).view(-1)
                    task_loss = loss_fn(logits, y.view(-1))
                if amp_enabled:
                    scaler.scale(task_loss).backward()
                else:
                    task_loss.backward()

                grad_scale = float(scaler.get_scale()) if amp_enabled else 1.0
                reg_term = torch.zeros((), device=device)
                with torch.no_grad():
                    for name, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        if lamda != 0.0:
                            delta = p.detach() - local_anchor[name]
                            p.grad.add_(delta, alpha=lamda * grad_scale)
                            reg_term = reg_term + (0.5 * lamda) * torch.sum(delta * delta)
                        if mu != 0.0:
                            p.grad.add_(p.detach(), alpha=mu * grad_scale)
                            reg_term = reg_term + (0.5 * mu) * torch.sum(p.detach() * p.detach())

                if amp_enabled:
                    scaler.step(personal_opt)
                    scaler.update()
                else:
                    personal_opt.step()
                obj_loss = task_loss.detach() + reg_term

                total_task_loss += float(task_loss.item()) * batch_n
                total_obj_loss += float(obj_loss.item()) * batch_n
                total_n += batch_n
                total_steps += 1

            # w_i <- w_i - lambda * eta * (w_i - theta)
            eta_lam = lamda * float(cfg.lr)
            with torch.no_grad():
                for name, p in model.named_parameters():
                    local_anchor[name].sub_(eta_lam * (local_anchor[name] - p.detach()))
            if show_progress:
                iterator.set_postfix(loss=f"{total_task_loss / max(total_n, 1):.4f}")

    # Upload local reference model w_i to server.
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(local_anchor[name])
    updated = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    aux = {
        "pfedme_personalized_state": {
            k: v.detach().cpu().clone() for k, v in local_anchor.items()
        }
    }
    metrics = {
        "optimizer": "sgd_pfedme_inner_amp" if amp_enabled else "sgd_pfedme_inner",
        "n_steps": int(total_steps),
        "avg_loss": float(total_task_loss / max(total_n, 1)),
        "avg_objective": float(total_obj_loss / max(total_n, 1)),
    }
    return updated, metrics, aux


def train_one_client(
    *,
    client_id: str,
    train_files: list[str],
    model_cfg: IOHModelConfig,
    global_state: Dict[str, torch.Tensor],
    cfg: LocalTrainConfig = LocalTrainConfig(),
    device: torch.device,
    show_progress: bool = False,
    train_dataset: WindowedNPZDataset | None = None,
    train_loader: DataLoader | None = None,
    n_examples: int | None = None,
    pos_weight: float | None = None,
    server_control: Dict[str, torch.Tensor] | None = None,
    client_control: Dict[str, torch.Tensor] | None = None,
) -> Tuple[Dict[str, torch.Tensor], int, Dict[str, Any], Dict[str, Any]]:
    """
    Train a single client locally starting from `global_state`.
    Returns (updated_state_dict, n_examples, metrics_dict).
    """
    if train_dataset is None and not train_files:
        return global_state, 0, {"client_id": str(client_id), "status": "empty"}, {}

    ds = train_dataset
    dl = train_loader
    if ds is None and dl is not None:
        ds_obj = getattr(dl, "dataset", None)
        if isinstance(ds_obj, WindowedNPZDataset):
            ds = ds_obj
    if ds is None and dl is None:
        ds = WindowedNPZDataset(
            train_files,
            use_clin="true",
            cache_in_memory=bool(cfg.cache_in_memory),
            max_cache_files=int(cfg.max_cache_files),
            cache_dtype=str(cfg.cache_dtype),
        )
    ds_len = int(n_examples) if n_examples is not None else (int(len(ds)) if ds is not None else 0)
    if ds_len == 0:
        return global_state, 0, {"client_id": str(client_id), "status": "empty"}, {}

    if dl is None:
        if ds is None:
            return global_state, 0, {"client_id": str(client_id), "status": "empty"}, {}
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
    algo = str(cfg.fl_algo).strip().lower()

    resolved_pos_weight = float(cfg.pos_weight) if cfg.pos_weight is not None else (
        float(pos_weight) if pos_weight is not None else infer_pos_weight(train_files)
    )
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([resolved_pos_weight], device=device))

    opt, opt_name = _build_local_optimizer(
        algo=algo,
        params=model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )

    if algo == "perfedavg":
        updated, algo_metrics = _train_perfedavg_fo(
            model=model,
            model_cfg=model_cfg,
            dl=dl,
            loss_fn=loss_fn,
            cfg=cfg,
            device=device,
            show_progress=bool(show_progress),
            client_id=str(client_id),
        )
        metrics = {
            "client_id": str(client_id),
            "status": "ok",
            "algo": str(algo),
            "optimizer": str(algo_metrics.get("optimizer", opt_name)),
            "n_examples": int(ds_len),
            "n_steps": int(algo_metrics.get("n_steps", 0)),
            "avg_loss": float(algo_metrics.get("avg_loss", float("nan"))),
            "avg_objective": float(algo_metrics.get("avg_objective", float("nan"))),
            "pos_weight": float(resolved_pos_weight),
        }
        return updated, int(ds_len), metrics, {}

    if algo == "pfedme":
        updated, algo_metrics, aux = _train_pfedme(
            model=model,
            dl=dl,
            loss_fn=loss_fn,
            cfg=cfg,
            device=device,
            show_progress=bool(show_progress),
            client_id=str(client_id),
        )
        metrics = {
            "client_id": str(client_id),
            "status": "ok",
            "algo": str(algo),
            "optimizer": str(algo_metrics.get("optimizer", opt_name)),
            "n_examples": int(ds_len),
            "n_steps": int(algo_metrics.get("n_steps", 0)),
            "avg_loss": float(algo_metrics.get("avg_loss", float("nan"))),
            "avg_objective": float(algo_metrics.get("avg_objective", float("nan"))),
            "pos_weight": float(resolved_pos_weight),
        }
        return updated, int(ds_len), metrics, aux

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    global_state_device: Dict[str, torch.Tensor] | None = None
    if algo in {"fedprox", "scaffold"}:
        global_state_device = _state_to_device(global_state, device=device)

    server_control_device: Dict[str, torch.Tensor] | None = None
    client_control_device: Dict[str, torch.Tensor] | None = None
    if algo == "scaffold":
        if server_control is None or client_control is None:
            raise ValueError("scaffold requires server_control and client_control")
        server_control_device = _state_to_device(server_control, device=device)
        client_control_device = _state_to_device(client_control, device=device)

    total_task_loss = 0.0
    total_obj_loss = 0.0
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
                task_loss = loss_fn(logits, y.view(-1))
                obj_loss = task_loss
                if algo == "fedprox":
                    assert global_state_device is not None
                    prox = torch.zeros((), device=device)
                    for name, param in model.named_parameters():
                        ref = global_state_device[name]
                        prox = prox + torch.sum((param - ref) ** 2)
                    obj_loss = task_loss + 0.5 * float(cfg.fedprox_mu) * prox
            scaler.scale(obj_loss).backward()
            if algo == "scaffold":
                assert server_control_device is not None
                assert client_control_device is not None
                scaler.unscale_(opt)
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    param.grad.add_(server_control_device[name] - client_control_device[name])
            scaler.step(opt)
            scaler.update()
            batch_n = int(y.shape[0])
            total_task_loss += float(task_loss.item()) * batch_n
            total_obj_loss += float(obj_loss.item()) * batch_n
            total_n += int(y.shape[0])
            total_steps += 1
            if show_progress:
                iterator.set_postfix(loss=f"{total_task_loss / max(total_n, 1):.4f}")

    updated = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    aux: Dict[str, Any] = {}
    if algo == "scaffold":
        if total_steps > 0 and float(cfg.lr) > 0.0:
            denom = float(total_steps) * float(cfg.lr)
            client_control_new: Dict[str, torch.Tensor] = {}
            client_control_delta: Dict[str, torch.Tensor] = {}
            assert server_control is not None
            assert client_control is not None
            for k, wt in global_state.items():
                ci_old = client_control[k].detach().cpu().clone()
                c = server_control[k].detach().cpu()
                wi = updated[k].detach().cpu()
                ci_new = ci_old - c + (wt.detach().cpu() - wi) / float(denom)
                client_control_new[k] = ci_new
                client_control_delta[k] = ci_new - ci_old
        else:
            assert client_control is not None
            client_control_new = {k: v.detach().cpu().clone() for k, v in client_control.items()}
            client_control_delta = {k: torch.zeros_like(v) for k, v in client_control_new.items()}
        aux = {
            "client_control_new": client_control_new,
            "client_control_delta": client_control_delta,
        }
    elif algo == "fednova":
        # For plain local SGD, FedNova's a_i equals the local step count.
        aux = {
            "fednova_local_normalizer": float(max(int(total_steps), 1)),
        }
    metrics = {
        "client_id": str(client_id),
        "status": "ok",
        "algo": str(algo),
        "optimizer": str(opt_name),
        "n_examples": int(ds_len),
        "n_steps": int(total_steps),
        "avg_loss": float(total_task_loss / max(total_n, 1)),
        "avg_objective": float(total_obj_loss / max(total_n, 1)),
        "pos_weight": float(resolved_pos_weight),
    }
    if algo == "fednova":
        metrics["fednova_local_normalizer"] = float(max(int(total_steps), 1))
    return updated, int(len(ds)), metrics, aux
