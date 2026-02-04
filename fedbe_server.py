from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root in sys.path when executed as `python fedbe_server.py`
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.dataset import WindowedNPZDataset, list_client_ids, list_npz_files
from common.fedbe_ensemble import fedavg_state, sample_teacher_states
from common.io import ensure_dir, get_git_hash, now_utc_iso, write_json
from common.ioh_model import IOHModelConfig, IOHNet
from common.trace import hash_state_dict, l2_diff_state_dict
from common.metrics import compute_binary_metrics, confusion_at_threshold, sigmoid_np
from common.utils import calc_comprehensive_metrics, set_seed
from datasets.unlabeled_dataset import UnlabeledNPZDataset
from federated.client import LocalTrainConfig, train_one_client
from scripts.make_server_unlabeled import create_server_unlabeled

LAST_META_PATH: Path | None = None


def _load_config(path: str) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except Exception:
        try:
            return json.loads(text)
        except Exception as e:
            raise RuntimeError("Failed to parse config. Install PyYAML or use JSON syntax.") from e


def _list_unlabeled_files(server_unlabeled_dir: str) -> List[str]:
    p = Path(server_unlabeled_dir)
    if not p.exists():
        return []
    return sorted([str(x) for x in p.glob("*.npz")])


def _make_unlabeled_if_needed(cfg: Dict[str, Any]) -> None:
    su_cfg = cfg.get("server_unlabeled", {})
    out_dir = str(cfg["data"]["server_unlabeled_dir"])
    make_if_missing = bool(su_cfg.get("make_if_missing", False))
    if not make_if_missing:
        return
    existing = _list_unlabeled_files(out_dir)
    if existing:
        return
    stats = create_server_unlabeled(
        federated_data_dir=str(cfg["data"]["federated_dir"]),
        out_dir=out_dir,
        frac=float(su_cfg.get("frac", 0.1)),
        seed=int(su_cfg.get("seed", 42)),
        mode=str(su_cfg.get("mode", "drop_y")),
    )
    print("[server_unlabeled] auto-created")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def _distill_loss(
    *,
    logits: torch.Tensor,
    teacher_prob: torch.Tensor,
    loss_type: str,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        temperature = 1.0
    logits_t = logits / float(temperature)
    if loss_type == "ce":
        return torch.nn.functional.binary_cross_entropy_with_logits(logits_t, teacher_prob)
    if loss_type == "kl":
        eps = 1e-6
        t = torch.clamp(teacher_prob, eps, 1.0 - eps)
        s = torch.sigmoid(logits_t)
        s = torch.clamp(s, eps, 1.0 - eps)
        kl = t * torch.log(t / s) + (1.0 - t) * torch.log((1.0 - t) / (1.0 - s))
        return kl.mean()
    raise ValueError(f"unknown loss_type: {loss_type}")


@torch.no_grad()
def _predict_logits(model: IOHNet, dl: DataLoader, *, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
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


def _build_client_models(
    *,
    client_states: List[Dict[str, torch.Tensor]],
    model_cfg: IOHModelConfig,
    device: torch.device,
) -> List[IOHNet]:
    models: List[IOHNet] = []
    for st in client_states:
        m = IOHNet(model_cfg).to(device)
        m.load_state_dict(st, strict=True)
        m.eval()
        models.append(m)
    return models


@torch.no_grad()
def _teacher_prob_from_models(
    *,
    models: List[IOHNet],
    x: tuple | torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        temperature = 1.0
    probs = []
    for m in models:
        logits = m(x).view(-1) / float(temperature)
        probs.append(torch.sigmoid(logits))
    return torch.stack(probs, dim=0).mean(dim=0)


def _ensemble_teacher_prob(
    *,
    teacher_states: List[Dict[str, torch.Tensor]],
    model_cfg: IOHModelConfig,
    x: tuple | torch.Tensor,
    device: torch.device,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        temperature = 1.0
    if not teacher_states:
        raise ValueError("no teacher states for ensemble")
    model = IOHNet(model_cfg).to(device)
    probs = []
    with torch.no_grad():
        for state in teacher_states:
            model.load_state_dict(state, strict=True)
            model.eval()
            logits = model(x).view(-1) / float(temperature)
            probs.append(torch.sigmoid(logits))
    return torch.stack(probs, dim=0).mean(dim=0).detach()


def _swa_update(
    *,
    swa_state: Dict[str, torch.Tensor] | None,
    model_state: Dict[str, torch.Tensor],
    swa_n: int,
) -> tuple[Dict[str, torch.Tensor], int]:
    if swa_state is None:
        out = {k: v.detach().cpu().clone() for k, v in model_state.items()}
        return out, 1
    denom = float(swa_n + 1)
    for k, v in model_state.items():
        swa_state[k] = (swa_state[k] * float(swa_n) + v.detach().cpu()) / denom
    return swa_state, swa_n + 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Federated Bayesian Ensemble (FedBE) server")
    ap.add_argument("--config", default="configs/fedbe.yaml")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--client-progress-bar", action="store_true", help="Show per-client batch progress bar.")
    ap.add_argument("--no-progress-bar", action="store_true", help="Disable progress bars.")
    ap.add_argument("--test-every-round", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--max-clients", type=int, default=0, help="Limit number of clients (dry run).")
    ap.add_argument("--max-client-files", type=int, default=0, help="Limit train files per client (dry run).")
    ap.add_argument("--max-unlabeled-batches", type=int, default=0, help="Limit unlabeled batches per epoch (dry run).")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    algo_name = "fedbe"
    set_seed(int(cfg.get("seed", 42)))
    torch.set_num_threads(max(1, min(os.cpu_count() or 4, 8)))
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    _make_unlabeled_if_needed(cfg)

    tag = args.run_name or cfg.get("run", {}).get("run_name") or cfg.get("experiment", {}).get("name") or "default"
    ts = now_utc_iso().replace("-", "").replace(":", "").replace("T", "_").replace("Z", "")
    run_id = f"{ts}_{algo_name}_{tag}"
    legacy_dir = os.environ.get("LEGACY_RUN_DIR")
    if legacy_dir:
        run_dir = ensure_dir(Path(legacy_dir))
    else:
        run_dir = ensure_dir(Path("runs") / str(run_id))
    ensure_dir(run_dir / "checkpoints")

    data_cfg = cfg.get("data", {})
    fed_dir = str(data_cfg.get("federated_dir", "federated_data"))
    train_split = str(data_cfg.get("train_split", "train"))
    test_split = str(data_cfg.get("test_split", "test"))

    client_ids = list_client_ids(fed_dir)
    client_train_files: Dict[str, List[str]] = {}
    for cid in client_ids:
        files = list_npz_files(fed_dir, train_split, client_id=str(cid))
        if files:
            client_train_files[str(cid)] = files
    client_ids = sorted(client_train_files.keys())
    if int(args.max_clients) > 0:
        client_ids = client_ids[: int(args.max_clients)]
    if not client_ids:
        raise SystemExit("No client train files found under data.federated_dir.")

    sample_file = next(iter(client_train_files.values()))[0]
    ds_sample = WindowedNPZDataset(
        [sample_file],
        use_clin="true",
        cache_in_memory=False,
        max_cache_files=int(cfg.get("train", {}).get("max_cache_files", 32)),
        cache_dtype=str(cfg.get("train", {}).get("cache_dtype", "float32")),
    )
    model_cfg = IOHModelConfig(
        in_channels=int(getattr(ds_sample, "wave_channels", 4) or 4),
        base_channels=int(cfg.get("model", {}).get("base_channels", 32)),
        dropout=float(cfg.get("model", {}).get("dropout", 0.1)),
        use_gru=bool(cfg.get("model", {}).get("use_gru", True)),
        gru_hidden=int(cfg.get("model", {}).get("gru_hidden", 64)),
        clin_dim=int(getattr(ds_sample, "clin_dim", 0) or 0),
    )

    global_model = IOHNet(model_cfg).to(device)
    global_state = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

    train_cfg = cfg.get("train", {})
    local_cfg = LocalTrainConfig(
        epochs=int(train_cfg.get("local_epochs", 1)),
        batch_size=int(train_cfg.get("batch_size", 64)),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        num_workers=int(train_cfg.get("num_workers", 0)),
        cache_in_memory=bool(train_cfg.get("cache_in_memory", False)),
        max_cache_files=int(train_cfg.get("max_cache_files", 32)),
        cache_dtype=str(train_cfg.get("cache_dtype", "float32")),
    )

    distill_cfg = cfg.get("distill", {})
    distill_epochs = int(distill_cfg.get("epochs", 1))
    distill_lr = float(distill_cfg.get("lr", 1e-3))
    distill_batch = int(distill_cfg.get("batch_size", 128))
    distill_temp = float(distill_cfg.get("temperature", 1.0))
    distill_loss = str(distill_cfg.get("loss_type", "kl")).lower()
    distill_check = bool(distill_cfg.get("check_loss_decrease", False))
    distill_check_lr = float(distill_cfg.get("check_lr", 1e-3))
    distill_check_steps = int(distill_cfg.get("check_steps", 3))

    fedbe_cfg = cfg.get("fedbe", {})
    teacher_cfg = fedbe_cfg.get("teacher", {})
    swa_cfg = fedbe_cfg.get("swa", {})
    update_mode = str(fedbe_cfg.get("update", "distill")).lower()
    init_mode = str(fedbe_cfg.get("init", "fedavg")).lower()
    weight_mode = str(fedbe_cfg.get("client_weight", "n_examples")).lower()
    teacher_type = str(teacher_cfg.get("type", "swag")).lower()
    add_fedavg_default = True if teacher_type == "swag" else False
    add_fedavg_teacher = bool(teacher_cfg.get("add_fedavg", add_fedavg_default))
    include_clients = bool(teacher_cfg.get("include_clients", False))
    warmup_rounds = int(teacher_cfg.get("warmup_rounds", -1))
    num_sample_teacher = int(teacher_cfg.get("num_samples", 10))
    sample_mode = str(teacher_cfg.get("sample_mode", "gaussian")).lower()
    dirichlet_alpha = float(teacher_cfg.get("dirichlet_alpha", 1.0))
    var_scale = float(teacher_cfg.get("var_scale", 0.1))
    swag_stepsize = float(teacher_cfg.get("swag_stepsize", 1.0))
    concentrate_num = int(teacher_cfg.get("concentrate_num", 1))
    swa_enabled = bool(swa_cfg.get("enabled", False))
    swa_start = int(swa_cfg.get("start_step", 500))
    swa_freq = int(swa_cfg.get("freq", 25))

    eval_cfg = cfg.get("eval", {})
    eval_threshold = float(eval_cfg.get("threshold", 0.5))
    teacher_every_round = bool(eval_cfg.get("teacher_every_round", False))
    teacher_max_batches = int(eval_cfg.get("teacher_max_batches", 0))
    teacher_temp = float(eval_cfg.get("teacher_temperature", distill_temp))
    selection_mode = str(eval_cfg.get("model_selection", "last")).lower()
    selection_metric = str(eval_cfg.get("selection_metric", "ece")).lower()
    per_client_every_round = bool(eval_cfg.get("per_client_every_round", False))

    run_meta: Dict[str, Any] = {
        "started_utc": now_utc_iso(),
        "git_hash": get_git_hash(PROJECT_ROOT),
        "device": str(device),
        "algo": str(algo_name),
        "run_id": str(run_id),
        "run_dir": str(run_dir),
        "legacy_run_dir": str(legacy_dir) if legacy_dir else None,
        "data": data_cfg,
        "seed": int(cfg.get("seed", 42)),
        "rounds": int(train_cfg.get("rounds", 1)),
        "local_cfg": asdict(local_cfg),
        "model": asdict(model_cfg),
        "distill": distill_cfg,
        "fedbe": fedbe_cfg,
        "clients": client_ids,
        "eval": eval_cfg,
        "compute_steps_def": "client_steps=sum(local_epochs * n_batches_per_client), distill_steps=sum(batches_seen_per_epoch)",
    }
    meta_path = run_dir / "meta.json"
    write_json(meta_path, run_meta)
    global LAST_META_PATH
    LAST_META_PATH = meta_path

    config_used_path = run_dir / "config_used.yaml"
    try:
        import yaml  # type: ignore

        config_used_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    except Exception:
        config_used_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    overlap_report_path = Path(str(data_cfg.get("server_unlabeled_dir", "server_unlabeled"))) / "overlap_report.json"
    if overlap_report_path.exists():
        try:
            target = run_dir / "overlap_report.json"
            target.write_text(overlap_report_path.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            run_meta["overlap_report_path"] = str(overlap_report_path)
            write_json(run_dir / "meta.json", run_meta)

    test_every_round = eval_cfg.get("test_every_round", True)
    if args.test_every_round is not None:
        test_every_round = bool(args.test_every_round)

    unlabeled_files = _list_unlabeled_files(str(data_cfg.get("server_unlabeled_dir", "server_unlabeled")))
    if not unlabeled_files:
        raise SystemExit("No server_unlabeled .npz files found. Run make_server_unlabeled.py first.")

    ds_unl = UnlabeledNPZDataset(
        unlabeled_files,
        use_clin="auto",
        cache_in_memory=bool(train_cfg.get("cache_in_memory", False)),
        max_cache_files=int(train_cfg.get("max_cache_files", 32)),
        cache_dtype=str(train_cfg.get("cache_dtype", "float32")),
    )
    dl_unl = DataLoader(
        ds_unl,
        batch_size=int(distill_batch),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(train_cfg.get("num_workers", 0)) > 0),
    )
    dl_unl_check = None
    if distill_check:
        dl_unl_check = DataLoader(
            ds_unl,
            batch_size=int(distill_batch),
            shuffle=False,
            num_workers=int(train_cfg.get("num_workers", 0)),
            pin_memory=(device.type == "cuda"),
            persistent_workers=(int(train_cfg.get("num_workers", 0)) > 0),
        )

    test_files = list_npz_files(fed_dir, test_split)
    dl_test = None
    if test_files:
        ds_test = WindowedNPZDataset(
            test_files,
            use_clin="true",
            cache_in_memory=bool(train_cfg.get("cache_in_memory", False)),
            max_cache_files=int(train_cfg.get("max_cache_files", 32)),
            cache_dtype=str(train_cfg.get("cache_dtype", "float32")),
        )
        dl_test = DataLoader(
            ds_test,
            batch_size=int(train_cfg.get("batch_size", 64)),
            shuffle=False,
            num_workers=int(train_cfg.get("num_workers", 0)),
            pin_memory=(device.type == "cuda"),
            persistent_workers=(int(train_cfg.get("num_workers", 0)) > 0),
        )

    history: List[Dict[str, Any]] = []
    compute_log: List[Dict[str, Any]] = []
    per_client_rounds: List[Dict[str, Any]] = []
    best = {"metric": None, "round": 0}
    model_trace_path = run_dir / "model_trace.jsonl"
    metrics_csv_path = run_dir / "metrics_round.csv"
    if not metrics_csv_path.exists():
        metrics_csv_path.write_text(
            "round,algo,auroc,auprc,ece,nll,brier,threshold,acc,f1,precision,recall\n",
            encoding="utf-8",
        )
    teacher_metrics_csv = run_dir / "teacher_metrics_round.csv"
    if teacher_every_round and not teacher_metrics_csv.exists():
        teacher_metrics_csv.write_text(
            "round,algo,auroc,auprc,ece,nll,brier,threshold,acc,f1,precision,recall\n",
            encoding="utf-8",
        )

    rounds = int(train_cfg.get("rounds", 1))
    prev_after_hash: str | None = None
    for rnd in range(1, rounds + 1):
        print(f"\n[round {rnd}/{rounds}]")

        global_in_hash = hash_state_dict(global_state)
        if prev_after_hash is not None and global_in_hash != prev_after_hash:
            raise RuntimeError("global_state_in_hash does not match previous global_state_after_distill_hash")
        with model_trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "round_start", "round": rnd, "global_state_in_hash": global_in_hash}) + "\n")

        client_states: List[Dict[str, torch.Tensor]] = []
        client_weights: List[int] = []
        per_client: List[Dict[str, Any]] = []
        match_count = 0
        trace_clients: List[Dict[str, Any]] = []
        iterator = client_ids
        for cid in iterator:
            train_files = client_train_files[str(cid)]
            if int(args.max_client_files) > 0:
                train_files = train_files[: int(args.max_client_files)]
            client_init_hash = hash_state_dict(global_state)
            if client_init_hash != global_in_hash:
                raise RuntimeError("client_init_hash != global_state_in_hash")
            match_count += 1
            if len(trace_clients) < 5:
                trace_clients.append({"client_id": str(cid), "client_init_hash": client_init_hash})
            updated, n_examples, metrics = train_one_client(
                client_id=str(cid),
                train_files=train_files,
                model_cfg=model_cfg,
                global_state=global_state,
                cfg=local_cfg,
                device=device,
                show_progress=bool(args.client_progress_bar),
            )
            if n_examples > 0:
                client_states.append(updated)
                client_weights.append(int(n_examples))
            per_client.append(metrics)
        if not client_states:
            raise SystemExit("No client updates produced; cannot distill.")

        if weight_mode == "uniform":
            weights = [1.0] * len(client_states)
        else:
            weights = client_weights if client_weights else [1.0] * len(client_states)
        fedavg_state_dict = fedavg_state(client_states, weights=weights)

        use_fedavg_init = init_mode == "fedavg"
        student_init_state = fedavg_state_dict if use_fedavg_init else global_state

        teacher_states: List[Dict[str, torch.Tensor]] = []
        if add_fedavg_teacher:
            teacher_states.append(fedavg_state_dict)

        if teacher_type == "swag" and (warmup_rounds < 0 or rnd > warmup_rounds):
            rng = np.random.default_rng(int(cfg.get("seed", 42)) + int(rnd))
            sampled = sample_teacher_states(
                base_state=global_state,
                teacher_states=client_states,
                num_samples=int(num_sample_teacher),
                mode=str(sample_mode),
                alpha=float(dirichlet_alpha),
                var_scale=float(var_scale),
                swag_stepsize=float(swag_stepsize),
                concentrate_num=int(concentrate_num),
                rng=rng,
            )
            teacher_states.extend(sampled)
        elif teacher_type == "swag":
            print("[fedbe] warmup: skip SWAG teacher sampling")

        if include_clients:
            teacher_states.extend(client_states)

        if not teacher_states:
            teacher_states = list(client_states)

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
                        "teacher_count": int(len(teacher_states)),
                        "teacher_type": str(teacher_type),
                        "teacher_sample_mode": str(sample_mode),
                        "teacher_num_sample": int(num_sample_teacher),
                        "teacher_add_fedavg": bool(add_fedavg_teacher),
                        "teacher_include_clients": bool(include_clients),
                    }
                )
                + "\n"
            )

        # Distillation on server_unlabeled
        student = IOHNet(model_cfg).to(device)
        student.load_state_dict(student_init_state, strict=True)
        student.train()
        student_before_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        student_before_hash = hash_state_dict(student_before_state)
        opt = torch.optim.AdamW(student.parameters(), lr=float(distill_lr), weight_decay=float(train_cfg.get("weight_decay", 1e-4)))
        scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
        autocast_device = "cuda" if device.type == "cuda" else "cpu"

        total_loss = 0.0
        total_n = 0
        epoch_losses: List[float] = []
        distill_steps = 0
        swa_state: Dict[str, torch.Tensor] | None = None
        swa_n = 0
        if distill_check and dl_unl_check is not None:
            try:
                batch = next(iter(dl_unl_check))
            except StopIteration:
                batch = None
            if batch is not None:
                if isinstance(batch, (tuple, list)):
                    x_check = tuple(t.to(device, non_blocking=True) for t in batch)
                else:
                    x_check = batch.to(device, non_blocking=True)
                teacher_prob = _ensemble_teacher_prob(
                    teacher_states=teacher_states,
                    model_cfg=model_cfg,
                    x=x_check,
                    device=device,
                    temperature=float(distill_temp),
                )
                with torch.no_grad():
                    student_prob = torch.sigmoid(student(x_check).view(-1) / float(max(distill_temp, 1.0)))
                    teacher_student_mae = float(torch.mean(torch.abs(student_prob - teacher_prob)).item())
                with model_trace_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "event": "distill_start",
                                "round": rnd,
                                "student_before_hash": student_before_hash,
                                "teacher_student_mae": float(teacher_student_mae),
                            }
                        )
                        + "\n"
                    )
                # Save/restore model so the check doesn't affect training
                saved_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
                check_opt = torch.optim.SGD(student.parameters(), lr=float(distill_check_lr))
                logits = student(x_check).view(-1)
                loss_before = _distill_loss(
                    logits=logits,
                    teacher_prob=teacher_prob,
                    loss_type=str(distill_loss),
                    temperature=float(distill_temp),
                )
                for _ in range(max(1, int(distill_check_steps))):
                    check_opt.zero_grad(set_to_none=True)
                    loss = _distill_loss(
                        logits=student(x_check).view(-1),
                        teacher_prob=teacher_prob,
                        loss_type=str(distill_loss),
                        temperature=float(distill_temp),
                    )
                    loss.backward()
                    check_opt.step()
                loss_after = _distill_loss(
                    logits=student(x_check).view(-1),
                    teacher_prob=teacher_prob,
                    loss_type=str(distill_loss),
                    temperature=float(distill_temp),
                )
                student.load_state_dict(saved_state, strict=True)
                print(
                    f"  distill check: loss_before={float(loss_before):.6f} loss_after={float(loss_after):.6f} steps={distill_check_steps}"
                )
        for ep in range(int(distill_epochs)):
            iterator = tqdm(
                dl_unl,
                total=len(dl_unl),
                desc=f"distill ep {ep + 1}/{int(distill_epochs)}",
                leave=False,
                disable=(bool(args.no_progress_bar)),
            )
            ep_loss = 0.0
            ep_n = 0
            batches_seen = 0
            for batch in iterator:
                if int(args.max_unlabeled_batches) > 0 and batches_seen >= int(args.max_unlabeled_batches):
                    break
                if isinstance(batch, (tuple, list)):
                    x = tuple(t.to(device, non_blocking=True) for t in batch)
                else:
                    x = batch.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=autocast_device, enabled=(device.type == "cuda")):
                    teacher_prob = _ensemble_teacher_prob(
                        teacher_states=teacher_states,
                        model_cfg=model_cfg,
                        x=x,
                        device=device,
                        temperature=float(distill_temp),
                    )
                    logits = student(x).view(-1)
                    if teacher_prob.shape != logits.shape:
                        raise RuntimeError(f"teacher_prob shape {teacher_prob.shape} != student logits shape {logits.shape}")
                    loss = _distill_loss(
                        logits=logits,
                        teacher_prob=teacher_prob,
                        loss_type=str(distill_loss),
                        temperature=float(distill_temp),
                    )
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                distill_steps += 1
                if swa_enabled and distill_steps >= int(swa_start) and (distill_steps - int(swa_start)) % max(int(swa_freq), 1) == 0:
                    swa_state, swa_n = _swa_update(swa_state=swa_state, model_state=student.state_dict(), swa_n=swa_n)
                total_loss += float(loss.item()) * int(teacher_prob.shape[0])
                total_n += int(teacher_prob.shape[0])
                ep_loss += float(loss.item()) * int(teacher_prob.shape[0])
                ep_n += int(teacher_prob.shape[0])
                batches_seen += 1
            if ep_n > 0:
                avg_ep_loss = float(ep_loss / ep_n)
                epoch_losses.append(avg_ep_loss)
                print(f"  distill ep {ep + 1}: avg_loss={avg_ep_loss:.6f} (batches={batches_seen})")

        raw_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
        student_update_l2 = l2_diff_state_dict(raw_state, student_before_state)
        distill_global_l2 = l2_diff_state_dict(raw_state, global_state)
        student_after_hash = hash_state_dict(raw_state)

        final_state = raw_state
        used_swa = False
        if update_mode == "fedavg":
            final_state = fedavg_state_dict
        elif swa_enabled and swa_state is not None:
            final_state = swa_state
            used_swa = True

        global_update_l2 = l2_diff_state_dict(final_state, global_state)
        global_state = final_state
        global_after_hash = hash_state_dict(global_state)
        if student_after_hash == student_before_hash:
            print("  [warn] distill update produced no parameter change (hash unchanged)")
        prev_after_hash = global_after_hash
        with model_trace_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "event": "round_end",
                        "round": rnd,
                        "global_state_after_distill_hash": global_after_hash,
                        "global_update_l2": float(global_update_l2),
                        "distill_update_l2": float(distill_global_l2),
                        "student_update_l2": float(student_update_l2),
                        "student_before_hash": student_before_hash,
                        "student_after_hash": student_after_hash,
                        "global_state_in_hash": global_in_hash,
                        "hash_changed": bool(global_after_hash != global_in_hash),
                        "update_mode": str(update_mode),
                        "used_swa": bool(used_swa),
                    }
                )
                + "\n"
            )

        row: Dict[str, Any] = {
            "round": int(rnd),
            "distill_loss": float(total_loss / max(total_n, 1)),
            "distill_epoch_losses": [float(v) for v in epoch_losses],
            "distill_update_l2": float(student_update_l2),
            "global_update_l2": float(global_update_l2),
            "n_clients": int(len(client_states)),
            "n_unlabeled": int(len(ds_unl)),
            "n_teachers": int(len(teacher_states)),
            "teacher_type": str(teacher_type),
            "teacher_sample_mode": str(sample_mode),
            "teacher_num_sample": int(num_sample_teacher),
            "teacher_add_fedavg": bool(add_fedavg_teacher),
            "teacher_include_clients": bool(include_clients),
            "update_mode": str(update_mode),
            "init_mode": str(init_mode),
            "weight_mode": str(weight_mode),
            "used_swa": bool(used_swa),
        }

        client_steps = 0
        for m in per_client:
            client_steps += int(m.get("n_steps", 0))
        compute_row = {
            "round": int(rnd),
            "client_steps": int(client_steps),
            "distill_steps": int(distill_steps),
            "total_steps": int(client_steps + distill_steps),
        }
        compute_log.append(compute_row)
        write_json(run_dir / "compute_log.json", compute_log)

        if test_every_round and dl_test is not None:
            student.load_state_dict(global_state, strict=True)
            student.eval()
            logits, y_true = _predict_logits(student, dl_test, device=device)
            prob = sigmoid_np(logits)
            metrics_pre = compute_binary_metrics(y_true, prob, n_bins=15)
            thr = float(eval_threshold)
            metrics = calc_comprehensive_metrics(y_true, prob, threshold=float(thr))
            row.update({f"test_{k}": float(v) for k, v in metrics.items()})
            metrics_csv_path.write_text(
                metrics_csv_path.read_text(encoding="utf-8")
                + f"{rnd},{algo_name},{metrics.get('auroc', '')},{metrics.get('auprc', '')},{metrics.get('ece', '')},{metrics.get('nll', '')},{metrics.get('brier', '')},{metrics.get('threshold', '')},{metrics.get('accuracy', '')},{metrics.get('f1', '')},{metrics.get('ppv', '')},{metrics.get('sensitivity', '')}\n",
                encoding="utf-8",
            )
            write_json(
                run_dir / f"round_{rnd:03d}_test.json",
                {
                    "round": int(rnd),
                    "n": int(metrics_pre.n),
                    "n_pos": int(metrics_pre.n_pos),
                    "n_neg": int(metrics_pre.n_neg),
                    "metrics_pre": asdict(metrics_pre),
                    "threshold": float(thr),
                    "threshold_method": "fixed",
                    "metrics_threshold": metrics,
                    "confusion_pre": confusion_at_threshold(y_true, prob, thr=float(thr)),
                },
            )
            if per_client_every_round:
                round_rows: List[Dict[str, Any]] = []
                for cid in client_ids:
                    files = list_npz_files(fed_dir, test_split, client_id=str(cid))
                    if not files:
                        continue
                    ds_c = WindowedNPZDataset(
                        files,
                        use_clin="true",
                        cache_in_memory=bool(train_cfg.get("cache_in_memory", False)),
                        max_cache_files=int(train_cfg.get("max_cache_files", 32)),
                        cache_dtype=str(train_cfg.get("cache_dtype", "float32")),
                    )
                    dl_c = DataLoader(
                        ds_c,
                        batch_size=int(train_cfg.get("batch_size", 64)),
                        shuffle=False,
                        num_workers=int(train_cfg.get("num_workers", 0)),
                        pin_memory=(device.type == "cuda"),
                        persistent_workers=(int(train_cfg.get("num_workers", 0)) > 0),
                    )
                    logits_c, y_c = _predict_logits(student, dl_c, device=device)
                    prob_c = sigmoid_np(logits_c)
                    m_c = compute_binary_metrics(y_c, prob_c, n_bins=15)
                    thr_c = float(thr)
                    m_thr = calc_comprehensive_metrics(y_c, prob_c, threshold=float(thr_c))
                    row_c = {
                        "round": int(rnd),
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
                        "threshold": float(thr_c),
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
                    import pandas as pd

                    pd.DataFrame(round_rows).to_csv(run_dir / f"round_{rnd:03d}_test_per_client.csv", index=False)
                    pd.DataFrame(per_client_rounds).to_csv(run_dir / "round_client_metrics.csv", index=False)
                    write_json(run_dir / "round_client_metrics.json", per_client_rounds)

            if selection_mode == "best":
                metric_val = None
                if hasattr(metrics_pre, selection_metric):
                    metric_val = getattr(metrics_pre, selection_metric)
                if metric_val is not None:
                    better = False
                    if best["metric"] is None:
                        better = True
                    else:
                        if selection_metric in {"nll", "brier", "ece"}:
                            better = float(metric_val) < float(best["metric"])
                        else:
                            better = float(metric_val) > float(best["metric"])
                    if better:
                        best = {"metric": float(metric_val), "round": int(rnd)}
                        torch.save(student.state_dict(), run_dir / "checkpoints" / "model_best.pt")

        if teacher_every_round and dl_test is not None:
            y_all = []
            prob_all = []
            batches_seen = 0
            for batch in dl_test:
                if teacher_max_batches > 0 and batches_seen >= teacher_max_batches:
                    break
                if isinstance(batch, (tuple, list)):
                    x, y = batch
                    x = tuple(t.to(device, non_blocking=True) for t in x)
                else:
                    x, y = batch, None
                    x = x.to(device, non_blocking=True)
                if y is None:
                    continue
                y_all.append(y.detach().cpu().view(-1).numpy())
                prob = _ensemble_teacher_prob(
                    teacher_states=teacher_states,
                    model_cfg=model_cfg,
                    x=x,
                    device=device,
                    temperature=float(teacher_temp),
                )
                prob_all.append(prob.detach().cpu().view(-1).numpy())
                batches_seen += 1
            if y_all and prob_all:
                y_true = np.concatenate(y_all, axis=0)
                prob = np.concatenate(prob_all, axis=0)
                m_pre = compute_binary_metrics(y_true, prob, n_bins=15)
                thr = float(eval_threshold)
                m_thr = calc_comprehensive_metrics(y_true, prob, threshold=float(thr))
                teacher_metrics_csv.write_text(
                    teacher_metrics_csv.read_text(encoding="utf-8")
                    + f"{rnd},teacher,{m_pre.auroc},{m_pre.auprc},{m_pre.ece},{m_pre.nll},{m_pre.brier},{thr},{m_thr.get('accuracy','')},{m_thr.get('f1','')},{m_thr.get('ppv','')},{m_thr.get('sensitivity','')}\n",
                    encoding="utf-8",
                )
                write_json(
                    run_dir / f"round_{rnd:03d}_teacher_test.json",
                    {
                        "round": int(rnd),
                        "n": int(m_pre.n),
                        "n_pos": int(m_pre.n_pos),
                        "n_neg": int(m_pre.n_neg),
                        "metrics_pre": asdict(m_pre),
                        "threshold": float(thr),
                        "threshold_method": "fixed",
                        "metrics_threshold": m_thr,
                        "confusion_pre": confusion_at_threshold(y_true, prob, thr=float(thr)),
                    },
                )

        history.append(row)
        if per_client:
            write_json(run_dir / f"round_{rnd:03d}_clients.json", per_client)
        write_json(run_dir / "history.json", history)

    # Save last model
    final_model = IOHNet(model_cfg)
    final_model.load_state_dict(global_state, strict=True)
    torch.save(final_model.state_dict(), run_dir / "checkpoints" / "model_last.pt")

    # Final test report
    if dl_test is not None:
        final_model = final_model.to(device)
        logits, y_true = _predict_logits(final_model, dl_test, device=device)
        prob = sigmoid_np(logits)
        metrics_pre = compute_binary_metrics(y_true, prob, n_bins=15)
        thr = float(eval_threshold)
        metrics = calc_comprehensive_metrics(y_true, prob, threshold=float(thr))
        write_json(
            run_dir / "test_report.json",
            {
                "n": int(len(y_true)),
                "metrics_pre": asdict(metrics_pre),
                "threshold": float(thr),
                "threshold_method": "fixed",
                "metrics_threshold": metrics,
                "confusion_pre": confusion_at_threshold(y_true, prob, thr=float(thr)),
            },
        )
        # Per-client TEST report (final model)
        per_client_reports: Dict[str, Any] = {}
        per_client_rows: List[Dict[str, Any]] = []
        for cid in client_ids:
            files = list_npz_files(fed_dir, test_split, client_id=str(cid))
            if not files:
                per_client_reports[str(cid)] = {"client_id": str(cid), "status": "no_files", "n": 0}
                per_client_rows.append({"client_id": str(cid), "status": "no_files", "n": 0})
                continue
            ds = WindowedNPZDataset(
                files,
                use_clin="true",
                cache_in_memory=bool(train_cfg.get("cache_in_memory", False)),
                max_cache_files=int(train_cfg.get("max_cache_files", 32)),
                cache_dtype=str(train_cfg.get("cache_dtype", "float32")),
            )
            dl = DataLoader(
                ds,
                batch_size=int(train_cfg.get("batch_size", 64)),
                shuffle=False,
                num_workers=int(train_cfg.get("num_workers", 0)),
                pin_memory=(device.type == "cuda"),
                persistent_workers=(int(train_cfg.get("num_workers", 0)) > 0),
            )
            logits_c, y_c = _predict_logits(final_model, dl, device=device)
            prob_c = sigmoid_np(logits_c)
            m_c = compute_binary_metrics(y_c, prob_c, n_bins=15)
            thr_c = float(thr)
            m_thr = calc_comprehensive_metrics(y_c, prob_c, threshold=float(thr_c))
            row_c = {
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
                "threshold": float(thr_c),
                "threshold_method": "fixed",
                "accuracy": float(m_thr.get("accuracy", float("nan"))),
                "f1": float(m_thr.get("f1", float("nan"))),
                "sensitivity": float(m_thr.get("sensitivity", float("nan"))),
                "specificity": float(m_thr.get("specificity", float("nan"))),
                "ppv": float(m_thr.get("ppv", float("nan"))),
                "npv": float(m_thr.get("npv", float("nan"))),
            }
            per_client_reports[str(cid)] = row_c
            per_client_rows.append(row_c)
        write_json(run_dir / "test_report_per_client.json", per_client_reports)
        try:
            import pandas as pd

            pd.DataFrame(per_client_rows).to_csv(run_dir / "test_report_per_client.csv", index=False)
        except Exception:
            pass

    write_json(run_dir / "history.json", history)
    print("[done]")


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
