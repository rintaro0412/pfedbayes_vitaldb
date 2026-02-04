from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

# Ensure project root in sys.path when executed as `python bayes_federated/pfedbayes_server.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bayes_federated.bayes_layers import BayesParams
from bayes_federated.eval import evaluate_split
from bayes_federated.models import BFLModel, build_bfl_model_from_point_checkpoint
from bayes_federated.pfedbayes_client import PFBayesClientConfig, train_client_pfedbayes
from bayes_federated.pfedbayes_utils import aggregate_bayes_dict
from common.checkpoint import capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint
from common.dataset import list_client_ids, list_npz_files, list_npz_files_by_client, scan_label_stats
from common.metrics import derived_from_confusion
from common.experiment import make_run_dir, save_env_snapshot, seed_everything
from common.io import read_json, write_json
from common.ioh_model import normalize_model_cfg


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


def _flatten_bayes_mu(params: Dict[str, BayesParams] | BayesParams) -> np.ndarray:
    if isinstance(params, BayesParams):
        parts = [
            params.weight_mu.detach().cpu().view(-1).numpy(),
            params.bias_mu.detach().cpu().view(-1).numpy(),
        ]
    else:
        parts = []
        for _, p in sorted(params.items(), key=lambda kv: kv[0]):
            parts.append(p.weight_mu.detach().cpu().view(-1).numpy())
            parts.append(p.bias_mu.detach().cpu().view(-1).numpy())
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


def _save_history(path: Path, rows: List[Dict[str, Any]]) -> None:
    import pandas as pd

    pd.DataFrame(rows).to_csv(path, index=False)


def _client_list(data_dir: str, train_split: str, existing: Path | None = None) -> List[str]:
    if existing is not None and existing.exists():
        return read_json(existing)
    out = []
    for cid in list_client_ids(data_dir):
        if list_npz_files(data_dir, train_split, client_id=str(cid)):
            out.append(str(cid))
    return sorted(out)


def _dataset_counts(data_dir: str, splits: Dict[str, str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for label, split in splits.items():
        files = list_npz_files(data_dir, split)
        pos, total = scan_label_stats(files) if files else (0, 0)
        out[str(label)] = {"split": str(split), "n": int(total), "n_pos": int(pos), "n_neg": int(total - pos)}
    return out


def _client_counts(data_dir: str, split: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cid in list_client_ids(data_dir):
        files = list_npz_files(data_dir, split, client_id=str(cid))
        if not files:
            continue
        pos, total = scan_label_stats(files)
        out[str(cid)] = {"n": int(total), "n_pos": int(pos), "n_neg": int(total - pos)}
    return out


def _infer_pos_weight(train_files: list[str]) -> float:
    n_pos, n_total = scan_label_stats(train_files)
    n_neg = int(n_total - n_pos)
    if n_pos <= 0:
        return 1.0
    return float(max(n_neg, 1) / max(n_pos, 1))


def _sample_clients(clients: List[str], *, rnd: int, fraction: float, sample_size: int, seed: int) -> List[str]:
    if not clients:
        return []
    if sample_size > 0:
        n_select = min(int(sample_size), len(clients))
    else:
        frac = float(fraction)
        if frac <= 0:
            frac = 1.0
        n_select = max(1, int(round(len(clients) * frac)))
    rng = np.random.default_rng(int(seed) + int(rnd))
    if n_select >= len(clients):
        return list(clients)
    return sorted(rng.choice(clients, size=n_select, replace=False).tolist())


def _format_seconds(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main() -> None:
    ap = argparse.ArgumentParser(description="pFedBayes server (personalized VI, full algorithm)")
    ap.add_argument("--config", default="configs/pfedbayes.yaml")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--client-progress-bar", action="store_true", help="Show per-client batch progress bar.")
    ap.add_argument("--no-progress-bar", action="store_true", help="Disable progress bars.")
    ap.add_argument("--save-test-pred-npz", default=None, help="Optional .npz to save per-sample test predictions/uncertainty.")
    ap.add_argument("--log-client-sim", action="store_true", help="Save per-round client cosine similarity matrix.")
    ap.add_argument("--test-every-round", action=argparse.BooleanOptionalAction, default=None)
    args = ap.parse_args()

    cfg = _load_config(args.config)
    resume = bool(args.resume) or bool(cfg.get("run", {}).get("resume", False))
    run_name = args.run_name or cfg.get("run", {}).get("run_name")
    run_dir = make_run_dir(cfg["run"]["out_dir"], run_name, resume=resume)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    save_env_snapshot(run_dir, cfg)
    seed_everything(int(cfg.get("seed", 42)), deterministic=True)

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    # Prepare base model and initial global params
    resume_fallback_cfg = None
    if resume:
        resume_ckpt = run_dir / "checkpoints" / "model_last.pt"
        if resume_ckpt.exists():
            ckpt = load_checkpoint(resume_ckpt, map_location="cpu")
            resume_fallback_cfg = normalize_model_cfg(ckpt.get("model_cfg", {}))
    backbone_cfg = cfg.get("backbone", {})
    full_bayes = bool(cfg.get("bayes", {}).get("full_bayes", False))
    param_type = str(cfg.get("bayes", {}).get("param_type", "logvar"))
    mu_init = str(cfg.get("bayes", {}).get("mu_init", "zeros"))
    init_rho = cfg.get("bayes", {}).get("init_rho", None)
    model, init_prior, used_point = build_bfl_model_from_point_checkpoint(
        backbone_cfg.get("checkpoint"),
        prior_sigma=float(cfg["bayes"]["prior_sigma"]),
        logvar_min=float(cfg["bayes"]["logvar_min"]),
        logvar_max=float(cfg["bayes"]["logvar_max"]),
        full_bayes=bool(full_bayes),
        fallback_cfg=resume_fallback_cfg,
        param_type=param_type,
        mu_init=mu_init,
        init_rho=(float(init_rho) if init_rho is not None else None),
    )
    base_state = model.state_dict()

    data_dir = cfg["data"]["data_dir"]
    train_split = str(cfg["data"]["train_split"])
    val_split = str(cfg["data"].get("val_split", "val"))
    test_split = str(cfg["data"]["test_split"])
    summary_path = Path(data_dir) / "summary.json"
    dataset_summary = None
    if summary_path.exists():
        try:
            dataset_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as e:
            dataset_summary = {"error": f"failed to read summary.json: {e}"}

    # Client list
    clients_path = run_dir / "clients.json"
    clients = _client_list(str(data_dir), train_split, existing=clients_path if resume else None)
    if not clients:
        raise SystemExit("No clients found in index.")
    if not resume:
        write_json(clients_path, clients)

    # Dataset counts
    val_files = list_npz_files(str(data_dir), val_split)
    val_fallback = False
    if not val_files:
        val_split = train_split
        val_files = list_npz_files(str(data_dir), val_split)
        val_fallback = True
    splits = {"train": train_split, "val": val_split, "test": test_split}
    counts = _dataset_counts(str(data_dir), splits)
    write_json(run_dir / "data_counts.json", counts)
    write_json(run_dir / "split_info.json", {"splits": splits, "val_fallback_to_train": bool(val_fallback)})
    client_counts = _client_counts(str(data_dir), train_split)
    write_json(run_dir / "client_counts.json", client_counts)
    if dataset_summary:
        keys = ["client_scheme", "merge_strategy", "opname_threshold", "min_client_cases", "clients"]
        write_json(run_dir / "dataset_summary.json", {k: dataset_summary.get(k) for k in keys if k in dataset_summary})

    # Resume state
    server_state_path = run_dir / "checkpoints" / "server_state.pt"
    start_round = 1
    global_params: Dict[str, BayesParams] = init_prior
    best = {"round": 0, "threshold": None}
    history: List[Dict[str, Any]] = []
    per_client_rounds: List[Dict[str, Any]] = []
    if resume and server_state_path.exists():
        state = load_checkpoint(server_state_path)
        start_round = int(state["round"]) + 1
        global_params = state["global_params"]
        best = state["best"]
        history = state.get("history", [])
        if "rng_state" in state:
            restore_rng_state(state["rng_state"])

    # Configs
    loss_cfg = cfg.get("loss", {})
    pos_weight_raw = loss_cfg.get("pos_weight", None)
    if isinstance(pos_weight_raw, str) and pos_weight_raw.strip().lower() in ("auto", "client"):
        pos_weight_val = None
    elif pos_weight_raw is None:
        pos_weight_val = None
    else:
        pos_weight_val = float(pos_weight_raw)
    pf_cfg = cfg.get("pfedbayes", {})
    base_train_cfg = PFBayesClientConfig(
        local_epochs=int(cfg["train"]["local_epochs"]),
        batch_size=int(cfg["train"]["batch_size"]),
        lr_q=float(cfg["train"]["lr_q"]),
        lr_w=float(cfg["train"]["lr_w"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        num_workers=int(cfg["train"]["num_workers"]),
        mc_train=int(cfg["train"]["mc_train"]),
        grad_clip=float(cfg["train"].get("grad_clip", 0.0)),
        grad_clip_w=float(cfg["train"].get("grad_clip_w", 0.0)),
        train_backbone=bool(cfg["backbone"].get("train_backbone", False)),
        seed=int(cfg.get("seed", 42)),
        loss_type=str(loss_cfg.get("type", "bce")),
        pos_weight=pos_weight_val,
        zeta=float(pf_cfg.get("zeta", 1.0)),
        max_steps=int(cfg["train"].get("max_steps", 0)),
        q_optim=str(cfg["train"].get("q_optim", "sgd")),
        w_optim=str(cfg["train"].get("w_optim", "sgd")),
        param_type=param_type,
    )
    auto_pos_weight = isinstance(pos_weight_raw, str) and pos_weight_raw.strip().lower() in ("auto", "client")
    eval_cfg = cfg.get("eval", {})
    eval_batch_size = int(eval_cfg.get("batch_size", 128))
    eval_num_workers = int(eval_cfg.get("num_workers", cfg["train"].get("num_workers", 0)))
    threshold_use_post = bool(eval_cfg.get("threshold_use_post", True))
    threshold_cfg = eval_cfg.get("threshold", {}) if isinstance(eval_cfg.get("threshold", {}), dict) else {}
    threshold_method = str(threshold_cfg.get("method", "fixed_threshold")).lower()
    test_every_round = bool(eval_cfg.get("test_every_round", False))
    if args.test_every_round is not None:
        test_every_round = bool(args.test_every_round)
    test_files = list_npz_files(str(data_dir), test_split)
    selection_enabled = str(eval_cfg.get("model_selection", "last")).lower() == "best"
    selection_source = str(eval_cfg.get("selection_source", "val")).lower()
    selection_metric = str(eval_cfg.get("selection_metric", "auroc")).lower()
    selection_use_post = bool(eval_cfg.get("selection_use_post", False))
    per_client_every_round = bool(eval_cfg.get("per_client_every_round", False))
    client_test_files = {}
    if per_client_every_round:
        client_test_files = list_npz_files_by_client(str(data_dir), test_split)

    rounds = int(cfg["train"]["rounds"])
    fixed_threshold = float(cfg.get("eval", {}).get("threshold", {}).get("fixed", 0.5))
    min_client_examples = int(cfg["clients"].get("min_examples", 1))
    sample_fraction = float(cfg["clients"].get("sample_fraction", 1.0))
    sample_size = int(cfg["clients"].get("sample_size", 0))
    server_beta = float(pf_cfg.get("server_beta", 1.0))
    weight_mode = str(pf_cfg.get("weight_mode", "uniform")).lower()
    logvar_min = float(cfg["bayes"]["logvar_min"]) if "logvar_min" in cfg.get("bayes", {}) and cfg["bayes"]["logvar_min"] is not None else None
    logvar_max = float(cfg["bayes"]["logvar_max"]) if "logvar_max" in cfg.get("bayes", {}) and cfg["bayes"]["logvar_max"] is not None else None
    show_progress = not bool(args.no_progress_bar)
    log_client_sim = bool(args.log_client_sim) or bool(cfg.get("run", {}).get("log_client_sim", False))

    best = {"round": 0, "metric": None}

    def _select_threshold_from_val(eval_model: BFLModel) -> float | None:
        if threshold_method != "youden_val":
            return None
        if not val_files:
            print("[WARN] youden_val requested but no val files found; using fixed threshold.")
            return float(fixed_threshold)
        rep_val = evaluate_split(
            model=eval_model,
            files=val_files,
            mc_eval=int(cfg["train"]["mc_eval"]),
            device=device,
            temperature=None,
            threshold=None,
            threshold_method="youden",
            recall_target=float(cfg["eval"]["threshold"].get("recall_target", 0.8)),
            fixed_threshold=float(fixed_threshold),
            threshold_use_post=bool(threshold_use_post),
            bootstrap_n=0,
            bootstrap_seed=int(cfg["eval"]["bootstrap"].get("seed", 42)),
            batch_size=eval_batch_size,
            num_workers=eval_num_workers,
        )
        return float(rep_val.get("threshold_selected", fixed_threshold))

    eval_elapsed_sum = 0.0
    eval_round_count = 0
    round_iter: Any = range(start_round, rounds + 1)
    if show_progress:
        round_iter = tqdm(
            round_iter,
            total=rounds,
            initial=max(start_round - 1, 0),
            desc="rounds",
            leave=True,
            position=0,
        )

    # Training rounds
    for rnd in round_iter:
        best_updated = False
        selected = _sample_clients(clients, rnd=rnd, fraction=sample_fraction, sample_size=sample_size, seed=int(cfg.get("seed", 42)))
        if not selected:
            raise SystemExit("No clients selected for round.")

        client_w_params: List[Dict[str, BayesParams]] = []
        client_weights: List[float] = []
        client_metrics: List[Dict[str, Any]] = []
        client_vecs: Dict[str, np.ndarray] = {}
        client_n: Dict[str, int] = {}
        train_loss_sum = 0.0
        train_loss_weight = 0

        client_iter = tqdm(
            selected,
            total=len(selected),
            desc=f"round {rnd}/{rounds} clients",
            leave=False,
            disable=(not show_progress),
            position=1 if show_progress else 0,
        )
        for cid in client_iter:
            client_dir = run_dir / "clients" / f"round_{rnd:03d}"
            client_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = client_dir / f"client_{cid}.pt"

            local_model = BFLModel(
                model.cfg,
                prior_sigma=float(cfg["bayes"]["prior_sigma"]),
                logvar_min=float(cfg["bayes"]["logvar_min"]),
                logvar_max=float(cfg["bayes"]["logvar_max"]),
                full_bayes=bool(full_bayes),
                param_type=param_type,
                mu_init=mu_init,
                init_rho=(float(init_rho) if init_rho is not None else None),
            )
            local_model.load_state_dict(base_state, strict=False)

            train_files = list_npz_files(str(data_dir), train_split, client_id=str(cid))
            if auto_pos_weight:
                client_pos_weight = _infer_pos_weight(train_files)
                train_cfg = PFBayesClientConfig(
                    **{**asdict(base_train_cfg), "pos_weight": float(client_pos_weight)}
                )
            else:
                train_cfg = base_train_cfg
            q_params, w_params, result, meta = train_client_pfedbayes(
                client_id=cid,
                train_files=train_files,
                model=local_model,
                global_params=global_params,
                cfg=train_cfg,
                device=device,
                logvar_min=logvar_min,
                logvar_max=logvar_max,
                show_progress=bool(args.client_progress_bar) and show_progress,
                resume_path=str(ckpt_path) if resume else None,
                save_path=str(ckpt_path),
            )

            client_metrics.append({**asdict(result), **meta})
            if result.n_examples >= min_client_examples:
                client_w_params.append(w_params)
                if weight_mode == "n_examples":
                    client_weights.append(float(result.n_examples))
                else:
                    client_weights.append(1.0)
                if log_client_sim and meta.get("status") == "ok":
                    client_vecs[str(cid)] = _flatten_bayes_mu(w_params)
                    client_n[str(cid)] = int(result.n_examples)
                train_loss_sum += float(result.avg_loss) * float(result.steps)
                train_loss_weight += int(result.steps)
            if meta.get("status") == "ok":
                client_iter.set_postfix(
                    client=str(cid),
                    n=int(result.n_examples),
                    loss=f"{float(result.avg_loss):.4f}",
                    w_kl=f"{float(result.avg_w_kl):.2f}",
                )
            else:
                client_iter.set_postfix(client=str(cid), status=str(meta.get("status")))

        if not client_w_params:
            raise SystemExit("No client updates available (min_examples filter).")

        # Server aggregation
        global_params = aggregate_bayes_dict(
            prev=global_params,
            locals=client_w_params,
            server_beta=server_beta,
            weights=client_weights,
            logvar_min=logvar_min,
            logvar_max=logvar_max,
            param_type=param_type,
        )

        # Validation
        eval_model = BFLModel(
            model.cfg,
            prior_sigma=float(cfg["bayes"]["prior_sigma"]),
            logvar_min=float(cfg["bayes"]["logvar_min"]),
            logvar_max=float(cfg["bayes"]["logvar_max"]),
            full_bayes=bool(full_bayes),
            param_type=param_type,
            mu_init=mu_init,
            init_rho=(float(init_rho) if init_rho is not None else None),
        )
        eval_model.load_state_dict(base_state, strict=False)
        eval_model.set_posterior(global_params)
        eval_model.set_prior(global_params)
        eval_model = eval_model.to(device)

        train_loss = float(train_loss_sum / max(train_loss_weight, 1))
        eval_total = rounds if (test_every_round and bool(test_files)) else 0
        eval_done = int(rnd - 1) if eval_total else 0
        msg = (
            f"[round {rnd:03d}] train_loss={train_loss:.4f} "
            f"used={len(client_w_params)}/{len(selected)} "
            f"progress total={rnd}/{rounds} eval={eval_done}/{eval_total}"
        )
        if show_progress:
            tqdm.write(msg)
        else:
            print(msg)
        sel_thr = float(fixed_threshold)
        history.append(
            {
                "round": int(rnd),
                "train_loss": float(train_loss),
                "threshold": float(sel_thr),
                "server_beta": float(server_beta),
                "zeta": float(train_cfg.zeta),
            }
        )
        if test_every_round and test_files:
            eval_started = time.perf_counter()
            if threshold_method == "youden":
                test_report = evaluate_split(
                    model=eval_model,
                    files=test_files,
                    mc_eval=int(cfg["train"]["mc_eval"]),
                    device=device,
                    temperature=None,
                    threshold=None,
                    threshold_method="youden",
                    recall_target=float(cfg["eval"]["threshold"].get("recall_target", 0.8)),
                    fixed_threshold=float(fixed_threshold),
                    threshold_use_post=bool(threshold_use_post),
                    bootstrap_n=0,
                    bootstrap_seed=int(cfg["eval"]["bootstrap"].get("seed", 42)),
                    batch_size=eval_batch_size,
                    num_workers=eval_num_workers,
                )
                sel_thr = float(test_report.get("threshold_selected", fixed_threshold))
            elif threshold_method == "youden_val":
                sel_thr = float(_select_threshold_from_val(eval_model) or fixed_threshold)
                test_report = evaluate_split(
                    model=eval_model,
                    files=test_files,
                    mc_eval=int(cfg["train"]["mc_eval"]),
                    device=device,
                    temperature=None,
                    threshold=float(sel_thr),
                    threshold_method="fixed_threshold",
                    recall_target=float(cfg["eval"]["threshold"].get("recall_target", 0.8)),
                    fixed_threshold=float(fixed_threshold),
                    threshold_use_post=bool(threshold_use_post),
                    bootstrap_n=0,
                    bootstrap_seed=int(cfg["eval"]["bootstrap"].get("seed", 42)),
                    batch_size=eval_batch_size,
                    num_workers=eval_num_workers,
                )
                test_report["threshold_selected"] = float(sel_thr)
                test_report["threshold_method"] = "youden_val"
            else:
                test_report = evaluate_split(
                    model=eval_model,
                    files=test_files,
                    mc_eval=int(cfg["train"]["mc_eval"]),
                    device=device,
                    temperature=None,
                    threshold=float(fixed_threshold),
                    threshold_method="fixed_threshold",
                    recall_target=float(cfg["eval"]["threshold"].get("recall_target", 0.8)),
                    fixed_threshold=float(fixed_threshold),
                    threshold_use_post=bool(threshold_use_post),
                    bootstrap_n=0,
                    bootstrap_seed=int(cfg["eval"]["bootstrap"].get("seed", 42)),
                    batch_size=eval_batch_size,
                    num_workers=eval_num_workers,
                )
            write_json(run_dir / f"round_{rnd:03d}_test.json", test_report)
            try:
                history[-1]["threshold"] = float(sel_thr)
                test_m = test_report.get("metrics_pre", {})
                history[-1]["test_auprc"] = float(test_m.get("auprc", float("nan")))
                history[-1]["test_auroc"] = float(test_m.get("auroc", float("nan")))
                history[-1]["test_brier"] = float(test_m.get("brier", float("nan")))
                history[-1]["test_nll"] = float(test_m.get("nll", float("nan")))
                history[-1]["test_ece"] = float(test_m.get("ece", float("nan")))
            except Exception:
                pass
            eval_elapsed_sum += max(0.0, time.perf_counter() - eval_started)
            eval_round_count += 1
            eval_done = int(rnd)
        if show_progress:
            postfix: Dict[str, str] = {
                "overall": f"{rnd}/{rounds}",
                "eval": f"{eval_done}/{eval_total}",
            }
            if eval_total > 0 and eval_done < eval_total and eval_round_count > 0:
                avg_eval = eval_elapsed_sum / max(eval_round_count, 1)
                postfix["eval_eta"] = _format_seconds(avg_eval * float(eval_total - eval_done))
            round_iter.set_postfix(postfix)
        if selection_enabled:
            sel_files = test_files if selection_source == "test" else val_files
            if not sel_files:
                if selection_source != "test":
                    print("[WARN] model_selection=best but val files missing; falling back to test.")
                    sel_files = test_files
            if sel_files:
                sel_report = evaluate_split(
                    model=eval_model,
                    files=sel_files,
                    mc_eval=int(cfg["train"]["mc_eval"]),
                    device=device,
                    temperature=None,
                    threshold=None,
                    threshold_method="youden",
                    recall_target=float(cfg["eval"]["threshold"].get("recall_target", 0.8)),
                    fixed_threshold=float(fixed_threshold),
                    threshold_use_post=bool(threshold_use_post),
                    bootstrap_n=0,
                    bootstrap_seed=int(cfg["eval"]["bootstrap"].get("seed", 42)),
                    batch_size=eval_batch_size,
                    num_workers=eval_num_workers,
                )
                metrics = sel_report.get("metrics_post" if selection_use_post else "metrics_pre", {}) or {}
                score = float(metrics.get(selection_metric, float("-inf")))
                prev = best.get("metric")
                if prev is None or score > float(prev):
                    best = {
                        "round": int(rnd),
                        "metric": float(score),
                        "source": selection_source,
                        "metric_name": selection_metric,
                    }
                    save_checkpoint(
                        run_dir / "checkpoints" / "model_best.pt",
                        {
                            "model_cfg": asdict(model.cfg),
                            "state_dict": eval_model.state_dict(),
                            "global_params": global_params,
                            "full_bayes": bool(full_bayes),
                        },
                    )
        if per_client_every_round and client_test_files:
            round_rows = []
            for cid, files in client_test_files.items():
                if not files:
                    continue
                rep = evaluate_split(
                    model=eval_model,
                    files=files,
                    mc_eval=int(cfg["train"]["mc_eval"]),
                    device=device,
                    temperature=None,
                    threshold=float(sel_thr),
                    threshold_method="fixed_threshold",
                    recall_target=float(cfg["eval"]["threshold"].get("recall_target", 0.8)),
                    fixed_threshold=float(fixed_threshold),
                    threshold_use_post=bool(threshold_use_post),
                    bootstrap_n=0,
                    bootstrap_seed=int(cfg["eval"]["bootstrap"].get("seed", 42)),
                    batch_size=eval_batch_size,
                    num_workers=eval_num_workers,
                )
                metrics = rep.get("metrics_pre", {}) or {}
                conf = rep.get("confusion_pre") or {}
                derived = derived_from_confusion(conf) if conf else {}
                row_c = {
                    "round": int(rnd),
                    "client_id": str(cid),
                    "n": int(rep.get("n", 0)),
                    "n_pos": int(rep.get("n_pos", 0)),
                    "n_neg": int(rep.get("n_neg", 0)),
                    "pos_rate": float(int(rep.get("n_pos", 0)) / max(int(rep.get("n", 0)), 1)),
                    "auprc": float(metrics.get("auprc", float("nan"))),
                    "auroc": float(metrics.get("auroc", float("nan"))),
                    "brier": float(metrics.get("brier", float("nan"))),
                    "nll": float(metrics.get("nll", float("nan"))),
                    "ece": float(metrics.get("ece", float("nan"))),
                    "threshold": float(sel_thr),
                    "accuracy": float(derived.get("accuracy", float("nan"))),
                    "f1": float(derived.get("f1", float("nan"))),
                    "sensitivity": float(conf.get("sensitivity", float("nan"))),
                    "specificity": float(conf.get("specificity", float("nan"))),
                    "ppv": float(conf.get("ppv", float("nan"))),
                    "npv": float(conf.get("npv", float("nan"))),
                }
                round_rows.append(row_c)
                per_client_rounds.append(row_c)
            if round_rows:
                import pandas as pd

                pd.DataFrame(round_rows).to_csv(run_dir / f"round_{rnd:03d}_test_per_client.csv", index=False)
                pd.DataFrame(per_client_rounds).to_csv(run_dir / "round_client_metrics.csv", index=False)
                write_json(run_dir / "round_client_metrics.json", per_client_rounds)
        _save_history(run_dir / "history.csv", history)
        write_json(run_dir / f"round_{rnd:03d}_clients.json", {"clients": client_metrics, "selected": selected})
        if log_client_sim and client_vecs:
            client_ids = list(client_vecs.keys())
            vecs = np.stack([client_vecs[cid] for cid in client_ids], axis=0)
            sim = _cosine_sim_matrix(vecs)
            write_json(
                run_dir / f"round_{rnd:03d}_client_similarity.json",
                {
                    "metric": "cosine_weight_mu_bias_mu",
                    "client_ids": client_ids,
                    "n_examples": client_n,
                    "matrix": sim.tolist(),
                },
            )

        if best_updated:
            pass
        # Always keep the latest checkpoint for last-round evaluation.
        save_checkpoint(
            run_dir / "checkpoints" / "model_last.pt",
            {
                "model_cfg": asdict(model.cfg),
                "state_dict": eval_model.state_dict(),
                "global_params": global_params,
                "full_bayes": bool(full_bayes),
            },
        )

        # Server state checkpoint
        save_checkpoint(
            server_state_path,
            {
                "round": int(rnd),
                "global_params": global_params,
                "best": best,
                "history": history,
                "rng_state": capture_rng_state(),
            },
        )

    # Final test evaluation with selected model (best or last)
    model_sel = "last"
    ckpt_name = "model_last.pt"
    if selection_enabled:
        best_ckpt = run_dir / "checkpoints" / "model_best.pt"
        if best_ckpt.exists():
            model_sel = "best"
            ckpt_name = "model_best.pt"
    ckpt_path = run_dir / "checkpoints" / ckpt_name
    if ckpt_path.exists():
        sel_state = load_checkpoint(ckpt_path)
        test_model = BFLModel(
            model.cfg,
            prior_sigma=float(cfg["bayes"]["prior_sigma"]),
            logvar_min=float(cfg["bayes"]["logvar_min"]),
            logvar_max=float(cfg["bayes"]["logvar_max"]),
            full_bayes=bool(full_bayes),
            param_type=param_type,
            mu_init=mu_init,
            init_rho=(float(init_rho) if init_rho is not None else None),
        )
        test_model.load_state_dict(sel_state["state_dict"], strict=True)
        test_model = test_model.to(device)

        test_files = list_npz_files(str(data_dir), test_split)
        last_entry = history[-1] if history else {}
        sel_thr = float(fixed_threshold)
        save_pred_path = args.save_test_pred_npz
        if save_pred_path is None:
            save_pred_cfg = cfg.get("eval", {}).get("save_test_pred_npz", None)
            if isinstance(save_pred_cfg, bool):
                if save_pred_cfg:
                    save_pred_path = str(run_dir / "test_predictions.npz")
            elif isinstance(save_pred_cfg, str):
                save_pred_path = save_pred_cfg
                if not Path(save_pred_path).is_absolute():
                    save_pred_path = str(run_dir / save_pred_path)
        if threshold_method == "youden":
            test_report = evaluate_split(
                model=test_model,
                files=test_files,
                mc_eval=int(cfg["train"]["mc_eval"]),
                device=device,
                temperature=None,
                threshold=None,
                threshold_method="youden",
                recall_target=float(cfg["eval"]["threshold"].get("recall_target", 0.8)),
                fixed_threshold=float(fixed_threshold),
                threshold_use_post=bool(threshold_use_post),
                bootstrap_n=int(cfg["eval"]["bootstrap"].get("n_boot", 1000)),
                bootstrap_seed=int(cfg["eval"]["bootstrap"].get("seed", 42)),
                batch_size=eval_batch_size,
                num_workers=eval_num_workers,
                save_pred_path=save_pred_path,
            )
            sel_thr = float(test_report.get("threshold_selected", fixed_threshold))
        elif threshold_method == "youden_val":
            sel_thr = float(_select_threshold_from_val(test_model) or fixed_threshold)
            test_report = evaluate_split(
                model=test_model,
                files=test_files,
                mc_eval=int(cfg["train"]["mc_eval"]),
                device=device,
                temperature=None,
                threshold=float(sel_thr),
                threshold_method="fixed_threshold",
                recall_target=float(cfg["eval"]["threshold"].get("recall_target", 0.8)),
                fixed_threshold=float(fixed_threshold),
                threshold_use_post=bool(threshold_use_post),
                bootstrap_n=int(cfg["eval"]["bootstrap"].get("n_boot", 1000)),
                bootstrap_seed=int(cfg["eval"]["bootstrap"].get("seed", 42)),
                batch_size=eval_batch_size,
                num_workers=eval_num_workers,
                save_pred_path=save_pred_path,
            )
            test_report["threshold_selected"] = float(sel_thr)
            test_report["threshold_method"] = "youden_val"
        else:
            test_report = evaluate_split(
                model=test_model,
                files=test_files,
                mc_eval=int(cfg["train"]["mc_eval"]),
                device=device,
                temperature=None,
                threshold=float(sel_thr),
                threshold_method="fixed_threshold",
                recall_target=float(cfg["eval"]["threshold"].get("recall_target", 0.8)),
                fixed_threshold=float(sel_thr),
                threshold_use_post=bool(threshold_use_post),
                bootstrap_n=int(cfg["eval"]["bootstrap"].get("n_boot", 1000)),
                bootstrap_seed=int(cfg["eval"]["bootstrap"].get("seed", 42)),
                batch_size=eval_batch_size,
                num_workers=eval_num_workers,
                save_pred_path=save_pred_path,
            )
        test_report["model_selected"] = str(model_sel)
        test_report["best"] = best
        write_json(run_dir / "test_report.json", test_report)

        # Per-client TEST report (fixed threshold; no bootstrap by default)
        per_client_reports: Dict[str, Any] = {}
        per_client_rows: List[Dict[str, Any]] = []
        for cid in clients:
            client_files = list_npz_files(str(data_dir), test_split, client_id=str(cid))
            if not client_files:
                per_client_reports[str(cid)] = {"client_id": str(cid), "status": "no_files", "n": 0}
                per_client_rows.append({"client_id": str(cid), "status": "no_files", "n": 0})
                continue
            rep = evaluate_split(
                model=test_model,
                files=client_files,
                mc_eval=int(cfg["train"]["mc_eval"]),
                device=device,
                temperature=None,
                threshold=float(sel_thr),
                threshold_method="fixed_threshold",
                recall_target=float(cfg["eval"]["threshold"].get("recall_target", 0.8)),
                fixed_threshold=float(sel_thr),
                threshold_use_post=bool(threshold_use_post),
                bootstrap_n=0,
                bootstrap_seed=int(cfg["eval"]["bootstrap"].get("seed", 42)),
                batch_size=eval_batch_size,
                num_workers=eval_num_workers,
            )
            rep["client_id"] = str(cid)
            rep["status"] = "ok"
            per_client_reports[str(cid)] = rep

            m_pre = rep.get("metrics_pre", {}) or {}
            per_client_rows.append(
                {
                    "client_id": str(cid),
                    "status": "ok",
                    "n": int(rep.get("n", 0)),
                    "n_pos": int(rep.get("n_pos", 0)),
                    "n_neg": int(rep.get("n_neg", 0)),
                    "auprc_pre": float(m_pre.get("auprc", float("nan"))),
                    "auroc_pre": float(m_pre.get("auroc", float("nan"))),
                    "brier_pre": float(m_pre.get("brier", float("nan"))),
                    "nll_pre": float(m_pre.get("nll", float("nan"))),
                    "ece_pre": float(m_pre.get("ece", float("nan"))),
                    "threshold": float(sel_thr),
                }
            )

        write_json(
            run_dir / "test_report_per_client.json",
            {
                "threshold": float(sel_thr),
                "clients": per_client_reports,
            },
        )
        import pandas as pd

        pd.DataFrame(per_client_rows).to_csv(run_dir / "test_report_per_client.csv", index=False)

    last_entry = history[-1] if history else {}
    selected = {
        "round": int(last_entry.get("round", rounds)),
        "threshold": float(fixed_threshold),
    }
    summary = {
        "best": best,
        "selected": {"mode": model_sel, **selected},
        "rounds": int(rounds),
        "used_point_init": bool(used_point),
        "train_backbone": bool(cfg["backbone"].get("train_backbone", False)),
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
