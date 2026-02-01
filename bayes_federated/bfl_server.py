from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

# Ensure project root in sys.path when executed as `python bayes_federated/bfl_server.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bayes_federated.agg import AggConfig, feduab_aggregate, feduab_aggregate_dict, poe_prior_corrected, poe_prior_corrected_dict
from bayes_federated.bfl_client import ClientTrainConfig, train_client_round
from bayes_federated.bayes_layers import BayesParams
from bayes_federated.eval import evaluate_split, mc_predict
from bayes_federated.models import BFLModel, build_bfl_model_from_point_checkpoint
from bayes_federated.vi import BetaConfig
from common.checkpoint import capture_rng_state, load_checkpoint, restore_rng_state, save_checkpoint
from common.experiment import make_run_dir, save_env_snapshot, seed_everything, seed_worker
from common.io import read_json, write_json
from common.dataset import WindowedNPZDataset, list_client_ids, list_npz_files, list_npz_files_by_client, scan_label_stats
from common.calibration import fit_temperature


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


def _dataset_counts(data_dir: str, train_split: str, val_split: str, test_split: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for split in [train_split, val_split, test_split]:
        files = list_npz_files(data_dir, split)
        pos, total = scan_label_stats(files) if files else (0, 0)
        out[str(split)] = {"n": int(total), "n_pos": int(pos), "n_neg": int(total - pos)}
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


def _load_labels_from_files(files: List[str]) -> np.ndarray:
    ys = []
    for p in files:
        with np.load(p, allow_pickle=False) as z:
            if "y" not in z:
                continue
            ys.append(np.asarray(z["y"], dtype=np.int64))
    if not ys:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(ys, axis=0)


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


def main() -> None:
    ap = argparse.ArgumentParser(description="BFL server (prior-corrected PoE aggregation)")
    ap.add_argument("--config", default="configs/bfl.yaml")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--client-progress-bar", action="store_true", help="Show per-client batch progress bar.")
    ap.add_argument("--no-progress-bar", action="store_true", help="Disable progress bars.")
    ap.add_argument("--log-client-sim", action="store_true", help="Save per-round client cosine similarity matrix.")
    ap.add_argument("--save-test-pred-npz", default=None, help="Optional .npz to save per-sample test predictions/uncertainty.")
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

    # Prepare base model
    backbone_cfg = cfg.get("backbone", {})
    full_bayes = bool(cfg.get("bayes", {}).get("full_bayes", False))
    model, init_prior, used_point = build_bfl_model_from_point_checkpoint(
        backbone_cfg.get("checkpoint"),
        prior_sigma=float(cfg["bayes"]["prior_sigma"]),
        logvar_min=float(cfg["bayes"]["logvar_min"]),
        logvar_max=float(cfg["bayes"]["logvar_max"]),
        full_bayes=bool(full_bayes),
    )
    base_state = model.state_dict()

    data_dir = cfg["data"]["data_dir"]
    summary_path = Path(data_dir) / "summary.json"
    dataset_summary = None
    if summary_path.exists():
        try:
            dataset_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as e:
            dataset_summary = {"error": f"failed to read summary.json: {e}"}

    # Client list
    clients_path = run_dir / "clients.json"
    clients = _client_list(str(data_dir), cfg["data"]["train_split"], existing=clients_path if resume else None)
    if not clients:
        raise SystemExit("No clients found in index.")
    if not resume:
        write_json(clients_path, clients)

    # Dataset counts
    counts = _dataset_counts(str(data_dir), cfg["data"]["train_split"], cfg["data"]["val_split"], cfg["data"]["test_split"])
    write_json(run_dir / "data_counts.json", counts)
    client_counts = _client_counts(str(data_dir), cfg["data"]["train_split"])
    write_json(run_dir / "client_counts.json", client_counts)
    if dataset_summary:
        keys = ["client_scheme", "merge_strategy", "opname_threshold", "min_client_cases", "clients"]
        write_json(run_dir / "dataset_summary.json", {k: dataset_summary.get(k) for k in keys if k in dataset_summary})

    # Resume state
    server_state_path = run_dir / "checkpoints" / "server_state.pt"
    start_round = 1
    global_params = init_prior
    best = {"round": 0, "val_auprc": -1.0, "temperature": None, "threshold": None}
    history = []
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
    train_cfg = ClientTrainConfig(
        local_epochs=int(cfg["train"]["local_epochs"]),
        batch_size=int(cfg["train"]["batch_size"]),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        num_workers=int(cfg["train"]["num_workers"]),
        mc_train=int(cfg["train"]["mc_train"]),
        grad_clip=float(cfg["train"].get("grad_clip", 0.0)),
        train_backbone=bool(cfg["backbone"].get("train_backbone", False)),
        seed=int(cfg.get("seed", 42)),
        loss_type=str(loss_cfg.get("type", "bce")),
        focal_gamma=float(loss_cfg.get("focal_gamma", 2.0)),
        focal_alpha=float(loss_cfg.get("focal_alpha", 0.25)),
        pos_weight=(float(loss_cfg["pos_weight"]) if "pos_weight" in loss_cfg and loss_cfg["pos_weight"] is not None else None),
    )
    eval_cfg = cfg.get("eval", {})
    eval_batch_size = int(eval_cfg.get("batch_size", 128))
    eval_num_workers = int(eval_cfg.get("num_workers", cfg["train"].get("num_workers", 0)))
    test_every_round = bool(eval_cfg.get("test_every_round", False))
    if args.test_every_round is not None:
        test_every_round = bool(args.test_every_round)
    test_files = list_npz_files(str(data_dir), cfg["data"]["test_split"])
    per_client_every_round = bool(eval_cfg.get("per_client_every_round", False))
    client_test_files = {}
    if per_client_every_round:
        client_test_files = list_npz_files_by_client(str(data_dir), cfg["data"]["test_split"])
    beta_cfg = BetaConfig(
        mode=str(cfg["beta"]["mode"]),
        anneal_steps=int(cfg["beta"].get("anneal_steps", 0)),
        base_beta=float(cfg["beta"].get("base_beta", 1.0)),
    )
    agg_cfg = AggConfig(
        precision_clamp=float(cfg["agg"]["precision_clamp"]),
        beta_mode=str(cfg["agg"].get("beta_mode", "normalized")),
        beta_value=float(cfg["agg"].get("beta_value", 1.0)),
    )
    agg_method = str(cfg.get("agg", {}).get("method", "poe")).lower()

    rounds = int(cfg["train"]["rounds"])
    min_client_examples = int(cfg["clients"].get("min_examples", 1))
    show_progress = not bool(args.no_progress_bar)

    # Training rounds
    for rnd in range(start_round, rounds + 1):
        client_post = []
        client_weights = []
        client_metrics = []
        client_vecs: Dict[str, np.ndarray] = {}
        client_n: Dict[str, int] = {}
        train_loss_sum = 0.0
        train_loss_weight = 0

        client_iter = tqdm(
            clients,
            total=len(clients),
            desc=f"round {rnd}/{rounds} clients",
            leave=False,
            disable=(not show_progress),
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
            )
            local_model.load_state_dict(base_state, strict=False)

            train_files = list_npz_files(str(data_dir), cfg["data"]["train_split"], client_id=str(cid))
            posterior, result, meta = train_client_round(
                client_id=cid,
                train_files=train_files,
                model=local_model,
                prior=global_params,
                cfg=train_cfg,
                beta_cfg=beta_cfg,
                device=device,
                show_progress=bool(args.client_progress_bar) and show_progress,
                resume_path=str(ckpt_path) if resume else None,
                save_path=str(ckpt_path),
            )

            client_metrics.append({**asdict(result), **meta})
            if meta.get("status") == "ok":
                client_iter.set_postfix(
                    client=str(cid),
                    n=int(result.n_examples),
                    loss=f"{float(result.avg_loss):.4f}",
                    beta=f"{float(result.avg_beta):.2e}",
                )
            else:
                client_iter.set_postfix(client=str(cid), status=str(meta.get("status")))
            if result.n_examples >= min_client_examples:
                client_post.append(posterior)
                client_weights.append(result.n_examples)
                if args.log_client_sim and meta.get("status") == "ok":
                    client_vecs[str(cid)] = _flatten_bayes_mu(posterior)
                    client_n[str(cid)] = int(result.n_examples)
                if meta.get("status") == "ok":
                    train_loss_sum += float(result.avg_loss) * float(result.steps)
                    train_loss_weight += int(result.steps)

        if not client_post:
            raise SystemExit("No client updates available (min_examples filter).")

        weights = [float(w) for w in client_weights]
        if agg_method == "feduab":
            if full_bayes:
                global_params, agg_stats = feduab_aggregate_dict(posteriors=client_post, weights=weights, cfg=agg_cfg)
            else:
                global_params, agg_stats = feduab_aggregate(posteriors=client_post, weights=weights, cfg=agg_cfg)
        elif agg_method == "poe":
            if full_bayes:
                global_params, agg_stats = poe_prior_corrected_dict(
                    prior=global_params, posteriors=client_post, weights=weights, cfg=agg_cfg
                )
            else:
                global_params, agg_stats = poe_prior_corrected(
                    prior=global_params, posteriors=client_post, weights=weights, cfg=agg_cfg
                )
        else:
            raise ValueError(f"Unknown agg.method: {agg_method}")

        # Validation
        eval_model = BFLModel(
            model.cfg,
            prior_sigma=float(cfg["bayes"]["prior_sigma"]),
            logvar_min=float(cfg["bayes"]["logvar_min"]),
            logvar_max=float(cfg["bayes"]["logvar_max"]),
            full_bayes=bool(full_bayes),
        )
        eval_model.load_state_dict(base_state, strict=False)
        eval_model.set_posterior(global_params)
        eval_model.set_prior(global_params)
        eval_model = eval_model.to(device)

        val_files = list_npz_files(str(data_dir), cfg["data"]["val_split"])
        if not val_files:
            val_files = list_npz_files(str(data_dir), cfg["data"]["train_split"])
        val_report_pre = evaluate_split(
            model=eval_model,
            files=val_files,
            mc_eval=int(cfg["train"]["mc_eval"]),
            device=device,
            temperature=None,
            threshold=None,
            threshold_method=str(cfg["eval"]["threshold"]["method"]),
            recall_target=float(cfg["eval"]["threshold"].get("recall_target", 0.8)),
            fixed_threshold=float(cfg["eval"]["threshold"].get("fixed", 0.5)),
            bootstrap_n=0,
            bootstrap_seed=int(cfg["eval"]["bootstrap"].get("seed", 42)),
            batch_size=eval_batch_size,
            num_workers=eval_num_workers,
        )

        # Fit temperature on val logits (mean of MC logits)
        ds_val = WindowedNPZDataset(val_files, use_clin="true", cache_in_memory=False, max_cache_files=32, cache_dtype="float32")
        dl_val = torch.utils.data.DataLoader(
            ds_val,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=eval_num_workers,
            pin_memory=(device.type == "cuda"),
            worker_init_fn=seed_worker,
            persistent_workers=(eval_num_workers > 0),
        )
        preds = mc_predict(eval_model, dl_val, mc_eval=int(cfg["train"]["mc_eval"]), device=device, temperature=None, return_y=True)
        logits_mean = preds["logits_mean"]
        y_true = preds["y_true"].astype(int)
        tfit = fit_temperature(logits_mean, y_true, device=str(device), max_iter=int(cfg["eval"]["calibration"].get("temperature_max_iter", 100)))

        val_report_post = evaluate_split(
            model=eval_model,
            files=val_files,
            mc_eval=int(cfg["train"]["mc_eval"]),
            device=device,
            temperature=float(tfit.temperature),
            threshold=None,
            threshold_method=str(cfg["eval"]["threshold"]["method"]),
            recall_target=float(cfg["eval"]["threshold"].get("recall_target", 0.8)),
            fixed_threshold=float(cfg["eval"]["threshold"].get("fixed", 0.5)),
            bootstrap_n=0,
            bootstrap_seed=int(cfg["eval"]["bootstrap"].get("seed", 42)),
            batch_size=eval_batch_size,
            num_workers=eval_num_workers,
        )

        val_auprc = val_report_post["metrics_post"]["auprc"] if "metrics_post" in val_report_post else val_report_pre["metrics_pre"]["auprc"]
        threshold = val_report_post.get("threshold_selected", val_report_pre.get("threshold_selected"))
        train_loss = float(train_loss_sum / max(train_loss_weight, 1))
        msg = (
            f"[round {rnd:03d}] train_loss={train_loss:.4f} val_auprc={float(val_auprc):.4f} "
            f"used={len(client_post)}/{len(clients)} temp={float(tfit.temperature):.4f}"
        )
        if show_progress:
            tqdm.write(msg)
        else:
            print(msg)
        history.append(
            {
                "round": int(rnd),
                "train_loss": float(train_loss),
                "val_auprc": float(val_auprc),
                "temperature": float(tfit.temperature),
                "threshold": float(threshold) if threshold is not None else None,
                "agg_clamped_raw": int(agg_stats.clamped_raw),
                "agg_clamped_global": int(agg_stats.clamped_global),
            }
        )
        if test_every_round and test_files:
            test_report = evaluate_split(
                model=eval_model,
                files=test_files,
                mc_eval=int(cfg["train"]["mc_eval"]),
                device=device,
                temperature=float(tfit.temperature),
                threshold=float(threshold) if threshold is not None else None,
                threshold_method=str(cfg["eval"]["threshold"]["method"]),
                recall_target=float(cfg["eval"]["threshold"].get("recall_target", 0.8)),
                fixed_threshold=float(cfg["eval"]["threshold"].get("fixed", 0.5)),
                bootstrap_n=0,
                bootstrap_seed=int(cfg["eval"]["bootstrap"].get("seed", 42)),
                batch_size=eval_batch_size,
                num_workers=eval_num_workers,
            )
            write_json(run_dir / f"round_{rnd:03d}_test.json", test_report)
            try:
                test_m = test_report.get("metrics_post", test_report.get("metrics_pre", {}))
                history[-1]["test_auprc"] = float(test_m.get("auprc", float("nan")))
                history[-1]["test_auroc"] = float(test_m.get("auroc", float("nan")))
            except Exception:
                pass
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
                    temperature=float(tfit.temperature),
                    threshold=float(threshold) if threshold is not None else None,
                    threshold_method=str(cfg["eval"]["threshold"]["method"]),
                    recall_target=float(cfg["eval"]["threshold"].get("recall_target", 0.8)),
                    fixed_threshold=float(cfg["eval"]["threshold"].get("fixed", 0.5)),
                    bootstrap_n=0,
                    bootstrap_seed=int(cfg["eval"]["bootstrap"].get("seed", 42)),
                    batch_size=eval_batch_size,
                    num_workers=eval_num_workers,
                )
                metrics = rep.get("metrics_post", rep.get("metrics_pre", {})) or {}
                row_c = {
                    "round": int(rnd),
                    "client_id": str(cid),
                    "n": int(rep.get("n", 0)),
                    "n_pos": int(rep.get("n_pos", 0)),
                    "n_neg": int(rep.get("n_neg", 0)),
                    "auprc": float(metrics.get("auprc", float("nan"))),
                    "auroc": float(metrics.get("auroc", float("nan"))),
                    "brier": float(metrics.get("brier", float("nan"))),
                    "nll": float(metrics.get("nll", float("nan"))),
                    "ece": float(metrics.get("ece", float("nan"))),
                }
                round_rows.append(row_c)
                per_client_rounds.append(row_c)
            if round_rows:
                import pandas as pd

                pd.DataFrame(round_rows).to_csv(run_dir / f"round_{rnd:03d}_test_per_client.csv", index=False)
                pd.DataFrame(per_client_rounds).to_csv(run_dir / "round_client_metrics.csv", index=False)
        _save_history(run_dir / "history.csv", history)
        write_json(run_dir / f"round_{rnd:03d}_val_pre.json", val_report_pre)
        write_json(run_dir / f"round_{rnd:03d}_val_post.json", val_report_post)
        write_json(
            run_dir / f"round_{rnd:03d}_val.json",
            {
                "n": int(val_report_pre.get("n", 0)),
                "n_pos": int(val_report_pre.get("n_pos", 0)),
                "n_neg": int(val_report_pre.get("n_neg", 0)),
                "metrics_pre": val_report_pre.get("metrics_pre"),
                "metrics_post": val_report_post.get("metrics_post"),
                "uncertainty": val_report_post.get("uncertainty", val_report_pre.get("uncertainty")),
                "threshold_selected": val_report_post.get("threshold_selected", val_report_pre.get("threshold_selected")),
                "confusion_pre": val_report_pre.get("confusion_pre"),
                "confusion_post": val_report_post.get("confusion_post"),
            },
        )
        write_json(run_dir / f"round_{rnd:03d}_clients.json", {"clients": client_metrics})
        if args.log_client_sim and client_vecs:
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

        if float(val_auprc) > float(best["val_auprc"]):
            best = {
                "round": int(rnd),
                "val_auprc": float(val_auprc),
                "temperature": float(tfit.temperature),
                "threshold": float(threshold) if threshold is not None else None,
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

    # Final test evaluation with best model
    best_ckpt = run_dir / "checkpoints" / "model_best.pt"
    if best_ckpt.exists():
        best_state = load_checkpoint(best_ckpt)
        test_model = BFLModel(
            model.cfg,
            prior_sigma=float(cfg["bayes"]["prior_sigma"]),
            logvar_min=float(cfg["bayes"]["logvar_min"]),
            logvar_max=float(cfg["bayes"]["logvar_max"]),
            full_bayes=bool(full_bayes),
        )
        test_model.load_state_dict(best_state["state_dict"], strict=True)
        test_model = test_model.to(device)

        test_files = list_npz_files(str(data_dir), cfg["data"]["test_split"])
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
        test_report = evaluate_split(
            model=test_model,
            files=test_files,
            mc_eval=int(cfg["train"]["mc_eval"]),
            device=device,
            temperature=float(best["temperature"]) if best["temperature"] is not None else None,
            threshold=float(best["threshold"]) if best["threshold"] is not None else None,
            threshold_method=str(cfg["eval"]["threshold"]["method"]),
            recall_target=float(cfg["eval"]["threshold"].get("recall_target", 0.8)),
            fixed_threshold=float(cfg["eval"]["threshold"].get("fixed", 0.5)),
            bootstrap_n=int(cfg["eval"]["bootstrap"].get("n_boot", 0)),
            bootstrap_seed=int(cfg["eval"]["bootstrap"].get("seed", 42)),
            save_pred_path=save_pred_path,
            batch_size=eval_batch_size,
            num_workers=eval_num_workers,
        )
        write_json(run_dir / "test_report.json", test_report)

        # Per-client TEST report (uses global temperature/threshold; no bootstrap by default)
        per_client_reports: Dict[str, Any] = {}
        per_client_rows: List[Dict[str, Any]] = []
        for cid in clients:
            client_files = list_npz_files(str(data_dir), cfg["data"]["test_split"], client_id=str(cid))
            if not client_files:
                per_client_reports[str(cid)] = {"client_id": str(cid), "status": "no_files", "n": 0}
                per_client_rows.append({"client_id": str(cid), "status": "no_files", "n": 0})
                continue

            rep = evaluate_split(
                model=test_model,
                files=client_files,
                mc_eval=int(cfg["train"]["mc_eval"]),
                device=device,
                temperature=float(best["temperature"]) if best["temperature"] is not None else None,
                threshold=float(best["threshold"]) if best["threshold"] is not None else None,
                threshold_method=str(cfg["eval"]["threshold"]["method"]),
                recall_target=float(cfg["eval"]["threshold"].get("recall_target", 0.8)),
                fixed_threshold=float(cfg["eval"]["threshold"].get("fixed", 0.5)),
                bootstrap_n=0,
                bootstrap_seed=int(cfg["eval"]["bootstrap"].get("seed", 42)),
                batch_size=eval_batch_size,
                num_workers=eval_num_workers,
            )
            rep["client_id"] = str(cid)
            rep["status"] = "ok"
            per_client_reports[str(cid)] = rep

            m_pre = rep.get("metrics_pre", {}) or {}
            m_post = rep.get("metrics_post", {}) or {}
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
                    "auprc_post": float(m_post.get("auprc", float("nan"))),
                    "auroc_post": float(m_post.get("auroc", float("nan"))),
                    "brier_post": float(m_post.get("brier", float("nan"))),
                    "nll_post": float(m_post.get("nll", float("nan"))),
                    "ece_post": float(m_post.get("ece", float("nan"))),
                    "temperature": float(best["temperature"]) if best["temperature"] is not None else None,
                    "threshold": float(best["threshold"]) if best["threshold"] is not None else None,
                }
            )

        write_json(
            run_dir / "test_report_per_client.json",
            {
                "temperature": float(best["temperature"]) if best["temperature"] is not None else None,
                "threshold": float(best["threshold"]) if best["threshold"] is not None else None,
                "clients": per_client_reports,
            },
        )
        import pandas as pd

        pd.DataFrame(per_client_rows).to_csv(run_dir / "test_report_per_client.csv", index=False)

    summary = {
        "best": best,
        "rounds": int(rounds),
        "used_point_init": bool(used_point),
        "train_backbone": bool(cfg["backbone"].get("train_backbone", False)),
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
