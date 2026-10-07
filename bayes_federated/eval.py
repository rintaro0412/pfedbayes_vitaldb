from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root in sys.path when executed as `python bayes_federated/eval.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bayes_federated.models import BFLModel
from common.checkpoint import load_checkpoint
from common.experiment import seed_worker
from common.io import read_json, write_json
from common.dataset import WindowedNPZDataset, list_client_ids, list_npz_files
from common.metrics import (
    auprc,
    auroc,
    bootstrap_group_ci,
    confusion_at_threshold,
    compute_binary_metrics,
)
from common.ioh_model import IOHModelConfig, normalize_model_cfg


def _group_ids(ds: WindowedNPZDataset) -> np.ndarray:
    return np.concatenate([np.full(n, cid, dtype=np.int64) for n, cid in zip(ds.file_sizes, ds.case_ids)])


@torch.no_grad()
def mc_predict(
    model: BFLModel,
    dl: DataLoader,
    *,
    mc_eval: int,
    device: torch.device,
    temperature: float | None = None,
    return_y: bool = False,
) -> Dict[str, np.ndarray]:
    model.eval()
    prob_mean_list = []
    prob_var_list = []
    prob_alea_list = []
    prob_epi_list = []
    prob_total_var_list = []
    entropy_list = []
    logits_mean_list = []
    prob_mean_cal_list = []
    y_list = []

    for x, y in dl:
        if isinstance(x, (tuple, list)):
            x = tuple(t.to(device, non_blocking=True) for t in x)
        else:
            x = x.to(device, non_blocking=True)
        logits_mc = model(x, sample=True, n_samples=int(mc_eval))
        if logits_mc.dim() == 1:
            logits_mc = logits_mc.unsqueeze(0).unsqueeze(-1)
        elif logits_mc.dim() == 2 and logits_mc.shape[-1] == 1:
            logits_mc = logits_mc.unsqueeze(0)
        logits_mc = logits_mc.squeeze(-1)  # (MC, B)
        logits_mean = logits_mc.mean(dim=0)
        probs_mc = torch.sigmoid(logits_mc)
        prob_mean = probs_mc.mean(dim=0)
        prob_var = probs_mc.var(dim=0, unbiased=False)
        prob_alea = (probs_mc * (1.0 - probs_mc)).mean(dim=0)
        prob_epi = prob_var
        prob_total_var = prob_alea + prob_epi
        eps = 1e-12
        entropy = -prob_mean * torch.log(prob_mean + eps) - (1.0 - prob_mean) * torch.log(1.0 - prob_mean + eps)

        prob_mean_list.append(prob_mean.detach().cpu().numpy())
        prob_var_list.append(prob_var.detach().cpu().numpy())
        prob_alea_list.append(prob_alea.detach().cpu().numpy())
        prob_epi_list.append(prob_epi.detach().cpu().numpy())
        prob_total_var_list.append(prob_total_var.detach().cpu().numpy())
        entropy_list.append(entropy.detach().cpu().numpy())
        logits_mean_list.append(logits_mean.detach().cpu().numpy())

        if temperature is not None:
            probs_cal = torch.sigmoid(logits_mc / float(temperature))
            prob_mean_cal = probs_cal.mean(dim=0)
            prob_mean_cal_list.append(prob_mean_cal.detach().cpu().numpy())
        if return_y:
            y_list.append(y.detach().cpu().view(-1).numpy())

    out = {
        "prob_mean": np.concatenate(prob_mean_list, axis=0),
        "prob_var": np.concatenate(prob_var_list, axis=0),
        "prob_alea": np.concatenate(prob_alea_list, axis=0),
        "prob_epi": np.concatenate(prob_epi_list, axis=0),
        "prob_total_var": np.concatenate(prob_total_var_list, axis=0),
        "entropy": np.concatenate(entropy_list, axis=0),
        "logits_mean": np.concatenate(logits_mean_list, axis=0),
    }
    if temperature is not None:
        out["prob_mean_cal"] = np.concatenate(prob_mean_cal_list, axis=0)
    if return_y:
        out["y_true"] = np.concatenate(y_list, axis=0) if y_list else np.zeros((0,), dtype=np.int64)
    return out


def evaluate_split(
    *,
    model: BFLModel,
    files: list[str],
    mc_eval: int,
    device: torch.device,
    temperature: float | None = None,
    threshold: float | None = None,
    fixed_threshold: float = 0.5,
    bootstrap_n: int = 0,
    bootstrap_seed: int = 42,
    save_pred_path: str | None = None,
    batch_size: int = 128,
    num_workers: int = 0,
) -> Dict[str, Any]:
    if not files:
        raise ValueError("No files provided for evaluation.")

    ds = WindowedNPZDataset(files, use_clin="true", cache_in_memory=False, max_cache_files=32, cache_dtype="float32")
    num_workers = int(num_workers)
    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        persistent_workers=(num_workers > 0),
    )

    pred = mc_predict(model, dl, mc_eval=mc_eval, device=device, temperature=temperature, return_y=True)
    y_true = pred["y_true"].astype(int)
    group = _group_ids(ds)

    prob = pred["prob_mean"]
    metrics_pre = compute_binary_metrics(y_true, prob, n_bins=15)
    report: Dict[str, Any] = {
        "n": int(len(y_true)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "metrics_pre": asdict(metrics_pre),
        "uncertainty": {
            "prob_var_mean": float(np.mean(pred["prob_var"])),
            "prob_var_std": float(np.std(pred["prob_var"])),
            "aleatoric_mean": float(np.mean(pred["prob_alea"])),
            "aleatoric_std": float(np.std(pred["prob_alea"])),
            "epistemic_mean": float(np.mean(pred["prob_epi"])),
            "epistemic_std": float(np.std(pred["prob_epi"])),
            "total_var_mean": float(np.mean(pred["prob_total_var"])),
            "total_var_std": float(np.std(pred["prob_total_var"])),
            "entropy_mean": float(np.mean(pred["entropy"])),
            "entropy_std": float(np.std(pred["entropy"])),
        },
    }

    if temperature is not None and "prob_mean_cal" in pred:
        prob_cal = pred["prob_mean_cal"]
        metrics_post = compute_binary_metrics(y_true, prob_cal, n_bins=15)
        report["metrics_post"] = asdict(metrics_post)
    else:
        prob_cal = None

    if threshold is None:
        threshold = float(fixed_threshold)
    report["threshold_selected"] = float(threshold)

    report["confusion_pre"] = confusion_at_threshold(y_true, prob, thr=float(threshold))
    if prob_cal is not None:
        report["confusion_post"] = confusion_at_threshold(y_true, prob_cal, thr=float(threshold))

    if int(bootstrap_n) > 0:
        report["bootstrap"] = {
            "n_boot": int(bootstrap_n),
            "seed": int(bootstrap_seed),
            "group": "caseid",
            "auprc_pre": bootstrap_group_ci(group_ids=group, y_true=y_true, prob=prob, metric_fn=auprc, n_boot=bootstrap_n, seed=bootstrap_seed),
            "auroc_pre": bootstrap_group_ci(group_ids=group, y_true=y_true, prob=prob, metric_fn=auroc, n_boot=bootstrap_n, seed=bootstrap_seed),
        }
        if prob_cal is not None:
            report["bootstrap"]["auprc_post"] = bootstrap_group_ci(group_ids=group, y_true=y_true, prob=prob_cal, metric_fn=auprc, n_boot=bootstrap_n, seed=bootstrap_seed)
            report["bootstrap"]["auroc_post"] = bootstrap_group_ci(group_ids=group, y_true=y_true, prob=prob_cal, metric_fn=auroc, n_boot=bootstrap_n, seed=bootstrap_seed)

    if save_pred_path:
        out_path = Path(save_pred_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "y_true": y_true.astype(np.int64, copy=False),
            "prob_mean": prob.astype(np.float64, copy=False),
            "prob_var": pred["prob_var"].astype(np.float64, copy=False),
            "prob_alea": pred["prob_alea"].astype(np.float64, copy=False),
            "prob_epi": pred["prob_epi"].astype(np.float64, copy=False),
            "prob_total_var": pred["prob_total_var"].astype(np.float64, copy=False),
            "entropy": pred["entropy"].astype(np.float64, copy=False),
            "case_id": group.astype(np.int64, copy=False),
        }
        if prob_cal is not None:
            payload["prob_mean_cal"] = prob_cal.astype(np.float64, copy=False)
        np.savez(out_path, **payload)

    return report


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError(f"PyYAML is required to read config: {path}") from exc
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return data


def _find_run_config(
    *,
    checkpoint: Path,
    explicit: str | None,
    run_dir: Path | None = None,
) -> tuple[Dict[str, Any], str | None]:
    if explicit:
        path = Path(explicit)
        return _load_config(path), str(path)

    candidates: list[Path] = []
    if run_dir is not None:
        candidates.append(run_dir / "config.json")
    candidates.extend(
        [
            checkpoint.parent / "config.json",
            checkpoint.parent.parent / "config.json",
        ]
    )
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return _load_config(path), str(path)
    return {}, None


def _bayes_model_options(ckpt: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    # ``bayes`` is retained for older configs; current pFedBayes runs use ``model``.
    options: Dict[str, Any] = {}
    legacy = config.get("bayes", {})
    current = config.get("model", {})
    if isinstance(legacy, dict):
        options.update(legacy)
    if isinstance(current, dict):
        options.update(current)

    full_bayes = bool(ckpt.get("full_bayes", options.get("full_bayes", False)))
    init_rho_raw = options.get("init_rho", None)
    return {
        "prior_sigma": float(options.get("prior_sigma", 0.1)),
        "logvar_min": float(options.get("logvar_min", -12.0)),
        "logvar_max": float(options.get("logvar_max", 6.0)),
        "full_bayes": full_bayes,
        "param_type": str(options.get("param_type", "logvar")),
        "mu_init": str(options.get("mu_init", "zeros")),
        "init_rho": float(init_rho_raw) if init_rho_raw is not None else None,
        "var_reduction_h": float(options.get("var_reduction_h", 1.0)),
    }


def _build_model(
    ckpt: Dict[str, Any],
    *,
    config: Dict[str, Any],
    device: torch.device,
) -> tuple[BFLModel, Dict[str, Any]]:
    raw_model_cfg = ckpt.get("model_cfg")
    if raw_model_cfg is None:
        raw_model_cfg = config.get("model", {})
    options = _bayes_model_options(ckpt, config)
    model = BFLModel(
        normalize_model_cfg(raw_model_cfg),
        prior_sigma=float(options["prior_sigma"]),
        logvar_min=float(options["logvar_min"]),
        logvar_max=float(options["logvar_max"]),
        full_bayes=bool(options["full_bayes"]),
        param_type=str(options["param_type"]),
        mu_init=str(options["mu_init"]),
        init_rho=options["init_rho"],
        var_reduction_h=float(options["var_reduction_h"]),
    )
    state_dict = ckpt.get("state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError("Global checkpoint does not contain a state_dict mapping.")
    model.load_state_dict(state_dict, strict=True)
    return model.to(device), options


def _selected_personalized_round(
    run_dir: Path,
    requested_round: int | None,
) -> tuple[int, str, Dict[str, Any]]:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"pFedBayes summary not found: {summary_path}")
    summary = read_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError(f"Invalid pFedBayes summary: {summary_path}")

    selected = summary.get("selected", {})
    selected = selected if isinstance(selected, dict) else {}
    selected_mode = str(selected.get("mode", "last")).lower()
    if selected_mode not in {"best", "last"}:
        raise ValueError(f"Unsupported selected.mode in {summary_path}: {selected_mode}")

    if requested_round is not None:
        round_idx = int(requested_round)
    elif selected_mode == "best":
        best = summary.get("best", {})
        best = best if isinstance(best, dict) else {}
        round_idx = int(best.get("round", 0) or 0)
    else:
        round_idx = int(selected.get("round", summary.get("rounds", 0)) or 0)
    if round_idx <= 0:
        raise ValueError(f"Could not resolve a positive personalized round from {summary_path}")
    return round_idx, selected_mode, summary


def _global_checkpoint_for_run(
    run_dir: Path,
    *,
    selected_mode: str,
    override: str | None,
) -> Path:
    if override:
        path = Path(override)
    else:
        name = "model_best.pt" if selected_mode == "best" else "model_last.pt"
        path = run_dir / "checkpoints" / name
    if not path.is_file():
        raise FileNotFoundError(f"Global checkpoint not found: {path}")
    return path


def _expected_clients(run_dir: Path, data_dir: str) -> list[str]:
    clients_path = run_dir / "clients.json"
    if clients_path.is_file():
        raw = read_json(clients_path)
        if not isinstance(raw, list) or not all(isinstance(v, str) for v in raw):
            raise ValueError(f"Invalid client list: {clients_path}")
        clients = sorted(set(raw))
    else:
        clients = list_client_ids(data_dir)
    if not clients:
        raise ValueError("No clients found for personalized evaluation.")
    return clients


def _macro_metrics(client_reports: Dict[str, Dict[str, Any]], key: str) -> Dict[str, float]:
    metric_names = ["auprc", "auroc", "brier", "nll", "ece"]
    out: Dict[str, float] = {}
    for metric in metric_names:
        values = []
        for report in client_reports.values():
            value = (report.get(key, {}) or {}).get(metric)
            if value is not None and np.isfinite(float(value)):
                values.append(float(value))
        out[metric] = float(np.mean(values)) if values else float("nan")
    return out


def _evaluate_personalized(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint_path: Path,
    round_idx: int,
    selected_mode: str,
    summary: Dict[str, Any],
    config: Dict[str, Any],
    config_source: str | None,
    device: torch.device,
) -> Dict[str, Any]:
    clients = _expected_clients(run_dir, str(args.data_dir))
    round_dir = run_dir / "clients" / f"round_{int(round_idx):03d}"

    # Validate the complete evaluation set before spending time on MC inference.
    asset_paths: Dict[str, tuple[Path, list[str]]] = {}
    missing: list[str] = []
    for client_id in clients:
        client_checkpoint = round_dir / f"client_{client_id}.pt"
        files = list_npz_files(str(args.data_dir), str(args.split), client_id=client_id)
        if not client_checkpoint.is_file():
            missing.append(f"checkpoint:{client_id}")
        if not files:
            missing.append(f"{args.split}_files:{client_id}")
        asset_paths[client_id] = (client_checkpoint, files)
    if missing:
        raise FileNotFoundError(
            f"Personalized evaluation requires every expected client at round {round_idx}; missing "
            + ", ".join(missing)
        )

    global_ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    model, model_options = _build_model(global_ckpt, config=config, device=device)
    global_state = global_ckpt["state_dict"]
    expected_posterior_keys = set(model.get_posterior().keys())

    assets: Dict[str, tuple[Path, list[str], Dict[str, Any]]] = {}
    invalid: list[str] = []
    for client_id in clients:
        client_checkpoint, files = asset_paths[client_id]
        try:
            client_ckpt = load_checkpoint(client_checkpoint, map_location="cpu")
        except Exception as exc:
            invalid.append(
                f"{client_id}: could not load {client_checkpoint}: {type(exc).__name__}: {exc}"
            )
            continue
        if not isinstance(client_ckpt, dict):
            invalid.append(f"{client_id}: checkpoint is not a mapping: {client_checkpoint}")
            continue

        client_errors: list[str] = []
        if client_ckpt.get("completed") is not True:
            client_errors.append(f"completed={client_ckpt.get('completed')!r}, expected True")
        saved_client_id = client_ckpt.get("client_id")
        if saved_client_id != client_id:
            client_errors.append(
                f"client_id={saved_client_id!r}, expected {client_id!r}"
            )

        posterior = client_ckpt.get("posterior")
        if not isinstance(posterior, dict):
            client_errors.append("posterior is missing or is not a mapping")
        else:
            posterior_keys = set(posterior.keys())
            if posterior_keys != expected_posterior_keys:
                client_errors.append(
                    "posterior keys mismatch: "
                    f"missing={sorted(expected_posterior_keys - posterior_keys)}, "
                    f"unexpected={sorted(posterior_keys - expected_posterior_keys)}"
                )

        if client_errors:
            invalid.append(f"{client_id} ({client_checkpoint}): " + "; ".join(client_errors))
        else:
            assets[client_id] = (client_checkpoint, files, posterior)

    if invalid:
        raise ValueError(
            f"Personalized checkpoint preflight failed before MC inference at round {round_idx}:\n- "
            + "\n- ".join(invalid)
        )

    reports: Dict[str, Dict[str, Any]] = {}
    for client_id in clients:
        client_checkpoint, files, posterior = assets[client_id]

        model.load_state_dict(global_state, strict=True)
        model.set_posterior(posterior)
        report = evaluate_split(
            model=model,
            files=files,
            mc_eval=int(args.mc_eval),
            device=device,
            temperature=(float(args.temperature) if args.temperature is not None else None),
            threshold=float(args.threshold),
            fixed_threshold=float(args.threshold),
            bootstrap_n=int(args.bootstrap_n),
            bootstrap_seed=int(args.bootstrap_seed),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
        )
        report.update(
            {
                "client_id": client_id,
                "model_scope": "personalized_q_i",
                "personalized_round": int(round_idx),
                "posterior_checkpoint": str(client_checkpoint),
            }
        )
        reports[client_id] = report

    result: Dict[str, Any] = {
        "evaluation_mode": "personalized",
        "model_scope": "one personalized q_i per matching client split",
        "run_dir": str(run_dir),
        "global_checkpoint": str(checkpoint_path),
        "config_source": config_source,
        "model_options": model_options,
        "split": str(args.split),
        "selected_mode": selected_mode,
        "personalized_round": int(round_idx),
        "summary_best": summary.get("best"),
        "n_clients": int(len(reports)),
        "client_macro_metrics_pre": _macro_metrics(reports, "metrics_pre"),
        "clients": reports,
    }
    if args.temperature is not None:
        result["client_macro_metrics_post"] = _macro_metrics(reports, "metrics_post")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="BFL global or personalized-posterior evaluation (MC, calibration, CI)")
    ap.add_argument("--data-dir", default="federated_data", help="Output of scripts/build_dataset.py")
    ap.add_argument("--checkpoint", default=None, help="Global BFL checkpoint. Required unless --personalized-run-dir is used.")
    ap.add_argument("--config", default=None, help="Optional pFedBayes YAML/JSON config. Defaults to config.json beside the run checkpoint.")
    ap.add_argument("--personalized-run-dir", default=None, help="Evaluate each saved personalized q_i on its matching client split.")
    ap.add_argument("--round", type=int, default=None, help="Personalized client-checkpoint round. Default: selected best round, or selected last round.")
    ap.add_argument("--split", default="test")
    ap.add_argument("--mc-eval", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument(
        "--bootstrap-n",
        type=int,
        default=None,
        help="Bootstrap replicates. Default: 1000 for global evaluation and 0 for personalized evaluation.",
    )
    ap.add_argument("--bootstrap-seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--output", default=None)
    ap.add_argument("--save-pred-npz", default=None, help="Optional .npz to save per-sample y/prob/uncertainty.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.personalized_run_dir:
        if args.save_pred_npz:
            raise SystemExit("--save-pred-npz is currently supported only for global --checkpoint evaluation.")
        run_dir = Path(args.personalized_run_dir)
        if not run_dir.is_dir():
            raise SystemExit(f"Personalized run directory not found: {run_dir}")

        args.bootstrap_n = 0 if args.bootstrap_n is None else int(args.bootstrap_n)
        if args.bootstrap_n < 0:
            raise SystemExit("--bootstrap-n must be >= 0")

        selected_round, selected_mode, summary = _selected_personalized_round(run_dir, None)
        if args.round is None:
            round_idx = selected_round
        else:
            round_idx = int(args.round)
            if round_idx <= 0:
                raise SystemExit("--round must be positive")
            if round_idx != selected_round and not args.checkpoint:
                raise SystemExit(
                    "--round differs from the summary-selected round; provide the corresponding "
                    f"--checkpoint explicitly (selected={selected_round}, requested={round_idx})."
                )
        checkpoint_path = _global_checkpoint_for_run(run_dir, selected_mode=selected_mode, override=args.checkpoint)
        config, config_source = _find_run_config(checkpoint=checkpoint_path, explicit=args.config, run_dir=run_dir)
        report = _evaluate_personalized(
            args=args,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            round_idx=round_idx,
            selected_mode=selected_mode,
            summary=summary,
            config=config,
            config_source=config_source,
            device=device,
        )
        out_path = args.output or f"bfl_eval_personalized_{args.split}_round_{round_idx:03d}.json"
        print(json.dumps(report["client_macro_metrics_pre"], indent=2))
        if "client_macro_metrics_post" in report:
            print("post-calibration client macro:")
            print(json.dumps(report["client_macro_metrics_post"], indent=2))
    else:
        if args.round is not None:
            raise SystemExit("--round requires --personalized-run-dir.")
        if not args.checkpoint:
            raise SystemExit("Either --checkpoint or --personalized-run-dir is required.")

        args.bootstrap_n = 1000 if args.bootstrap_n is None else int(args.bootstrap_n)
        if args.bootstrap_n < 0:
            raise SystemExit("--bootstrap-n must be >= 0")

        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_file():
            raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
        ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
        config, config_source = _find_run_config(checkpoint=checkpoint_path, explicit=args.config)
        model, model_options = _build_model(ckpt, config=config, device=device)

        files = list_npz_files(args.data_dir, args.split)
        report = evaluate_split(
            model=model,
            files=files,
            mc_eval=int(args.mc_eval),
            device=device,
            temperature=(float(args.temperature) if args.temperature is not None else None),
            threshold=float(args.threshold),
            fixed_threshold=float(args.threshold),
            bootstrap_n=int(args.bootstrap_n),
            bootstrap_seed=int(args.bootstrap_seed),
            save_pred_path=(str(args.save_pred_npz) if args.save_pred_npz else None),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
        )
        report.update(
            {
                "evaluation_mode": "global",
                "model_scope": "one global posterior on the pooled split",
                "checkpoint": str(checkpoint_path),
                "config_source": config_source,
                "model_options": model_options,
                "split": str(args.split),
            }
        )
        # Keep the legacy default filename for backward compatibility.
        out_path = args.output or f"bfl_eval_{args.split}.json"
        print(json.dumps(report["metrics_pre"], indent=2))
        if "metrics_post" in report:
            print("post-calibration:")
            print(json.dumps(report["metrics_post"], indent=2))

    write_json(out_path, report)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
