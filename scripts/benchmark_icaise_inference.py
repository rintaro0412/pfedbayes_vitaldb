from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Ensure project root in sys.path when executed as `python scripts/benchmark_icaise_inference.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bayes_federated.eval import mc_predict
from bayes_federated.models import BFLModel
from common.dataset import WindowedNPZDataset, list_client_ids, list_npz_files
from common.ioh_model import IOHNet, normalize_model_cfg
from common.metrics import compute_binary_metrics, sigmoid_np


@dataclass(frozen=True)
class MethodSpec:
    method: str
    run_dir: str
    kind: str
    mc_eval: int = 1
    config_path: str = ""


DEFAULT_METHODS: tuple[MethodSpec, ...] = (
    MethodSpec("Centralized", "runs/centralized/seed0", "point", config_path="configs/centralized.yaml"),
    MethodSpec("Local", "runs/local/seed42", "local_point", config_path="configs/local.yaml"),
    MethodSpec("FedAvg", "runs/fedavg/seed42", "point", config_path="configs/fedavg.yaml"),
    MethodSpec("FedProx", "runs/fedprox/seed42", "point", config_path="configs/fedprox.yaml"),
    MethodSpec("SCAFFOLD", "runs/scaffold/seed42", "point", config_path="configs/scaffold.yaml"),
    MethodSpec("FedNova", "runs/fednova/seed42", "point", config_path="configs/fednova.yaml"),
    MethodSpec("Per-FedAvg", "runs/perfedavg/seed42", "point", config_path="configs/perfedavg.yaml"),
    MethodSpec("pFedMe", "runs/pfedme/seed42", "point", config_path="configs/pfedme.yaml"),
    MethodSpec("pFedBayes", "runs/pfedbayes/seed42", "bayes", 25, "configs/pfedbayes.yaml"),
)


def _device_from_arg(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    return device


def _ru_maxrss_mb() -> float:
    # Linux reports ru_maxrss in KiB. macOS reports bytes; this environment is Linux.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_mb(device: torch.device, cpu_before_mb: float) -> Dict[str, float]:
    cpu_peak = max(0.0, _ru_maxrss_mb() - float(cpu_before_mb))
    out = {"cpu_peak_delta_mb": float(cpu_peak), "gpu_peak_allocated_mb": 0.0, "gpu_peak_reserved_mb": 0.0}
    if device.type == "cuda":
        out["gpu_peak_allocated_mb"] = float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
        out["gpu_peak_reserved_mb"] = float(torch.cuda.max_memory_reserved(device)) / (1024.0**2)
    return out


def _make_loader(
    files: Sequence[str],
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_batches: int,
) -> tuple[DataLoader, int | None]:
    ds = WindowedNPZDataset(files, use_clin="true", cache_in_memory=False, max_cache_files=32, cache_dtype="float32")
    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(num_workers) > 0),
    )
    n_limit = None
    if int(max_batches) > 0:
        n_limit = min(len(ds), int(max_batches) * int(batch_size))
    return dl, n_limit


def _checkpoint_path(run_dir: Path) -> Path:
    best = run_dir / "checkpoints" / "model_best.pt"
    if best.exists():
        return best
    last = run_dir / "checkpoints" / "model_last.pt"
    if last.exists():
        return last
    raise FileNotFoundError(f"checkpoint not found under {run_dir / 'checkpoints'}")


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _load_point_model(ckpt_path: Path, device: torch.device) -> IOHNet:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = IOHNet(normalize_model_cfg(ckpt.get("model_cfg", {}))).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


def _load_bayes_model(ckpt_path: Path, device: torch.device, *, config_path: Path | None = None) -> BFLModel:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = _load_yaml(config_path) if config_path is not None else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    model = BFLModel(
        normalize_model_cfg(ckpt.get("model_cfg", {})),
        prior_sigma=float(model_cfg.get("prior_sigma", 0.1)),
        logvar_min=float(model_cfg.get("logvar_min", -12.0)),
        logvar_max=float(model_cfg.get("logvar_max", 6.0)),
        full_bayes=bool(ckpt.get("full_bayes", False)),
        param_type=str(model_cfg.get("param_type", "logvar")),
        mu_init=str(model_cfg.get("mu_init", "zeros")),
        init_rho=(float(model_cfg["init_rho"]) if model_cfg.get("init_rho") is not None else None),
        var_reduction_h=float(model_cfg.get("var_reduction_h", 1.0)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def _predict_point(
    model: IOHNet,
    dl: DataLoader,
    *,
    device: torch.device,
    max_batches: int,
) -> tuple[np.ndarray, np.ndarray]:
    logits_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    for batch_idx, (x, y) in enumerate(dl):
        if int(max_batches) > 0 and batch_idx >= int(max_batches):
            break
        if isinstance(x, (tuple, list)):
            x = tuple(t.to(device, non_blocking=True) for t in x)
        else:
            x = x.to(device, non_blocking=True)
        logits = model(x).detach().cpu().view(-1).numpy()
        logits_all.append(logits)
        y_all.append(y.detach().cpu().view(-1).numpy())
    if not logits_all:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.concatenate(logits_all, axis=0), np.concatenate(y_all, axis=0)


def _point_once(
    *,
    model: IOHNet,
    dl: DataLoader,
    device: torch.device,
    max_batches: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    _sync(device)
    started = time.perf_counter()
    logits, y = _predict_point(model, dl, device=device, max_batches=max_batches)
    _sync(device)
    return float(time.perf_counter() - started), logits, y


def _bayes_once(
    *,
    model: BFLModel,
    dl: DataLoader,
    device: torch.device,
    mc_eval: int,
    max_batches: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    # mc_predict does not expose max_batches, so use a small wrapper DataLoader iterator when needed.
    if int(max_batches) > 0:
        limited_batches = []
        for idx, batch in enumerate(dl):
            if idx >= int(max_batches):
                break
            limited_batches.append(batch)

        class _Limited:
            def __iter__(self):
                return iter(limited_batches)

        eval_dl: Iterable[Any] = _Limited()
    else:
        eval_dl = dl
    _sync(device)
    started = time.perf_counter()
    pred = mc_predict(model, eval_dl, mc_eval=int(mc_eval), device=device, temperature=None, return_y=True)  # type: ignore[arg-type]
    _sync(device)
    return float(time.perf_counter() - started), pred["prob_mean"], pred["y_true"]


def _summarize_times(times: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(times), dtype=np.float64)
    return {
        "repeats": int(arr.size),
        "time_s_mean": float(np.mean(arr)),
        "time_s_std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "time_s_min": float(np.min(arr)),
        "time_s_max": float(np.max(arr)),
    }


def _metrics_row(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0 or prob.size == 0:
        return {"auprc": float("nan"), "auroc": float("nan"), "brier": float("nan"), "nll": float("nan"), "ece": float("nan")}
    m = compute_binary_metrics(y_true.astype(int), prob.astype(float), n_bins=15)
    return {"auprc": float(m.auprc), "auroc": float(m.auroc), "brier": float(m.brier), "nll": float(m.nll), "ece": float(m.ece)}


def _benchmark_point_method(
    *,
    spec: MethodSpec,
    repo_root: Path,
    data_dir: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    repeats: int,
    warmup: int,
    max_batches: int,
    device: torch.device,
) -> Dict[str, Any]:
    run_dir = repo_root / spec.run_dir
    ckpt_path = _checkpoint_path(run_dir)
    model = _load_point_model(ckpt_path, device=device)
    files = list_npz_files(str(data_dir), split)
    if not files:
        raise FileNotFoundError(f"no files for {data_dir} split={split}")
    dl, n_limit = _make_loader(files, batch_size=batch_size, num_workers=num_workers, device=device, max_batches=max_batches)
    for _ in range(int(warmup)):
        _point_once(model=model, dl=dl, device=device, max_batches=max_batches)
    _reset_peak_memory(device)
    cpu_before = _ru_maxrss_mb()
    times: List[float] = []
    last_logits = np.zeros((0,), dtype=np.float32)
    last_y = np.zeros((0,), dtype=np.int64)
    for _ in range(int(repeats)):
        elapsed, last_logits, last_y = _point_once(model=model, dl=dl, device=device, max_batches=max_batches)
        times.append(elapsed)
    prob = sigmoid_np(last_logits)
    mem = _peak_memory_mb(device, cpu_before)
    n = int(last_y.size if n_limit is None else min(last_y.size, n_limit))
    row: Dict[str, Any] = {
        "method": spec.method,
        "kind": spec.kind,
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "data_dir": str(data_dir),
        "split": str(split),
        "device": str(device),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "mc_eval": 1,
        "n_samples": n,
        "throughput_samples_per_s_mean": float(n / max(np.mean(times), 1e-12)),
        "latency_ms_per_sample_mean": float(np.mean(times) * 1000.0 / max(n, 1)),
        **_summarize_times(times),
        **mem,
        **_metrics_row(last_y, prob),
    }
    return row


def _client_id_from_checkpoint(path: Path) -> str:
    name = path.name
    prefix = "client_"
    for suffix in ("_best.pt", "_last.pt"):
        if name.startswith(prefix) and name.endswith(suffix):
            return name[len(prefix) : -len(suffix)]
    raise ValueError(f"cannot parse client id from {path}")


def _benchmark_local_method(
    *,
    spec: MethodSpec,
    repo_root: Path,
    data_dir: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    repeats: int,
    warmup: int,
    max_batches: int,
    device: torch.device,
    per_client_csv: Path,
) -> Dict[str, Any]:
    run_dir = repo_root / spec.run_dir
    ckpt_dir = run_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("client_*_best.pt"))
    if not ckpts:
        ckpts = sorted(ckpt_dir.glob("client_*_last.pt"))
    if not ckpts:
        raise FileNotFoundError(f"local client checkpoints not found under {ckpt_dir}")

    client_rows: List[Dict[str, Any]] = []
    total_times = [0.0 for _ in range(int(repeats))]
    y_all: List[np.ndarray] = []
    p_all: List[np.ndarray] = []
    total_n = 0
    _reset_peak_memory(device)
    cpu_before = _ru_maxrss_mb()

    valid_clients = set(list_client_ids(str(data_dir)))
    for ckpt_path in ckpts:
        cid = _client_id_from_checkpoint(ckpt_path)
        if valid_clients and cid not in valid_clients:
            continue
        files = list_npz_files(str(data_dir), split, client_id=cid)
        if not files:
            continue
        model = _load_point_model(ckpt_path, device=device)
        dl, _ = _make_loader(files, batch_size=batch_size, num_workers=num_workers, device=device, max_batches=max_batches)
        for _ in range(int(warmup)):
            _point_once(model=model, dl=dl, device=device, max_batches=max_batches)
        last_logits = np.zeros((0,), dtype=np.float32)
        last_y = np.zeros((0,), dtype=np.int64)
        client_times: List[float] = []
        for repeat_idx in range(int(repeats)):
            elapsed, last_logits, last_y = _point_once(model=model, dl=dl, device=device, max_batches=max_batches)
            client_times.append(elapsed)
            total_times[repeat_idx] += elapsed
        prob = sigmoid_np(last_logits)
        total_n += int(last_y.size)
        y_all.append(last_y)
        p_all.append(prob)
        client_rows.append(
            {
                "method": spec.method,
                "client_id": cid,
                "checkpoint": str(ckpt_path),
                "n_samples": int(last_y.size),
                "time_s_mean": float(np.mean(client_times)),
                "throughput_samples_per_s_mean": float(last_y.size / max(np.mean(client_times), 1e-12)),
                "latency_ms_per_sample_mean": float(np.mean(client_times) * 1000.0 / max(last_y.size, 1)),
                **_metrics_row(last_y, prob),
            }
        )
        del model
        gc.collect()

    if total_n <= 0:
        raise RuntimeError("no local client samples evaluated")
    y = np.concatenate(y_all, axis=0)
    prob = np.concatenate(p_all, axis=0)
    mem = _peak_memory_mb(device, cpu_before)
    client_csv = per_client_csv
    client_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(client_rows).to_csv(client_csv, index=False)
    return {
        "method": spec.method,
        "kind": spec.kind,
        "run_dir": str(run_dir),
        "checkpoint": "per-client model_best.pt",
        "data_dir": str(data_dir),
        "split": str(split),
        "device": str(device),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "mc_eval": 1,
        "n_samples": int(total_n),
        "n_clients": int(len(client_rows)),
        "per_client_csv": str(client_csv),
        "throughput_samples_per_s_mean": float(total_n / max(np.mean(total_times), 1e-12)),
        "latency_ms_per_sample_mean": float(np.mean(total_times) * 1000.0 / max(total_n, 1)),
        **_summarize_times(total_times),
        **mem,
        **_metrics_row(y, prob),
    }


def _benchmark_bayes_method(
    *,
    spec: MethodSpec,
    repo_root: Path,
    data_dir: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    repeats: int,
    warmup: int,
    max_batches: int,
    device: torch.device,
) -> Dict[str, Any]:
    run_dir = repo_root / spec.run_dir
    ckpt_path = _checkpoint_path(run_dir)
    config_path = (repo_root / spec.config_path).resolve() if spec.config_path else None
    model = _load_bayes_model(ckpt_path, device=device, config_path=config_path)
    files = list_npz_files(str(data_dir), split)
    if not files:
        raise FileNotFoundError(f"no files for {data_dir} split={split}")
    dl, n_limit = _make_loader(files, batch_size=batch_size, num_workers=num_workers, device=device, max_batches=max_batches)
    for _ in range(int(warmup)):
        _bayes_once(model=model, dl=dl, device=device, mc_eval=int(spec.mc_eval), max_batches=max_batches)
    _reset_peak_memory(device)
    cpu_before = _ru_maxrss_mb()
    times: List[float] = []
    last_prob = np.zeros((0,), dtype=np.float32)
    last_y = np.zeros((0,), dtype=np.int64)
    for _ in range(int(repeats)):
        elapsed, last_prob, last_y = _bayes_once(model=model, dl=dl, device=device, mc_eval=int(spec.mc_eval), max_batches=max_batches)
        times.append(elapsed)
    mem = _peak_memory_mb(device, cpu_before)
    n = int(last_y.size if n_limit is None else min(last_y.size, n_limit))
    return {
        "method": spec.method,
        "kind": spec.kind,
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "data_dir": str(data_dir),
        "split": str(split),
        "device": str(device),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "mc_eval": int(spec.mc_eval),
        "n_samples": n,
        "throughput_samples_per_s_mean": float(n / max(np.mean(times), 1e-12)),
        "latency_ms_per_sample_mean": float(np.mean(times) * 1000.0 / max(n, 1)),
        **_summarize_times(times),
        **mem,
        **_metrics_row(last_y, last_prob),
    }


def _parse_method_specs(path: Path | None) -> Sequence[MethodSpec]:
    if path is None:
        return DEFAULT_METHODS
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("methods")
    if not isinstance(data, list):
        raise ValueError("--methods-json must contain a list or an object with methods")
    specs: List[MethodSpec] = []
    for row in data:
        if not isinstance(row, dict):
            raise ValueError("method spec rows must be objects")
        label = row.get("method", row.get("label"))
        run_dir = row.get("run_dir", row.get("source_run_dir"))
        if not label or not run_dir:
            raise ValueError("method spec requires method/label and run_dir/source_run_dir")
        specs.append(
            MethodSpec(
                method=str(label),
                run_dir=str(run_dir),
                kind=str(row.get("inference_kind", row.get("kind", "point"))),
                mc_eval=int(row.get("mc_eval", 1)),
                config_path=str(row.get("config_path", row.get("config", ""))),
            )
        )
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark inference latency and memory from existing ICAISE checkpoints.")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--data-dir", default="federated_data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--methods-json", default=None, help="Optional JSON list with method/run_dir/kind/mc_eval entries.")
    ap.add_argument("--methods", default="", help="Comma-separated method names to include. Default: all.")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-batches", type=int, default=0, help="Limit batches per method for smoke tests. 0 means full split.")
    ap.add_argument("--out-csv", default="tables/icaise_inference_benchmark.csv")
    ap.add_argument("--out-json", default="tables/icaise_inference_benchmark.json")
    ap.add_argument("--fail-on-error", action="store_true", help="Return failure after saving diagnostics if any selected method failed.")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    data_dir = (repo_root / args.data_dir).resolve()
    out_csv = (repo_root / args.out_csv).resolve()
    device = _device_from_arg(str(args.device))
    methods_json = Path(args.methods_json).resolve() if args.methods_json else None
    specs = list(_parse_method_specs(methods_json))
    include = {m.strip() for m in str(args.methods).split(",") if m.strip()}
    if include:
        specs = [s for s in specs if s.method in include]
    if not specs:
        raise SystemExit("No methods selected.")

    torch.set_grad_enabled(False)
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for spec in specs:
        started = time.perf_counter()
        try:
            if spec.kind == "point":
                row = _benchmark_point_method(
                    spec=spec,
                    repo_root=repo_root,
                    data_dir=data_dir,
                    split=str(args.split),
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                    repeats=int(args.repeats),
                    warmup=int(args.warmup),
                    max_batches=int(args.max_batches),
                    device=device,
                )
            elif spec.kind == "local_point":
                row = _benchmark_local_method(
                    spec=spec,
                    repo_root=repo_root,
                    data_dir=data_dir,
                    split=str(args.split),
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                    repeats=int(args.repeats),
                    warmup=int(args.warmup),
                    max_batches=int(args.max_batches),
                    device=device,
                    per_client_csv=out_csv.with_name(f"{out_csv.stem}_local_per_client.csv"),
                )
            elif spec.kind == "bayes":
                row = _benchmark_bayes_method(
                    spec=spec,
                    repo_root=repo_root,
                    data_dir=data_dir,
                    split=str(args.split),
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                    repeats=int(args.repeats),
                    warmup=int(args.warmup),
                    max_batches=int(args.max_batches),
                    device=device,
                )
            else:
                raise ValueError(f"unknown kind: {spec.kind}")
            row["benchmark_wall_time_s"] = float(time.perf_counter() - started)
            rows.append(row)
            print(f"[OK] {spec.method}: {row['time_s_mean']:.4f}s mean, {row['latency_ms_per_sample_mean']:.4f} ms/sample")
        except Exception as exc:
            warning = f"{spec.method}: {type(exc).__name__}: {exc}"
            warnings.append(warning)
            print(f"[WARN] {warning}")

    if not rows:
        raise SystemExit("No benchmark rows were produced.")
    out_csv = (repo_root / args.out_csv).resolve()
    out_json = (repo_root / args.out_json).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    payload = {
        "rows": rows,
        "warnings": warnings,
        "config": {
            "data_dir": str(data_dir),
            "split": str(args.split),
            "device": str(device),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "warmup": int(args.warmup),
            "repeats": int(args.repeats),
            "max_batches": int(args.max_batches),
            "pid": int(os.getpid()),
        },
        "notes": [
            "Model loading time is excluded from measured inference time.",
            "Local uses each client-specific checkpoint on that client's test split.",
            "CPU peak memory is reported as process ru_maxrss delta and should be treated as approximate.",
            "GPU memory uses torch.cuda max allocated/reserved when CUDA is used.",
        ],
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved CSV: {out_csv}")
    print(f"Saved JSON: {out_json}")
    if warnings:
        print("[WARNINGS]")
        for warning in warnings:
            print(f"- {warning}")
        if args.fail_on_error:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
