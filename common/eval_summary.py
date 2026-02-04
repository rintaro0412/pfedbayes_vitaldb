from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

from common.io import ensure_dir, write_json
from common.metrics import BinaryMetrics, compute_binary_metrics

METRIC_KEYS = ("auroc", "auprc", "ece", "nll", "brier")


def metrics_from_binary(y_true: np.ndarray, prob: np.ndarray, *, n_bins: int = 15) -> Dict[str, float]:
    m: BinaryMetrics = compute_binary_metrics(y_true, prob, n_bins=int(n_bins))
    out = asdict(m)
    return {k: float(out[k]) for k in out}


def _filter_clients(per_client: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for cid, row in per_client.items():
        try:
            n = int(row.get("n", 0))
        except Exception:
            n = 0
        if n <= 0:
            continue
        out[str(cid)] = row
    return out


def client_stats(per_client: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    per_client = _filter_clients(per_client)
    vals: Dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
    for row in per_client.values():
        for k in METRIC_KEYS:
            try:
                vals[k].append(float(row.get(k, float("nan"))))
            except Exception:
                vals[k].append(float("nan"))

    macro: Dict[str, float] = {}
    std: Dict[str, float] = {}
    minv: Dict[str, float] = {}
    for k, arr in vals.items():
        if not arr:
            macro[k] = float("nan")
            std[k] = float("nan")
            minv[k] = float("nan")
            continue
        a = np.asarray(arr, dtype=np.float64)
        macro[k] = float(np.nanmean(a))
        std[k] = float(np.nanstd(a))
        minv[k] = float(np.nanmin(a))

    return {
        "macro": macro,
        "std": std,
        "min": minv,
        "n_clients": int(len(per_client)),
    }


def build_mode_report(
    *,
    overall: Dict[str, Any],
    per_client: Dict[str, Dict[str, Any]],
    note: str | None = None,
) -> Dict[str, Any]:
    report = {
        "overall": overall,
        "per_client": per_client,
        "client_stats": client_stats(per_client),
    }
    if note:
        report["note"] = str(note)
    return report


def write_eval_outputs(
    *,
    run_dir: str | Path,
    algo: str,
    modes: Dict[str, Dict[str, Any]],
    prefix: str = "eval_modes",
) -> None:
    run_dir = Path(run_dir)
    ensure_dir(run_dir)
    write_json(run_dir / f"{prefix}.json", {"algo": str(algo), "modes": modes})

    # Overall CSV
    overall_path = run_dir / f"{prefix}_overall.csv"
    with overall_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "n", "n_pos", "n_neg", *METRIC_KEYS])
        for mode, rep in modes.items():
            overall = rep.get("overall", {})
            row = [
                str(mode),
                overall.get("n", ""),
                overall.get("n_pos", ""),
                overall.get("n_neg", ""),
            ]
            for k in METRIC_KEYS:
                row.append(overall.get(k, ""))
            writer.writerow(row)

    # Per-client CSV
    per_client_path = run_dir / f"{prefix}_per_client.csv"
    with per_client_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "client_id", "n", "n_pos", "n_neg", *METRIC_KEYS])
        for mode, rep in modes.items():
            per_client = rep.get("per_client", {})
            for cid, row in per_client.items():
                out_row = [
                    str(mode),
                    str(cid),
                    row.get("n", ""),
                    row.get("n_pos", ""),
                    row.get("n_neg", ""),
                ]
                for k in METRIC_KEYS:
                    out_row.append(row.get(k, ""))
                writer.writerow(out_row)

    # Client stats CSV
    stats_path = run_dir / f"{prefix}_client_stats.csv"
    with stats_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "stat", *METRIC_KEYS, "n_clients"])
        for mode, rep in modes.items():
            stats = rep.get("client_stats", {})
            for stat_name in ("macro", "min", "std"):
                row_dict = stats.get(stat_name, {})
                row = [str(mode), stat_name]
                for k in METRIC_KEYS:
                    row.append(row_dict.get(k, ""))
                row.append(stats.get("n_clients", ""))
                writer.writerow(row)

