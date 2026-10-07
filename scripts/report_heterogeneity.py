from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Ensure project root in sys.path when executed as `python scripts/report_heterogeneity.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.dataset import list_client_ids, list_npz_files, parse_caseid_from_path
from common.experiment import make_run_dir, save_env_snapshot, seed_everything
from common.io import now_utc_iso, write_json


CLIN_COLS = [
    "age",
    "sex_M",
    "bmi",
    "asa",
    "emop",
    "preop_htn",
    "preop_hb",
    "preop_bun",
    "preop_cr",
    "preop_alb",
    "preop_na",
    "preop_k",
]

WAVE_COLS_FALLBACK = ["wave_0", "wave_1", "wave_2", "wave_3"]
WAVE_COLS_IOH = ["hr", "spo2", "etco2", "fio2"]
WAVE_STATS = ["mean", "std", "min", "max", "q25", "q50", "q75", "absdiff"]


def _load_splits(text: str) -> list[str]:
    out = [s.strip() for s in str(text).split(",") if s.strip()]
    if not out:
        raise ValueError("No splits specified.")
    return out


def _wave_feature_names(n_channels: int) -> list[str]:
    base = WAVE_COLS_IOH if int(n_channels) == 4 else [f"wave_{i}" for i in range(int(n_channels))]
    names: list[str] = []
    for ch in base:
        for stat in WAVE_STATS:
            names.append(f"{ch}_{stat}")
    return names


def _clin_feature_names(n_cols: int) -> list[str]:
    if int(n_cols) <= len(CLIN_COLS):
        return CLIN_COLS[: int(n_cols)]
    out = list(CLIN_COLS)
    for i in range(len(CLIN_COLS), int(n_cols)):
        out.append(f"clin_{i:02d}")
    return out


def _extract_wave_features(x_wave: np.ndarray) -> np.ndarray:
    x = np.asarray(x_wave, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"x_wave must be (N,C,T), got {x.shape}")
    q25, q50, q75 = np.quantile(x, q=[0.25, 0.5, 0.75], axis=2)
    absdiff = np.abs(np.diff(x, axis=2)).mean(axis=2)
    blocks = [
        x.mean(axis=2),
        x.std(axis=2),
        x.min(axis=2),
        x.max(axis=2),
        q25,
        q50,
        q75,
        absdiff,
    ]
    feat = np.concatenate(blocks, axis=1)
    return np.asarray(feat, dtype=np.float32)


def _sanitize_features(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32)
    if not np.isfinite(out).all():
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _eta_squared(x: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    grand = float(np.mean(x))
    ss_total = float(np.sum((x - grand) ** 2))
    if ss_total <= 0:
        return 0.0
    ss_between = 0.0
    for cls in range(int(n_classes)):
        mask = y == cls
        if not np.any(mask):
            continue
        mu = float(np.mean(x[mask]))
        ss_between += float(mask.sum()) * (mu - grand) ** 2
    return float(max(0.0, min(1.0, ss_between / ss_total)))


def _pairwise_sqdist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xx = np.sum(x * x, axis=1, keepdims=True)
    yy = np.sum(y * y, axis=1, keepdims=True).T
    d2 = xx + yy - 2.0 * (x @ y.T)
    return np.maximum(d2, 0.0)


def _median_bandwidth(x: np.ndarray, *, rng: np.random.Generator, max_samples: int) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] > int(max_samples):
        idx = rng.choice(x.shape[0], size=int(max_samples), replace=False)
        x = x[idx]
    d2 = _pairwise_sqdist(x, x)
    tri = d2[np.triu_indices_from(d2, k=1)]
    tri = tri[np.isfinite(tri) & (tri > 0)]
    if tri.size == 0:
        return 1.0
    med = float(np.median(tri))
    if not np.isfinite(med) or med <= 0:
        return 1.0
    return med


def _rbf_mmd2_biased(x: np.ndarray, y: np.ndarray, *, gamma: float) -> float:
    kxx = np.exp(-gamma * _pairwise_sqdist(x, x))
    kyy = np.exp(-gamma * _pairwise_sqdist(y, y))
    kxy = np.exp(-gamma * _pairwise_sqdist(x, y))
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_distribution(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"min": float("nan"), "p25": float("nan"), "median": float("nan"), "p75": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan")}
    return {
        "min": float(np.min(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "p75": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _build_markdown(
    *,
    report: Dict[str, Any],
    top_feature_rows: Sequence[Dict[str, Any]],
    mmd_rows: Sequence[Dict[str, Any]],
    clf_rows: Sequence[Dict[str, Any]],
) -> str:
    overview = report["overview"]
    label = report["label_distribution"]
    size = report["client_size_distribution"]
    nearest = sorted(mmd_rows, key=lambda r: float(r["mmd2"]))[:5]
    farthest = sorted(mmd_rows, key=lambda r: float(r["mmd2"]), reverse=True)[:5]
    lines = [
        "# Heterogeneity Report",
        "",
        f"- generated_utc: {report['finished_utc']}",
        f"- data_dir: {report['data_dir']}",
        f"- splits: {', '.join(report['splits'])}",
        f"- clients: {overview['n_clients']}",
        f"- samples: {overview['n_samples']}",
        f"- feature_dim: {overview['feature_dim']}",
        "",
        "## Overview",
        "",
        f"- client_size_cv: {size['cv']:.4f}",
        f"- client_size_gini: {size['gini']:.4f}",
        f"- global_positive_rate: {label['global_positive_rate']:.4f}",
        f"- client_positive_rate_std: {label['client_positive_rate_std_unweighted']:.4f}",
        f"- weighted_abs_deviation_from_global: {label['weighted_abs_deviation']:.4f}",
        "",
        "## Client Identification",
        "",
        f"- folds: {report['client_identification']['n_folds']}",
        f"- accuracy_mean: {report['client_identification']['accuracy_mean']:.4f}",
        f"- balanced_accuracy_mean: {report['client_identification']['balanced_accuracy_mean']:.4f}",
        f"- macro_f1_mean: {report['client_identification']['macro_f1_mean']:.4f}",
        f"- majority_baseline: {report['client_identification']['majority_baseline']:.4f}",
        f"- chance_baseline: {report['client_identification']['chance_baseline']:.4f}",
        "",
        "## Top Features",
        "",
        "| rank | feature | eta_squared | |",
        "|---:|---|---:|---|",
    ]
    for i, row in enumerate(top_feature_rows[:10], start=1):
        lines.append(f"| {i} | {row['feature']} | {float(row['eta_squared']):.4f} | |")
    lines.extend(
        [
            "",
            "## Pairwise MMD",
            "",
            f"- kernel_bandwidth_sq_median: {report['mmd']['kernel_bandwidth_sq_median']:.6f}",
            f"- mmd_mean: {report['mmd']['mmd_mean']:.6f}",
            f"- mmd_std: {report['mmd']['mmd_std']:.6f}",
            "",
            "### Nearest Pairs",
            "",
        ]
    )
    for row in nearest:
        lines.append(f"- {row['client_a']} vs {row['client_b']}: {float(row['mmd2']):.6f}")
    lines.extend(["", "### Farthest Pairs", ""])
    for row in farthest:
        lines.append(f"- {row['client_a']} vs {row['client_b']}: {float(row['mmd2']):.6f}")
    lines.extend(["", "## Fold Metrics", ""])
    for row in clf_rows:
        lines.append(
            f"- fold {row['fold']}: accuracy={float(row['accuracy']):.4f}, "
            f"balanced_accuracy={float(row['balanced_accuracy']):.4f}, macro_f1={float(row['macro_f1']):.4f}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Report client heterogeneity with feature summaries, MMD, and client identification accuracy")
    ap.add_argument("--data-dir", default="federated_data")
    ap.add_argument("--splits", default="train,val,test", help="Comma-separated splits to include")
    ap.add_argument("--out-dir", default="runs/heterogeneity")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mmd-max-samples-per-client", type=int, default=256)
    ap.add_argument("--mmd-bandwidth-samples", type=int, default=2048)
    ap.add_argument("--classifier-folds", type=int, default=5)
    args = ap.parse_args()

    splits = _load_splits(args.splits)
    seed_everything(int(args.seed), deterministic=True)
    rng = np.random.default_rng(int(args.seed))

    run_dir = make_run_dir(args.out_dir, args.run_name, resume=False)
    cfg = {
        "data_dir": str(args.data_dir),
        "splits": list(splits),
        "seed": int(args.seed),
        "mmd_max_samples_per_client": int(args.mmd_max_samples_per_client),
        "mmd_bandwidth_samples": int(args.mmd_bandwidth_samples),
        "classifier_folds": int(args.classifier_folds),
    }
    save_env_snapshot(run_dir, cfg)

    clients = sorted(list_client_ids(str(args.data_dir)))
    if not clients:
        raise SystemExit(f"No clients found under {args.data_dir}")

    features_parts: list[np.ndarray] = []
    labels_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    groups_parts: list[np.ndarray] = []
    sample_client_parts: list[np.ndarray] = []
    sample_split_parts: list[np.ndarray] = []
    case_rows: list[Dict[str, Any]] = []
    file_rows: list[Dict[str, Any]] = []
    wave_feature_names: list[str] | None = None
    clin_feature_names: list[str] | None = None
    total_files = 0

    for client_idx, client_id in enumerate(clients):
        for split in splits:
            files = list_npz_files(str(args.data_dir), split, client_id=client_id)
            total_files += len(files)
            for path_str in files:
                path = Path(path_str)
                caseid = parse_caseid_from_path(str(path))
                if caseid is None:
                    raise ValueError(f"Could not parse caseid from {path}")
                with np.load(path, allow_pickle=False) as z:
                    x_wave = np.asarray(z["x_wave"], dtype=np.float32)
                    y = np.asarray(z["y"], dtype=np.int64).reshape(-1)
                    x_clin = np.asarray(z["x_clin"], dtype=np.float32) if "x_clin" in z else None
                if x_wave.shape[0] != y.shape[0]:
                    raise ValueError(f"x_wave/y sample mismatch in {path}")
                wave_feat = _extract_wave_features(x_wave)
                if wave_feature_names is None:
                    wave_feature_names = _wave_feature_names(int(x_wave.shape[1]))
                if x_clin is not None:
                    if x_clin.ndim != 2 or x_clin.shape[0] != y.shape[0]:
                        raise ValueError(f"x_clin shape mismatch in {path}: {x_clin.shape}")
                    if clin_feature_names is None:
                        clin_feature_names = _clin_feature_names(int(x_clin.shape[1]))
                    feat = np.concatenate([wave_feat, np.asarray(x_clin, dtype=np.float32)], axis=1)
                else:
                    feat = wave_feat
                    if clin_feature_names is None:
                        clin_feature_names = []
                feat = _sanitize_features(feat)

                features_parts.append(feat)
                labels_parts.append(np.full(y.shape[0], int(client_idx), dtype=np.int64))
                y_parts.append(y.astype(np.int64, copy=False))
                groups_parts.append(np.full(y.shape[0], int(caseid), dtype=np.int64))
                sample_client_parts.append(np.asarray([client_id] * y.shape[0], dtype=object))
                sample_split_parts.append(np.asarray([split] * y.shape[0], dtype=object))

                pos = int((y > 0).sum())
                case_rows.append(
                    {
                        "client_id": client_id,
                        "split": split,
                        "caseid": int(caseid),
                        "n_samples": int(y.shape[0]),
                        "n_pos": int(pos),
                        "n_neg": int(y.shape[0] - pos),
                        "positive_rate": float(pos / max(int(y.shape[0]), 1)),
                    }
                )
                file_rows.append(
                    {
                        "client_id": client_id,
                        "split": split,
                        "caseid": int(caseid),
                        "path": str(path),
                        "n_samples": int(y.shape[0]),
                    }
                )

    if not features_parts:
        raise SystemExit("No samples found. Check --data-dir and --splits.")

    feature_names = list(wave_feature_names or []) + list(clin_feature_names or [])
    x_all = np.concatenate(features_parts, axis=0)
    label_idx = np.concatenate(labels_parts, axis=0)
    y_all = np.concatenate(y_parts, axis=0)
    group_ids = np.concatenate(groups_parts, axis=0)
    sample_clients = np.concatenate(sample_client_parts, axis=0)
    sample_splits = np.concatenate(sample_split_parts, axis=0)

    if x_all.shape[1] != len(feature_names):
        raise RuntimeError(f"feature dimension mismatch: {x_all.shape[1]} != {len(feature_names)}")

    n_clients = len(clients)
    n_samples = int(x_all.shape[0])

    per_client_rows: list[Dict[str, Any]] = []
    per_feature_rows: list[Dict[str, Any]] = []
    client_prev: list[float] = []
    client_sizes: list[int] = []
    global_pos_rate = float(np.mean(y_all > 0))

    for client_idx, client_id in enumerate(clients):
        mask = label_idx == int(client_idx)
        x_c = x_all[mask]
        y_c = y_all[mask]
        client_sizes.append(int(mask.sum()))
        prev = float(np.mean(y_c > 0))
        client_prev.append(prev)
        row: Dict[str, Any] = {
            "client_id": client_id,
            "n_samples": int(mask.sum()),
            "n_cases": int(np.unique(group_ids[mask]).size),
            "positive_rate": prev,
        }
        mu = np.mean(x_c, axis=0)
        sd = np.std(x_c, axis=0)
        for name, val in zip(feature_names, mu.tolist()):
            row[f"{name}__mean"] = float(val)
        for name, val in zip(feature_names, sd.tolist()):
            row[f"{name}__std"] = float(val)
        per_client_rows.append(row)

    for feat_idx, feat_name in enumerate(feature_names):
        eta = _eta_squared(x_all[:, feat_idx], label_idx, n_clients)
        per_feature_rows.append(
            {
                "feature": feat_name,
                "eta_squared": float(eta),
                "global_mean": float(np.mean(x_all[:, feat_idx])),
                "global_std": float(np.std(x_all[:, feat_idx])),
            }
        )
    per_feature_rows.sort(key=lambda r: float(r["eta_squared"]), reverse=True)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_all)
    bandwidth_sq = _median_bandwidth(x_scaled, rng=rng, max_samples=int(args.mmd_bandwidth_samples))
    gamma = 1.0 / max(2.0 * bandwidth_sq, 1e-12)

    pairwise_mmd_rows: list[Dict[str, Any]] = []
    sampled_by_client: dict[int, np.ndarray] = {}
    for client_idx in range(n_clients):
        idx = np.flatnonzero(label_idx == int(client_idx))
        if idx.size > int(args.mmd_max_samples_per_client):
            idx = rng.choice(idx, size=int(args.mmd_max_samples_per_client), replace=False)
        sampled_by_client[int(client_idx)] = x_scaled[idx]
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            mmd2 = _rbf_mmd2_biased(sampled_by_client[i], sampled_by_client[j], gamma=gamma)
            pairwise_mmd_rows.append(
                {
                    "client_a": clients[i],
                    "client_b": clients[j],
                    "mmd2": float(mmd2),
                    "n_a": int(sampled_by_client[i].shape[0]),
                    "n_b": int(sampled_by_client[j].shape[0]),
                }
            )

    clf_pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced"),
    )
    n_splits = int(min(int(args.classifier_folds), np.unique(group_ids).size))
    if n_splits < 2:
        raise SystemExit("Not enough groups for cross-validation.")
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=int(args.seed))
    clf_rows: list[Dict[str, Any]] = []
    for fold_idx, (tr_idx, te_idx) in enumerate(sgkf.split(x_all, label_idx, groups=group_ids), start=1):
        clf_pipe.fit(x_all[tr_idx], label_idx[tr_idx])
        pred = clf_pipe.predict(x_all[te_idx])
        clf_rows.append(
            {
                "fold": int(fold_idx),
                "n_train": int(tr_idx.size),
                "n_test": int(te_idx.size),
                "accuracy": float(accuracy_score(label_idx[te_idx], pred)),
                "balanced_accuracy": float(balanced_accuracy_score(label_idx[te_idx], pred)),
                "macro_f1": float(f1_score(label_idx[te_idx], pred, average="macro")),
            }
        )

    def _gini(xs: Sequence[int]) -> float:
        arr = np.asarray(list(xs), dtype=np.float64)
        if arr.size == 0 or float(arr.sum()) <= 0:
            return 0.0
        arr = np.sort(arr)
        idx = np.arange(1, arr.size + 1, dtype=np.float64)
        return float((2.0 * np.sum(idx * arr)) / (arr.size * np.sum(arr)) - (arr.size + 1.0) / arr.size)

    def _js_bernoulli(p: float, q: float) -> float:
        eps = 1e-12
        p = min(max(float(p), eps), 1.0 - eps)
        q = min(max(float(q), eps), 1.0 - eps)
        m = 0.5 * (p + q)
        return 0.5 * (p * math.log(p / m) + (1.0 - p) * math.log((1.0 - p) / (1.0 - m))) + 0.5 * (
            q * math.log(q / m) + (1.0 - q) * math.log((1.0 - q) / (1.0 - m))
        )

    weighted_abs_dev = float(np.average(np.abs(np.asarray(client_prev) - global_pos_rate), weights=np.asarray(client_sizes)))
    weighted_js = float(np.average([_js_bernoulli(p, global_pos_rate) for p in client_prev], weights=np.asarray(client_sizes)))
    client_size_cv = float(np.std(client_sizes) / max(np.mean(client_sizes), 1e-12))
    majority_baseline = float(max(client_sizes) / max(sum(client_sizes), 1))
    chance_baseline = float(1.0 / max(n_clients, 1))

    overview = {
        "n_clients": int(n_clients),
        "n_samples": int(n_samples),
        "n_cases": int(np.unique(group_ids).size),
        "n_files": int(total_files),
        "feature_dim": int(x_all.shape[1]),
        "wave_feature_dim": int(len(wave_feature_names or [])),
        "clinical_feature_dim": int(len(clin_feature_names or [])),
    }
    label_distribution = {
        "global_positive_rate": float(global_pos_rate),
        "client_positive_rate_min": float(np.min(client_prev)),
        "client_positive_rate_median": float(np.median(client_prev)),
        "client_positive_rate_max": float(np.max(client_prev)),
        "client_positive_rate_std_unweighted": float(np.std(client_prev)),
        "weighted_abs_deviation": float(weighted_abs_dev),
        "weighted_js_divergence_nats": float(weighted_js),
    }
    client_size_distribution = {
        "min": int(np.min(client_sizes)),
        "median": float(np.median(client_sizes)),
        "max": int(np.max(client_sizes)),
        "mean": float(np.mean(client_sizes)),
        "std": float(np.std(client_sizes)),
        "cv": float(client_size_cv),
        "gini": float(_gini(client_sizes)),
    }
    mmd_vals = [float(r["mmd2"]) for r in pairwise_mmd_rows]
    client_identification = {
        "n_folds": int(len(clf_rows)),
        "accuracy_mean": float(np.mean([float(r["accuracy"]) for r in clf_rows])),
        "accuracy_std": float(np.std([float(r["accuracy"]) for r in clf_rows])),
        "balanced_accuracy_mean": float(np.mean([float(r["balanced_accuracy"]) for r in clf_rows])),
        "balanced_accuracy_std": float(np.std([float(r["balanced_accuracy"]) for r in clf_rows])),
        "macro_f1_mean": float(np.mean([float(r["macro_f1"]) for r in clf_rows])),
        "macro_f1_std": float(np.std([float(r["macro_f1"]) for r in clf_rows])),
        "majority_baseline": float(majority_baseline),
        "chance_baseline": float(chance_baseline),
    }
    report: Dict[str, Any] = {
        "started_utc": now_utc_iso(),
        "data_dir": str(args.data_dir),
        "splits": list(splits),
        "overview": overview,
        "client_size_distribution": client_size_distribution,
        "label_distribution": label_distribution,
        "client_identification": client_identification,
        "mmd": {
            "kernel_bandwidth_sq_median": float(bandwidth_sq),
            "gamma": float(gamma),
            "mmd_mean": float(np.mean(mmd_vals)),
            "mmd_std": float(np.std(mmd_vals)),
            "mmd_min": float(np.min(mmd_vals)),
            "mmd_max": float(np.max(mmd_vals)),
        },
        "top_features": per_feature_rows[:20],
        "finished_utc": now_utc_iso(),
    }

    report_path = run_dir / "heterogeneity_report.json"
    write_json(report_path, report)
    _write_csv(run_dir / "per_client_feature_summary.csv", per_client_rows)
    _write_csv(run_dir / "feature_effects.csv", per_feature_rows)
    _write_csv(run_dir / "pairwise_mmd.csv", pairwise_mmd_rows)
    _write_csv(run_dir / "client_identification_folds.csv", clf_rows)
    _write_csv(run_dir / "case_summary.csv", case_rows)
    _write_csv(run_dir / "file_summary.csv", file_rows)

    markdown = _build_markdown(report=report, top_feature_rows=per_feature_rows, mmd_rows=pairwise_mmd_rows, clf_rows=clf_rows)
    (run_dir / "REPORT.md").write_text(markdown, encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
