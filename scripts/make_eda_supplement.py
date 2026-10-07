"""Generate robustness and heterogeneity supplements for hypoxemia EDA."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage
from scipy.spatial.distance import pdist, squareform

from make_eda_figures import (
    CHANNELS,
    COLORS,
    FEATURES,
    LABEL_NAMES,
    RAW_COLUMNS,
    markdown_table,
    save_figure,
    short_client,
    standardized_effect,
    style_axis,
    waveform_features,
)

CLINICAL = [
    "Age", "Sex=M", "BMI", "ASA", "Emergency", "Preop HTN",
    "Hb", "BUN", "Creatinine", "Albumin", "Na", "K",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate supplemental EDA figures for the hypoxemia dataset")
    p.add_argument("--data-dir", default="federated_data")
    p.add_argument("--raw-dir", default="vitaldb_data")
    p.add_argument("--client-stats", default="outputs_dataset_stats/federated_client_stats.csv")
    p.add_argument("--figure-dir", default="figures/eda")
    p.add_argument("--table-dir", default="tables/eda")
    p.add_argument("--cases-per-client-label", type=int, default=20)
    p.add_argument("--split-cases-per-client-label", type=int, default=6)
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--clusters", type=int, default=4)
    p.add_argument("--max-lag-sec", type=int, default=5)
    p.add_argument("--raw-cases-per-client", type=int, default=1)
    p.add_argument("--raw-max-minutes", type=int, default=60)
    return p.parse_args()


def sample_windows(
    data_dir: Path,
    split: str,
    *,
    cases_per_client_label: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    waves: list[np.ndarray] = []
    labels: list[int] = []
    clients_out: list[str] = []
    cases_out: list[str] = []
    clinical: list[np.ndarray] = []

    clients = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    for client in clients:
        files = sorted((data_dir / client / split).glob("*.npz"))
        order = rng.permutation(len(files)) if files else np.empty(0, dtype=int)
        counts = {0: 0, 1: 0}
        for file_idx in order.tolist():
            if all(counts[label] >= cases_per_client_label for label in (0, 1)):
                break
            path = files[int(file_idx)]
            try:
                with np.load(path, allow_pickle=False) as z:
                    y_arr = np.asarray(z["y"], dtype=np.int64)
                    x_wave = z["x_wave"]
                    x_clin = z["x_clin"]
                    for label in (0, 1):
                        if counts[label] >= cases_per_client_label:
                            continue
                        candidates = np.flatnonzero(y_arr == label)
                        if candidates.size == 0:
                            continue
                        selected = int(rng.choice(candidates))
                        wave = np.asarray(x_wave[selected], dtype=np.float32)
                        clin = np.asarray(x_clin[selected], dtype=np.float32)
                        if wave.shape != (4, 3000) or clin.ndim != 1:
                            continue
                        if not np.isfinite(wave).all() or not np.isfinite(clin).all():
                            continue
                        waves.append(wave.reshape(4, 30, 100).mean(axis=2))
                        labels.append(label)
                        clients_out.append(client)
                        cases_out.append(f"{client}/{path.stem}")
                        clinical.append(clin)
                        counts[label] += 1
            except (OSError, ValueError, KeyError):
                continue
    if not waves:
        raise RuntimeError(f"No windows sampled from {data_dir} split={split}")
    return {
        "waves": np.stack(waves).astype(np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "clients": np.asarray(clients_out, dtype=object),
        "cases": np.asarray(cases_out, dtype=object),
        "clinical": np.stack(clinical).astype(np.float32),
    }


def effect_matrix(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = np.full((4, 4), np.nan, dtype=np.float64)
    for ch in range(4):
        for feat in range(4):
            out[ch, feat] = standardized_effect(
                features[labels == 1, ch, feat],
                features[labels == 0, ch, feat],
            )
    return out


def case_cluster_bootstrap(
    features: np.ndarray,
    labels: np.ndarray,
    clients: np.ndarray,
    cases: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed + 404)
    client_case_rows: dict[str, dict[str, np.ndarray]] = {}
    for client in sorted(set(clients.tolist())):
        c_mask = clients == client
        mapping: dict[str, np.ndarray] = {}
        for case in sorted(set(cases[c_mask].tolist())):
            mapping[case] = np.flatnonzero(cases == case)
        client_case_rows[client] = mapping

    boot = np.full((n_bootstrap, 4, 4), np.nan, dtype=np.float64)
    for b in range(n_bootstrap):
        selected_rows: list[int] = []
        for mapping in client_case_rows.values():
            keys = list(mapping)
            sampled = rng.choice(keys, size=len(keys), replace=True)
            for key in sampled.tolist():
                selected_rows.extend(mapping[str(key)].tolist())
        idx = np.asarray(selected_rows, dtype=np.int64)
        boot[b] = effect_matrix(features[idx], labels[idx])
    return boot


def plot_bootstrap_effects(
    estimate: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    figure_dir: Path,
) -> None:
    vmax = max(0.5, float(np.nanmax(np.abs(np.stack([lo, hi])))))
    fig, ax = plt.subplots(figsize=(11.0, 5.7))
    im = ax.imshow(estimate, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(4), FEATURES)
    ax.set_yticks(np.arange(4), CHANNELS)
    for i in range(4):
        for j in range(4):
            color = "white" if abs(estimate[i, j]) > 0.57 * vmax else "black"
            ax.text(j, i, f"{estimate[i,j]:+.2f}\n[{lo[i,j]:+.2f}, {hi[i,j]:+.2f}]", ha="center", va="center", fontsize=8.5, color=color)
    ax.set_title("Case-cluster bootstrap for waveform-feature differences\nEstimate and 95% interval; positive minus negative", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Standardized mean difference")
    fig.tight_layout()
    save_figure(fig, figure_dir, "fig07_wave_feature_bootstrap")


def plot_seed_stability(seed_effects: dict[int, np.ndarray], figure_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), sharex=True)
    x = np.arange(4)
    for ch, ax in enumerate(axes.ravel()):
        for seed, matrix in seed_effects.items():
            ax.plot(x, matrix[ch], marker="o", linewidth=1.5, label=f"seed {seed}")
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_title(CHANNELS[ch])
        ax.set_xticks(x, FEATURES, rotation=15)
        ax.set_ylabel("Standardized mean difference")
        style_axis(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Waveform-feature effect stability across deterministic samples", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, figure_dir, "fig08_wave_feature_seed_stability")


def client_clustering(
    client_df: pd.DataFrame,
    *,
    n_clusters: int,
    figure_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    columns = ["train_cases", "train_pos_ratio", "wave_jsd_sum", "clinical_mean_abs_z"]
    values = client_df[columns].to_numpy(dtype=np.float64)
    z = (values - values.mean(axis=0)) / np.maximum(values.std(axis=0), 1e-12)
    tree = linkage(z, method="ward", metric="euclidean")
    order = leaves_list(tree)
    dist = squareform(pdist(z, metric="euclidean"))
    cluster = fcluster(tree, t=n_clusters, criterion="maxclust")

    fig = plt.figure(figsize=(12.0, 12.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 5.0], hspace=0.08)
    ax_d = fig.add_subplot(gs[0])
    dendrogram(tree, ax=ax_d, no_labels=True, color_threshold=None, above_threshold_color="0.35")
    ax_d.set_ylabel("Ward distance")
    ax_d.set_xticks([])
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    ax_h = fig.add_subplot(gs[1])
    matrix = dist[np.ix_(order, order)]
    im = ax_h.imshow(matrix, cmap="viridis", aspect="equal")
    labels = [short_client(v) for v in client_df.iloc[order]["client_id"]]
    ax_h.set_xticks(np.arange(len(order)), labels, rotation=90, fontsize=6.1)
    ax_h.set_yticks(np.arange(len(order)), labels, fontsize=6.1)
    ax_h.set_title("Pairwise distance after standardizing four heterogeneity dimensions")
    cbar = fig.colorbar(im, ax=ax_h, shrink=0.72)
    cbar.set_label("Euclidean distance")
    fig.suptitle("Exploratory client clustering", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.23, right=0.93, bottom=0.25, top=0.95)
    save_figure(fig, figure_dir, "fig09_client_distance_clustering")

    out = client_df[["client_id", *columns]].copy()
    out["exploratory_cluster"] = cluster
    out["short_client"] = out["client_id"].map(short_client)
    return out, dist


def within_window_cross_correlation(
    waves: np.ndarray,
    labels: np.ndarray,
    *,
    max_lag: int,
    figure_dir: Path,
) -> list[dict[str, object]]:
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    lags = np.arange(-max_lag, max_lag + 1)
    curves: dict[tuple[int, int, int], np.ndarray] = {}
    rows: list[dict[str, object]] = []

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.3), sharex=True, sharey=True)
    for ax, (a, b) in zip(axes.ravel(), pairs):
        for label in (0, 1):
            subset = waves[labels == label]
            medians: list[float] = []
            for lag in lags.tolist():
                if lag >= 0:
                    x = subset[:, a, : subset.shape[2] - lag if lag else None]
                    y = subset[:, b, lag:]
                else:
                    x = subset[:, a, -lag:]
                    y = subset[:, b, : subset.shape[2] + lag]
                x0 = x - x.mean(axis=1, keepdims=True)
                y0 = y - y.mean(axis=1, keepdims=True)
                denom = np.sqrt(np.sum(x0 * x0, axis=1) * np.sum(y0 * y0, axis=1))
                corr = np.divide(np.sum(x0 * y0, axis=1), denom, out=np.full(len(x0), np.nan), where=denom > 0)
                medians.append(float(np.nanmedian(corr)))
            curve = np.asarray(medians)
            curves[(a, b, label)] = curve
            best = int(np.nanargmax(np.abs(curve)))
            rows.append(
                {
                    "signal_pair": f"{CHANNELS[a]}-{CHANNELS[b]}",
                    "label": LABEL_NAMES[label],
                    "best_lag_sec": int(lags[best]),
                    "median_correlation": float(curve[best]),
                    "zero_lag_correlation": float(curve[np.flatnonzero(lags == 0)[0]]),
                }
            )
            ax.plot(lags, curve, marker="o", markersize=3, color=COLORS[label], label=LABEL_NAMES[label])
        ax.axhline(0, color="black", linewidth=0.7, linestyle=":")
        ax.axvline(0, color="0.5", linewidth=0.7, linestyle=":")
        ax.set_title(f"{CHANNELS[a]} vs {CHANNELS[b]}")
        ax.set_xlabel("Lag of second signal (s)")
        ax.set_ylabel("Median within-window r")
        style_axis(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Within-window cross-correlation by label\nPositive lag shifts the second signal later", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, figure_dir, "fig10_signal_lag_correlations")
    return rows


def smd_against_train(train: np.ndarray, target: np.ndarray) -> np.ndarray:
    mean_diff = target.mean(axis=0) - train.mean(axis=0)
    pooled = np.sqrt((train.var(axis=0, ddof=1) + target.var(axis=0, ddof=1)) / 2.0)
    return np.divide(mean_diff, pooled, out=np.zeros_like(mean_diff, dtype=np.float64), where=pooled > 0)


def split_shift(
    split_samples: dict[str, dict[str, np.ndarray]],
    *,
    figure_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    wave_arrays: dict[str, np.ndarray] = {}
    clin_arrays: dict[str, np.ndarray] = {}
    for split, sample in split_samples.items():
        wave_arrays[split] = waveform_features(sample["waves"]).reshape(len(sample["waves"]), -1)
        # Clinical rows can occur twice when a case contributes both labels; keep one per case.
        _, unique_idx = np.unique(sample["cases"], return_index=True)
        clin_arrays[split] = sample["clinical"][np.sort(unique_idx)]

    wave_shift = np.stack([
        smd_against_train(wave_arrays["train"], wave_arrays["val"]),
        smd_against_train(wave_arrays["train"], wave_arrays["test"]),
    ], axis=1)
    clin_shift = np.stack([
        smd_against_train(clin_arrays["train"], clin_arrays["val"]),
        smd_against_train(clin_arrays["train"], clin_arrays["test"]),
    ], axis=1)

    wave_names = [f"{ch}/{feat}" for ch in CHANNELS for feat in FEATURES]
    vmax = max(0.3, float(np.max(np.abs(np.concatenate([wave_shift.ravel(), clin_shift.ravel()])))))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 8.4), gridspec_kw={"width_ratios": [1.0, 1.0]})
    im0 = ax0.imshow(wave_shift, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax0.set_xticks([0, 1], ["Val - Train", "Test - Train"])
    ax0.set_yticks(np.arange(len(wave_names)), wave_names, fontsize=7.5)
    ax0.set_title("Waveform features")
    im1 = ax1.imshow(clin_shift, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax1.set_xticks([0, 1], ["Val - Train", "Test - Train"])
    ax1.set_yticks(np.arange(len(CLINICAL)), CLINICAL, fontsize=7.5)
    ax1.set_title("Clinical variables")
    cbar = fig.colorbar(im1, ax=[ax0, ax1], shrink=0.72, pad=0.08)
    cbar.set_label("Standardized mean difference")
    fig.suptitle("Input-distribution shift across dataset splits\nDeterministic, label-balanced case sample", fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.14, right=0.86, wspace=0.45, bottom=0.08, top=0.90)
    save_figure(fig, figure_dir, "fig11_split_input_shift")
    return wave_shift, clin_shift


def raw_case_1hz(raw_path: Path, *, max_rows: int) -> np.ndarray | None:
    parts: list[np.ndarray] = []
    rows_read = 0
    try:
        for chunk in pd.read_csv(raw_path, usecols=RAW_COLUMNS, chunksize=100_000):
            remaining = max_rows - rows_read
            if remaining <= 0:
                break
            arr = chunk[RAW_COLUMNS].to_numpy(dtype=np.float64)[:remaining]
            n_complete = (len(arr) // 100) * 100
            if n_complete:
                blocks = arr[:n_complete].reshape(-1, 100, 4)
                finite = np.isfinite(blocks)
                counts = finite.sum(axis=1)
                sums = np.where(finite, blocks, 0.0).sum(axis=1)
                means = np.full(counts.shape, np.nan, dtype=np.float64)
                np.divide(sums, counts, out=means, where=counts > 0)
                parts.append(means)
            rows_read += len(arr)
            if rows_read >= max_rows:
                break
    except (OSError, ValueError):
        return None
    return np.concatenate(parts) if parts else None


def raw_missingness(
    data_dir: Path,
    raw_dir: Path,
    *,
    cases_per_client: int,
    max_minutes: int,
    seed: int,
    figure_dir: Path,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed + 1771)
    records: list[tuple[str, str, np.ndarray]] = []
    max_rows = max_minutes * 60 * 100
    for client_path in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        files = sorted((client_path / "train").glob("*.npz"))
        order = rng.permutation(len(files)) if files else []
        used = 0
        for idx in list(order):
            if used >= cases_per_client:
                break
            match = re.search(r"case_(\d+)$", files[int(idx)].stem)
            if not match:
                continue
            raw_path = raw_dir / f"case_{match.group(1)}.csv.gz"
            if not raw_path.exists():
                continue
            arr = raw_case_1hz(raw_path, max_rows=max_rows)
            if arr is None or arr.size == 0:
                continue
            records.append((client_path.name, match.group(1), arr))
            used += 1

    n_cases = len(records)
    minute = np.full((n_cases, max_minutes, 4), np.nan, dtype=np.float64)
    overall = np.full((n_cases, 4), np.nan, dtype=np.float64)
    rows: list[dict[str, object]] = []
    for i, (client, case_id, arr) in enumerate(records):
        overall[i] = np.mean(~np.isfinite(arr), axis=0)
        for m in range(min(max_minutes, int(math.ceil(len(arr) / 60)))):
            block = arr[m * 60 : (m + 1) * 60]
            if len(block):
                minute[i, m] = np.mean(~np.isfinite(block), axis=0)
        row: dict[str, object] = {"client_id": client, "case_id": case_id}
        for ch, name in enumerate(CHANNELS):
            row[f"{name}_missing_rate"] = float(overall[i, ch])
        rows.append(row)

    order = np.argsort(np.nanmean(overall, axis=1))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.0, 8.8), gridspec_kw={"width_ratios": [1.2, 1.0]})
    x = np.arange(max_minutes)
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for ch, name in enumerate(CHANNELS):
        med = np.nanmedian(minute[:, :, ch], axis=0)
        q25 = np.nanquantile(minute[:, :, ch], 0.25, axis=0)
        q75 = np.nanquantile(minute[:, :, ch], 0.75, axis=0)
        ax0.plot(x, 100 * med, color=palette[ch], label=name, linewidth=1.8)
        ax0.fill_between(x, 100 * q25, 100 * q75, color=palette[ch], alpha=0.10)
    ax0.set_xlabel("Minutes from recording start")
    ax0.set_ylabel("No-value 1-s blocks (%)")
    ax0.set_ylim(0, 103)
    ax0.set_title("Temporal raw sparsity\nMedian and interquartile range")
    ax0.legend(frameon=False)
    style_axis(ax0)

    im = ax1.imshow(100 * overall[order], cmap="magma_r", vmin=0, vmax=100, aspect="auto")
    ax1.set_xticks(np.arange(4), CHANNELS, rotation=20)
    ax1.set_yticks(np.arange(n_cases), [short_client(records[i][0]) for i in order], fontsize=6.5)
    ax1.set_title("Case-level raw sparsity")
    cbar = fig.colorbar(im, ax=ax1, shrink=0.75)
    cbar.set_label("No-value 1-s blocks (%)")
    fig.suptitle("Raw 1-s-block sparsity before interpolation", fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.09, right=0.93, wspace=0.30, bottom=0.08, top=0.90)
    save_figure(fig, figure_dir, "fig12_raw_missingness")
    return rows


def write_outputs(
    table_dir: Path,
    *,
    estimate: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    seed_effects: dict[int, np.ndarray],
    cluster_df: pd.DataFrame,
    lag_rows: list[dict[str, object]],
    wave_shift: np.ndarray,
    clin_shift: np.ndarray,
    missing_rows: list[dict[str, object]],
) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ch in range(4):
        for feat in range(4):
            rows.append([CHANNELS[ch], FEATURES[feat], f"{estimate[ch,feat]:+.3f}", f"{lo[ch,feat]:+.3f}", f"{hi[ch,feat]:+.3f}"])
    (table_dir / "table05_wave_feature_bootstrap.md").write_text(
        "# Case-cluster bootstrap intervals\n\n" + markdown_table(["Signal", "Feature", "Estimate", "2.5%", "97.5%"], rows)
    )

    seed_rows = []
    for ch in range(4):
        for feat in range(4):
            vals = np.array([m[ch, feat] for m in seed_effects.values()])
            seed_rows.append([CHANNELS[ch], FEATURES[feat], f"{vals.mean():+.3f}", f"{vals.min():+.3f}", f"{vals.max():+.3f}"])
    (table_dir / "table06_seed_stability.md").write_text(
        "# Effect stability across seeds\n\n" + markdown_table(["Signal", "Feature", "Mean", "Min", "Max"], seed_rows)
    )

    cluster_rows = [
        [int(r.exploratory_cluster), short_client(r.client_id), int(r.train_cases), f"{100*r.train_pos_ratio:.2f}%", f"{r.wave_jsd_sum:.3f}", f"{r.clinical_mean_abs_z:.3f}"]
        for r in cluster_df.sort_values(["exploratory_cluster", "client_id"]).itertuples()
    ]
    (table_dir / "table07_exploratory_client_clusters.md").write_text(
        "# Exploratory client clusters\n\nCluster count is fixed for exploration, not selected as a final model.\n\n"
        + markdown_table(["Cluster", "Client", "Cases", "Positive rate", "Wave JSD", "Clinical shift"], cluster_rows)
    )

    lag_table = markdown_table(
        ["Signal pair", "Label", "Best lag (s)", "Correlation at best lag", "Zero-lag correlation"],
        [[r["signal_pair"], r["label"], r["best_lag_sec"], f"{r['median_correlation']:+.3f}", f"{r['zero_lag_correlation']:+.3f}"] for r in lag_rows],
    )
    (table_dir / "table08_signal_lag_correlations.md").write_text("# Within-window lag correlations\n\n" + lag_table)

    wave_names = [f"{ch}/{feat}" for ch in CHANNELS for feat in FEATURES]
    shift_rows = [["Wave", name, f"{wave_shift[i,0]:+.3f}", f"{wave_shift[i,1]:+.3f}"] for i, name in enumerate(wave_names)]
    shift_rows += [["Clinical", name, f"{clin_shift[i,0]:+.3f}", f"{clin_shift[i,1]:+.3f}"] for i, name in enumerate(CLINICAL)]
    (table_dir / "table09_split_input_shift.md").write_text(
        "# Split input-distribution shift\n\n" + markdown_table(["Type", "Feature", "Val - Train", "Test - Train"], shift_rows)
    )

    missing_table = markdown_table(
        ["Client", "Case", *[f"{ch} no-value blocks" for ch in CHANNELS]],
        [[short_client(r["client_id"]), r["case_id"], *[f"{100*float(r[f'{ch}_missing_rate']):.1f}%" for ch in CHANNELS]] for r in missing_rows],
    )
    (table_dir / "table10_raw_case_missingness.md").write_text("# Raw case 1-s-block sparsity before interpolation\n\n" + missing_table)


def main() -> None:
    args = parse_args()
    seeds = [int(v.strip()) for v in str(args.seeds).split(",") if v.strip()]
    if not seeds:
        raise SystemExit("--seeds must contain at least one integer")
    data_dir = Path(args.data_dir)
    raw_dir = Path(args.raw_dir)
    figure_dir = Path(args.figure_dir)
    table_dir = Path(args.table_dir)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.2, "pdf.fonttype": 42, "ps.fonttype": 42})

    samples: dict[int, dict[str, np.ndarray]] = {}
    seed_effects: dict[int, np.ndarray] = {}
    for seed in seeds:
        sample = sample_windows(data_dir, "train", cases_per_client_label=args.cases_per_client_label, seed=seed)
        samples[seed] = sample
        seed_effects[seed] = effect_matrix(waveform_features(sample["waves"]), sample["labels"])
        print(f"seed={seed}: windows={len(sample['waves'])}, cases={len(set(sample['cases'].tolist()))}")

    primary = samples[seeds[0]]
    primary_features = waveform_features(primary["waves"])
    estimate = seed_effects[seeds[0]]
    boot = case_cluster_bootstrap(
        primary_features,
        primary["labels"],
        primary["clients"],
        primary["cases"],
        n_bootstrap=args.bootstrap,
        seed=seeds[0],
    )
    lo, hi = np.nanquantile(boot, [0.025, 0.975], axis=0)
    plot_bootstrap_effects(estimate, lo, hi, figure_dir)
    plot_seed_stability(seed_effects, figure_dir)

    client_df = pd.read_csv(args.client_stats)
    cluster_df, client_dist = client_clustering(client_df, n_clusters=args.clusters, figure_dir=figure_dir)
    lag_rows = within_window_cross_correlation(primary["waves"], primary["labels"], max_lag=args.max_lag_sec, figure_dir=figure_dir)

    split_samples = {
        split: sample_windows(
            data_dir,
            split,
            cases_per_client_label=args.split_cases_per_client_label,
            seed=seeds[0] + i * 100,
        )
        for i, split in enumerate(["train", "val", "test"])
    }
    wave_shift, clin_shift = split_shift(split_samples, figure_dir=figure_dir)
    missing_rows = raw_missingness(
        data_dir,
        raw_dir,
        cases_per_client=args.raw_cases_per_client,
        max_minutes=args.raw_max_minutes,
        seed=seeds[0],
        figure_dir=figure_dir,
    )

    write_outputs(
        table_dir,
        estimate=estimate,
        lo=lo,
        hi=hi,
        seed_effects=seed_effects,
        cluster_df=cluster_df,
        lag_rows=lag_rows,
        wave_shift=wave_shift,
        clin_shift=clin_shift,
        missing_rows=missing_rows,
    )

    largest_shift_idx = int(np.argmax(np.abs(np.concatenate([wave_shift.ravel(), clin_shift.ravel()]))))
    metadata = {
        "data_dir": str(data_dir.resolve()),
        "raw_dir": str(raw_dir.resolve()),
        "seeds": seeds,
        "cases_per_client_label": int(args.cases_per_client_label),
        "bootstrap_replicates": int(args.bootstrap),
        "bootstrap_unit": "case, stratified by client; all sampled labels for the case move together",
        "primary_sample": {
            "windows": int(len(primary["waves"])),
            "unique_cases": int(len(set(primary["cases"].tolist()))),
            "negative_windows": int((primary["labels"] == 0).sum()),
            "positive_windows": int((primary["labels"] == 1).sum()),
        },
        "exploratory_clusters": int(args.clusters),
        "cluster_sizes": {str(k): int(v) for k, v in cluster_df["exploratory_cluster"].value_counts().sort_index().items()},
        "max_abs_split_shift": float(np.max(np.abs(np.concatenate([wave_shift.ravel(), clin_shift.ravel()])))),
        "split_shift_samples": {
            split: {
                "windows": int(len(sample["waves"])),
                "unique_cases": int(len(set(sample["cases"].tolist()))),
                "negative_windows": int((sample["labels"] == 0).sum()),
                "positive_windows": int((sample["labels"] == 1).sum()),
            }
            for split, sample in split_samples.items()
        },
        "raw_missing_cases": int(len(missing_rows)),
        "outputs": sorted(str(p) for p in [*figure_dir.glob("fig0[7-9]*.*"), *figure_dir.glob("fig1[0-2]*.*"), *table_dir.glob("table0[5-9]*.md"), *table_dir.glob("table10*.md")]),
    }
    (figure_dir / "eda_supplement_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
