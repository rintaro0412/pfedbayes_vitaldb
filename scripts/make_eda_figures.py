"""Generate explanatory EDA figures for the current hypoxemia dataset.

The client/split figures use the existing full-dataset summaries. Waveform
figures use a deterministic, case-level sample from the training split to avoid
loading the full 5.9 GB dataset. Raw signal distributions use 1 Hz subsamples
from a deterministic sample of training cases.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CHANNELS = ["HR", "SpO2", "ETCO2", "FIO2"]
RAW_COLUMNS = [
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2",
    "Primus/ETCO2",
    "Primus/FIO2",
]
RAW_UNITS = ["bpm", "%", "mmHg", "%"]
FEATURES = ["Mean", "SD", "Slope", "Abs. diff."]
COLORS = {0: "#0072B2", 1: "#D55E00"}
LABEL_NAMES = {0: "Negative", 1: "Positive"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate EDA figures from the hypoxemia dataset")
    p.add_argument("--data-dir", default="federated_data")
    p.add_argument("--raw-dir", default="vitaldb_data")
    p.add_argument("--dataset-summary", default="federated_data/summary.json")
    p.add_argument("--client-stats", default="outputs_dataset_stats/federated_client_stats.csv")
    p.add_argument("--figure-dir", default="figures/eda")
    p.add_argument("--table-dir", default="tables/eda")
    p.add_argument("--cases-per-client-label", type=int, default=12)
    p.add_argument("--raw-cases-per-client", type=int, default=1)
    p.add_argument("--raw-max-minutes", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def short_client(name: str) -> str:
    text = str(name)
    text = text.replace("General_surgery__", "GS/")
    text = text.replace("Thoracic_surgery__", "TS/")
    text = text.replace("Gynecology__", "GYN/")
    text = text.replace("Urology__", "URO/")
    return text.replace("_", " ")


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.7, zorder=0)


def sample_case_windows(
    data_dir: Path,
    *,
    cases_per_client_label: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    rng = np.random.default_rng(seed)
    waves: list[np.ndarray] = []
    labels: list[int] = []
    client_names: list[str] = []
    rows: list[dict[str, object]] = []

    clients = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    for client in clients:
        files = sorted((data_dir / client / "train").glob("*.npz"))
        order = rng.permutation(len(files)) if files else np.empty(0, dtype=int)
        counts = {0: 0, 1: 0}
        sampled_cases = {0: set(), 1: set()}
        for file_idx in order.tolist():
            if all(counts[y] >= cases_per_client_label for y in (0, 1)):
                break
            path = files[int(file_idx)]
            try:
                with np.load(path, allow_pickle=False) as z:
                    y_arr = np.asarray(z["y"], dtype=np.int64)
                    for label in (0, 1):
                        if counts[label] >= cases_per_client_label:
                            continue
                        idx = np.flatnonzero(y_arr == label)
                        if idx.size == 0:
                            continue
                        selected = int(rng.choice(idx))
                        wave = np.asarray(z["x_wave"][selected], dtype=np.float32)
                        if wave.shape != (4, 3000) or not np.isfinite(wave).all():
                            continue
                        # 100 Hz -> 1 Hz for interpretable temporal profiles.
                        wave_1hz = wave.reshape(4, 30, 100).mean(axis=2)
                        waves.append(wave_1hz)
                        labels.append(label)
                        client_names.append(client)
                        counts[label] += 1
                        sampled_cases[label].add(path.stem)
            except (OSError, ValueError, KeyError):
                continue
        rows.append(
            {
                "client_id": client,
                "negative_cases": len(sampled_cases[0]),
                "positive_cases": len(sampled_cases[1]),
            }
        )

    if not waves:
        raise RuntimeError(f"No valid windows sampled under {data_dir}")
    return (
        np.stack(waves).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(client_names, dtype=object),
        rows,
    )


def waveform_features(waves: np.ndarray) -> np.ndarray:
    n, c, t = waves.shape
    x = np.arange(t, dtype=np.float64)
    x_centered = x - x.mean()
    denom = float(np.sum(x_centered**2))
    means = waves.mean(axis=2)
    sds = waves.std(axis=2)
    slopes = np.sum((waves - means[:, :, None]) * x_centered[None, None, :], axis=2) / denom
    absdiff = np.abs(np.diff(waves, axis=2)).mean(axis=2)
    return np.stack([means, sds, slopes, absdiff], axis=2).reshape(n, c, 4)


def standardized_effect(pos: np.ndarray, neg: np.ndarray) -> float:
    if pos.size < 2 or neg.size < 2:
        return float("nan")
    pooled = math.sqrt((float(pos.var(ddof=1)) + float(neg.var(ddof=1))) / 2.0)
    if pooled <= 0:
        return 0.0
    return float((pos.mean() - neg.mean()) / pooled)


def plot_client_label_distribution(client_df: pd.DataFrame, out_dir: Path) -> dict[str, object]:
    df = client_df.sort_values("train_pos_ratio", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))
    labels = [short_client(v) for v in df["client_id"]]
    global_rate = float(df["train_pos"].sum() / (df["train_pos"].sum() + df["train_neg"].sum()))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.5, 10.5), sharey=True)
    ax0.barh(y, df["train_cases"], color="#56B4E9", zorder=2)
    ax0.set_xlabel("Training cases")
    ax0.set_yticks(y, labels, fontsize=7.4)
    ax0.set_title("Client size")
    style_axis(ax0)

    colors = [COLORS[0] if v < global_rate else COLORS[1] for v in df["train_pos_ratio"]]
    ax1.barh(y, 100 * df["train_pos_ratio"], color=colors, zorder=2)
    ax1.axvline(100 * global_rate, color="black", linestyle="--", linewidth=1.2, label=f"Global: {100*global_rate:.1f}%")
    ax1.set_xlabel("Positive-window rate (%)")
    ax1.set_title("Label distribution")
    ax1.legend(frameon=False, loc="lower right")
    style_axis(ax1)
    fig.suptitle("Training-client size and hypoxemia label heterogeneity", fontsize=14, fontweight="bold")
    fig.subplots_adjust(left=0.30, wspace=0.12, top=0.92)
    save_figure(fig, out_dir, "fig01_client_size_and_label_rate")
    return {
        "global_train_positive_rate": global_rate,
        "min_client": str(df.iloc[0]["client_id"]),
        "min_client_positive_rate": float(df.iloc[0]["train_pos_ratio"]),
        "max_client": str(df.iloc[-1]["client_id"]),
        "max_client_positive_rate": float(df.iloc[-1]["train_pos_ratio"]),
    }


def plot_temporal_profiles(waves: np.ndarray, labels: np.ndarray, out_dir: Path) -> None:
    time = np.arange(waves.shape[2])
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), sharex=True)
    for ch, ax in enumerate(axes.ravel()):
        for label in (0, 1):
            arr = waves[labels == label, ch, :]
            med = np.median(arr, axis=0)
            q25, q75 = np.quantile(arr, [0.25, 0.75], axis=0)
            ax.plot(time, med, color=COLORS[label], linewidth=2.0, label=f"{LABEL_NAMES[label]} (n={len(arr)})")
            ax.fill_between(time, q25, q75, color=COLORS[label], alpha=0.18, linewidth=0)
        ax.axhline(0, color="0.45", linewidth=0.8, linestyle=":")
        ax.set_title(CHANNELS[ch])
        ax.set_ylabel("Case-standardized value")
        ax.set_xlim(0, 29)
        style_axis(ax)
    for ax in axes[-1]:
        ax.set_xlabel("Seconds within the 30-s input window")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Sampled 30-s waveform profiles by label\nMedian and interquartile range; one window per case and label", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, out_dir, "fig02_waveform_profiles_by_label")


def plot_feature_effects(waves: np.ndarray, labels: np.ndarray, out_dir: Path) -> np.ndarray:
    features = waveform_features(waves)
    effects = np.zeros((4, 4), dtype=np.float64)
    for ch in range(4):
        for feat in range(4):
            effects[ch, feat] = standardized_effect(features[labels == 1, ch, feat], features[labels == 0, ch, feat])

    vmax = max(0.5, float(np.nanmax(np.abs(effects))))
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    im = ax.imshow(effects, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(4), FEATURES)
    ax.set_yticks(np.arange(4), CHANNELS)
    for i in range(4):
        for j in range(4):
            value = effects[i, j]
            color = "white" if abs(value) > 0.55 * vmax else "black"
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", color=color, fontsize=10)
    ax.set_title("Positive vs negative waveform-feature differences\nStandardized mean difference; positive values are higher in positive windows", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Standardized mean difference")
    fig.tight_layout()
    save_figure(fig, out_dir, "fig03_wave_feature_effects")
    return effects


def plot_client_heterogeneity_heatmap(client_df: pd.DataFrame, out_dir: Path) -> None:
    cols = ["train_cases", "train_pos_ratio", "wave_jsd_sum", "clinical_mean_abs_z"]
    names = ["Cases", "Positive rate", "Wave JSD", "Clinical shift"]
    df = client_df.sort_values("train_pos_ratio", ascending=True).reset_index(drop=True)
    values = df[cols].to_numpy(dtype=np.float64)
    z = (values - values.mean(axis=0, keepdims=True)) / np.maximum(values.std(axis=0, keepdims=True), 1e-12)
    vmax = max(1.0, float(np.nanmax(np.abs(z))))

    fig, ax = plt.subplots(figsize=(8.8, 10.5))
    im = ax.imshow(z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(cols)), names, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(df)), [short_client(v) for v in df["client_id"]], fontsize=7.4)
    ax.set_title("Client heterogeneity across complementary dimensions\nColumn-wise z-scores", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label("Across-client z-score")
    fig.tight_layout()
    save_figure(fig, out_dir, "fig04_client_heterogeneity_heatmap")


def plot_split_composition(summary: dict[str, object], out_dir: Path) -> list[dict[str, object]]:
    detail = summary["splits_detail"]
    splits = ["train", "val", "test"]
    cases = np.array([detail[s]["cases_written"] for s in splits], dtype=float)
    pos = np.array([detail[s]["pos_windows"] for s in splits], dtype=float)
    neg = np.array([detail[s]["neg_windows"] for s in splits], dtype=float)
    rates = pos / (pos + neg)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2))
    axes[0].bar(splits, cases, color="#56B4E9")
    axes[0].set_ylabel("Cases")
    axes[0].set_title("Cases")
    for i, v in enumerate(cases):
        axes[0].text(i, v, f"{int(v):,}", ha="center", va="bottom", fontsize=9)
    style_axis(axes[0])

    axes[1].bar(splits, neg, color=COLORS[0], label="Negative")
    axes[1].bar(splits, pos, bottom=neg, color=COLORS[1], label="Positive")
    axes[1].set_ylabel("Windows")
    axes[1].set_title("Window counts")
    axes[1].legend(frameon=False, fontsize=8)
    style_axis(axes[1])

    axes[2].bar(splits, 100 * rates, color="#009E73")
    axes[2].set_ylim(0, max(45, float(100 * rates.max() + 5)))
    axes[2].set_ylabel("Positive-window rate (%)")
    axes[2].set_title("Label balance")
    for i, v in enumerate(rates):
        axes[2].text(i, 100 * v, f"{100*v:.2f}%", ha="center", va="bottom", fontsize=9)
    style_axis(axes[2])

    fig.suptitle("Dataset composition by split", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, out_dir, "fig05_split_composition")

    return [
        {
            "split": s,
            "cases": int(cases[i]),
            "positive_windows": int(pos[i]),
            "negative_windows": int(neg[i]),
            "total_windows": int(pos[i] + neg[i]),
            "positive_rate": float(rates[i]),
        }
        for i, s in enumerate(splits)
    ]


def case_id_from_npz(path: Path) -> int | None:
    match = re.search(r"case_(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def sample_raw_signals(
    data_dir: Path,
    raw_dir: Path,
    *,
    cases_per_client: int,
    max_minutes: int,
    seed: int,
) -> tuple[list[np.ndarray], list[dict[str, object]], int]:
    rng = np.random.default_rng(seed + 991)
    value_parts: list[list[np.ndarray]] = [[] for _ in RAW_COLUMNS]
    observed = np.zeros(4, dtype=np.int64)
    total = np.zeros(4, dtype=np.int64)
    sampled_case_count = 0
    max_rows = max(1, int(max_minutes)) * 60 * 100

    clients = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    for client in clients:
        files = sorted((data_dir / client / "train").glob("*.npz"))
        if not files:
            continue
        order = rng.permutation(len(files))
        used = 0
        for idx in order.tolist():
            if used >= cases_per_client:
                break
            case_id = case_id_from_npz(files[int(idx)])
            if case_id is None:
                continue
            raw_path = raw_dir / f"case_{case_id}.csv.gz"
            if not raw_path.exists():
                continue
            one_hz_parts: list[np.ndarray] = []
            rows_read = 0
            try:
                for chunk in pd.read_csv(raw_path, usecols=RAW_COLUMNS, chunksize=100_000):
                    remaining = max_rows - rows_read
                    if remaining <= 0:
                        break
                    arr = chunk[RAW_COLUMNS].to_numpy(dtype=np.float64)[:remaining]
                    n_complete = (len(arr) // 100) * 100
                    if n_complete > 0:
                        blocks = arr[:n_complete].reshape(-1, 100, len(RAW_COLUMNS))
                        finite = np.isfinite(blocks)
                        counts = finite.sum(axis=1)
                        sums = np.where(finite, blocks, 0.0).sum(axis=1)
                        means = np.full(counts.shape, np.nan, dtype=np.float64)
                        np.divide(sums, counts, out=means, where=counts > 0)
                        one_hz_parts.append(means)
                    rows_read += len(arr)
                    if rows_read >= max_rows:
                        break
            except (OSError, ValueError):
                continue
            if not one_hz_parts:
                continue
            sample = np.concatenate(one_hz_parts, axis=0)
            for ch in range(len(RAW_COLUMNS)):
                arr = sample[:, ch]
                total[ch] += arr.size
                finite_values = arr[np.isfinite(arr)]
                observed[ch] += finite_values.size
                if finite_values.size:
                    value_parts[ch].append(finite_values)
            sampled_case_count += 1
            used += 1

    values = [np.concatenate(parts) if parts else np.empty(0, dtype=np.float64) for parts in value_parts]
    rows: list[dict[str, object]] = []
    for ch, arr in enumerate(values):
        if arr.size:
            p01, p25, med, p75, p99 = np.quantile(arr, [0.01, 0.25, 0.50, 0.75, 0.99])
        else:
            p01 = p25 = med = p75 = p99 = float("nan")
        rows.append(
            {
                "signal": CHANNELS[ch],
                "unit": RAW_UNITS[ch],
                "n_observed_1hz": int(observed[ch]),
                "missing_rate": float(1.0 - observed[ch] / total[ch]) if total[ch] else float("nan"),
                "p01": float(p01),
                "p25": float(p25),
                "median": float(med),
                "p75": float(p75),
                "p99": float(p99),
            }
        )
    return values, rows, sampled_case_count


def plot_raw_distributions(values: list[np.ndarray], rows: list[dict[str, object]], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    for ch, ax in enumerate(axes.ravel()):
        arr = values[ch]
        row = rows[ch]
        if arr.size:
            lo, hi = np.quantile(arr, [0.01, 0.99])
            clipped = arr[(arr >= lo) & (arr <= hi)]
            ax.hist(clipped, bins=60, density=True, color="#0072B2", alpha=0.78)
            ax.axvline(float(row["median"]), color="#D55E00", linewidth=1.8, label=f"Median: {float(row['median']):.1f}")
            ax.legend(frameon=False, fontsize=8)
            ax.set_xlim(lo, hi)
        ax.set_title(f"{CHANNELS[ch]} ({RAW_UNITS[ch]})")
        ax.set_xlabel("Raw value (1st-99th percentile shown)")
        ax.set_ylabel("Density")
        style_axis(ax)
    fig.suptitle("Raw physiologic-signal distributions\nDeterministic 1-Hz subsample of training cases", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, out_dir, "fig06_raw_signal_distributions")


def markdown_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def write_tables(
    table_dir: Path,
    split_rows: list[dict[str, object]],
    raw_rows: list[dict[str, object]],
    client_df: pd.DataFrame,
    effects: np.ndarray,
) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    composition = markdown_table(
        ["Split", "Cases", "Positive windows", "Negative windows", "Total windows", "Positive rate"],
        [
            [r["split"], f"{r['cases']:,}", f"{r['positive_windows']:,}", f"{r['negative_windows']:,}", f"{r['total_windows']:,}", f"{100*float(r['positive_rate']):.2f}%"]
            for r in split_rows
        ],
    )
    (table_dir / "table01_dataset_composition.md").write_text("# Dataset composition\n\n" + composition)

    raw_table = markdown_table(
        ["Signal", "Unit", "Observed 1-Hz points", "Missing", "P1", "P25", "Median", "P75", "P99"],
        [
            [r["signal"], r["unit"], f"{r['n_observed_1hz']:,}", f"{100*float(r['missing_rate']):.1f}%", *[f"{float(r[k]):.2f}" for k in ("p01", "p25", "median", "p75", "p99")]]
            for r in raw_rows
        ],
    )
    (table_dir / "table02_raw_signal_summary.md").write_text("# Raw signal summary\n\n" + raw_table)

    low = client_df.nsmallest(5, "train_pos_ratio")
    high = client_df.nlargest(5, "train_pos_ratio")
    extremes = pd.concat([low.assign(group="Lowest"), high.assign(group="Highest")])
    client_table = markdown_table(
        ["Group", "Client", "Train cases", "Train windows", "Positive rate", "Wave JSD", "Clinical shift"],
        [
            [r.group, short_client(r.client_id), f"{int(r.train_cases):,}", f"{int(r.train_windows):,}", f"{100*float(r.train_pos_ratio):.2f}%", f"{float(r.wave_jsd_sum):.3f}", f"{float(r.clinical_mean_abs_z):.3f}"]
            for r in extremes.itertuples()
        ],
    )
    (table_dir / "table03_client_extremes.md").write_text("# Clients with extreme positive-window rates\n\n" + client_table)

    effect_table = markdown_table(
        ["Signal", *FEATURES],
        [[CHANNELS[i], *[f"{effects[i, j]:+.3f}" for j in range(4)]] for i in range(4)],
    )
    (table_dir / "table04_wave_feature_effects.md").write_text("# Positive vs negative standardized effects\n\n" + effect_table)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    raw_dir = Path(args.raw_dir)
    figure_dir = Path(args.figure_dir)
    table_dir = Path(args.table_dir)
    summary = json.loads(Path(args.dataset_summary).read_text())
    client_df = pd.read_csv(args.client_stats)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    client_findings = plot_client_label_distribution(client_df, figure_dir)
    waves, labels, sampled_clients, sample_rows = sample_case_windows(
        data_dir,
        cases_per_client_label=int(args.cases_per_client_label),
        seed=int(args.seed),
    )
    plot_temporal_profiles(waves, labels, figure_dir)
    effects = plot_feature_effects(waves, labels, figure_dir)
    plot_client_heterogeneity_heatmap(client_df, figure_dir)
    split_rows = plot_split_composition(summary, figure_dir)
    raw_values, raw_rows, raw_case_count = sample_raw_signals(
        data_dir,
        raw_dir,
        cases_per_client=int(args.raw_cases_per_client),
        max_minutes=int(args.raw_max_minutes),
        seed=int(args.seed),
    )
    plot_raw_distributions(raw_values, raw_rows, figure_dir)
    write_tables(table_dir, split_rows, raw_rows, client_df, effects)

    effect_index = np.unravel_index(np.nanargmax(np.abs(effects)), effects.shape)
    metadata = {
        "data_dir": str(data_dir.resolve()),
        "raw_dir": str(raw_dir.resolve()),
        "seed": int(args.seed),
        "full_dataset": {
            "clients": int(len(client_df)),
            "case_files": int(summary["written_case_files"]),
            "positive_windows": int(summary["written_pos_windows"]),
            "negative_windows": int(summary["written_neg_windows"]),
        },
        "waveform_sample": {
            "sampling_unit": "at most one window per case and label",
            "cases_per_client_label_target": int(args.cases_per_client_label),
            "windows": int(len(waves)),
            "negative_windows": int((labels == 0).sum()),
            "positive_windows": int((labels == 1).sum()),
            "represented_clients": int(len(set(sampled_clients.tolist()))),
            "per_client": sample_rows,
            "stored_values": "case-wide channel-standardized values",
            "temporal_downsampling": "100 Hz to 1 Hz by mean",
        },
        "raw_sample": {
            "cases_per_client_target": int(args.raw_cases_per_client),
            "sampled_cases": int(raw_case_count),
            "max_minutes_per_case": int(args.raw_max_minutes),
            "temporal_sampling": "mean of each 100-row block (1 Hz from 100 Hz)",
            "signals": raw_rows,
        },
        "observed_findings": {
            **client_findings,
            "largest_absolute_wave_feature_effect": {
                "signal": CHANNELS[int(effect_index[0])],
                "feature": FEATURES[int(effect_index[1])],
                "standardized_effect": float(effects[effect_index]),
            },
        },
        "outputs": sorted(str(p) for p in [*figure_dir.glob("fig*.*"), *table_dir.glob("*.md")]),
    }
    (figure_dir / "eda_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
