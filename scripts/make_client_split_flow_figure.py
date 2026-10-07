from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


def _node(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    body: str,
    *,
    stats: str | None = None,
    facecolor: str = "white",
    edgecolor: str = "black",
    lw: float = 1.15,
    title_size: float = 9.1,
    body_size: float = 7.2,
    stats_size: float = 5.15,
) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(
        Rectangle(
            (x - w / 2, y - h / 2),
            w,
            h,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=lw,
        )
    )
    ax.text(x, y + h * 0.29, title, ha="center", va="center", fontsize=title_size, fontweight="bold")
    ax.text(x, y + h * 0.00, body, ha="center", va="center", fontsize=body_size, linespacing=1.12)
    if stats:
        ax.text(x, y - h * 0.34, stats, ha="center", va="center", fontsize=stats_size, color="0.35", linespacing=1.0)


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    rad: float = 0.0,
    label: str | None = None,
    label_xy: tuple[float, float] | None = None,
    color: str = "black",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.05,
            color=color,
            shrinkA=2,
            shrinkB=2,
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if label:
        lx, ly = label_xy if label_xy is not None else ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.03)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=7.4, color=color)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_int(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except Exception:
        return "n/a"


def _delta(before: Any, after: Any) -> str:
    try:
        return f"{_fmt_int(before)} -> {_fmt_int(after)}"
    except Exception:
        return f"{_fmt_int(before)} -> {_fmt_int(after)}"


def _drop(before: Any, after: Any) -> str:
    try:
        return f"drop {_fmt_int(int(before) - int(after))}"
    except Exception:
        return "drop n/a"


def _count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def _count_age_exclusions(path: Path) -> int | None:
    if not path.exists():
        return None
    n = 0
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                if float(row.get("age", "")) < 18:
                    n += 1
            except Exception:
                n += 1
    return n


def _counts(
    summary_path: Path,
    download_run_path: Path,
    clinical_csv_path: Path,
    total_cases_override: int | None,
) -> dict[str, str]:
    summary = _load_json(summary_path)
    download_run = _load_json(download_run_path)

    dl_stats = download_run.get("stats") if isinstance(download_run.get("stats"), dict) else {}
    downloaded_cases = dl_stats.get("downloaded") or dl_stats.get("to_download")
    raw_cases = int(total_cases_override) if total_cases_override is not None else _count_csv_rows(clinical_csv_path)
    if raw_cases is None:
        raw_cases = downloaded_cases

    clients = summary.get("clients") if isinstance(summary.get("clients"), dict) else {}
    splits = summary.get("splits") if isinstance(summary.get("splits"), dict) else {}
    split_detail = summary.get("splits_detail") if isinstance(summary.get("splits_detail"), dict) else {}
    drop = summary.get("window_drop_estimate") if isinstance(summary.get("window_drop_estimate"), dict) else {}

    written_cases = 0
    if split_detail:
        written_cases = sum(int(v.get("cases_written", 0)) for v in split_detail.values() if isinstance(v, dict))

    pos_windows = summary.get("written_pos_windows", drop.get("pos_written"))
    neg_windows = summary.get("written_neg_windows", drop.get("neg_written"))
    total_windows = None
    if pos_windows is not None and neg_windows is not None:
        total_windows = int(pos_windows) + int(neg_windows)
    planned_windows = None
    if drop.get("pos_units_est") is not None and drop.get("neg_target") is not None:
        planned_windows = int(drop.get("pos_units_est")) + int(drop.get("neg_target"))

    age_excluded = _count_age_exclusions(clinical_csv_path)
    track_excluded = None
    if raw_cases is not None and downloaded_cases is not None and age_excluded is not None:
        track_excluded = int(raw_cases) - int(downloaded_cases) - int(age_excluded)
    eligible_cases = summary.get("eligible_cases")
    split_cases = sum(int(v) for v in splits.values()) if splits else None

    return {
        "raw": f"n={_fmt_int(raw_cases)}",
        "signals": _delta(raw_cases, downloaded_cases) + "\n" + _drop(raw_cases, downloaded_cases),
        "labels": _delta(downloaded_cases, summary.get("eligible_cases"))
        + "\n"
        + _drop(downloaded_cases, summary.get("eligible_cases"))
        + "; events="
        + _fmt_int(drop.get("pos_events")),
        "windows": "pos="
        + _fmt_int(drop.get("pos_units_est"))
        + "\nneg="
        + _fmt_int(drop.get("neg_target")),
        "qc": _delta(summary.get("eligible_cases"), written_cases)
        + "; "
        + _drop(summary.get("eligible_cases"), written_cases)
        + "\nwin drop "
        + _fmt_int(int(planned_windows) - int(total_windows)) if planned_windows is not None and total_windows is not None else "",
        "metadata": f"n={_fmt_int(raw_cases)}",
        "client": _delta(summary.get("eligible_cases"), summary.get("eligible_cases"))
        + "; "
        + _drop(summary.get("eligible_cases"), summary.get("eligible_cases"))
        + "\nC="
        + _fmt_int(len(clients))
        + " clients",
        "split": _delta(summary.get("eligible_cases"), sum(int(v) for v in splits.values()) if splits else None)
        + "; "
        + _drop(summary.get("eligible_cases"), sum(int(v) for v in splits.values()) if splits else None)
        + "\ntr/val/te="
        + "/".join(_fmt_int(splits.get(k)) for k in ("train", "val", "test")),
        "output": "n="
        + _fmt_int(written_cases)
        + "; C="
        + _fmt_int(len(clients))
        + "\nwindows="
        + _fmt_int(total_windows),
        "raw_n": _fmt_int(raw_cases),
        "downloaded_n": _fmt_int(downloaded_cases),
        "download_drop": _fmt_int(int(raw_cases) - int(downloaded_cases)) if raw_cases is not None and downloaded_cases is not None else "n/a",
        "age_drop": _fmt_int(age_excluded),
        "track_drop": _fmt_int(track_excluded),
        "eligible_n": _fmt_int(eligible_cases),
        "eligible_drop": _fmt_int(int(downloaded_cases) - int(eligible_cases)) if downloaded_cases is not None and eligible_cases is not None else "n/a",
        "events_n": _fmt_int(drop.get("pos_events")),
        "pos_units": _fmt_int(drop.get("pos_units_est")),
        "neg_target": _fmt_int(drop.get("neg_target")),
        "planned_windows": _fmt_int(planned_windows),
        "written_cases": _fmt_int(written_cases),
        "written_case_drop": _fmt_int(int(eligible_cases) - int(written_cases)) if eligible_cases is not None else "n/a",
        "written_windows": _fmt_int(total_windows),
        "window_drop": _fmt_int(int(planned_windows) - int(total_windows)) if planned_windows is not None and total_windows is not None else "n/a",
        "clients_n": _fmt_int(len(clients)),
        "split_drop": _fmt_int(int(eligible_cases) - int(split_cases)) if eligible_cases is not None and split_cases is not None else "n/a",
        "train_n": _fmt_int(splits.get("train")),
        "val_n": _fmt_int(splits.get("val")),
        "test_n": _fmt_int(splits.get("test")),
    }


def draw(
    out_path: Path,
    *,
    summary_path: Path,
    download_run_path: Path,
    clinical_csv_path: Path,
    total_cases_override: int | None,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(8.4, 10.2), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Palette matched to the FedOcw figures: white background, green client/data
    # nodes, light-blue process arrows/output, and orange exclusion branches.
    main_fill = "#f3faf4"
    main_edge = "#2ca25f"
    process_fill = "#f7fbff"
    output_fill = "#edf5ff"
    output_edge = "#6fa8dc"
    side_fill = "#fff7ed"
    side_edge = "#d97924"
    main_arrow = "#7fa6cf"
    side_arrow = "#c66a1e"
    counts = _counts(summary_path, download_run_path, clinical_csv_path, total_cases_override)

    ax.text(
        0.08,
        0.965,
        "Data construction and client split flow",
        ha="left",
        va="center",
        fontsize=13.0,
        fontweight="bold",
    )
    ax.text(
        0.08,
        0.935,
        "Cohort attrition from VitalDB metadata to federated client-level datasets.",
        ha="left",
        va="center",
        fontsize=8.6,
        color="0.25",
    )

    main_x = 0.36
    side_x = 0.77
    main_w = 0.48
    main_h = 0.092
    side_w = 0.30
    side_h = 0.072
    ys = {
        "raw": 0.875,
        "signals": 0.755,
        "eligible": 0.635,
        "windows": 0.515,
        "qc": 0.395,
        "client": 0.275,
        "split": 0.155,
    }

    _node(
        ax,
        (main_x, ys["raw"]),
        (main_w, main_h),
        "VitalDB cohort",
        "clinical_data.csv",
        stats=f"n = {counts['raw_n']} cases",
        facecolor=main_fill,
        edgecolor=main_edge,
        title_size=9.0,
        body_size=7.4,
        stats_size=6.4,
    )
    _node(
        ax,
        (main_x, ys["signals"]),
        (main_w, main_h),
        "Signal-complete cases",
        "HR / SpO2 / ETCO2 / FIO2 at 100 Hz",
        stats=f"{counts['raw_n']} -> {counts['downloaded_n']}  (drop {counts['download_drop']})",
        facecolor=process_fill,
        edgecolor=main_edge,
        title_size=9.0,
        body_size=7.2,
        stats_size=6.2,
    )
    _node(
        ax,
        (main_x, ys["eligible"]),
        (main_w, main_h),
        "Label-eligible cases",
        r"$\bar{s}_{1Hz}(t)\leq92,\; L\geq60$s" + "\n"
        + r"$\bar{s}_{1Hz}(t)\geq95,\; L\geq20$min",
        stats=f"{counts['downloaded_n']} -> {counts['eligible_n']}  (drop {counts['eligible_drop']}); events = {counts['events_n']}",
        facecolor=process_fill,
        edgecolor=main_edge,
        title_size=9.0,
        body_size=7.0,
        stats_size=5.9,
    )
    _node(
        ax,
        (main_x, ys["windows"]),
        (main_w, main_h),
        "Window construction",
        r"$X_i=[t_p-30s,t_p]$" + "\n" + r"$t_p=t_0-5$min,  $y_i\in\{0,1\}$",
        stats=f"positive = {counts['pos_units']}; negative target = {counts['neg_target']}",
        facecolor=process_fill,
        edgecolor=main_edge,
        title_size=9.0,
        body_size=7.0,
        stats_size=6.0,
    )
    _node(
        ax,
        (main_x, ys["qc"]),
        (main_w, main_h),
        "Quality control and writing",
        "finite inputs; SpO2 in [50, 100]",
        stats=f"{counts['eligible_n']} -> {counts['written_cases']} cases (drop {counts['written_case_drop']}); "
        + f"{counts['planned_windows']} -> {counts['written_windows']} windows",
        facecolor=process_fill,
        edgecolor=main_edge,
        title_size=9.0,
        body_size=7.0,
        stats_size=5.75,
    )
    _node(
        ax,
        (main_x, ys["client"]),
        (main_w, main_h),
        "Client assignment",
        r"$c_i=g(\mathrm{department},\mathrm{opname},\mathrm{optype})$",
        stats=f"{counts['eligible_n']} -> {counts['eligible_n']} cases (drop 0); C = {counts['clients_n']} clients",
        facecolor=main_fill,
        edgecolor=main_edge,
        title_size=9.0,
        body_size=7.0,
        stats_size=6.0,
    )
    _node(
        ax,
        (main_x, ys["split"]),
        (main_w, main_h),
        "Within-client split",
        r"$D_c=D_c^{tr}\cup D_c^{val}\cup D_c^{te}$; 70 / 10 / 20",
        stats=f"{counts['eligible_n']} -> {counts['eligible_n']} cases (drop {counts['split_drop']}); "
        + f"train/val/test = {counts['train_n']} / {counts['val_n']} / {counts['test_n']}",
        facecolor=output_fill,
        edgecolor=output_edge,
        lw=1.25,
        title_size=9.0,
        body_size=7.0,
        stats_size=5.75,
    )

    # Exclusion / side summary boxes.
    _node(
        ax,
        (side_x, 0.815),
        (side_w, side_h),
        "Excluded before download",
        "age < 18 or missing required tracks",
        stats=f"n = {counts['download_drop']}  (age {counts['age_drop']}; tracks {counts['track_drop']})",
        facecolor=side_fill,
        edgecolor=side_edge,
        title_size=7.2,
        body_size=5.8,
        stats_size=5.4,
    )
    _node(
        ax,
        (side_x, 0.695),
        (side_w, side_h),
        "Excluded in dataset construction",
        "no usable label/window under criteria",
        stats=f"n = {counts['eligible_drop']}",
        facecolor=side_fill,
        edgecolor=side_edge,
        title_size=7.2,
        body_size=5.8,
        stats_size=5.4,
    )
    _node(
        ax,
        (side_x, 0.455),
        (side_w, side_h),
        "Dropped by QC / writing",
        "invalid windows or no written sample",
        stats=f"cases = {counts['written_case_drop']}; windows = {counts['window_drop']}",
        facecolor=side_fill,
        edgecolor=side_edge,
        title_size=7.2,
        body_size=5.8,
        stats_size=5.4,
    )
    _node(
        ax,
        (side_x, ys["split"]),
        (side_w, 0.078),
        "Output dataset",
        "federated_data/<client>/<split>/case_*.npz",
        stats=f"written cases = {counts['written_cases']}; windows = {counts['written_windows']}; C = {counts['clients_n']}",
        facecolor=output_fill,
        edgecolor=output_edge,
        lw=1.25,
        title_size=7.2,
        body_size=5.5,
        stats_size=5.1,
    )

    for upper, lower in [
        ("raw", "signals"),
        ("signals", "eligible"),
        ("eligible", "windows"),
        ("windows", "qc"),
        ("qc", "client"),
        ("client", "split"),
    ]:
        _arrow(ax, (main_x, ys[upper] - main_h / 2), (main_x, ys[lower] + main_h / 2), color=main_arrow)

    _arrow(ax, (main_x + main_w / 2, 0.815), (side_x - side_w / 2, 0.815), color=side_arrow)
    _arrow(ax, (main_x + main_w / 2, 0.695), (side_x - side_w / 2, 0.695), color=side_arrow)
    _arrow(ax, (main_x + main_w / 2, 0.455), (side_x - side_w / 2, 0.455), color=side_arrow)
    _arrow(ax, (main_x + main_w / 2, ys["split"]), (side_x - side_w / 2, ys["split"]), color=main_arrow)

    ax.text(
        0.08,
        0.045,
        r"Positive onset $t_0$: first second of sustained hypoxemia. Negative windows: stable normoxemia segments.",
        ha="left",
        va="center",
        fontsize=7.2,
        color="0.25",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a minimal method-overview figure up to client splitting.")
    parser.add_argument("--out", type=Path, default=Path("figures/fig1_client_split_flow.pdf"))
    parser.add_argument("--summary", type=Path, default=Path("federated_data/summary.json"))
    parser.add_argument("--download-run", type=Path, default=Path("download_run.json"))
    parser.add_argument("--clinical-csv", type=Path, default=Path("clinical_data.csv"))
    parser.add_argument("--total-cases", type=int, default=None, help="Override total VitalDB case count shown at the start.")
    args = parser.parse_args()
    draw(
        args.out,
        summary_path=args.summary,
        download_run_path=args.download_run,
        clinical_csv_path=args.clinical_csv,
        total_cases_override=args.total_cases,
    )


if __name__ == "__main__":
    main()
