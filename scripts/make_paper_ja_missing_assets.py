from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MethodSpec:
    label: str
    config_name: str
    run_dir: str
    # Checkpoint scope is separate from aggregation; client macro is not personalized inference.
    model_scope: str = "global"


METHOD_SPECS: Tuple[MethodSpec, ...] = (
    MethodSpec("Centralized", "centralized.yaml", "runs/centralized/seed0"),
    MethodSpec("Local", "local.yaml", "runs/local/seed42", "client_specific"),
    MethodSpec("FedAvg", "fedavg.yaml", "runs/fedavg/seed42"),
    MethodSpec("FedProx", "fedprox.yaml", "runs/fedprox/seed42"),
    MethodSpec("SCAFFOLD", "scaffold.yaml", "runs/scaffold/seed42"),
    MethodSpec("FedNova", "fednova.yaml", "runs/fednova/seed42"),
    MethodSpec("Per-FedAvg", "perfedavg.yaml", "runs/perfedavg/seed42"),
    MethodSpec("pFedMe", "pfedme.yaml", "runs/pfedme/seed42"),
    MethodSpec("pFedBayes", "pfedbayes.yaml", "runs/pfedbayes/seed42"),
)


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # type: ignore

    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _parse_method_specs(path: Path | None, *, repo_root: Path) -> Tuple[MethodSpec, ...]:
    if path is None:
        return METHOD_SPECS
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = _load_yaml(path)
    rows = data.get("methods") if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise ValueError("Method manifest must contain a nonempty methods list.")
    methods: List[MethodSpec] = []
    labels: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Method entries must be objects.")
        label = str(row.get("label", row.get("method", row.get("key", "")))).strip()
        run_dir = str(row.get("run_dir", row.get("source_run_dir", ""))).strip()
        config = str(row.get("config", row.get("config_path", ""))).strip()
        if not label or not run_dir or not config:
            raise ValueError("Each method requires a label, source_run_dir/run_dir, and config/config_path.")
        if label in labels:
            raise ValueError(f"Duplicate method label: {label}")
        scope = str(row.get("model_scope", "client_specific" if label == "Local" else "global"))
        if scope not in {"global", "client_specific"}:
            raise ValueError(f"Unsupported model_scope for {label}: {scope}")
        labels.add(label)
        methods.append(MethodSpec(label, str(repo_root / config), run_dir, scope))
    return tuple(methods)


def _normalize_saved_config(config: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("Saved run config must be an object.")
    args = config.get("args")
    if isinstance(args, dict):
        # Point-estimate trainers save resolved CLI arguments instead of nested YAML.
        return {"train": args, "eval": args}
    if "train" not in config and isinstance(config.get("hyper"), dict):
        train = dict(config["hyper"])
        for key in ("rounds", "epochs", "seed"):
            if key in config:
                train[key] = config[key]
        return {"train": train, "eval": config.get("eval", config["hyper"])}
    return config


def _method_config(
    method: MethodSpec, *, repo_root: Path, configs_dir: Path, require_saved: bool,
) -> Tuple[Dict[str, Any], Path, str]:
    run_dir = repo_root / method.run_dir
    if require_saved and not run_dir.is_dir():
        raise FileNotFoundError(f"Source run not found for {method.label}: {run_dir}")
    for name in ("config.json", "config_used.yaml", "run_config.json"):
        path = run_dir / name
        if path.is_file():
            config = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else _load_yaml(path)
            return _normalize_saved_config(config), path, "saved_run_config"
    if require_saved:
        raise FileNotFoundError(f"Saved run config not found for {method.label}: {run_dir}")
    path = configs_dir / method.config_name
    return _normalize_saved_config(_load_yaml(path)), path, "current_config_fallback"


def _normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Local reports use raw names; *_pre explicitly denotes uncalibrated metrics elsewhere.
    df = df.copy()
    for metric in ("auprc", "auroc", "brier", "nll", "ece"):
        target = f"{metric}_pre"
        if target not in df.columns and metric in df.columns:
            df[target] = df[metric]
    return df


def _save_table(df: pd.DataFrame, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def _save_manifest(tables_dir: Path, logs: Dict[str, Any]) -> Path:
    path = tables_dir / "paper_ja_missing_assets_manifest.json"
    path.write_text(json.dumps(logs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _set_clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_flow_figure(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 3), dpi=220)
    ax.axis("off")

    steps = [
        "VitalDB Download",
        "Window Extraction",
        "Client Split",
        "Model Training",
        "Evaluation",
        "Paper Tables/Figures",
    ]
    x_positions = np.linspace(0.06, 0.94, len(steps))
    y = 0.52

    for i, (x, name) in enumerate(zip(x_positions, steps)):
        rect = plt.Rectangle((x - 0.07, y - 0.10), 0.14, 0.20, fill=False, linewidth=1.8, edgecolor="black")
        ax.add_patch(rect)
        ax.text(x, y, name, ha="center", va="center", fontsize=10)
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x_positions[i + 1] - 0.08, y),
                xytext=(x + 0.08, y),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="black"),
            )

    ax.set_title("Figure 1: Research Flow", fontsize=12, pad=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=220)
    plt.close(fig)


def _draw_method_overview(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4), dpi=220)
    ax.axis("off")

    box_w = 0.24
    box_h = 0.30
    boxes = {
        "Centralized": (0.12, 0.58),
        "FedAvg": (0.44, 0.58),
        "pFedBayes": (0.76, 0.58),
        "Server": (0.44, 0.20),
    }

    for name, (x, y) in boxes.items():
        rect = plt.Rectangle((x - box_w / 2, y - box_h / 2), box_w, box_h, fill=False, lw=1.8, ec="black")
        ax.add_patch(rect)
        ax.text(x, y, name, ha="center", va="center", fontsize=11)

    ax.text(0.12, 0.38, "All data pooled", ha="center", va="center", fontsize=9)
    ax.text(0.44, 0.38, "Point estimate\naggregate", ha="center", va="center", fontsize=9)
    ax.text(0.76, 0.38, "Posterior (mu,sigma)\naggregate", ha="center", va="center", fontsize=9)

    ax.annotate("", xy=(0.44, 0.35), xytext=(0.44, 0.50), arrowprops=dict(arrowstyle="<->", lw=1.8))
    ax.annotate("", xy=(0.76, 0.35), xytext=(0.60, 0.35), arrowprops=dict(arrowstyle="->", lw=1.8))
    ax.annotate("", xy=(0.60, 0.35), xytext=(0.44, 0.35), arrowprops=dict(arrowstyle="->", lw=1.8))

    ax.set_title("Figure 3: Method Overview (Centralized / FedAvg / pFedBayes)", fontsize=12, pad=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=220)
    plt.close(fig)


def _draw_noniid_from_summary(summary_json: Path, out_path: Path) -> Dict[str, Any]:
    js = json.loads(summary_json.read_text(encoding="utf-8"))
    clients = js.get("clients_detail") or {}
    if not isinstance(clients, dict) or not clients:
        raise ValueError(f"clients_detail missing in {summary_json}")

    def _label_from_client_id(client_id: str) -> str:
        cid = str(client_id)
        if "__" in cid:
            dept, proc = cid.split("__", 1)
        else:
            dept, proc = cid, ""
        dept = dept.replace("_", " ")
        proc = proc.replace("_", " ")
        if len(proc) > 36:
            proc = proc[:33] + "..."
        if proc:
            return f"{dept} | {proc}"
        return dept

    rows = []
    for cid, rec in clients.items():
        pos = int(rec.get("pos_windows", 0))
        neg = int(rec.get("neg_windows", 0))
        n = pos + neg
        rows.append(
            {
                "client_id": str(cid),
                "cases": int(rec.get("cases", 0)),
                "n_windows": int(n),
                "pos_windows": int(pos),
                "pos_rate": (float(pos) / float(max(n, 1))),
            }
        )
    df = pd.DataFrame(rows).sort_values("cases", ascending=False).reset_index(drop=True)
    df["client_label"] = [_label_from_client_id(cid) for cid in df["client_id"].astype(str).tolist()]

    n_clients = int(len(df))
    fig_h = max(7.0, 0.30 * n_clients + 1.8)
    fig, (ax_cases, ax_pos) = plt.subplots(
        1,
        2,
        figsize=(15, fig_h),
        dpi=220,
        sharey=True,
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )
    y = np.arange(n_clients)

    case_vals = df["cases"].to_numpy(dtype=float)
    pos_vals = df["pos_rate"].to_numpy(dtype=float)

    ax_cases.barh(y, case_vals, color="lightgray", edgecolor="black", linewidth=0.7)
    ax_cases.set_yticks(y)
    ax_cases.set_yticklabels(df["client_label"].tolist(), fontsize=8)
    ax_cases.invert_yaxis()
    ax_cases.set_xlabel("Cases")
    ax_cases.set_title("Cases per client", fontsize=11)
    ax_cases.grid(True, axis="x", linestyle="--", alpha=0.35)
    _set_clean_axes(ax_cases)

    x_offset = max(float(np.nanmax(case_vals)) * 0.012, 0.8)
    for yi, cv in zip(y, case_vals):
        ax_cases.text(float(cv) + x_offset, yi, f"{int(cv)}", va="center", ha="left", fontsize=7, color="dimgray")

    ax_pos.plot(pos_vals, y, color="black", marker="o", linewidth=1.4, markersize=3.2)
    ax_pos.set_xlabel("Positive rate")
    ax_pos.set_title("Positive rate per client", fontsize=11)
    ax_pos.set_xlim(0.0, max(0.5, float(np.nanmax(pos_vals)) * 1.12))
    ax_pos.grid(True, axis="x", linestyle="--", alpha=0.35)
    _set_clean_axes(ax_pos)

    fig.suptitle("Figure 2: Non-IID Client Distribution", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=220)
    plt.close(fig)
    return {"n_clients": int(len(df)), "min_cases": int(df["cases"].min()), "max_cases": int(df["cases"].max())}


def _extract_lr(cfg: Dict[str, Any], *, method: str) -> str:
    train = cfg.get("train", {}) if isinstance(cfg, dict) else {}
    if method == "pFedBayes":
        lr_q = train.get("lr_q")
        lr_w = train.get("lr_w")
        if lr_q is None and lr_w is None:
            return ""
        return f"lr_q={lr_q}, lr_w={lr_w}"
    lr = train.get("lr")
    return "" if lr is None else str(lr)


def _build_table2(
    configs_dir: Path, methods: Iterable[MethodSpec], *,
    repo_root: Path = Path("."), require_saved: bool = False,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for m in methods:
        cfg, config_path, config_source = _method_config(
            m, repo_root=repo_root, configs_dir=configs_dir, require_saved=require_saved,
        )
        train = cfg.get("train", {}) if isinstance(cfg, dict) else {}
        eval_cfg = cfg.get("eval", {}) if isinstance(cfg, dict) else {}

        rounds = train.get("rounds")
        if rounds is None:
            rounds = train.get("epochs")
        local_epochs = train.get("local_epochs")
        if m.label in {"Centralized", "Local"} and local_epochs is None:
            local_epochs = "-"

        rows.append(
            {
                "method": m.label,
                "rounds_or_epochs": rounds if rounds is not None else "",
                "local_epochs": local_epochs if local_epochs is not None else "",
                "batch_size": train.get("batch_size", ""),
                "seed": train.get("seed", cfg.get("seed", "")),
                "learning_rate": _extract_lr(cfg, method=m.label),
                "model_selection": eval_cfg.get("model_selection", ""),
                "selection_metric": eval_cfg.get("selection_metric", ""),
                "mc_train": train.get("mc_train", ""),
                "mc_eval": train.get("mc_eval", ""),
                "config_path": str(config_path),
                "config_source": config_source,
                "source_run_dir": m.run_dir,
                "model_scope": m.model_scope,
            }
        )
    return pd.DataFrame(rows)


def _load_per_client_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or "client_id" not in df.columns:
        return None
    cols = [str(c).strip() for c in df.columns]
    df.columns = cols
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower().eq("ok")].copy()
    return _normalize_metric_columns(df)


def _method_per_client_paths(repo_root: Path, methods: Iterable[MethodSpec]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for m in methods:
        out[m.label] = repo_root / m.run_dir / "test_report_per_client.csv"
    return out


def _metric_col(metric: str) -> str:
    m = str(metric).lower().strip()
    if m.endswith("_pre"):
        return m
    return f"{m}_pre"


def _compute_table4(
    per_client: Dict[str, pd.DataFrame], metrics: List[str], *,
    model_scopes: Dict[str, str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, Any]] = []
    long_rows: List[Dict[str, Any]] = []
    for method, df in per_client.items():
        df = _normalize_metric_columns(df)
        scope = (model_scopes or {}).get(method, "client_specific" if method == "Local" else "global")
        row: Dict[str, Any] = {"method": method, "model_scope": scope, "aggregation": "client_macro"}
        for metric in metrics:
            col = _metric_col(metric)
            if col not in df.columns:
                row[f"macro_{metric}"] = float("nan")
                row[f"median_{metric}"] = float("nan")
                row[f"iqr_{metric}"] = float("nan")
                row[f"min_{metric}"] = float("nan")
                row[f"max_{metric}"] = float("nan")
                continue
            vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size == 0:
                row[f"macro_{metric}"] = float("nan")
                row[f"median_{metric}"] = float("nan")
                row[f"iqr_{metric}"] = float("nan")
                row[f"min_{metric}"] = float("nan")
                row[f"max_{metric}"] = float("nan")
                continue
            q1, q3 = np.percentile(vals, [25.0, 75.0])
            row[f"macro_{metric}"] = float(np.mean(vals))
            row[f"median_{metric}"] = float(np.median(vals))
            row[f"iqr_{metric}"] = float(q3 - q1)
            row[f"min_{metric}"] = float(np.min(vals))
            row[f"max_{metric}"] = float(np.max(vals))
            long_rows.append(
                {
                    "method": method,
                    "model_scope": scope,
                    "aggregation": "client_macro",
                    "metric": metric,
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "iqr": float(q3 - q1),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "n_clients": int(vals.size),
                }
            )

        auprc_col = _metric_col("auprc")
        if auprc_col in df.columns:
            vals = pd.to_numeric(df[auprc_col], errors="coerce").dropna()
            row["worst_client_auprc"] = float(vals.min()) if len(vals) else float("nan")
            row["n_clients"] = int(len(vals))
        else:
            row["worst_client_auprc"] = float("nan")
            row["n_clients"] = 0
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    if not df_summary.empty:
        df_summary = df_summary.sort_values("method").reset_index(drop=True)
    df_long = pd.DataFrame(long_rows)
    if not df_long.empty:
        df_long = df_long.sort_values(["metric", "method"]).reset_index(drop=True)
    return df_summary, df_long


def _build_forest_from_summary_csv(summary_csv: Path, metric: str) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    need_cols = {"comparator", "metric", "diff_pfedbayes_minus_comparator", "diff_ci_low", "diff_ci_high"}
    if not need_cols.issubset(df.columns):
        raise ValueError(f"columns missing in {summary_csv}")
    d = df[df["metric"].astype(str).str.lower().eq(metric.lower())].copy()
    if d.empty:
        raise ValueError(f"metric '{metric}' not found in {summary_csv}")
    d = d.rename(
        columns={
            "comparator": "method",
            "diff_pfedbayes_minus_comparator": "diff",
            "diff_ci_low": "ci_low",
            "diff_ci_high": "ci_high",
        }
    )[["method", "diff", "ci_low", "ci_high"]]
    d["source"] = "summary_csv"
    return d


def _bootstrap_ci_mean(diffs: np.ndarray, n_boot: int, rng: np.random.Generator) -> Tuple[float, float]:
    if diffs.size == 0:
        return float("nan"), float("nan")
    if diffs.size == 1:
        v = float(diffs[0])
        return v, v
    boots = []
    n = diffs.size
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        boots.append(float(np.mean(diffs[idx])))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _build_forest_fallback(
    per_client: Dict[str, pd.DataFrame],
    metric: str,
    *,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    target = "pFedBayes"
    col = _metric_col(metric)
    if target not in per_client or col not in per_client[target].columns:
        raise ValueError("pFedBayes data missing for fallback forest")

    ref = per_client[target][["client_id", col]].copy().rename(columns={col: "pfedbayes"})
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))
    for method, df in per_client.items():
        if method == target:
            continue
        if col not in df.columns:
            continue
        cmp_df = df[["client_id", col]].copy().rename(columns={col: "cmp"})
        m = pd.merge(ref, cmp_df, on="client_id", how="inner")
        if m.empty:
            continue
        diffs = (pd.to_numeric(m["pfedbayes"], errors="coerce") - pd.to_numeric(m["cmp"], errors="coerce")).dropna().to_numpy(dtype=float)
        if diffs.size == 0:
            continue
        ci_low, ci_high = _bootstrap_ci_mean(diffs, int(n_boot), rng)
        rows.append(
            {
                "method": method,
                "diff": float(np.mean(diffs)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "source": "client_bootstrap",
            }
        )
    if not rows:
        raise ValueError("no comparator rows for fallback forest")
    return pd.DataFrame(rows)


def _draw_forest(df: pd.DataFrame, out_path: Path, metric: str) -> None:
    d = df.copy()
    d = d.sort_values("diff", ascending=True).reset_index(drop=True)
    y = np.arange(len(d))
    x = d["diff"].to_numpy(dtype=float)
    lo = d["ci_low"].to_numpy(dtype=float)
    hi = d["ci_high"].to_numpy(dtype=float)
    xerr = np.vstack([x - lo, hi - x])

    fig, ax = plt.subplots(figsize=(8, max(3.2, 0.45 * len(d) + 1.2)), dpi=220)
    ax.errorbar(x, y, xerr=xerr, fmt="o", color="black", ecolor="black", capsize=3, lw=1.4)
    ax.axvline(0.0, color="gray", linestyle="--", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(d["method"].astype(str).tolist())
    ax.set_xlabel(f"pFedBayes - comparator ({metric})")
    ax.set_title("Figure 5: pFedBayes Forest Plot", fontsize=12)
    _set_clean_axes(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=220)
    plt.close(fig)


def _draw_heatmap(per_client: Dict[str, pd.DataFrame], out_path: Path, metric: str) -> Dict[str, Any]:
    col = _metric_col(metric)
    methods = sorted(per_client.keys())
    if not methods:
        raise ValueError("no per-client data for heatmap")

    clients = sorted({str(c) for d in per_client.values() for c in d["client_id"].astype(str).tolist()})
    if not clients:
        raise ValueError("no clients for heatmap")

    mat = np.full((len(clients), len(methods)), np.nan, dtype=float)
    for j, method in enumerate(methods):
        d = per_client[method]
        if col not in d.columns:
            continue
        s = pd.Series(pd.to_numeric(d[col], errors="coerce").to_numpy(dtype=float), index=d["client_id"].astype(str))
        for i, cid in enumerate(clients):
            if cid in s.index:
                mat[i, j] = float(s.loc[cid])

    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(methods) + 2), max(7, 0.18 * len(clients) + 2.5)), dpi=220)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(col)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(clients)))
    ax.set_yticklabels(clients, fontsize=7)
    ax.set_title(f"Figure 9: Client Heatmap ({col})", fontsize=12)
    ax.set_xlabel("Method")
    ax.set_ylabel("Client")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=220)
    plt.close(fig)
    return {"n_clients": int(len(clients)), "n_methods": int(len(methods)), "metric": col}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build missing paper_ja assets: Figure 1/2/3/5/9 and Table 2/4. "
            "Outputs are created under --figures-dir and --tables-dir."
        )
    )
    ap.add_argument("--repo-root", default=".")
    method_source = ap.add_mutually_exclusive_group()
    method_source.add_argument("--manifest", default=None, help="Experiment JSON/YAML with methods, source_run_dir, config, and model_scope.")
    method_source.add_argument("--methods-json", default=None, help="JSON method list; source_run_dir/run_dir and config/config_path are required.")
    ap.add_argument("--configs-dir", default="configs")
    ap.add_argument("--summary-json", default="federated_data/summary.json")
    ap.add_argument("--significance-csv", default="outputs_plan/significance_pfedbayes_vs_methods/summary.csv")
    ap.add_argument("--figures-dir", default="figures")
    ap.add_argument("--tables-dir", default="tables")
    ap.add_argument("--tables-only", action="store_true", help="Generate method settings and client-macro tables with provenance; skip figures and forest bootstrap.")
    ap.add_argument("--forest-metric", default="auprc", choices=["auprc", "auroc", "brier", "nll", "ece"])
    ap.add_argument("--heatmap-metric", default="auprc", choices=["auprc", "auroc", "brier", "nll", "ece"])
    ap.add_argument("--forest-bootstrap-n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    configs_dir = (repo_root / args.configs_dir).resolve()
    summary_json = (repo_root / args.summary_json).resolve()
    significance_csv = (repo_root / args.significance_csv).resolve()
    figures_dir = (repo_root / args.figures_dir).resolve()
    tables_dir = (repo_root / args.tables_dir).resolve()

    source_path = args.manifest or args.methods_json
    source_path = (repo_root / source_path).resolve() if source_path else None
    methods = _parse_method_specs(source_path, repo_root=repo_root)
    table2 = _build_table2(
        configs_dir=configs_dir, methods=methods, repo_root=repo_root,
        require_saved=source_path is not None,
    )

    logs: Dict[str, Any] = {
        "generated": {}, "warnings": [],
        "method_manifest": str(source_path) if source_path else None,
        "aggregation": "client_macro",
        "model_scopes": {m.label: m.model_scope for m in methods},
        "config_provenance": table2[["method", "config_path", "config_source", "source_run_dir"]].to_dict(orient="records"),
        "per_client_sources": {},
    }
    for row in logs["config_provenance"]:
        if row["config_source"] == "current_config_fallback":
            logs["warnings"].append(f"using current config without saved run config: {row['method']}: {row['config_path']}")

    # Table 2
    table2_csv = tables_dir / "table2_method_settings.csv"

    # Per-client inputs (for Table 4, Figure 5 fallback, Figure 9)
    per_client_paths = _method_per_client_paths(repo_root=repo_root, methods=methods)
    per_client_data: Dict[str, pd.DataFrame] = {}
    for method, path in per_client_paths.items():
        d = _load_per_client_csv(path)
        if d is None:
            if source_path is not None:
                raise FileNotFoundError(f"Per-client CSV not available for {method}: {path}")
            logs["warnings"].append(f"missing per-client CSV: {method}: {path}")
            continue
        per_client_data[method] = d
        logs["per_client_sources"][method] = str(path)

    if not args.tables_only:
        figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    _save_table(table2, table2_csv)
    logs["generated"]["table2"] = str(table2_csv)

    # Table 4
    table4_main, table4_long = _compute_table4(
        per_client=per_client_data,
        metrics=["auprc", "auroc", "brier", "nll", "ece"],
        model_scopes=logs["model_scopes"],
    )
    table4_csv = tables_dir / "table4_client_macro.csv"
    _save_table(table4_main, table4_csv)
    logs["generated"]["table4"] = str(table4_csv)
    if not table4_long.empty:
        table4_long_csv = tables_dir / "table4_client_macro_stats_long.csv"
        _save_table(table4_long, table4_long_csv)
        logs["generated"]["table4_long"] = str(table4_long_csv)

    if args.tables_only:
        manifest = _save_manifest(tables_dir, logs)
        print(f"Saved Table 2: {table2_csv}")
        print(f"Saved Table 4: {table4_csv}")
        print(f"Saved manifest: {manifest}")
        return

    # Figure 1 / 3
    fig1 = figures_dir / "fig1_research_flow.pdf"
    _draw_flow_figure(fig1)
    logs["generated"]["fig1"] = str(fig1)

    fig3 = figures_dir / "fig3_method_overview.pdf"
    _draw_method_overview(fig3)
    logs["generated"]["fig3"] = str(fig3)

    # Figure 2
    fig2 = figures_dir / "fig2_noniid_clients.pdf"
    try:
        info2 = _draw_noniid_from_summary(summary_json=summary_json, out_path=fig2)
        logs["generated"]["fig2"] = {"path": str(fig2), "info": info2}
    except Exception as exc:
        logs["warnings"].append(f"failed fig2: {exc}")

    # Figure 5
    fig5 = figures_dir / "fig5_forest.pdf"
    try:
        if significance_csv.exists():
            forest_df = _build_forest_from_summary_csv(significance_csv, metric=str(args.forest_metric))
        else:
            forest_df = _build_forest_fallback(
                per_client=per_client_data,
                metric=str(args.forest_metric),
                n_boot=int(args.forest_bootstrap_n),
                seed=int(args.seed),
            )
        forest_csv = tables_dir / "fig5_forest_data.csv"
        _save_table(forest_df, forest_csv)
        _draw_forest(forest_df, out_path=fig5, metric=str(args.forest_metric))
        logs["generated"]["fig5"] = {"path": str(fig5), "data": str(forest_csv)}
    except Exception as exc:
        logs["warnings"].append(f"failed fig5: {exc}")

    # Figure 9
    fig9 = figures_dir / "fig9_client_heatmap.pdf"
    try:
        info9 = _draw_heatmap(per_client=per_client_data, out_path=fig9, metric=str(args.heatmap_metric))
        logs["generated"]["fig9"] = {"path": str(fig9), "info": info9}
    except Exception as exc:
        logs["warnings"].append(f"failed fig9: {exc}")

    manifest = _save_manifest(tables_dir, logs)

    print(f"Saved Table 2: {table2_csv}")
    print(f"Saved Table 4: {table4_csv}")
    if "fig1" in logs["generated"]:
        print(f"Saved Figure 1: {fig1}")
    if "fig2" in logs["generated"]:
        print(f"Saved Figure 2: {fig2}")
    if "fig3" in logs["generated"]:
        print(f"Saved Figure 3: {fig3}")
    if "fig5" in logs["generated"]:
        print(f"Saved Figure 5: {fig5}")
    if "fig9" in logs["generated"]:
        print(f"Saved Figure 9: {fig9}")
    print(f"Saved manifest: {manifest}")

    if logs["warnings"]:
        print("[WARNINGS]")
        for w in logs["warnings"]:
            print(f"- {w}")


if __name__ == "__main__":
    main()
