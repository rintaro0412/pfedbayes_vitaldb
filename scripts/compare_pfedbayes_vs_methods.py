from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


DEFAULT_METHODS: Tuple[Tuple[str, str, str], ...] = (
    ("Centralized", "ioh", "runs/centralized/seed0"),
    ("FedAvg", "ioh", "runs/fedavg/seed42"),
    ("FedProx", "ioh", "runs/fedprox/seed42"),
    ("SCAFFOLD", "ioh", "runs/scaffold/seed42"),
    ("FedNova", "ioh", "runs/fednova/seed42"),
    ("Per-FedAvg", "ioh", "runs/perfedavg/seed42"),
    ("pFedMe", "ioh", "runs/pfedme/seed42"),
)

HIGHER_IS_BETTER = {"auprc", "auroc"}
LOWER_IS_BETTER = {"ece", "brier", "nll"}


def _parse_method(raw: str) -> Tuple[str, str, str]:
    parts = str(raw).split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--method must be formatted as 'Label:kind:run_dir', e.g. 'FedAvg:ioh:runs/fedavg/seed42'"
        )
    label, kind, run_dir = (p.strip() for p in parts)
    if not label or not kind or not run_dir:
        raise argparse.ArgumentTypeError("--method label, kind, and run_dir must be non-empty")
    if kind not in {"ioh", "bfl"}:
        raise argparse.ArgumentTypeError("--method kind must be 'ioh' or 'bfl'")
    return label, kind, run_dir


def _safe_name(label: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(label))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "method"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _add_adjusted_p_values(rows: List[Dict[str, Any]]) -> None:
    """Add per-metric Holm and Benjamini-Hochberg adjusted p-values in place."""
    by_metric: Dict[str, List[Tuple[int, float]]] = {}
    for idx, row in enumerate(rows):
        p = _finite_float(row.get("p_two_sided"))
        if p is None:
            row["p_holm"] = None
            row["p_bh"] = None
            continue
        by_metric.setdefault(str(row.get("metric")), []).append((idx, p))

    for pairs in by_metric.values():
        ordered = sorted(pairs, key=lambda item: item[1])
        m = len(ordered)
        if m == 0:
            continue

        # Holm-Bonferroni: step-down, monotone non-decreasing adjusted p-values.
        running = 0.0
        for rank, (idx, p) in enumerate(ordered, start=1):
            adj = min(1.0, (m - rank + 1) * p)
            running = max(running, adj)
            rows[idx]["p_holm"] = running

        # Benjamini-Hochberg: step-up, monotone non-decreasing in sorted order.
        running_bh = 1.0
        bh_values: Dict[int, float] = {}
        for rank, (idx, p) in reversed(list(enumerate(ordered, start=1))):
            adj = min(1.0, (m / rank) * p)
            running_bh = min(running_bh, adj)
            bh_values[idx] = running_bh
        for idx, _ in ordered:
            rows[idx]["p_bh"] = bh_values[idx]


def _summarize_compare(label: str, path: Path) -> List[Dict[str, Any]]:
    payload = _load_json(path)
    rows: List[Dict[str, Any]] = []
    comparison = payload.get("comparison", {}) or {}
    for metric, rec in sorted(comparison.items()):
        if not isinstance(rec, dict):
            continue
        diff = _finite_float(rec.get("diff"))
        better = None
        if diff is not None:
            if metric in HIGHER_IS_BETTER:
                better = diff > 0.0
            elif metric in LOWER_IS_BETTER:
                better = diff < 0.0
        rows.append(
            {
                "comparator": label,
                "metric": metric,
                "comparator_value": rec.get("a"),
                "pfedbayes_value": rec.get("b"),
                "diff_pfedbayes_minus_comparator": rec.get("diff"),
                "diff_ci_low": rec.get("diff_p2_5"),
                "diff_ci_high": rec.get("diff_p97_5"),
                "p_two_sided": rec.get("p_two_sided"),
                "p_holm": None,
                "p_bh": None,
                "p_greater": rec.get("p_greater"),
                "p_less": rec.get("p_less"),
                "pfedbayes_better": better,
                "source_json": str(path),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run paired case-level bootstrap comparisons with pFedBayes as the reference. "
            "Each comparison calls scripts/compare_significance.py with method A=comparator and method B=pFedBayes, "
            "so reported diffs are pFedBayes minus comparator."
        )
    )
    ap.add_argument("--data-dir", default="federated_data")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--pfedbayes-run", default="runs/pfedbayes/seed42")
    ap.add_argument("--pfedbayes-label", default="pFedBayes")
    ap.add_argument(
        "--method",
        action="append",
        type=_parse_method,
        default=[],
        help="Comparator formatted as 'Label:kind:run_dir'. May be repeated.",
    )
    ap.add_argument(
        "--include-defaults",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include existing default comparator run dirs: Centralized, FedAvg, FedProx, SCAFFOLD, FedNova, Per-FedAvg, pFedMe.",
    )
    ap.add_argument("--variant", default="pre", choices=["pre", "post"])
    ap.add_argument("--mc-eval", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--bootstrap-n", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=42)
    ap.add_argument("--out-dir", default="outputs_plan/significance_pfedbayes_vs_methods")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods: List[Tuple[str, str, str]] = []
    if bool(args.include_defaults):
        methods.extend([m for m in DEFAULT_METHODS if Path(m[2]).exists()])
    methods.extend(args.method or [])

    if not methods:
        raise SystemExit("No comparator methods found. Use --method or --include-defaults.")
    if not Path(args.pfedbayes_run).exists():
        raise SystemExit(f"pFedBayes run not found: {args.pfedbayes_run}")

    all_rows: List[Dict[str, Any]] = []
    manifest: Dict[str, Any] = {
        "data_dir": str(args.data_dir),
        "split": str(args.split),
        "pfedbayes_run": str(args.pfedbayes_run),
        "variant": str(args.variant),
        "bootstrap_n": int(args.bootstrap_n),
        "bootstrap_seed": int(args.bootstrap_seed),
        "comparisons": [],
    }

    for label, kind, run_dir in methods:
        run_path = Path(run_dir)
        if not run_path.exists():
            print(f"[skip] {label}: run dir not found: {run_dir}", file=sys.stderr)
            continue

        out_json = out_dir / f"pfedbayes_vs_{_safe_name(label)}_{args.split}_{args.variant}.json"
        cmd = [
            str(args.python),
            "scripts/compare_significance.py",
            "--data-dir",
            str(args.data_dir),
            "--split",
            str(args.split),
            "--a-kind",
            str(kind),
            "--a-run-dir",
            str(run_path),
            "--a-label",
            str(label),
            "--b-kind",
            "bfl",
            "--b-run-dir",
            str(args.pfedbayes_run),
            "--b-label",
            str(args.pfedbayes_label),
            "--variant",
            str(args.variant),
            "--mc-eval",
            str(int(args.mc_eval)),
            "--batch-size",
            str(int(args.batch_size)),
            "--num-workers",
            str(int(args.num_workers)),
            "--bootstrap-n",
            str(int(args.bootstrap_n)),
            "--bootstrap-seed",
            str(int(args.bootstrap_seed)),
            "--out",
            str(out_json),
        ]
        manifest["comparisons"].append({"label": label, "kind": kind, "run_dir": str(run_path), "out_json": str(out_json), "cmd": cmd})
        print(" ".join(cmd))
        if bool(args.dry_run):
            continue

        subprocess.run(cmd, check=True)
        all_rows.extend(_summarize_compare(label, out_json))

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not bool(args.dry_run):
        _add_adjusted_p_values(all_rows)
        summary_json = out_dir / "summary.json"
        summary_csv = out_dir / "summary.csv"
        summary_json.write_text(json.dumps({"rows": all_rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "comparator",
                "metric",
                "comparator_value",
                "pfedbayes_value",
                "diff_pfedbayes_minus_comparator",
                "diff_ci_low",
                "diff_ci_high",
                "p_two_sided",
                "p_holm",
                "p_bh",
                "p_greater",
                "p_less",
                "pfedbayes_better",
                "source_json",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved summary: {summary_csv}")
        print(f"Saved manifest: {manifest_path}")
    else:
        print(f"Saved dry-run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
