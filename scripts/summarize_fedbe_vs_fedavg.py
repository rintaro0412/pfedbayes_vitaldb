from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _summarize(df: pd.DataFrame, *, label: str) -> Dict[str, Any]:
    df = df.sort_values("round")
    out: Dict[str, Any] = {"label": label}
    last = df.iloc[-1].to_dict()
    out["last_round"] = int(last.get("round", -1))
    out["last"] = {k: last.get(k) for k in ["auroc", "auprc", "ece", "nll", "brier", "threshold", "acc", "f1", "precision", "recall"]}

    # best metrics
    if "auroc" in df:
        r = df.loc[df["auroc"].idxmax()]
        out["best_auroc"] = {"round": int(r["round"]), "value": float(r["auroc"])}
    if "auprc" in df:
        r = df.loc[df["auprc"].idxmax()]
        out["best_auprc"] = {"round": int(r["round"]), "value": float(r["auprc"])}
    if "ece" in df:
        r = df.loc[df["ece"].idxmin()]
        out["best_ece"] = {"round": int(r["round"]), "value": float(r["ece"])}
    if "nll" in df:
        r = df.loc[df["nll"].idxmin()]
        out["best_nll"] = {"round": int(r["round"]), "value": float(r["nll"])}
    if "brier" in df:
        r = df.loc[df["brier"].idxmin()]
        out["best_brier"] = {"round": int(r["round"]), "value": float(r["brier"])}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize FedAvg vs FedBE round metrics")
    ap.add_argument("--fedavg-run", required=True)
    ap.add_argument("--fedbe-run", required=True)
    ap.add_argument("--out-json", default="outputs/fedbe_vs_fedavg_summary.json")
    ap.add_argument("--out-csv", default="outputs/fedbe_vs_fedavg_summary.csv")
    args = ap.parse_args()

    fedavg = _load(Path(args.fedavg_run) / "metrics_round.csv")
    fedbe = _load(Path(args.fedbe_run) / "metrics_round.csv")

    summ = {
        "fedavg": _summarize(fedavg, label="FedAvg"),
        "fedbe": _summarize(fedbe, label="FedBE"),
        "fedavg_run": str(args.fedavg_run),
        "fedbe_run": str(args.fedbe_run),
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summ, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # flat CSV (last + bests)
    rows = []
    for key in ["fedavg", "fedbe"]:
        s = summ[key]
        row = {"label": s.get("label"), "last_round": s.get("last_round")}
        for k, v in (s.get("last") or {}).items():
            row[f"last_{k}"] = v
        for k in ["best_auroc", "best_auprc", "best_ece", "best_nll", "best_brier"]:
            d = s.get(k, {})
            row[f"{k}_round"] = d.get("round")
            row[f"{k}_value"] = d.get("value")
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)


if __name__ == "__main__":
    main()
