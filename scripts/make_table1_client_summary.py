from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Ensure project root in sys.path when executed as `python scripts/make_table1_client_summary.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.io import now_utc_iso, write_json


DEPARTMENT_JA = {
    "General surgery": "一般外科",
    "Thoracic surgery": "胸部外科",
    "Gynecology": "婦人科",
    "Urology": "泌尿器科",
}


OPTYPE_JA = {
    "Anterior resection": "前方切除",
    "Biliary/Pancreas": "胆嚢・膵臓",
    "Breast": "乳腺",
    "Breast conserving surgery": "乳房温存術",
    "Cholecystectomy": "胆嚢摘出",
    "Colorectal": "大腸",
    "Distal gastrectomy": "幽門側胃切除",
    "Excision": "切除",
    "Exploratory laparotomy": "試験開腹",
    "Hemicolectomy": "半結腸切除",
    "Hepatic": "肝臓",
    "Hernia repair": "ヘルニア修復",
    "Ileostomy repair": "回腸瘻修復",
    "Ligation and stripping": "結紮・ストリッピング",
    "Low anterior resection": "低位前方切除",
    "Lung lobectomy": "肺葉切除",
    "Lung wedge resection": "肺部分切除",
    "Major resection": "大切除",
    "Minor resection": "小切除",
    "Others": "その他",
    "Pylorus preserving pancreaticoduodenectomy": "幽門輪温存膵頭十二指腸切除",
    "Stomach": "胃",
    "Thyroid lobectomy": "甲状腺葉切除",
    "Total thyroidectomy": "甲状腺全摘",
    "Transplantation": "移植",
    "Vascular": "血管",
    "OtherSurgery": "その他",
}


def _load_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _scan_counts(data_dir: Path, splits: List[str]) -> Dict[str, Dict[str, Dict[str, float]]]:
    clients = sorted([p.name for p in data_dir.iterdir() if p.is_dir()])
    stats: Dict[str, Dict[str, Dict[str, float]]] = {c: {s: {"n_files": 0.0, "n_windows": 0.0, "n_pos": 0.0, "n_neg": 0.0} for s in splits} for c in clients}
    for c in clients:
        for s in splits:
            files = sorted((data_dir / c / s).glob("*.npz"))
            stats[c][s]["n_files"] = float(len(files))
            for f in files:
                with np.load(f, allow_pickle=False) as z:
                    if "y" not in z:
                        continue
                    y = np.asarray(z["y"], dtype=np.int64)
                    n = int(y.size)
                    pos = int((y > 0).sum())
                    stats[c][s]["n_windows"] += float(n)
                    stats[c][s]["n_pos"] += float(pos)
                    stats[c][s]["n_neg"] += float(n - pos)
    return stats


def _pooled(stats: Dict[str, Dict[str, Dict[str, float]]], splits: List[str]) -> Dict[str, Dict[str, int]]:
    pooled: Dict[str, Dict[str, int]] = {}
    for c, per_split in stats.items():
        n_files = int(sum(per_split[s]["n_files"] for s in splits))
        n_windows = int(sum(per_split[s]["n_windows"] for s in splits))
        n_pos = int(sum(per_split[s]["n_pos"] for s in splits))
        n_neg = int(sum(per_split[s]["n_neg"] for s in splits))
        pooled[c] = {
            "n_files": n_files,
            "n_windows": n_windows,
            "n_pos": n_pos,
            "n_neg": n_neg,
        }
    return pooled


def _pretty_client_id(client_id: str) -> str:
    if "__" not in client_id:
        return client_id.replace("_", " ")
    dept_raw, optype_raw = client_id.split("__", 1)
    dept = dept_raw.replace("_", " ")
    optype = optype_raw.replace("_", " ")
    dept_ja = DEPARTMENT_JA.get(dept, dept)
    optype_ja = OPTYPE_JA.get(optype, optype)
    return f"{dept} × {optype}\n({dept_ja} × {optype_ja})"


def _fmt_mean_sd(values: pd.Series, digits: int = 1) -> str:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return ""
    mean = float(vals.mean())
    sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def _to_markdown_table(df: pd.DataFrame) -> str:
    def esc(value: Any) -> str:
        text = "" if pd.isna(value) else str(value)
        return text.replace("\n", "<br>").replace("|", "\\|")

    headers = [esc(c) for c in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(esc(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def _build_attached_like_table(case_summary_csv: Path) -> tuple[pd.DataFrame, List[Dict[str, Any]]]:
    df = pd.read_csv(case_summary_csv)
    required = {"client_id", "split", "caseid", "case_has_pos", "clin_age", "clin_bmi", "clin_asa"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise SystemExit(f"{case_summary_csv} is missing required columns: {', '.join(missing)}")

    rows: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []
    split_order = ["train", "val", "test"]
    for client_id, g in df.groupby("client_id", sort=False):
        split_counts = {s: int((g["split"].astype(str) == s).sum()) for s in split_order}
        total_cases = int(sum(split_counts.values()))
        positive_cases = int(pd.to_numeric(g["case_has_pos"], errors="coerce").fillna(0).astype(int).sum())
        positive_rate = (positive_cases / total_cases) if total_cases else float("nan")
        raw = {
            "client_id": str(client_id),
            "client_label": _pretty_client_id(str(client_id)),
            "train_cases": split_counts["train"],
            "val_cases": split_counts["val"],
            "test_cases": split_counts["test"],
            "total_cases": total_cases,
            "positive_cases": positive_cases,
            "positive_rate": positive_rate,
            "age_mean": float(pd.to_numeric(g["clin_age"], errors="coerce").mean()),
            "age_sd": float(pd.to_numeric(g["clin_age"], errors="coerce").std(ddof=1)),
            "bmi_mean": float(pd.to_numeric(g["clin_bmi"], errors="coerce").mean()),
            "bmi_sd": float(pd.to_numeric(g["clin_bmi"], errors="coerce").std(ddof=1)),
            "asa_mean": float(pd.to_numeric(g["clin_asa"], errors="coerce").mean()),
            "asa_sd": float(pd.to_numeric(g["clin_asa"], errors="coerce").std(ddof=1)),
        }
        raw_rows.append(raw)
        rows.append(
            {
                "診療科・術式詳細 (Client)": raw["client_label"],
                "症例数内訳 (train/val/test)": f"{split_counts['train']} / {split_counts['val']} / {split_counts['test']}",
                "総症例数": total_cases,
                "陽性症例数": positive_cases,
                "陽性率": round(positive_rate, 3) if total_cases else "",
                "年齢 (平均 ± SD)": _fmt_mean_sd(g["clin_age"]),
                "BMI (平均 ± SD)": _fmt_mean_sd(g["clin_bmi"]),
                "ASA (平均 ± SD)": _fmt_mean_sd(g["clin_asa"]),
            }
        )

    order = sorted(range(len(raw_rows)), key=lambda i: (-raw_rows[i]["total_cases"], raw_rows[i]["client_id"]))
    rows = [rows[i] for i in order]
    raw_rows = [raw_rows[i] for i in order]

    total_n = int(len(df))
    total_pos = int(pd.to_numeric(df["case_has_pos"], errors="coerce").fillna(0).astype(int).sum())
    total_split_counts = {s: int((df["split"].astype(str) == s).sum()) for s in split_order}
    rows.append(
        {
            "診療科・術式詳細 (Client)": "Total(合計)",
            "症例数内訳 (train/val/test)": f"{total_split_counts['train']} / {total_split_counts['val']} / {total_split_counts['test']}",
            "総症例数": total_n,
            "陽性症例数": total_pos,
            "陽性率": round(total_pos / total_n, 3) if total_n else "",
            "年齢 (平均 ± SD)": _fmt_mean_sd(df["clin_age"]),
            "BMI (平均 ± SD)": _fmt_mean_sd(df["clin_bmi"]),
            "ASA (平均 ± SD)": _fmt_mean_sd(df["clin_asa"]),
        }
    )
    raw_rows.append(
        {
            "client_id": "Total",
            "client_label": "Total(合計)",
            "train_cases": total_split_counts["train"],
            "val_cases": total_split_counts["val"],
            "test_cases": total_split_counts["test"],
            "total_cases": total_n,
            "positive_cases": total_pos,
            "positive_rate": (total_pos / total_n) if total_n else float("nan"),
            "age_mean": float(pd.to_numeric(df["clin_age"], errors="coerce").mean()),
            "age_sd": float(pd.to_numeric(df["clin_age"], errors="coerce").std(ddof=1)),
            "bmi_mean": float(pd.to_numeric(df["clin_bmi"], errors="coerce").mean()),
            "bmi_sd": float(pd.to_numeric(df["clin_bmi"], errors="coerce").std(ddof=1)),
            "asa_mean": float(pd.to_numeric(df["clin_asa"], errors="coerce").mean()),
            "asa_sd": float(pd.to_numeric(df["clin_asa"], errors="coerce").std(ddof=1)),
        }
    )
    return pd.DataFrame(rows), raw_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Table1 client summary from existing artifacts")
    ap.add_argument("--data-dir", default="federated_data")
    ap.add_argument("--noniid-json", default="tmp_noniid_report.json")
    ap.add_argument("--summary-json", default="federated_data/summary.json")
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--split-unit", default="case", help="Split unit label to record in table (default=case)")
    ap.add_argument("--out-csv", default="outputs/table1_client_summary.csv")
    ap.add_argument("--out-json", default="outputs/table1_client_summary.json")
    ap.add_argument(
        "--attached-like",
        action="store_true",
        help="Create the post-client-split table in the same columns as the attached example.",
    )
    ap.add_argument(
        "--case-summary-csv",
        default="outputs_dataset_stats/federated_case_window_stats.csv",
        help="Case-level split summary CSV used by --attached-like.",
    )
    ap.add_argument("--out-md", default=None, help="Optional Markdown output path for --attached-like.")
    args = ap.parse_args()

    if args.attached_like:
        out_csv = Path(args.out_csv)
        out_json = Path(args.out_json)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        df, raw_rows = _build_attached_like_table(Path(args.case_summary_csv))
        df.to_csv(out_csv, index=False)
        if args.out_md:
            out_md = Path(args.out_md)
            out_md.parent.mkdir(parents=True, exist_ok=True)
            out_md.write_text(_to_markdown_table(df) + "\n", encoding="utf-8")
        payload = {
            "started_utc": now_utc_iso(),
            "case_summary_csv": str(args.case_summary_csv),
            "notes": "Rows are computed after client train/val/test splitting from the case-level written dataset summary. Positive cases use case_has_pos.",
            "rows": raw_rows,
            "finished_utc": now_utc_iso(),
        }
        write_json(out_json, payload)
        print(f"Saved CSV: {out_csv}")
        print(f"Saved JSON: {out_json}")
        if args.out_md:
            print(f"Saved Markdown: {args.out_md}")
        return

    data_dir = Path(args.data_dir)
    splits = [s.strip() for s in str(args.splits).split(",") if s.strip()]
    noniid = _load_json(Path(args.noniid_json))
    summary = _load_json(Path(args.summary_json))

    if noniid and "per_client_per_split" in noniid:
        stats = noniid["per_client_per_split"]
    else:
        stats = _scan_counts(data_dir, splits)

    pooled = _pooled(stats, splits)

    exclusion_rate = None
    missing_rate = None
    missing_rates_by_col = None
    if summary:
        raw_cases = summary.get("exclusion", {}).get("raw_cases")
        eligible_cases = summary.get("eligible_cases")
        if raw_cases:
            try:
                exclusion_rate = 1.0 - (float(eligible_cases) / float(raw_cases))
            except Exception:
                exclusion_rate = None
        missing_counts = summary.get("missing", {}).get("counts", {})
        missing_rates_by_col = summary.get("missing", {}).get("rates")
        after_basic = summary.get("exclusion", {}).get("after_basic_filter")
        if missing_counts and after_basic:
            total_missing = float(sum(float(v) for v in missing_counts.values()))
            denom = float(after_basic) * float(len(missing_counts))
            missing_rate = (total_missing / denom) if denom > 0 else None

    rows = []
    for c in sorted(pooled.keys()):
        p = pooled[c]
        n_windows = int(p.get("n_windows", 0))
        n_pos = int(p.get("n_pos", 0))
        pos_rate = (float(n_pos) / float(n_windows)) if n_windows > 0 else float("nan")
        row = {
            "client_id": c,
            "n_samples": int(n_windows),
            "n_pos": int(n_pos),
            "pos_rate": float(pos_rate),
            "train_cases": int(stats[c]["train"]["n_files"]) if "train" in stats[c] else 0,
            "val_cases": int(stats[c]["val"]["n_files"]) if "val" in stats[c] else 0,
            "test_cases": int(stats[c]["test"]["n_files"]) if "test" in stats[c] else 0,
            "train_windows": int(stats[c]["train"]["n_windows"]) if "train" in stats[c] else 0,
            "val_windows": int(stats[c]["val"]["n_windows"]) if "val" in stats[c] else 0,
            "test_windows": int(stats[c]["test"]["n_windows"]) if "test" in stats[c] else 0,
            "exclusion_rate": exclusion_rate,
            "missing_rate": missing_rate,
            "split_unit": str(args.split_unit),
        }
        rows.append(row)

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    payload = {
        "started_utc": now_utc_iso(),
        "data_dir": str(data_dir),
        "splits": splits,
        "split_unit": str(args.split_unit),
        "exclusion_rate": exclusion_rate,
        "missing_rate": missing_rate,
        "missing_rates_by_col": missing_rates_by_col,
        "rows": rows,
        "finished_utc": now_utc_iso(),
    }
    write_json(out_json, payload)
    print(f"Saved CSV: {out_csv}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
