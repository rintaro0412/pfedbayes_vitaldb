import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import vitaldb
from tqdm import tqdm

# 設定 (VitalDB hypoxemia prediction)
SAVE_DIR = "./vitaldb_data"
CLINICAL_SAVE_PATH = "clinical_data.csv"

# 低酸素イベント予測では、症例を多く残すため ABP/MAP は必須にしない。
# 数値トラックは 100Hz グリッドに前方補間され、既存の 1D CNN 入力として扱う。
INPUT_TRACKS = [
    'Solar8000/HR',           # 心拍数
    'Solar8000/PLETH_SPO2',   # SpO2: 入力およびラベル作成
    'Primus/ETCO2',           # 呼気終末 CO2
    'Primus/FIO2',            # 吸入酸素濃度
]
LABEL_TRACK = 'Solar8000/PLETH_SPO2'
ALL_TRACKS = list(dict.fromkeys(INPUT_TRACKS))

# 既定では手術種別・麻酔種別で除外しない。必要なら CLI で指定する。
EXCLUDED_OPTYPES: list[str] = []

DEFAULT_WORKERS = os.cpu_count() or 4
CSV_COMPRESSION = {"method": "gzip", "compresslevel": 1}
TRACK_INDEX = {name: i for i, name in enumerate(ALL_TRACKS)}

REQUIRED_TRACKS = list(ALL_TRACKS)


def _split_arg_list(value: str) -> list[str]:
    return [x.strip() for x in str(value or "").replace(",", " ").split() if x.strip()]

def _load_case_data(case_id):
    return vitaldb.load_case(case_id, ALL_TRACKS, interval=1/100)

def download_case(case_id, save_dir: str, force: bool):
    save_path = os.path.join(save_dir, f"case_{case_id}.csv.gz")
    
    if (not force) and os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        return "skipped", case_id

    try:
        data = _load_case_data(case_id)

        if data is None or data.size == 0 or data.shape[0] < 60000:
            return "short", case_id
        
        missing = []
        for trk in REQUIRED_TRACKS:
            idx = TRACK_INDEX.get(trk)
            if idx is None or idx >= data.shape[1]:
                missing.append(trk)
                continue
            if np.isnan(data[:, idx]).all():
                missing.append(trk)
        if missing:
            return "missing_required", case_id

        if data.dtype != np.float32:
            data = data.astype(np.float32, copy=False)
        df = pd.DataFrame(data, columns=ALL_TRACKS)
        df.to_csv(save_path, index=False, compression=CSV_COMPRESSION)
        return "downloaded", case_id

    except Exception:
        return "failed", case_id

def main():
    parser = argparse.ArgumentParser(description="VitalDB Downloader (hypoxemia prediction)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--executor", choices=["process", "thread"], default="process")
    parser.add_argument("--no-prefilter-required", action="store_true", help="Do not prefilter cases by REQUIRED_TRACKS")
    parser.add_argument("--run-log", default="download_run.json", help="Write run config and system specs to JSON")
    parser.add_argument("--save-dir", default=SAVE_DIR)
    parser.add_argument("--clinical-save-path", default=CLINICAL_SAVE_PATH)
    parser.add_argument("--max-cases", type=int, default=0, help="Limit downloads after filtering (0=all)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing case files")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle target cases before limiting")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ane-type",
        default="",
        help="Comma/space-separated anesthesia types to keep (default: all types).",
    )
    parser.add_argument(
        "--exclude-optypes",
        default=",".join(EXCLUDED_OPTYPES),
        help="Comma/space-separated optype values to exclude (default: none).",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    start_ts = time.time()
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _mem_total_kb() -> int | None:
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1])
        except Exception:
            return None
        return None

    def _safe_version(mod) -> str | None:
        return getattr(mod, "__version__", None)

    ane_types = _split_arg_list(args.ane_type)
    excluded_optypes = _split_arg_list(args.exclude_optypes)

    run_meta = {
        "started_utc": _now_iso(),
        "config": {
            "save_dir": str(args.save_dir),
            "clinical_save_path": str(args.clinical_save_path),
            "workers": int(args.workers),
            "executor": str(args.executor),
            "prefilter_required": bool(not args.no_prefilter_required),
            "all_tracks": list(ALL_TRACKS),
            "required_tracks": list(REQUIRED_TRACKS),
            "label_track": str(LABEL_TRACK),
            "ane_type_filter": list(ane_types),
            "excluded_optypes": list(excluded_optypes),
            "max_cases": int(args.max_cases),
            "force": bool(args.force),
            "shuffle": bool(args.shuffle),
            "seed": int(args.seed),
        },
        "system": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "cpu_count": os.cpu_count(),
            "mem_total_kb": _mem_total_kb(),
            "numpy": _safe_version(np),
            "pandas": _safe_version(pd),
            "vitaldb": _safe_version(vitaldb),
        },
        "stats": {},
        "finished_utc": None,
        "elapsed_sec": None,
    }

    print("Fetching case list and clinical info...")
    try:
        df_cases = pd.read_csv("https://api.vitaldb.net/cases")
        df_cases.to_csv(args.clinical_save_path, index=False)
        print(f"Saved clinical metadata to {args.clinical_save_path}")
    except Exception as e:
        print(f"Failed to fetch case list: {e}")
        return

    target_mask = df_cases['age'] >= 18
    if ane_types:
        target_mask = target_mask & df_cases['ane_type'].isin(ane_types)
    if excluded_optypes:
        target_mask = target_mask & (~df_cases['optype'].isin(excluded_optypes))
    target_cases = df_cases[target_mask]['caseid'].tolist()

    if not args.no_prefilter_required:
        print("Prefiltering cases by required tracks...")
        try:
            eligible = set(vitaldb.find_cases(REQUIRED_TRACKS))
            target_cases = [cid for cid in target_cases if cid in eligible]
            print(f"Eligible Target Cases: {len(target_cases)}")
        except Exception as e:
            print(f"Prefilter failed: {e}")

    if args.shuffle:
        rng = np.random.default_rng(int(args.seed))
        rng.shuffle(target_cases)
    else:
        target_cases = sorted(target_cases)
    if int(args.max_cases) > 0:
        target_cases = target_cases[:int(args.max_cases)]
        print(f"Target Cases (limited): {len(target_cases)}")

    existing_files = set()
    if os.path.exists(args.save_dir):
        for f in os.listdir(args.save_dir):
            if f.endswith(".csv.gz"):
                try:
                    cid = int(f.replace("case_", "").replace(".csv.gz", ""))
                    existing_files.add(cid)
                except: pass

    if args.force:
        existing_files = set()
    to_download = sorted(list(set(target_cases) - existing_files))
    print(f"To Download: {len(to_download)}")
    run_meta["stats"]["to_download"] = int(len(to_download))

    if not to_download:
        run_meta["stats"].update({"downloaded": 0, "skipped": 0, "short": 0, "missing_required": 0, "failed": 0})
        run_meta["finished_utc"] = _now_iso()
        run_meta["elapsed_sec"] = float(time.time() - start_ts)
        if args.run_log:
            with open(args.run_log, "w", encoding="utf-8") as f:
                json.dump(run_meta, f, ensure_ascii=False, indent=2)
        return

    print(f"Starting download with {args.workers} {args.executor} workers...")
    stats = {"downloaded": 0, "skipped": 0, "short": 0, "missing_required": 0, "failed": 0}
    
    executor_cls = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor
    with executor_cls(max_workers=args.workers) as executor:
        futures = {executor.submit(download_case, cid, args.save_dir, args.force): cid for cid in to_download}
        with tqdm(total=len(futures), unit="file") as pbar:
            for future in as_completed(futures):
                status, cid = future.result()
                stats[status] = stats.get(status, 0) + 1
                pbar.update(1)

    print("\nSummary:", stats)
    run_meta["stats"].update(stats)
    run_meta["finished_utc"] = _now_iso()
    run_meta["elapsed_sec"] = float(time.time() - start_ts)
    if args.run_log:
        with open(args.run_log, "w", encoding="utf-8") as f:
            json.dump(run_meta, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
