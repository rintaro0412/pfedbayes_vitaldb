"""
Shim et al., 2025 (Medicina) に「できるだけ」合わせて学習用セグメントを生成する。

主要条件（本文より）
- hypotension event: MAP <= 65 が 1 分超
- positive: 各 hypotension event の 5 分前の 30 秒波形（入力）で予測（horizon=5min, window=60s）
- negative: MAP > 65 が 20 分超続く "non-hypotensive segment" から抽出し、各 segment から 1 または 2 個の入力を取り、
  全体の negative 数が positive に近くなるようにする
- artifact 除外:
  (i) ABP 波形のピーク間隔（心拍周期）が生理範囲から外れるセグメントを除外
  (ii) MAP < 20 または > 200 のセグメントを除外

注意
- 「心拍周期の生理範囲」の具体閾値は本文に明示されていないため、デフォルトとして
  0.3〜2.0 秒（= 30〜200 bpm 相当）を採用する（引数で変更可）
- 入力波形は ABP/ECG/PPG/ETCO2 の 4 波形（Shim 論文に合わせる）
- 本リポジトリの federated 学習のため、client 分割と train/test=80/20（val なし）を採用
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import shutil
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from tqdm import tqdm

# 設定 (Medicina 2025 Paper Config)
METADATA_PATH = 'clinical_data.csv'
SOURCE_DATA_DIR = './vitaldb_data'
OUTPUT_BASE_DIR = './federated_data'
FILE_EXTENSION = '.csv.gz'

# NOTE: Uncompressed .npz gets extremely large (e.g. tens of GB). Default to compressed.
_COMPRESS_ENV = os.environ.get("BUILD_DATASET_COMPRESS", "1").strip().lower()
USE_COMPRESSED = _COMPRESS_ENV in ("1", "true", "yes")

_DEFAULT_WORKERS = min(os.cpu_count() or 4, 14)
_ENV_WORKERS = os.environ.get("BUILD_DATASET_WORKERS")
if _ENV_WORKERS:
    try:
        NUM_WORKERS = max(1, int(_ENV_WORKERS))
    except ValueError:
        NUM_WORKERS = _DEFAULT_WORKERS
else:
    NUM_WORKERS = _DEFAULT_WORKERS
SPLIT_RATIOS = {'train': 0.8, 'val': 0.0, 'test': 0.2}
ALL_SPLITS = ("train", "val", "test")
ACTIVE_SPLITS = tuple(k for k in ALL_SPLITS if float(SPLIT_RATIOS.get(k, 0.0)) > 0.0)
RANDOM_SEED = 42

FS = 100  # Hz (download で 100Hz 化済みを想定)

WINDOW_SEC = 30
HORIZON_MIN = 5

MAP_HYPOTEN = 65.0
HYPOTEN_MIN_DUR_SEC = 60
NORM_MIN_DUR_SEC = 20 * 60

MAP_MIN_VALID = 20.0
MAP_MAX_VALID = 200.0

EXCLUDED_OPTYPES = ['Transplantation', 'Cardiac Surgery']

# 波形は4種類
WAVEFORMS = {
    'ABP': 'SNUADC/ART',
    'ECG': 'SNUADC/ECG_II',
    'PPG': 'SNUADC/PLETH',
}
ETCO2_TRACKS = [
    'Primus/CO2',
    'SNUADC/ETCO2',
    'SNUADC/CO2',
]
LABEL_TRACK = 'Solar8000/ART_MBP'

CLINICAL_COLS = [
    "age",
    "sex",
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

# ==========================================
# 関数定義
# ==========================================

def get_client_id(row):
    dept = row['department']
    if pd.isna(dept): return None
    dept = str(dept).strip()
    if dept.startswith('General'): return 'General_surgery'
    if dept.startswith('Gynecology'): return 'Gynecology'
    if dept.startswith('Thoracic'): return 'Thoracic_surgery'
    if dept.startswith('Urology'): return 'Urology'
    return None

def robust_split(df_client, *, seed: int):
    total = len(df_client)
    n_train, n_val, n_test = _compute_split_counts(total)
    if total < 3:
        empty = df_client.iloc[:0].copy()
        return df_client, empty, empty

    df_shuffled = df_client.sample(frac=1, random_state=int(seed)).reset_index(drop=True)
    return df_shuffled.iloc[:n_train], df_shuffled.iloc[n_train:n_train+n_val], df_shuffled.iloc[n_train+n_val:]

def _compute_split_counts(total: int) -> Tuple[int, int, int]:
    if total < 3:
        return total, 0, 0
    n_val = int(total * SPLIT_RATIOS['val'])
    n_test = int(total * SPLIT_RATIOS['test'])
    if SPLIT_RATIOS.get('val', 0.0) > 0 and n_val == 0:
        n_val = 1
    if SPLIT_RATIOS.get('test', 0.0) > 0 and n_test == 0:
        n_test = 1
    n_train = max(1, total - n_val - n_test)
    if total - n_val - n_test <= 0:
        n_train = total - n_test
        n_val = 0
    return n_train, n_val, n_test

def stratified_split_by_pos(df_client, *, pos_by_case: Dict[int, int], seed: int):
    total = len(df_client)
    n_train, n_val, n_test = _compute_split_counts(total)
    if total < 3:
        empty = df_client.iloc[:0].copy()
        return df_client, empty, empty

    case_ids = [int(cid) for cid in df_client["caseid"].astype(int).tolist()]
    total_pos = sum(int(pos_by_case.get(cid, 0)) for cid in case_ids)
    if total_pos <= 0:
        return robust_split(df_client, seed=int(seed))

    targets_cases = {"train": n_train, "val": n_val, "test": n_test}
    raw_targets = {k: float(total_pos) * float(SPLIT_RATIOS.get(k, 0.0)) for k in ALL_SPLITS}
    target_pos = {k: 0 for k in ALL_SPLITS}
    for k in ALL_SPLITS:
        if targets_cases[k] > 0:
            target_pos[k] = int(raw_targets.get(k, 0.0))
    remainder = int(total_pos) - sum(int(v) for v in target_pos.values())
    if remainder > 0:
        fracs = [
            (raw_targets[k] - float(target_pos[k]), k)
            for k in ALL_SPLITS
            if targets_cases[k] > 0
        ]
        fracs.sort(reverse=True)
        for i in range(int(remainder)):
            _, k = fracs[int(i) % len(fracs)]
            target_pos[k] = int(target_pos[k]) + 1

    rng = np.random.default_rng(int(seed))
    cases = [(cid, int(pos_by_case.get(cid, 0)), float(rng.random())) for cid in case_ids]
    cases.sort(key=lambda x: (int(x[1]), float(x[2])), reverse=True)
    curr_cases = {"train": 0, "val": 0, "test": 0}
    curr_pos = {"train": 0, "val": 0, "test": 0}
    assign: Dict[int, str] = {}
    for cid, n_pos, _ in cases:
        best_split = None
        best_key = None
        for split in ALL_SPLITS:
            if curr_cases[split] >= targets_cases[split]:
                continue
            pos_def = int(target_pos[split]) - int(curr_pos[split])
            case_def = int(targets_cases[split]) - int(curr_cases[split])
            key = (int(pos_def), int(case_def), float(rng.random()))
            if best_key is None or key > best_key:
                best_key = key
                best_split = split
        if best_split is None:
            for split in ALL_SPLITS:
                if curr_cases[split] < targets_cases[split]:
                    best_split = split
                    break
        assign[int(cid)] = str(best_split)
        curr_cases[str(best_split)] = int(curr_cases[str(best_split)]) + 1
        curr_pos[str(best_split)] = int(curr_pos[str(best_split)]) + int(n_pos)

    df_train = df_client[df_client["caseid"].astype(int).isin([cid for cid, sp in assign.items() if sp == "train"])].copy()
    df_val = df_client[df_client["caseid"].astype(int).isin([cid for cid, sp in assign.items() if sp == "val"])].copy()
    df_test = df_client[df_client["caseid"].astype(int).isin([cid for cid, sp in assign.items() if sp == "test"])].copy()
    return df_train, df_val, df_test
def find_intervals(mask_1hz: np.ndarray) -> List[Tuple[int, int]]:
    """mask_1hz の True 連続区間 [start,end) を返す（秒単位）"""
    out: List[Tuple[int, int]] = []
    n = int(mask_1hz.size)
    i = 0
    while i < n:
        if not bool(mask_1hz[i]):
            i += 1
            continue
        j = i + 1
        while j < n and bool(mask_1hz[j]):
            j += 1
        out.append((int(i), int(j)))
        i = j
    return out


def to_1hz_mean(x_100hz: np.ndarray, fs: int) -> np.ndarray:
    n = (int(x_100hz.size) // int(fs)) * int(fs)
    if n <= 0:
        return np.empty((0,), dtype=np.float32)
    return x_100hz[:n].reshape(-1, int(fs)).mean(axis=1).astype(np.float32, copy=False)


def detect_peaks_simple(x: np.ndarray, fs: int, min_dist_sec: float = 0.25) -> np.ndarray:
    """
    scipy に依存しない簡易ピーク検出。
    - 符号変化（上昇→下降）で局所最大
    - 近接ピークは距離制約で間引く
    """
    if x.size < 3:
        return np.empty((0,), dtype=np.int32)

    dx = np.diff(x)
    candidates = np.where((dx[:-1] > 0) & (dx[1:] <= 0))[0] + 1
    if candidates.size == 0:
        return np.empty((0,), dtype=np.int32)

    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-6
    amp_thr = med + 2.0 * mad
    candidates = candidates[x[candidates] >= amp_thr]
    if candidates.size == 0:
        return np.empty((0,), dtype=np.int32)

    min_dist = int(float(min_dist_sec) * int(fs))
    kept = [int(candidates[0])]
    last = kept[0]
    for idx in candidates[1:]:
        idx = int(idx)
        if idx - last >= min_dist:
            kept.append(idx)
            last = idx
    return np.asarray(kept, dtype=np.int32)


def segment_ok(
    abp_seg: np.ndarray,
    mbp_seg: np.ndarray,
    fs: int,
    cycle_min_sec: float,
    cycle_max_sec: float,
) -> bool:
    """
    Shim 論文の artifact 除外に対応。
    - MAP が [20,200] を外れたら除外
    - ABP ピーク間隔（心拍周期）が生理範囲外なら除外（閾値は本文に明記無し）
    """
    if abp_seg.size == 0 or mbp_seg.size == 0:
        return False

    if np.nanmin(mbp_seg) < MAP_MIN_VALID or np.nanmax(mbp_seg) > MAP_MAX_VALID:
        return False

    peaks = detect_peaks_simple(abp_seg, fs)
    if peaks.size < 2:
        return False

    cycle = np.diff(peaks) / float(fs)
    bad_frac = float(np.mean((cycle < float(cycle_min_sec)) | (cycle > float(cycle_max_sec))))
    return bad_frac <= 0.10


def extract_pos_events(map_1hz: np.ndarray) -> List[int]:
    """hypotension interval の開始秒を event として返す"""
    valid = np.isfinite(map_1hz) & (map_1hz >= MAP_MIN_VALID) & (map_1hz <= MAP_MAX_VALID)
    hyp = (map_1hz <= MAP_HYPOTEN) & valid
    intervals = find_intervals(hyp)
    events: List[int] = []
    for s, e in intervals:
        if (e - s) >= HYPOTEN_MIN_DUR_SEC:
            events.append(int(s))
    return events


def extract_norm_segments(map_1hz: np.ndarray) -> List[Tuple[int, int]]:
    """MAP>65 が 20 分超の区間（秒 [start,end)）"""
    valid = np.isfinite(map_1hz) & (map_1hz >= MAP_MIN_VALID) & (map_1hz <= MAP_MAX_VALID)
    norm = (map_1hz > MAP_HYPOTEN) & valid
    intervals = find_intervals(norm)
    out: List[Tuple[int, int]] = []
    for s, e in intervals:
        if (e - s) >= NORM_MIN_DUR_SEC:
            out.append((int(s), int(e)))
    return out


def _read_case_series(case_path: str, cols: List[str]) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(case_path, compression="gzip", usecols=cols)
    except Exception:
        return None
    if not all(c in df.columns for c in cols):
        return None
    df[cols] = df[cols].ffill().bfill()
    return df

def _has_valid_etco2(case_path: str) -> bool:
    """ETCO2 列が存在し、有限値が1つでもあれば True."""
    for etco2_col in ETCO2_TRACKS:
        df = _read_case_series(case_path, [etco2_col])
        if df is None:
            continue
        etco2 = df[etco2_col].to_numpy(dtype=np.float32, copy=True)
        if np.isfinite(etco2).any():
            return True
    return False


def _etco2_worker(args: Tuple[int, str]) -> Tuple[int, bool]:
    idx, case_path = args
    try:
        return int(idx), bool(_has_valid_etco2(str(case_path)))
    except Exception:
        return int(idx), False


def _load_case_arrays(
    case_path: str,
    *,
    require_etco2: bool,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    base_cols = [LABEL_TRACK, WAVEFORMS["ABP"], WAVEFORMS["ECG"], WAVEFORMS["PPG"]]
    for etco2_col in ETCO2_TRACKS:
        cols = base_cols + [etco2_col]
        df = _read_case_series(case_path, cols)
        if df is None:
            continue
        mbp = df[LABEL_TRACK].to_numpy(dtype=np.float32, copy=True)
        abp = df[WAVEFORMS["ABP"]].to_numpy(dtype=np.float32, copy=True)
        ecg = df[WAVEFORMS["ECG"]].to_numpy(dtype=np.float32, copy=True)
        ppg = df[WAVEFORMS["PPG"]].to_numpy(dtype=np.float32, copy=True)
        etco2 = df[etco2_col].to_numpy(dtype=np.float32, copy=True)
        if require_etco2:
            if not np.isfinite(etco2).any():
                continue
            return abp, ecg, ppg, etco2, mbp
        if not np.isfinite(etco2).any():
            etco2 = np.zeros_like(mbp)
        return abp, ecg, ppg, etco2, mbp

    if not require_etco2:
        df = _read_case_series(case_path, base_cols)
        if df is None:
            return None
        mbp = df[LABEL_TRACK].to_numpy(dtype=np.float32, copy=True)
        abp = df[WAVEFORMS["ABP"]].to_numpy(dtype=np.float32, copy=True)
        ecg = df[WAVEFORMS["ECG"]].to_numpy(dtype=np.float32, copy=True)
        ppg = df[WAVEFORMS["PPG"]].to_numpy(dtype=np.float32, copy=True)
        etco2 = np.zeros_like(mbp)
        return abp, ecg, ppg, etco2, mbp

    return None


def _load_mbp_only(case_path: str) -> Optional[np.ndarray]:
    df = _read_case_series(case_path, [LABEL_TRACK])
    if df is None:
        return None
    return df[LABEL_TRACK].to_numpy(dtype=np.float32, copy=True)


def _pass1_worker(task: Tuple[int, str]) -> Tuple[int, int, List[Tuple[int, int]]]:
    cid, case_path = task
    try:
        mbp_100 = _load_mbp_only(case_path)
        if mbp_100 is None:
            return int(cid), 0, []
        map_1hz = to_1hz_mean(mbp_100, FS)
        if map_1hz.size == 0:
            return int(cid), 0, []
        pos_events = extract_pos_events(map_1hz)
        norm_segs = extract_norm_segments(map_1hz)
        norm_segs = [(int(s), int(e)) for s, e in norm_segs]
        return int(cid), int(len(pos_events)), norm_segs
    except Exception:
        return int(cid), 0, []


@dataclass(frozen=True)
class CaseSegments:
    x_wave: np.ndarray  # (N, C, T)
    x_clin: np.ndarray  # (N, D)
    y: np.ndarray       # (N,)
    t_event: np.ndarray # (N,) seconds
    is_pos: np.ndarray  # (N,) bool


def build_case_segments(
    *,
    case_path: str,
    clin_vec: np.ndarray,
    assigned_k_by_normseg: Dict[Tuple[int, int], int],
    rng: np.random.Generator,
    cycle_min_sec: float,
    cycle_max_sec: float,
    instance_norm: bool,
    require_etco2: bool,
) -> Optional[CaseSegments]:
    loaded = _load_case_arrays(case_path, require_etco2=require_etco2)
    if loaded is None:
        return None
    abp_100, ecg_100, ppg_100, etco2_100, mbp_100 = loaded

    map_1hz = to_1hz_mean(mbp_100, FS)
    if map_1hz.size == 0:
        return None

    pos_events = extract_pos_events(map_1hz)
    norm_segs = extract_norm_segments(map_1hz)

    T_input = int(WINDOW_SEC)
    H = int(HORIZON_MIN * 60)

    waves_full = np.stack([abp_100, ecg_100, ppg_100, etco2_100], axis=0)  # (4, T_total)
    if instance_norm:
        mean = waves_full.mean(axis=1, keepdims=True)
        std = waves_full.std(axis=1, keepdims=True) + 1e-6
        waves_full = (waves_full - mean) / std

    waves: List[np.ndarray] = []
    labels: List[int] = []
    t_event: List[int] = []
    is_pos: List[bool] = []

    # positives
    for ev in pos_events:
        input_end = int(ev - H)
        input_start = int(input_end - T_input)
        if input_start < 0:
            continue
        a = int(input_start * FS)
        b = int(input_end * FS)
        if a < 0 or b > waves_full.shape[1]:
            continue
        wave = waves_full[:, a:b]
        mbp_seg = mbp_100[a:b]
        if not segment_ok(abp_100[a:b], mbp_seg, FS, cycle_min_sec, cycle_max_sec):
            continue
        if np.isnan(wave).any() or np.isnan(mbp_seg).any():
            continue
        waves.append(wave.astype(np.float32, copy=False))
        labels.append(1)
        t_event.append(int(ev))
        is_pos.append(True)

    # negatives: norm segment ごとに 1 or 2 個
    for (s, e) in norm_segs:
        k = int(assigned_k_by_normseg.get((int(s), int(e)), 0))
        if k <= 0:
            continue

        min_ev = int(s + H + T_input)
        max_ev = int(e - 1)  # e is exclusive
        if max_ev < min_ev:
            continue

        for _ in range(k):
            ev = int(rng.integers(min_ev, max_ev + 1))
            input_end = int(ev - H)
            input_start = int(input_end - T_input)
            if input_start < int(s):
                continue
            a = int(input_start * FS)
            b = int(input_end * FS)
            if a < 0 or b > waves_full.shape[1]:
                continue
            wave = waves_full[:, a:b]
            mbp_seg = mbp_100[a:b]
            if not segment_ok(abp_100[a:b], mbp_seg, FS, cycle_min_sec, cycle_max_sec):
                continue
            if np.isnan(wave).any() or np.isnan(mbp_seg).any():
                continue
            waves.append(wave.astype(np.float32, copy=False))
            labels.append(0)
            t_event.append(int(ev))
            is_pos.append(False)

    if not waves:
        return None

    x_wave = np.stack(waves, axis=0).astype(np.float32, copy=False)  # (N, 4, 6000)
    x_clin = np.repeat(clin_vec[None, :], repeats=int(x_wave.shape[0]), axis=0).astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=np.int64)
    t_arr = np.asarray(t_event, dtype=np.int64)
    is_pos_arr = np.asarray(is_pos, dtype=bool)
    return CaseSegments(x_wave=x_wave, x_clin=x_clin, y=y, t_event=t_arr, is_pos=is_pos_arr)

def convert_worker(args):
    case_id, src_path, dst_path, clin_vec, assigned_by_seg, seed, cycle_min_sec, cycle_max_sec, instance_norm, require_etco2 = args
    try:
        rng = np.random.default_rng(int(seed) ^ (int(case_id) * 1000003))
        segs = build_case_segments(
            case_path=str(src_path),
            clin_vec=np.asarray(clin_vec, dtype=np.float32),
            assigned_k_by_normseg=dict(assigned_by_seg),
            rng=rng,
            cycle_min_sec=float(cycle_min_sec),
            cycle_max_sec=float(cycle_max_sec),
            instance_norm=bool(instance_norm),
            require_etco2=bool(require_etco2),
        )
        if segs is None:
            return int(case_id), False, 0, 0

        out_dir = os.path.dirname(dst_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        if USE_COMPRESSED:
            np.savez_compressed(
                dst_path,
                x_wave=segs.x_wave,
                x_clin=segs.x_clin,
                y=segs.y,
                t_event=segs.t_event,
                is_pos=segs.is_pos,
            )
        else:
            np.savez(
                dst_path,
                x_wave=segs.x_wave,
                x_clin=segs.x_clin,
                y=segs.y,
                t_event=segs.t_event,
                is_pos=segs.is_pos,
            )
        n_pos = int(segs.y.sum())
        n_neg = int((segs.y == 0).sum())
        return int(case_id), True, n_pos, n_neg
    except Exception:
        return int(case_id), False, 0, 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build federated dataset (Shim 2025-style segmentation)")
    p.add_argument("--clinical-csv", default=METADATA_PATH)
    p.add_argument("--wave-dir", default=SOURCE_DATA_DIR, help="case_*.csv.gz があるディレクトリ")
    p.add_argument("--out-dir", default=OUTPUT_BASE_DIR)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--cycle-min-sec", type=float, default=0.3)
    p.add_argument("--cycle-max-sec", type=float, default=2.0)
    p.add_argument("--age-min", type=float, default=18.0)
    p.add_argument("--instance-norm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--require-etco2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="ETCO2 波形が無い（全 NaN を含む）症例を除外するか",
    )
    p.add_argument("--opname-threshold", type=int, default=150)
    p.add_argument("--min-client-cases", type=int, default=150)
    p.add_argument(
        "--min-client-pos",
        type=int,
        default=10,
        help="After split, drop clients whose train/test positive events are below this (0 to disable).",
    )
    p.add_argument(
        "--client-scheme",
        choices=["department", "opname_optype"],
        default="opname_optype",
        help="Client grouping rule (department=legacy, opname_optype=new default).",
    )
    p.add_argument(
        "--merge-strategy",
        choices=["dept_pool", "dept_min", "none"],
        default="dept_pool",
        help=(
            "How to merge small clients when using opname_optype "
            "(dept_pool=pool per department, dept_min=absorb into smallest large client within department, "
            "none=do not merge; small clients are dropped by min_client_cases)."
        ),
    )
    p.add_argument(
        "--exclude-clients",
        default="",
        help="Comma/space-separated client_id list to drop after assignment (e.g., Urology__OtherSurgery).",
    )
    return p.parse_args()

def main():
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))

    print("Build Dataset (Shim 2025-style)")
    print(f"  save: np.savez (compressed={USE_COMPRESSED})")
    print(f"  out_dir: {args.out_dir}")
    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/3] Loading Metadata...")
    df_meta_raw = pd.read_csv(args.clinical_csv)
    df_meta_raw.columns = [str(c).replace('clinical_data.csv', '').strip() for c in df_meta_raw.columns]

    missing_cols = [c for c in CLINICAL_COLS if c not in df_meta_raw.columns]
    if missing_cols:
        raise RuntimeError(f"clinical_data.csv に必要列がありません: {missing_cols}")

    required_cols = ["caseid", "ane_type", "optype", "opname", "department"]
    missing_required = [c for c in required_cols if c not in df_meta_raw.columns]
    if missing_required:
        raise RuntimeError(f"clinical_data.csv に必要列が不足しています: {missing_required}")
    
    mask = (
        (df_meta_raw['age'] >= float(args.age_min)) & 
        (df_meta_raw['ane_type'] == 'General') & 
        (~df_meta_raw['optype'].isin(EXCLUDED_OPTYPES))
    )
    df_meta_masked = df_meta_raw[mask].copy()
    n_raw_cases = int(len(df_meta_raw))
    n_after_basic_filter = int(len(df_meta_masked))
    missing_check_cols = list(dict.fromkeys(list(CLINICAL_COLS) + required_cols))
    missing_counts = {c: int(df_meta_masked[c].isna().sum()) for c in missing_check_cols}
    missing_rates = {
        c: (float(missing_counts[c]) / float(n_after_basic_filter) if n_after_basic_filter > 0 else None)
        for c in missing_check_cols
    }

    df_meta_non_missing_clin = df_meta_masked.dropna(subset=CLINICAL_COLS).copy()
    n_after_dropna_clin = int(len(df_meta_non_missing_clin))
    df_meta = df_meta_non_missing_clin.dropna(subset=["caseid"]).copy()
    n_after_dropna_caseid = int(len(df_meta))
    
    wave_dir = str(args.wave_dir)
    df_meta["case_path"] = df_meta["caseid"].apply(lambda cid: os.path.join(wave_dir, f"case_{int(cid)}{FILE_EXTENSION}"))
    df_valid = df_meta[df_meta["case_path"].apply(lambda p: os.path.exists(p))].copy()
    n_missing_wave = int(n_after_dropna_caseid - len(df_valid))
    df_valid["caseid"] = df_valid["caseid"].astype(int)
    print(f"  Target Cases: {len(df_valid)}")
    n_missing_etco2 = 0
    if args.require_etco2:
        print("  Checking ETCO2 waveform availability...")
        keep_mask = [False] * int(len(df_valid))
        if NUM_WORKERS <= 1:
            for idx, row in enumerate(tqdm(df_valid.itertuples(index=False), total=len(df_valid), desc="etco2")):
                case_path = str(getattr(row, "case_path"))
                ok = _has_valid_etco2(case_path)
                keep_mask[int(idx)] = bool(ok)
        else:
            print(f"  etco2 workers: {NUM_WORKERS}")
            tasks = [(int(i), str(getattr(row, "case_path"))) for i, row in enumerate(df_valid.itertuples(index=False))]
            with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = [executor.submit(_etco2_worker, t) for t in tasks]
                for f in tqdm(as_completed(futures), total=len(futures), desc="etco2"):
                    idx, ok = f.result()
                    keep_mask[int(idx)] = bool(ok)
        n_missing_etco2 = int(len(df_valid) - int(sum(keep_mask)))
        if n_missing_etco2 > 0:
            df_valid = df_valid.loc[keep_mask].copy()
        print(f"  Target Cases (after ETCO2 filter): {len(df_valid)}")

    def normalize_client_name(name: str) -> str:
        text = "" if name is None else str(name).strip()
        text = re.sub(r"[^\w]+", "_", text)
        text = re.sub(r"_+", "_", text)
        text = text.strip("_")
        return text if text else "Unknown"

    warnings_small_departments: List[Tuple[str, int]] = []

    missing_client_id = 0
    if args.client_scheme == "department":
        df_valid["client_id"] = df_valid.apply(get_client_id, axis=1)
        missing_client_id = int(df_valid["client_id"].isna().sum())
        df_valid = df_valid.dropna(subset=["client_id"]).copy()
        merged_clients: Dict[str, int] = {}
    else:
        for col in ["opname", "optype"]:
            missing_count = int(df_valid[col].isna().sum())
            if missing_count > 0:
                example_cases = df_valid.loc[df_valid[col].isna(), "caseid"].head(5).tolist()
                raise RuntimeError(
                    f"clinical_data.csv の {col} に欠損があります (rows={missing_count}, caseid例={example_cases})"
                )
        missing_dept = int(df_valid["department"].isna().sum())
        if missing_dept > 0:
            example_cases = df_valid.loc[df_valid["department"].isna(), "caseid"].head(5).tolist()
            raise RuntimeError(
                f"clinical_data.csv の department に欠損があります (rows={missing_dept}, caseid例={example_cases})"
            )

        opname_counts = df_valid["opname"].value_counts()

        def pick_client_raw(row) -> str:
            opname = row.opname
            optype = row.optype
            opname_count = int(opname_counts.get(opname, 0))
            raw = opname if opname_count >= int(args.opname_threshold) else optype
            return normalize_client_name(raw)

        df_valid["client_raw"] = df_valid.apply(pick_client_raw, axis=1)
        df_valid["department_norm"] = df_valid["department"].apply(normalize_client_name)

        counts_by_dept_raw = (
            df_valid[["department", "client_raw", "caseid"]]
            .drop_duplicates()
            .groupby(["department", "client_raw"])["caseid"]
            .nunique()
            .to_dict()
        )
        min_client_cases = int(args.min_client_cases)

        merged_summary: Dict[str, Dict[str, int]] = {}

        if args.merge_strategy == "none":
            def map_final(row):
                dept_norm = row.department_norm
                client_raw = row.client_raw
                return f"{dept_norm}__{client_raw}"
        elif args.merge_strategy == "dept_pool":
            def map_final(row):
                dept = row.department
                dept_norm = row.department_norm
                client_raw = row.client_raw
                count = int(counts_by_dept_raw.get((dept, client_raw), 0))
                if count >= min_client_cases:
                    return f"{dept_norm}__{client_raw}"
                target = f"{dept_norm}__OtherSurgery"
                merged_summary.setdefault(target, {})[client_raw] = count
                return target
        else:
            counts_by_dept = (
                df_valid[["department", "client_raw", "caseid"]]
                .drop_duplicates()
                .groupby("department")["client_raw"]
                .apply(lambda s: s.value_counts().to_dict())
            )
            # Precompute smallest sufficient client per department
            smallest_sufficient: Dict[str, Tuple[str, int]] = {}
            for dept, counts in counts_by_dept.items():
                eligible = [(cr, cnt) for cr, cnt in counts.items() if int(cnt) >= min_client_cases]
                if eligible:
                    target = min(eligible, key=lambda t: (int(t[1]), t[0]))
                    smallest_sufficient[dept] = (target[0], int(target[1]))

            def map_final(row):
                dept = row.department
                dept_norm = row.department_norm
                client_raw = row.client_raw
                count = int(counts_by_dept_raw.get((dept, client_raw), 0))
                if count >= min_client_cases:
                    return f"{dept_norm}__{client_raw}"
                if dept in smallest_sufficient:
                    target_raw = smallest_sufficient[dept][0]
                    target = f"{dept_norm}__{target_raw}"
                    merged_summary.setdefault(target, {})[client_raw] = count
                    return target
                target = f"{dept_norm}__OtherSurgery"
                merged_summary.setdefault(target, {})[client_raw] = count
                return target

        df_valid["client_id"] = df_valid.apply(map_final, axis=1)

        if args.merge_strategy == "dept_pool":
            client_case_counts = (
                df_valid[["client_id", "caseid"]]
                .drop_duplicates()
                .groupby("client_id")["caseid"]
                .nunique()
                .to_dict()
            )
            dept_client_counts = (
                df_valid[["department", "department_norm", "client_id", "caseid"]]
                .drop_duplicates()
                .groupby(["department", "department_norm", "client_id"])["caseid"]
                .nunique()
                .reset_index()
            )

            for dept in dept_client_counts["department"].unique():
                dept_rows = dept_client_counts[dept_client_counts["department"] == dept]
                dept_norm = str(dept_rows["department_norm"].iloc[0])
                other_id = f"{dept_norm}__OtherSurgery"
                other_count = client_case_counts.get(other_id)
                if other_count is None or other_count >= min_client_cases:
                    continue
                eligible = [
                    (row.client_id, int(row.caseid))
                    for row in dept_rows.itertuples(index=False)
                    if row.client_id != other_id and int(row.caseid) >= min_client_cases
                ]
                if eligible:
                    target = min(eligible, key=lambda t: (int(t[1]), t[0]))[0]
                    df_valid.loc[df_valid["client_id"] == other_id, "client_id"] = target
                    if other_id in merged_summary:
                        merged_summary.setdefault(target, {})
                        for src, cnt in merged_summary.pop(other_id).items():
                            merged_summary[target][src] = cnt
                else:
                    warnings_small_departments.append((dept_norm, int(other_count)))

        merged_clients = merged_summary

    exclude_clients: List[str] = []
    excluded_counts: Dict[str, int] = {}
    raw_exclude = str(args.exclude_clients or "").strip()
    if raw_exclude:
        exclude_clients = [c for c in re.split(r"[,\s]+", raw_exclude) if c]
    if exclude_clients:
        exclude_set = set(exclude_clients)
        excluded_df = df_valid[df_valid["client_id"].isin(exclude_set)]
        if not excluded_df.empty:
            excluded_counts = (
                excluded_df[["client_id", "caseid"]]
                .drop_duplicates()
                .groupby("client_id")["caseid"]
                .nunique()
                .to_dict()
            )
            df_valid = df_valid[~df_valid["client_id"].isin(exclude_set)].copy()
        if warnings_small_departments:
            filtered: List[Tuple[str, int]] = []
            for dept_norm, cnt in warnings_small_departments:
                other_id = f"{dept_norm}__OtherSurgery"
                if other_id in exclude_set:
                    continue
                filtered.append((dept_norm, int(cnt)))
            warnings_small_departments = filtered

    excluded_small_clients: Dict[str, int] = {}
    # Recompute counts based on unique caseid
    final_counts = (
        df_valid[["client_id", "caseid"]]
        .drop_duplicates()
        .groupby("client_id")["caseid"]
        .nunique()
        .to_dict()
    )
    min_client_cases = int(args.min_client_cases)
    small_final_clients = {cid: cnt for cid, cnt in final_counts.items() if cnt < min_client_cases}
    if small_final_clients:
        excluded_small_clients = {cid: int(cnt) for cid, cnt in small_final_clients.items()}
        df_valid = df_valid[~df_valid["client_id"].isin(set(excluded_small_clients))].copy()
        final_counts = (
            df_valid[["client_id", "caseid"]]
            .drop_duplicates()
            .groupby("client_id")["caseid"]
            .nunique()
            .to_dict()
        )

    print("\n[client assignment]")
    print(f"  scheme: {args.client_scheme}")
    print(f"  opname_threshold: {args.opname_threshold}")
    print(f"  min_client_cases: {args.min_client_cases}")
    print(f"  min_client_pos (per split): {args.min_client_pos}")
    print(f"  merge_strategy: {args.merge_strategy}")
    print(f"  final clients: {len(final_counts)}")
    for cid, cnt in sorted(final_counts.items(), key=lambda kv: kv[1], reverse=True):
        print(f"    {cid}: {cnt}")
    if exclude_clients:
        if excluded_counts:
            dropped = sum(int(v) for v in excluded_counts.values())
            dropped_str = ", ".join([f"{cid}={cnt}" for cid, cnt in sorted(excluded_counts.items())])
            print(f"  [exclude] dropped clients: {dropped_str} (cases={dropped})")
        else:
            print(f"  [exclude] no matching clients found for: {', '.join(sorted(exclude_clients))}")
    if excluded_small_clients:
        dropped = sum(int(v) for v in excluded_small_clients.values())
        dropped_str = ", ".join([f"{cid}={cnt}" for cid, cnt in sorted(excluded_small_clients.items())])
        print(f"  [drop] below min_client_cases: {dropped_str} (cases={dropped})")
    if args.client_scheme == "opname_optype":
        if args.merge_strategy != "none":
            if warnings_small_departments:
                print("  [warning] departments without eligible absorber for OtherSurgery:")
                for dept_norm, cnt in warnings_small_departments:
                    print(f"    {dept_norm}: OtherSurgery={cnt}")

            # Build merged summary from final assignment (deduplicated)
            raw_counts_case = (
                df_valid[["department", "client_raw", "caseid"]]
                .drop_duplicates()
                .groupby(["department", "client_raw"])["caseid"]
                .nunique()
            )
            small_raws = raw_counts_case[raw_counts_case < min_client_cases]
            raw_to_client = (
                df_valid[["department", "client_raw", "client_id"]]
                .drop_duplicates()
                .groupby(["department", "client_raw"])["client_id"]
                .agg(lambda s: s.iloc[0])
            )
            merged_summary_out: Dict[str, List[Tuple[str, int]]] = {}
            for (dept, client_raw), cnt in small_raws.items():
                target = raw_to_client.get((dept, client_raw))
                if target is None:
                    continue
                merged_summary_out.setdefault(target, []).append((client_raw, int(cnt)))

            if merged_summary_out:
                print("  merged summary:")
                for target, items in sorted(merged_summary_out.items(), key=lambda kv: kv[0]):
                    merged_str = ", ".join([f"{src} {cnt}" for src, cnt in sorted(items, key=lambda kv: kv[1], reverse=True)])
                    print(f"    into {target}: {merged_str}")
            else:
                print("  merged summary: None")
        else:
            print("  merged summary: None (merge_strategy=none)")

    # pass-1: count pos events and collect norm segments for negative assignment
    print("\n[2/3] pass1: counting events / collecting norm segments...")
    pos_by_case: Dict[int, int] = {}
    norm_segments_all: List[Tuple[int, int, int]] = []  # (caseid, s, e)
    pass1_tasks = [
        (int(getattr(row, "caseid")), str(getattr(row, "case_path")))
        for row in df_valid.itertuples(index=False)
    ]
    if NUM_WORKERS <= 1:
        for cid, case_path in tqdm(pass1_tasks, total=len(pass1_tasks), desc="pass1"):
            _, n_pos, norm_segs = _pass1_worker((cid, case_path))
            pos_by_case[int(cid)] = int(n_pos)
            for (s, e) in norm_segs:
                norm_segments_all.append((int(cid), int(s), int(e)))
    else:
        print(f"  pass1 workers: {NUM_WORKERS}")
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(_pass1_worker, t) for t in pass1_tasks]
            for f in tqdm(as_completed(futures), total=len(futures), desc="pass1"):
                cid, n_pos, norm_segs = f.result()
                pos_by_case[int(cid)] = int(n_pos)
                for (s, e) in norm_segs:
                    norm_segments_all.append((int(cid), int(s), int(e)))

    # Split by client (train/test=80/20, valなし) with pos_event stratification
    split_map: Dict[int, str] = {}
    for client_id, group_df in df_valid.groupby("client_id"):
        train_df, val_df, test_df = stratified_split_by_pos(group_df, pos_by_case=pos_by_case, seed=int(args.seed))
        for cid in train_df["caseid"].astype(int).tolist():
            split_map[int(cid)] = "train"
        for cid in val_df["caseid"].astype(int).tolist():
            split_map[int(cid)] = "val"
        for cid in test_df["caseid"].astype(int).tolist():
            split_map[int(cid)] = "test"
    df_valid["split"] = df_valid["caseid"].map(split_map)
    missing_split = int(df_valid["split"].isna().sum())
    df_valid = df_valid.dropna(subset=["split"]).copy()

    excluded_low_pos_clients: Dict[str, Dict[str, int]] = {}
    min_client_pos = int(args.min_client_pos)
    if min_client_pos > 0:
        client_pos_by_split: Dict[str, Dict[str, int]] = {}
        for row in df_valid.itertuples(index=False):
            client_id = str(getattr(row, "client_id"))
            split = str(getattr(row, "split"))
            case_id = int(getattr(row, "caseid"))
            cstat = client_pos_by_split.setdefault(client_id, {"train": 0, "val": 0, "test": 0})
            cstat[split] = int(cstat.get(split, 0)) + int(pos_by_case.get(case_id, 0))

        low_pos_clients = {
            cid: counts
            for cid, counts in client_pos_by_split.items()
            if any(int(counts.get(s, 0)) < min_client_pos for s in ACTIVE_SPLITS)
        }
        if low_pos_clients:
            excluded_low_pos_clients = {
                cid: {s: int(counts.get(s, 0)) for s in ACTIVE_SPLITS}
                for cid, counts in low_pos_clients.items()
            }
            df_valid = df_valid[~df_valid["client_id"].isin(set(low_pos_clients))].copy()

            keep_caseids = set(df_valid["caseid"].astype(int).tolist())
            pos_by_case = {cid: int(n) for cid, n in pos_by_case.items() if int(cid) in keep_caseids}
            norm_segments_all = [(cid, s, e) for (cid, s, e) in norm_segments_all if int(cid) in keep_caseids]

            final_counts = (
                df_valid[["client_id", "caseid"]]
                .drop_duplicates()
                .groupby("client_id")["caseid"]
                .nunique()
                .to_dict()
            )
            dropped_str = ", ".join(
                [
                    f"{cid}(" + ", ".join([f"{s}={cnt.get(s, 0)}" for s in ACTIVE_SPLITS]) + ")"
                    for cid, cnt in sorted(excluded_low_pos_clients.items())
                ]
            )
            dropped_pos = sum(int(v.get(s, 0)) for v in excluded_low_pos_clients.values() for s in ACTIVE_SPLITS)
            print(f"  [drop] below min_client_pos per split: {dropped_str} (pos_events={dropped_pos})")
            print(f"  final clients (after min_client_pos): {len(final_counts)}")
            for cid, cnt in sorted(final_counts.items(), key=lambda kv: kv[1], reverse=True):
                print(f"    {cid}: {cnt}")
    if excluded_counts or excluded_small_clients or excluded_low_pos_clients:
        print("\n[client drop summary]")
        if excluded_counts:
            dropped = sum(int(v) for v in excluded_counts.values())
            dropped_str = ", ".join([f"{cid}={cnt}" for cid, cnt in sorted(excluded_counts.items())])
            print(f"  exclude_clients: {dropped_str} (cases={dropped})")
        if excluded_small_clients:
            dropped = sum(int(v) for v in excluded_small_clients.values())
            dropped_str = ", ".join([f"{cid}={cnt}" for cid, cnt in sorted(excluded_small_clients.items())])
            print(f"  min_client_cases: {dropped_str} (cases={dropped})")
        if excluded_low_pos_clients:
            dropped = sum(int(v.get(s, 0)) for v in excluded_low_pos_clients.values() for s in ACTIVE_SPLITS)
            dropped_str = ", ".join(
                [
                    f"{cid}(" + ", ".join([f"{s}={cnt.get(s, 0)}" for s in ACTIVE_SPLITS]) + ")"
                    for cid, cnt in sorted(excluded_low_pos_clients.items())
                ]
            )
            print(f"  min_client_pos: {dropped_str} (pos_events={dropped})")

    n_norm = int(len(norm_segments_all))
    total_pos = sum(int(v) for v in pos_by_case.values())
    assigned_k: Dict[Tuple[int, int, int], int] = {(cid, s, e): 1 for (cid, s, e) in norm_segments_all}

    total_neg_est = n_norm
    if total_neg_est < total_pos and n_norm > 0:
        extra = min(int(total_pos - total_neg_est), int(n_norm))
        chosen = rng.choice(n_norm, size=int(extra), replace=False) if extra > 0 else np.array([], dtype=int)
        for i in chosen.tolist():
            cid, s, e = norm_segments_all[int(i)]
            assigned_k[(int(cid), int(s), int(e))] = 2
        total_neg_est = n_norm + int(extra)

    assigned_by_case: Dict[int, Dict[Tuple[int, int], int]] = {}
    for (cid, s, e), k in assigned_k.items():
        d = assigned_by_case.setdefault(int(cid), {})
        d[(int(s), int(e))] = int(k)

    # Build tasks
    print("\n[3/3] Generating segments (.npz)...")
    convert_tasks = []
    case_meta: Dict[int, Tuple[str, str]] = {}

    def clin_to_vec(row) -> np.ndarray:
        sex = 1.0 if str(row.sex).upper().startswith("M") else 0.0
        return np.array(
            [
                float(row.age),
                float(sex),
                float(row.bmi),
                float(row.asa),
                float(row.emop),
                float(row.preop_htn),
                float(row.preop_hb),
                float(row.preop_bun),
                float(row.preop_cr),
                float(row.preop_alb),
                float(row.preop_na),
                float(row.preop_k),
            ],
            dtype=np.float32,
        )

    out_base = str(args.out_dir)
    for row in df_valid.itertuples(index=False):
        cid = int(getattr(row, "caseid"))
        client_id = str(getattr(row, "client_id"))
        split = str(getattr(row, "split"))
        src = str(getattr(row, "case_path"))
        dst = os.path.join(out_base, client_id, split, f"case_{cid}.npz")
        case_meta[int(cid)] = (str(client_id), str(split))
        assigned = assigned_by_case.get(cid, {})
        convert_tasks.append(
            (
                cid,
                src,
                dst,
                clin_to_vec(row),
                assigned,
                int(args.seed),
                float(args.cycle_min_sec),
                float(args.cycle_max_sec),
                bool(args.instance_norm),
                bool(args.require_etco2),
            )
        )

    print(f"  cases: {len(convert_tasks)} (pos_events={total_pos}, norm_segs={n_norm}, neg_est={total_neg_est})")
    success_count = 0
    pos_written = 0
    neg_written = 0
    written_cases_by_split: Dict[str, int] = {}
    written_windows_by_split: Dict[str, Dict[str, int]] = {}
    written_clients: Dict[str, Dict[str, object]] = {}
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(convert_worker, t) for t in convert_tasks]
        for f in tqdm(as_completed(futures), total=len(futures)):
            case_id, ok, n_pos, n_neg = f.result()
            if ok:
                success_count += 1
                pos_written += int(n_pos)
                neg_written += int(n_neg)
                meta = case_meta.get(int(case_id))
                if meta:
                    client_id, split = meta
                    written_cases_by_split[split] = int(written_cases_by_split.get(split, 0)) + 1
                    ws = written_windows_by_split.setdefault(split, {"pos_windows": 0, "neg_windows": 0})
                    ws["pos_windows"] = int(ws["pos_windows"]) + int(n_pos)
                    ws["neg_windows"] = int(ws["neg_windows"]) + int(n_neg)
                    cstat = written_clients.setdefault(
                        client_id,
                        {"cases_written": 0, "pos_windows": 0, "neg_windows": 0, "splits": {}},
                    )
                    cstat["cases_written"] = int(cstat["cases_written"]) + 1
                    cstat["pos_windows"] = int(cstat["pos_windows"]) + int(n_pos)
                    cstat["neg_windows"] = int(cstat["neg_windows"]) + int(n_neg)
                    cs = cstat["splits"].setdefault(split, {"cases_written": 0, "pos_windows": 0, "neg_windows": 0})
                    cs["cases_written"] = int(cs["cases_written"]) + 1
                    cs["pos_windows"] = int(cs["pos_windows"]) + int(n_pos)
                    cs["neg_windows"] = int(cs["neg_windows"]) + int(n_neg)
    failed_cases = int(len(convert_tasks) - success_count)
    pos_dropped_est = int(max(int(total_pos) - int(pos_written), 0))
    neg_dropped_est = int(max(int(total_neg_est) - int(neg_written), 0))

    case_stats = df_valid[["caseid", "client_id", "split"]].drop_duplicates().copy()
    case_stats["pos_events"] = case_stats["caseid"].map(pos_by_case).fillna(0).astype(int)
    split_case_counts = case_stats.groupby("split")["caseid"].nunique().to_dict()
    client_case_counts = case_stats.groupby("client_id")["caseid"].nunique().to_dict()
    client_split_case_counts = case_stats.groupby(["client_id", "split"])["caseid"].nunique().to_dict()
    split_pos_events = case_stats.groupby("split")["pos_events"].sum().to_dict()
    client_pos_events = case_stats.groupby("client_id")["pos_events"].sum().to_dict()
    client_split_pos_events = case_stats.groupby(["client_id", "split"])["pos_events"].sum().to_dict()
    splits_detail: Dict[str, Dict[str, int]] = {}
    for split in ACTIVE_SPLITS:
        ws = written_windows_by_split.get(split, {"pos_windows": 0, "neg_windows": 0})
        splits_detail[split] = {
            "cases": int(split_case_counts.get(split, 0)),
            "cases_written": int(written_cases_by_split.get(split, 0)),
            "pos_windows": int(ws.get("pos_windows", 0)),
            "neg_windows": int(ws.get("neg_windows", 0)),
            "pos_events": int(split_pos_events.get(split, 0)),
        }
    clients_detail: Dict[str, Dict[str, object]] = {}
    for client_id in sorted(client_case_counts.keys()):
        cstat = written_clients.get(client_id, {"cases_written": 0, "pos_windows": 0, "neg_windows": 0, "splits": {}})
        split_stats: Dict[str, Dict[str, int]] = {}
        for split in ACTIVE_SPLITS:
            key = (client_id, split)
            wsplit = cstat.get("splits", {}).get(split, {"cases_written": 0, "pos_windows": 0, "neg_windows": 0})
            split_stats[split] = {
                "cases": int(client_split_case_counts.get(key, 0)),
                "cases_written": int(wsplit.get("cases_written", 0)),
                "pos_windows": int(wsplit.get("pos_windows", 0)),
                "neg_windows": int(wsplit.get("neg_windows", 0)),
                "pos_events": int(client_split_pos_events.get(key, 0)),
            }
        clients_detail[client_id] = {
            "cases": int(client_case_counts.get(client_id, 0)),
            "cases_written": int(cstat.get("cases_written", 0)),
            "pos_windows": int(cstat.get("pos_windows", 0)),
            "neg_windows": int(cstat.get("neg_windows", 0)),
            "pos_events": int(client_pos_events.get(client_id, 0)),
            "splits": split_stats,
        }

    summary = {
        "eligible_cases": int(len(df_valid)),
        "clients": {cid: int(count) for cid, count in sorted(final_counts.items())},
        "splits": {split: int((df_valid["split"] == split).sum()) for split in ACTIVE_SPLITS},
        "splits_detail": splits_detail,
        "clients_detail": clients_detail,
        "pass1_total_pos_events": int(total_pos),
        "pass1_total_norm_segments": int(n_norm),
        "assigned_total_neg_windows_est": int(total_neg_est),
        "written_case_files": int(success_count),
        "written_pos_windows": int(pos_written),
        "written_neg_windows": int(neg_written),
        "exclusion": {
            "raw_cases": int(n_raw_cases),
            "after_basic_filter": int(n_after_basic_filter),
            "dropped_basic_filter": int(n_raw_cases - n_after_basic_filter),
            "after_dropna_clinical": int(n_after_dropna_clin),
            "dropped_missing_clinical": int(n_after_basic_filter - n_after_dropna_clin),
            "after_dropna_caseid": int(n_after_dropna_caseid),
            "dropped_missing_caseid": int(n_after_dropna_clin - n_after_dropna_caseid),
            "missing_waveform_files": int(n_missing_wave),
            "missing_etco2_waveform": int(n_missing_etco2),
            "missing_client_id": int(missing_client_id),
            "missing_split": int(missing_split),
            "failed_cases_no_segments": int(failed_cases),
        },
        "missing": {
            "counts": {k: int(v) for k, v in missing_counts.items()},
            "rates": {k: (float(v) if v is not None else None) for k, v in missing_rates.items()},
        },
        "window_drop_estimate": {
            "pos_events": int(total_pos),
            "neg_est": int(total_neg_est),
            "pos_written": int(pos_written),
            "neg_written": int(neg_written),
            "pos_dropped_est": int(pos_dropped_est),
            "neg_dropped_est": int(neg_dropped_est),
        },
        "client_scheme": str(args.client_scheme),
        "merge_strategy": str(args.merge_strategy),
        "opname_threshold": int(args.opname_threshold),
        "min_client_cases": int(args.min_client_cases),
        "min_client_pos": int(args.min_client_pos),
        "excluded_clients": {cid: int(cnt) for cid, cnt in sorted(excluded_counts.items())} if excluded_counts else {},
        "excluded_small_clients": {cid: int(cnt) for cid, cnt in sorted(excluded_small_clients.items())} if excluded_small_clients else {},
        "excluded_low_pos_clients": {cid: int(cnt) for cid, cnt in sorted(excluded_low_pos_clients.items())} if excluded_low_pos_clients else {},
        "notes": {
            "waveforms": ["ABP", "ECG", "PPG", "ETCO2"],
            "tracks": {"ABP": WAVEFORMS["ABP"], "ECG": WAVEFORMS["ECG"], "PPG": WAVEFORMS["PPG"], "MBP": LABEL_TRACK},
            "fs_hz": int(FS),
            "window_sec": int(WINDOW_SEC),
            "horizon_min": int(HORIZON_MIN),
            "map_threshold": float(MAP_HYPOTEN),
            "hypotension_min_duration_sec": int(HYPOTEN_MIN_DUR_SEC),
            "normotension_min_duration_sec": int(NORM_MIN_DUR_SEC),
            "artifact_map_range": [float(MAP_MIN_VALID), float(MAP_MAX_VALID)],
            "cycle_len_sec": [float(args.cycle_min_sec), float(args.cycle_max_sec)],
            "instance_norm": bool(args.instance_norm),
            "require_etco2": bool(args.require_etco2),
            "clinical_cols": list(CLINICAL_COLS),
            "paper_covariate_missing": ["WBC"],
        },
    }
    with open(os.path.join(out_base, "summary.json"), "w", encoding="utf-8") as fsum:
        json.dump(summary, fsum, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print(f" Dataset Build Complete. Created {success_count} .npz files.")
    print(f" Windows: pos={pos_written}, neg={neg_written}")
    print(f" Output: {args.out_dir}")
    print(f" Summary: {os.path.join(out_base, 'summary.json')}")
    print("="*60)

if __name__ == "__main__":
    main()
