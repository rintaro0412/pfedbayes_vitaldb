from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

# Ensure project root in sys.path when executed as `python scripts/pack_contiguous_dataset.py`
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.dataset import list_client_ids  # noqa: E402


@dataclass(frozen=True)
class CaseFile:
    path: str
    caseid: Optional[int]
    n_samples: int


@dataclass(frozen=True)
class CasePiece:
    case_idx: int
    start: int
    n_samples: int


def _parse_caseid(path: str) -> Optional[int]:
    m = re.search(r"case_(\d+)", os.path.basename(path))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _split_arg_list(value: str) -> List[str]:
    return [x.strip() for x in str(value or "").replace(",", " ").split() if x.strip()]


def _collect_split_dirs(data_dir: Path, split: str, clients: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    include_set = set(str(c) for c in clients) if clients else None
    for cid in list_client_ids(str(data_dir)):
        if include_set is not None and str(cid) not in include_set:
            continue
        p = data_dir / str(cid) / str(split)
        if p.is_dir():
            out.append(p)
    root_split = data_dir / str(split)
    if root_split.is_dir():
        out.append(root_split)
    return sorted(out)


def _scan_cases(npz_files: Sequence[str]) -> tuple[List[CaseFile], int, int, bool, int]:
    cases: List[CaseFile] = []
    wave_channels: int | None = None
    window_size: int | None = None
    has_clin: bool | None = None
    clin_dim: int = 0

    for p in npz_files:
        with np.load(p, allow_pickle=False, mmap_mode="r") as z:
            if "x_wave" not in z or "y" not in z:
                raise ValueError(f"missing x_wave/y in {p}")
            x_wave = z["x_wave"]
            y = z["y"]
            if x_wave.ndim != 3:
                raise ValueError(f"x_wave must be (N,C,T) in {p}, got {x_wave.shape}")
            if y.ndim != 1:
                raise ValueError(f"y must be (N,) in {p}, got {y.shape}")
            if int(x_wave.shape[0]) != int(y.shape[0]):
                raise ValueError(f"x_wave/y sample mismatch in {p}: {x_wave.shape[0]} != {y.shape[0]}")

            c = int(x_wave.shape[1])
            t = int(x_wave.shape[2])
            n = int(y.shape[0])
            cur_has_clin = ("x_clin" in z)
            if cur_has_clin:
                x_clin = z["x_clin"]
                if x_clin.ndim != 2:
                    raise ValueError(f"x_clin must be (N,F) in {p}, got {x_clin.shape}")
                if int(x_clin.shape[0]) != int(n):
                    raise ValueError(f"x_clin/y sample mismatch in {p}: {x_clin.shape[0]} != {n}")
                cur_clin_dim = int(x_clin.shape[1])
            else:
                cur_clin_dim = 0

            if wave_channels is None:
                wave_channels = c
                window_size = t
                has_clin = bool(cur_has_clin)
                clin_dim = int(cur_clin_dim)
            else:
                if int(c) != int(wave_channels):
                    raise ValueError(f"channel mismatch in {p}: {c} != {wave_channels}")
                if int(t) != int(window_size):
                    raise ValueError(f"window size mismatch in {p}: {t} != {window_size}")
                if bool(cur_has_clin) != bool(has_clin):
                    raise ValueError(f"x_clin presence mismatch in {p}")
                if bool(cur_has_clin) and int(cur_clin_dim) != int(clin_dim):
                    raise ValueError(f"clinical dim mismatch in {p}: {cur_clin_dim} != {clin_dim}")

            if n <= 0:
                continue
            cases.append(CaseFile(path=str(p), caseid=_parse_caseid(str(p)), n_samples=n))

    if wave_channels is None or window_size is None or has_clin is None:
        raise ValueError("no valid cases found")
    return cases, int(wave_channels), int(window_size), bool(has_clin), int(clin_dim)


def _plan_shards(cases: Sequence[CaseFile], max_samples_per_shard: int) -> List[List[CasePiece]]:
    max_n = max(1, int(max_samples_per_shard))
    shards: List[List[CasePiece]] = []
    cur: List[CasePiece] = []
    cur_n = 0

    for case_idx, case in enumerate(cases):
        start = 0
        remain = int(case.n_samples)
        while remain > 0:
            cap = int(max_n - cur_n)
            if cap <= 0:
                shards.append(cur)
                cur = []
                cur_n = 0
                cap = max_n
            take = int(min(remain, cap))
            cur.append(CasePiece(case_idx=case_idx, start=start, n_samples=take))
            cur_n += take
            start += take
            remain -= take
            if cur_n >= max_n:
                shards.append(cur)
                cur = []
                cur_n = 0
    if cur:
        shards.append(cur)
    return shards


def _write_contiguous_for_split(
    split_dir: Path,
    npz_files: Sequence[str],
    *,
    max_samples_per_shard: int,
    overwrite: bool,
    remove_npz: bool,
) -> Dict[str, int]:
    if not npz_files:
        return {"cases": 0, "samples": 0, "shards": 0}

    contig_dir = split_dir / "contiguous"
    manifest_path = contig_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        return {"cases": 0, "samples": 0, "shards": 0}

    if contig_dir.exists() and overwrite:
        shutil.rmtree(contig_dir)
    contig_dir.mkdir(parents=True, exist_ok=True)

    cases, c, t, has_clin, clin_dim = _scan_cases(npz_files)
    if not cases:
        return {"cases": 0, "samples": 0, "shards": 0}

    shards = _plan_shards(cases, max_samples_per_shard=int(max_samples_per_shard))
    total_samples = int(sum(case.n_samples for case in cases))
    shard_entries: List[Dict[str, object]] = []
    case_segments: List[Dict[str, object]] = []

    for shard_idx, pieces in enumerate(tqdm(shards, desc=f"pack {split_dir}", leave=False)):
        n_shard = int(sum(piece.n_samples for piece in pieces))
        x_wave_name = f"shard_{shard_idx:05d}_x_wave.npy"
        y_name = f"shard_{shard_idx:05d}_y.npy"
        caseid_name = f"shard_{shard_idx:05d}_caseid.npy"
        x_clin_name = f"shard_{shard_idx:05d}_x_clin.npy" if has_clin else None

        x_wave_mm = np.lib.format.open_memmap(
            str(contig_dir / x_wave_name),
            mode="w+",
            dtype=np.float32,
            shape=(n_shard, c, t),
        )
        y_mm = np.lib.format.open_memmap(
            str(contig_dir / y_name),
            mode="w+",
            dtype=np.int64,
            shape=(n_shard,),
        )
        caseid_mm = np.lib.format.open_memmap(
            str(contig_dir / caseid_name),
            mode="w+",
            dtype=np.int64,
            shape=(n_shard,),
        )
        x_clin_mm = None
        if has_clin:
            x_clin_mm = np.lib.format.open_memmap(
                str(contig_dir / str(x_clin_name)),
                mode="w+",
                dtype=np.float32,
                shape=(n_shard, clin_dim),
            )

        write_pos = 0
        for piece in pieces:
            case = cases[int(piece.case_idx)]
            n_take = int(piece.n_samples)
            lo = int(piece.start)
            hi = int(lo + n_take)
            with np.load(case.path, allow_pickle=False) as z:
                x_wave_chunk = np.asarray(z["x_wave"][lo:hi], dtype=np.float32)
                y_chunk = np.asarray(z["y"][lo:hi], dtype=np.int64)
                if has_clin:
                    x_clin_chunk = np.asarray(z["x_clin"][lo:hi], dtype=np.float32)
                else:
                    x_clin_chunk = None

            w_lo = int(write_pos)
            w_hi = int(w_lo + n_take)
            x_wave_mm[w_lo:w_hi] = x_wave_chunk
            y_mm[w_lo:w_hi] = y_chunk
            fill_caseid = int(case.caseid) if case.caseid is not None else int(piece.case_idx)
            caseid_mm[w_lo:w_hi] = fill_caseid
            if has_clin and x_clin_mm is not None and x_clin_chunk is not None:
                x_clin_mm[w_lo:w_hi] = x_clin_chunk

            case_segments.append(
                {
                    "caseid": fill_caseid,
                    "n_samples": n_take,
                    "shard": int(shard_idx),
                    "offset": int(w_lo),
                    "source_file": os.path.basename(case.path),
                }
            )
            write_pos = w_hi

        x_wave_mm.flush()
        y_mm.flush()
        caseid_mm.flush()
        del x_wave_mm
        del y_mm
        del caseid_mm
        if x_clin_mm is not None:
            x_clin_mm.flush()
            del x_clin_mm

        shard_obj: Dict[str, object] = {
            "id": int(shard_idx),
            "n_samples": int(n_shard),
            "x_wave": x_wave_name,
            "y": y_name,
            "caseid": caseid_name,
        }
        if has_clin and x_clin_name is not None:
            shard_obj["x_clin"] = str(x_clin_name)
        shard_entries.append(shard_obj)

    manifest = {
        "format": "windowed_contiguous_v1",
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "split_dir": str(split_dir),
        "total_samples": int(total_samples),
        "total_cases": int(len(cases)),
        "wave_channels": int(c),
        "window_size": int(t),
        "has_clin": bool(has_clin),
        "clin_dim": int(clin_dim if has_clin else 0),
        "shards": shard_entries,
        "case_segments": case_segments,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if remove_npz:
        for p in npz_files:
            try:
                os.remove(p)
            except Exception:
                pass

    return {"cases": int(len(cases)), "samples": int(total_samples), "shards": int(len(shards))}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pack per-case .npz dataset into contiguous memory-mapped shards")
    ap.add_argument("--data-dir", default="federated_data")
    ap.add_argument("--splits", default="train,val,test", help="Comma-separated splits to pack")
    ap.add_argument("--clients", default="", help="Optional comma-separated client IDs to limit")
    ap.add_argument("--max-samples-per-shard", type=int, default=131072)
    ap.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--remove-npz", action=argparse.BooleanOptionalAction, default=False)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"data_dir not found: {data_dir}")

    splits = _split_arg_list(args.splits)
    if not splits:
        raise SystemExit("--splits is empty")
    clients = _split_arg_list(args.clients)

    total_cases = 0
    total_samples = 0
    total_shards = 0
    total_dirs = 0

    print("Pack contiguous dataset")
    print(f"  data_dir: {data_dir}")
    print(f"  splits: {splits}")
    print(f"  clients: {clients if clients else 'ALL'}")
    print(f"  max_samples_per_shard: {int(args.max_samples_per_shard)}")
    print(f"  overwrite: {bool(args.overwrite)}")
    print(f"  remove_npz: {bool(args.remove_npz)}")

    for split in splits:
        split_dirs = _collect_split_dirs(data_dir, split, clients)
        for split_dir in split_dirs:
            npz_files = sorted(glob.glob(str(split_dir / "*.npz")))
            if not npz_files:
                continue
            total_dirs += 1
            stats = _write_contiguous_for_split(
                split_dir,
                npz_files,
                max_samples_per_shard=int(args.max_samples_per_shard),
                overwrite=bool(args.overwrite),
                remove_npz=bool(args.remove_npz),
            )
            total_cases += int(stats["cases"])
            total_samples += int(stats["samples"])
            total_shards += int(stats["shards"])
            if int(stats["shards"]) > 0:
                print(
                    f"  packed: {split_dir} "
                    f"cases={int(stats['cases'])} samples={int(stats['samples'])} shards={int(stats['shards'])}"
                )

    print("\nDone")
    print(f"  packed_dirs: {int(total_dirs)}")
    print(f"  cases: {int(total_cases)}")
    print(f"  samples: {int(total_samples)}")
    print(f"  shards: {int(total_shards)}")


if __name__ == "__main__":
    main()
