from __future__ import annotations

import glob
import json
import os
import re
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def parse_caseid_from_path(path: str) -> Optional[int]:
    base = os.path.basename(path)
    m = re.search(r"case_(\d+)", base)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def list_client_ids(data_dir: str) -> List[str]:
    if not os.path.isdir(data_dir):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, name)
        if not os.path.isdir(p):
            continue
        if os.path.isdir(os.path.join(p, "train")) or os.path.isdir(os.path.join(p, "val")) or os.path.isdir(os.path.join(p, "test")):
            out.append(name)
    return out


def list_npz_files(data_dir: str, split: str, client_id: str | None = None) -> List[str]:
    split = str(split)
    out: List[str] = []
    if client_id:
        split_dir = os.path.join(str(data_dir), str(client_id), split)
        npz = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
        if npz:
            return npz
        manifest = contiguous_manifest_path(split_dir)
        if os.path.isfile(manifest):
            return [manifest]
        return []

    split_dirs: List[str] = []
    split_dirs.extend(sorted(glob.glob(os.path.join(str(data_dir), "*", split))))
    split_dirs.append(os.path.join(str(data_dir), split))
    seen_dirs = set()
    for d in split_dirs:
        if d in seen_dirs:
            continue
        seen_dirs.add(d)
        if not os.path.isdir(d):
            continue
        npz = sorted(glob.glob(os.path.join(d, "*.npz")))
        if npz:
            out.extend(npz)
            continue
        manifest = contiguous_manifest_path(d)
        if os.path.isfile(manifest):
            out.append(manifest)
    return sorted(list(set(out)))


def list_npz_files_by_client(data_dir: str, split: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for cid in list_client_ids(data_dir):
        files = list_npz_files(data_dir, split, client_id=cid)
        if files:
            out[cid] = files
    return out


def contiguous_manifest_path(split_dir: str) -> str:
    return os.path.join(str(split_dir), "contiguous", "manifest.json")


def has_contiguous_manifest(split_dir: str) -> bool:
    return os.path.isfile(contiguous_manifest_path(split_dir))


def scan_label_stats(files: Sequence[str]) -> Tuple[int, int]:
    pos = 0
    total = 0
    for p in files:
        p_str = str(p)
        if (os.path.basename(p_str) == "manifest.json") and (os.path.basename(os.path.dirname(p_str)) == "contiguous"):
            with open(p_str, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if str(manifest.get("format", "")) != "windowed_contiguous_v1":
                raise ValueError(f"unsupported contiguous manifest format in {p_str}: {manifest.get('format')}")
            base = os.path.dirname(p_str)
            for shard in manifest.get("shards", []):
                y_path = os.path.join(base, str(shard["y"]))
                y = np.load(y_path, allow_pickle=False, mmap_mode="r")
                total += int(y.shape[0])
                if y.size:
                    pos += int((np.asarray(y) > 0).sum())
            continue
        with np.load(p, allow_pickle=False) as z:
            if "y" not in z:
                raise ValueError(f"missing 'y' in {p}")
            y = np.asarray(z["y"])
            total += int(y.size)
            if y.size:
                pos += int((y > 0).sum())
    return pos, total


@dataclass(frozen=True)
class FileStats:
    n_samples: int
    caseid: Optional[int]
    kind: str
    source_path: str
    shard_idx: int = -1
    shard_offset: int = 0


class WindowedNPZDataset(Dataset):
    """
    Dataset for windowed .npz files created by scripts/build_dataset.py.
    Each file must contain: x_wave (N, C, T), y (N,), optional x_clin (N, F).
    """

    def __init__(
        self,
        files: Sequence[str],
        *,
        require_window_size: int | None = None,
        use_clin: str | bool = "auto",
        cache_in_memory: bool = False,
        max_cache_files: int = 32,
        cache_dtype: str = "float32",
        return_meta: bool = False,
    ) -> None:
        super().__init__()
        self.files = self._rewrite_files_with_contiguous_manifests([str(p) for p in files])
        if not self.files:
            raise ValueError("no .npz files provided")

        self.return_meta = bool(return_meta)
        # Backward compatibility only: file caching is intentionally disabled.
        self.cache_in_memory = False
        self.max_cache_files = 0
        self.cache_dtype = "float32"

        self._contig_shards: List[Dict[str, Any]] = []
        self.file_stats: List[FileStats] = []
        has_clin = None
        wave_channels = None
        window_size = None
        clin_dim = None
        for p in self.files:
            if self._is_contiguous_manifest(p):
                manifest = self._read_contiguous_manifest(p)
                m_channels = int(manifest["wave_channels"])
                m_window = int(manifest["window_size"])
                m_has_clin = bool(manifest.get("has_clin", False))
                m_clin_dim = int(manifest.get("clin_dim", 0))
                if wave_channels is None:
                    wave_channels = m_channels
                    window_size = m_window
                else:
                    if int(m_channels) != int(wave_channels):
                        raise ValueError(f"channel mismatch in {p}: {m_channels} != {wave_channels}")
                    if int(m_window) != int(window_size):
                        raise ValueError(f"window size mismatch in {p}: {m_window} != {window_size}")
                if require_window_size is not None and int(m_window) != int(require_window_size):
                    raise ValueError(f"window size mismatch in {p}: {m_window} != {require_window_size}")
                if has_clin is None:
                    has_clin = bool(m_has_clin)
                elif bool(m_has_clin) != bool(has_clin):
                    raise ValueError(f"x_clin presence mismatch across files: {p}")
                if m_has_clin:
                    if clin_dim is None:
                        clin_dim = int(m_clin_dim)
                    elif int(m_clin_dim) != int(clin_dim):
                        raise ValueError(f"clinical dim mismatch in {p}: {m_clin_dim} != {clin_dim}")

                base = os.path.dirname(p)
                shard_base = len(self._contig_shards)
                for shard in manifest["shards"]:
                    x_wave_path = os.path.join(base, str(shard["x_wave"]))
                    y_path = os.path.join(base, str(shard["y"]))
                    x_wave_mm = np.load(x_wave_path, allow_pickle=False, mmap_mode="r")
                    y_mm = np.load(y_path, allow_pickle=False, mmap_mode="r")
                    if x_wave_mm.ndim != 3:
                        raise ValueError(f"x_wave must be (N,C,T) in {x_wave_path}, got {x_wave_mm.shape}")
                    if y_mm.ndim != 1:
                        raise ValueError(f"y must be (N,) in {y_path}, got {y_mm.shape}")
                    if int(x_wave_mm.shape[0]) != int(y_mm.shape[0]):
                        raise ValueError(f"x_wave/y sample mismatch in shard {x_wave_path}")
                    x_clin_mm = None
                    if m_has_clin:
                        if "x_clin" not in shard:
                            raise ValueError(f"missing x_clin entry in manifest shard: {p}")
                        x_clin_path = os.path.join(base, str(shard["x_clin"]))
                        x_clin_mm = np.load(x_clin_path, allow_pickle=False, mmap_mode="r")
                        if x_clin_mm.ndim != 2:
                            raise ValueError(f"x_clin must be (N,F) in {x_clin_path}, got {x_clin_mm.shape}")
                        if int(x_clin_mm.shape[0]) != int(y_mm.shape[0]):
                            raise ValueError(f"x_clin/y sample mismatch in shard {x_clin_path}")
                    self._contig_shards.append({"x_wave": x_wave_mm, "y": y_mm, "x_clin": x_clin_mm})

                case_segments = manifest.get("case_segments", [])
                if case_segments:
                    for seg in case_segments:
                        n = int(seg.get("n_samples", 0))
                        if n <= 0:
                            continue
                        caseid_raw = seg.get("caseid", None)
                        caseid = int(caseid_raw) if caseid_raw is not None else None
                        self.file_stats.append(
                            FileStats(
                                n_samples=n,
                                caseid=caseid,
                                kind="contig",
                                source_path=str(p),
                                shard_idx=int(shard_base + int(seg["shard"])),
                                shard_offset=int(seg["offset"]),
                            )
                        )
                else:
                    for i, shard in enumerate(manifest["shards"]):
                        n = int(shard.get("n_samples", 0))
                        if n <= 0:
                            continue
                        self.file_stats.append(
                            FileStats(
                                n_samples=n,
                                caseid=None,
                                kind="contig",
                                source_path=str(p),
                                shard_idx=int(shard_base + i),
                                shard_offset=0,
                            )
                        )
                continue

            with np.load(p, allow_pickle=False, mmap_mode="r") as z:
                if "x_wave" not in z or "y" not in z:
                    raise ValueError(f"missing x_wave/y in {p}")
                x_wave = z["x_wave"]
                if x_wave.ndim != 3:
                    raise ValueError(f"x_wave must be (N,C,T) in {p}, got {x_wave.shape}")
                if wave_channels is None:
                    wave_channels = int(x_wave.shape[1])
                    window_size = int(x_wave.shape[2])
                else:
                    if int(x_wave.shape[1]) != int(wave_channels):
                        raise ValueError(f"channel mismatch in {p}: {x_wave.shape[1]} != {wave_channels}")
                    if int(x_wave.shape[2]) != int(window_size):
                        raise ValueError(f"window size mismatch in {p}: {x_wave.shape[2]} != {window_size}")
                if require_window_size is not None and int(x_wave.shape[2]) != int(require_window_size):
                    raise ValueError(f"window size mismatch in {p}: {x_wave.shape[2]} != {require_window_size}")
                y = z["y"]
                n = int(y.shape[0])
                if has_clin is None:
                    has_clin = ("x_clin" in z)
                elif ("x_clin" in z) != bool(has_clin):
                    raise ValueError(f"x_clin presence mismatch across files: {p}")
                if "x_clin" in z:
                    x_clin = z["x_clin"]
                    if x_clin.ndim != 2:
                        raise ValueError(f"x_clin must be (N,F) in {p}, got {x_clin.shape}")
                    if int(x_clin.shape[0]) != int(n):
                        raise ValueError(f"x_clin samples mismatch in {p}: {x_clin.shape[0]} != {n}")
                    if clin_dim is None:
                        clin_dim = int(x_clin.shape[1])
                    elif int(x_clin.shape[1]) != int(clin_dim):
                        raise ValueError(f"clinical dim mismatch in {p}: {x_clin.shape[1]} != {clin_dim}")
                self.file_stats.append(
                    FileStats(
                        n_samples=n,
                        caseid=parse_caseid_from_path(p),
                        kind="npz",
                        source_path=str(p),
                    )
                )

        if has_clin is None:
            has_clin = False

        if isinstance(use_clin, str):
            use_clin = use_clin.lower()
            if use_clin == "auto":
                self.use_clin = bool(has_clin)
            elif use_clin in ("true", "1", "yes"):
                self.use_clin = True
            else:
                self.use_clin = False
        else:
            self.use_clin = bool(use_clin)

        if self.use_clin and not has_clin:
            raise ValueError("use_clin=True but x_clin is not present in files")

        self.wave_channels = int(wave_channels) if wave_channels is not None else 0
        self.window_size = int(window_size) if window_size is not None else 0
        if self.use_clin and clin_dim is None:
            raise ValueError("use_clin=True but clinical feature dimension is unavailable")
        self.clin_dim = int(clin_dim) if (self.use_clin and clin_dim is not None) else 0

        self._sizes = np.array([fs.n_samples for fs in self.file_stats], dtype=np.int64)
        self._cum = np.cumsum(self._sizes)
        if self._cum.size <= 0:
            raise ValueError("no samples found in files")

    @staticmethod
    def _is_contiguous_manifest(path: str) -> bool:
        p = str(path)
        return (os.path.basename(p) == "manifest.json") and (os.path.basename(os.path.dirname(p)) == "contiguous")

    @staticmethod
    def _rewrite_files_with_contiguous_manifests(files: Sequence[str]) -> List[str]:
        by_split_dir: Dict[str, List[str]] = {}
        passthrough: List[str] = []
        for raw in files:
            p = str(raw)
            if WindowedNPZDataset._is_contiguous_manifest(p):
                passthrough.append(p)
                continue
            if p.endswith(".npz"):
                by_split_dir.setdefault(os.path.dirname(p), []).append(p)
            else:
                passthrough.append(p)

        selected: List[str] = []
        for split_dir in sorted(by_split_dir.keys()):
            manifest = contiguous_manifest_path(split_dir)
            if os.path.isfile(manifest):
                selected.append(manifest)
            else:
                selected.extend(sorted(set(by_split_dir[split_dir])))
        selected.extend(sorted(set(passthrough)))
        return selected

    @staticmethod
    def _read_contiguous_manifest(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError(f"invalid contiguous manifest (not dict): {path}")
        if str(obj.get("format", "")) != "windowed_contiguous_v1":
            raise ValueError(f"unsupported contiguous manifest format in {path}: {obj.get('format')}")
        if "shards" not in obj or not isinstance(obj["shards"], list):
            raise ValueError(f"invalid contiguous manifest shards in {path}")
        return obj

    @property
    def file_sizes(self) -> List[int]:
        return [int(x) for x in self._sizes.tolist()]

    @property
    def case_ids(self) -> List[int]:
        out: List[int] = []
        for i, fs in enumerate(self.file_stats):
            if fs.caseid is not None:
                out.append(int(fs.caseid))
            else:
                out.append(int(i))
        return out

    def __len__(self) -> int:
        return int(self._cum[-1]) if self._cum.size else 0

    def _locate(self, idx: int) -> Tuple[int, int]:
        idx = int(idx)
        file_idx = int(bisect_right(self._cum, idx))
        prev = int(self._cum[file_idx - 1]) if file_idx > 0 else 0
        local = idx - prev
        return file_idx, local

    def _load_sample(self, file_idx: int, local: int) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
        fs = self.file_stats[int(file_idx)]
        if fs.kind == "npz":
            path = fs.source_path
            with np.load(path, allow_pickle=False) as z:
                x = np.array(z["x_wave"][local], dtype=np.float32, copy=True)
                y = float(np.asarray(z["y"][local]))
                clin: Optional[np.ndarray] = None
                if self.use_clin:
                    if "x_clin" not in z:
                        raise ValueError(f"missing x_clin in {path}")
                    clin = np.array(z["x_clin"][local], dtype=np.float32, copy=True)
            return x, y, clin
        if fs.kind == "contig":
            shard = self._contig_shards[int(fs.shard_idx)]
            pos = int(fs.shard_offset) + int(local)
            x = np.array(shard["x_wave"][pos], dtype=np.float32, copy=True)
            y = float(np.asarray(shard["y"][pos]))
            clin = None
            if self.use_clin:
                x_clin_arr = shard.get("x_clin", None)
                if x_clin_arr is None:
                    raise ValueError(f"missing x_clin in contiguous shard: {fs.source_path}")
                clin = np.array(x_clin_arr[pos], dtype=np.float32, copy=True)
            return x, y, clin
        raise ValueError(f"unknown file stat kind: {fs.kind}")

    def __getitem__(self, idx: int):
        file_idx, local = self._locate(idx)
        x, y, clin = self._load_sample(file_idx, local)

        x_t = torch.from_numpy(x).float()
        y_t = torch.tensor([y], dtype=torch.float32)

        if self.use_clin:
            if clin is None:
                raise ValueError(f"missing x_clin for file index {file_idx}")
            clin_t = torch.from_numpy(clin).float()
            payload = ((x_t, clin_t), y_t)
        else:
            payload = (x_t, y_t)

        if not self.return_meta:
            return payload

        fs = self.file_stats[file_idx]
        caseid = fs.caseid
        meta = {"caseid": int(caseid) if caseid is not None else int(file_idx), "file": fs.source_path}
        if self.use_clin:
            return payload[0], payload[1], meta  # ((x, clin), y, meta)
        return payload[0], payload[1], meta  # (x, y, meta)


class FederatedWindowDataset(WindowedNPZDataset):
    """
    Backward-compatible alias used by older scripts.
    """


class VitalDBDataset(WindowedNPZDataset):
    """
    Minimal compatibility wrapper. If x_wave exists, behaves like WindowedNPZDataset.
    Timeseries-only .npz files are not supported in this pipeline.
    """
