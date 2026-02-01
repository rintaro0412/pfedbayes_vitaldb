from __future__ import annotations

import glob
import os
import re
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    patterns: List[str] = []
    if client_id:
        patterns.append(os.path.join(str(data_dir), str(client_id), split, "*.npz"))
    else:
        patterns.append(os.path.join(str(data_dir), "*", split, "*.npz"))
        patterns.append(os.path.join(str(data_dir), split, "*.npz"))
    out: List[str] = []
    for pat in patterns:
        out.extend(glob.glob(pat))
    return sorted(list(set(out)))


def list_npz_files_by_client(data_dir: str, split: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for cid in list_client_ids(data_dir):
        files = list_npz_files(data_dir, split, client_id=cid)
        if files:
            out[cid] = files
    return out


def scan_label_stats(files: Sequence[str]) -> Tuple[int, int]:
    pos = 0
    total = 0
    for p in files:
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


class _FileCache:
    def __init__(self, max_files: int):
        self.max_files = int(max_files)
        self._cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()

    def get(self, path: str) -> Optional[Dict[str, np.ndarray]]:
        if path not in self._cache:
            return None
        self._cache.move_to_end(path)
        return self._cache[path]

    def put(self, path: str, data: Dict[str, np.ndarray]) -> None:
        if self.max_files <= 0:
            return
        self._cache[path] = data
        self._cache.move_to_end(path)
        if len(self._cache) > self.max_files:
            self._cache.popitem(last=False)


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
        self.files = [str(p) for p in files]
        if not self.files:
            raise ValueError("no .npz files provided")

        self.return_meta = bool(return_meta)
        self.cache_in_memory = bool(cache_in_memory)
        self.cache_dtype = str(cache_dtype)
        self.cache = _FileCache(max_files=int(max_cache_files))

        self.file_stats: List[FileStats] = []
        has_clin = None
        wave_channels = None
        window_size = None
        clin_dim = None
        for p in self.files:
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
                self.file_stats.append(FileStats(n_samples=n, caseid=parse_caseid_from_path(p)))

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

    def _load_file(self, file_idx: int) -> Dict[str, np.ndarray]:
        path = self.files[int(file_idx)]
        cached = self.cache.get(path)
        if cached is not None:
            return cached

        with np.load(path, allow_pickle=False) as z:
            x_wave = z["x_wave"]
            y = z["y"]
            x_clin = z["x_clin"] if ("x_clin" in z and self.use_clin) else None

        if self.cache_in_memory:
            dtype = np.float16 if self.cache_dtype == "float16" else np.float32
            x_wave = np.asarray(x_wave, dtype=dtype)
            y = np.asarray(y, dtype=np.int64)
            if x_clin is not None:
                x_clin = np.asarray(x_clin, dtype=np.float32)
        data = {"x_wave": x_wave, "y": y}
        if x_clin is not None:
            data["x_clin"] = x_clin
        self.cache.put(path, data)
        return data

    def __getitem__(self, idx: int):
        file_idx, local = self._locate(idx)
        data = self._load_file(file_idx)
        x = np.asarray(data["x_wave"][local], dtype=np.float32)
        y = float(np.asarray(data["y"][local]))

        x_t = torch.from_numpy(x).float()
        y_t = torch.tensor([y], dtype=torch.float32)

        if self.use_clin:
            clin = np.asarray(data["x_clin"][local], dtype=np.float32)
            clin_t = torch.from_numpy(clin).float()
            payload = ((x_t, clin_t), y_t)
        else:
            payload = (x_t, y_t)

        if not self.return_meta:
            return payload

        caseid = self.file_stats[file_idx].caseid
        meta = {"caseid": int(caseid) if caseid is not None else int(file_idx), "file": self.files[file_idx]}
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
