from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class WindowSpec:
    fs_wave: int = 100
    window_sec: int = 30
    lead_sec: int = 300

    @property
    def window_samples(self) -> int:
        return int(self.fs_wave * self.window_sec)


class _CaseCache:
    def __init__(self, max_cases: int):
        self.max_cases = int(max_cases)
        self._cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()

    def get(self, path: str) -> Optional[Dict[str, np.ndarray]]:
        if path not in self._cache:
            return None
        self._cache.move_to_end(path)
        return self._cache[path]

    def put(self, path: str, data: Dict[str, np.ndarray]) -> None:
        self._cache[path] = data
        self._cache.move_to_end(path)
        if self.max_cases > 0 and len(self._cache) > self.max_cases:
            self._cache.popitem(last=False)


def _fill_nan_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if np.isfinite(x).all():
        return x
    if np.isfinite(x).any():
        med = float(np.nanmedian(x))
    else:
        med = 0.0
    return np.where(np.isfinite(x), x, med).astype(np.float32, copy=False)


def _z_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = float(x.mean())
    std = float(x.std())
    if not np.isfinite(std) or std < eps:
        std = eps
    return ((x - mean) / std).astype(np.float32, copy=False)


class IOHWindowDataset(Dataset):
    """
    Slice windows from per-case processed .npz using an index CSV.

    Index CSV requirements:
      - processed_path
      - caseid
      - win_start_sec
      - win_end_sec
      - label (0/1)
      - split (train/val/test)
    """

    def __init__(
        self,
        index_csv: str | Path,
        *,
        split: str,
        sample_kind: str = "anchor",
        window: WindowSpec = WindowSpec(),
        cache_cases: int = 8,
        return_meta: bool = False,
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        df = pd.read_csv(index_csv)
        df.columns = [str(c).strip() for c in df.columns]
        for col in ["processed_path", "caseid", "win_start_sec", "win_end_sec", "label", "split"]:
            if col not in df.columns:
                raise ValueError(f"index missing required column '{col}': {index_csv}")

        df = df[df["split"] == str(split)].copy()
        if "sample_kind" in df.columns:
            df = df[df["sample_kind"] == str(sample_kind)].copy()
        if filters:
            for k, v in filters.items():
                if k not in df.columns:
                    continue
                if isinstance(v, (list, tuple, set)):
                    df = df[df[k].isin(list(v))].copy()
                else:
                    df = df[df[k] == v].copy()

        df = df.reset_index(drop=True)
        self.df = df
        self.window = window
        self.cache = _CaseCache(max_cases=cache_cases)
        self.return_meta = bool(return_meta)

    def __len__(self) -> int:
        return int(len(self.df))

    def _load_case(self, path: str) -> Dict[str, np.ndarray]:
        cached = self.cache.get(path)
        if cached is not None:
            return cached

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"processed case not found: {p}")

        with np.load(p, allow_pickle=False) as z:
            out = {
                "abp_100hz": np.asarray(z["abp_100hz"], dtype=np.float32),
                "ecg_100hz": np.asarray(z["ecg_100hz"], dtype=np.float32),
                "ppg_100hz": np.asarray(z["ppg_100hz"], dtype=np.float32),
                "fs_wave": int(z["fs_wave"]) if "fs_wave" in z else 100,
            }
        self.cache.put(str(p), out)
        return out

    def __getitem__(self, idx: int):
        row = self.df.iloc[int(idx)]
        case_path = str(row["processed_path"])
        data = self._load_case(case_path)

        fs = int(data.get("fs_wave", 100))
        s_sec = int(row["win_start_sec"])
        e_sec = int(row["win_end_sec"])
        s = s_sec * fs
        e = e_sec * fs

        abp = data["abp_100hz"][s:e]
        ecg = data["ecg_100hz"][s:e]
        ppg = data["ppg_100hz"][s:e]

        # NaN fill then per-window z-normalization (per channel)
        abp = _z_norm(_fill_nan_1d(abp))
        ecg = _z_norm(_fill_nan_1d(ecg))
        ppg = _z_norm(_fill_nan_1d(ppg))

        x = np.stack([abp, ecg, ppg], axis=0)  # (3, T)
        y = float(row["label"])

        x_t = torch.from_numpy(x).float()
        y_t = torch.tensor([y], dtype=torch.float32)

        if not self.return_meta:
            return x_t, y_t

        meta: Dict[str, Any] = {
            "caseid": int(row["caseid"]),
        }
        for k in ["subjectid", "client_id", "anchor_time_sec", "sample_kind"]:
            if k in row:
                meta[k] = row[k]
        return x_t, y_t, meta
