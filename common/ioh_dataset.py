from __future__ import annotations

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
      - split (train/test)
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
        # Backward compatibility only: case caching is intentionally disabled.
        self.cache_cases = 0
        self.return_meta = bool(return_meta)

    def __len__(self) -> int:
        return int(len(self.df))

    def _load_window(self, path: str, s_sec: int, e_sec: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"processed case not found: {p}")

        with np.load(p, allow_pickle=False) as z:
            fs = int(z["fs_wave"]) if "fs_wave" in z else 100
            s = int(s_sec) * fs
            e = int(e_sec) * fs
            abp = np.asarray(z["abp_100hz"][s:e], dtype=np.float32)
            ecg = np.asarray(z["ecg_100hz"][s:e], dtype=np.float32)
            ppg = np.asarray(z["ppg_100hz"][s:e], dtype=np.float32)
        return abp, ecg, ppg

    def __getitem__(self, idx: int):
        row = self.df.iloc[int(idx)]
        case_path = str(row["processed_path"])
        s_sec = int(row["win_start_sec"])
        e_sec = int(row["win_end_sec"])
        abp, ecg, ppg = self._load_window(case_path, s_sec, e_sec)

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
