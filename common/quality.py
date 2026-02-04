from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import numpy as np


def _nan_ratio(x: np.ndarray) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return 1.0
    return float(np.isnan(x).mean())


def _range(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    if not np.isfinite(x).any():
        return float("nan")
    return float(np.nanmax(x) - np.nanmin(x))


def _fill_nan_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if np.isfinite(x).all():
        return x
    if np.isfinite(x).any():
        med = float(np.nanmedian(x))
    else:
        med = 0.0
    return np.where(np.isfinite(x), x, med).astype(np.float32, copy=False)


def _smooth_1d(x: np.ndarray, n: int) -> np.ndarray:
    if int(n) <= 1:
        return x
    n = int(n)
    kernel = np.ones(n, dtype=np.float32) / float(n)
    return np.convolve(x, kernel, mode="same")


def _max_true_run(mask: np.ndarray) -> int:
    max_run = 0
    run = 0
    for v in mask:
        if v:
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    return int(max_run)


def _max_nan_run_sec(x: np.ndarray, fs: int) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    nan_mask = ~np.isfinite(x)
    max_run = _max_true_run(nan_mask)
    return float(max_run) / float(fs)


def _max_flatline_sec(x: np.ndarray, fs: int, eps: float) -> float:
    x = np.asarray(x)
    if x.size < 2:
        return 0.0
    finite = np.isfinite(x)
    if not np.any(finite):
        return float("inf")
    diff = np.abs(np.diff(x.astype(np.float64)))
    same = (diff <= float(eps)) & finite[1:] & finite[:-1]
    max_run = _max_true_run(same)
    if max_run <= 0:
        return 0.0
    # run length in samples = max_run + 1 (because diff length is n-1)
    return float(max_run + 1) / float(fs)


def compute_valid_sec_mask_1hz(
    *,
    map_1hz: np.ndarray,
    hr_1hz: np.ndarray,
    spo2_1hz: np.ndarray,
    abp_100hz: np.ndarray,
    fs_wave: int = 100,
    map_range: Tuple[float, float] = (30.0, 200.0),
    abp_range: Tuple[float, float] = (0.0, 300.0),
    hr_range: Tuple[float, float] = (30.0, 200.0),
    spo2_range: Tuple[float, float] = (70.0, 100.0),
) -> np.ndarray:
    """
    Build 1Hz valid mask for quality gating.

    Notes:
      - Default numeric ranges: MAP 30-200, ABP 0-300 (instantaneous), HR 30-200, SpO2 70-100.
      - Numeric series: any NaN -> invalid second.
      - ABP range check is applied per second (any sample outside range -> invalid).
      - Waveform missingness/flatline checks are handled at window level.
    """
    map_1hz = np.asarray(map_1hz, dtype=np.float32)
    hr_1hz = np.asarray(hr_1hz, dtype=np.float32)
    spo2_1hz = np.asarray(spo2_1hz, dtype=np.float32)
    abp = np.asarray(abp_100hz, dtype=np.float32)

    n_sec = int(min(map_1hz.shape[0], hr_1hz.shape[0], spo2_1hz.shape[0], abp.shape[0] // fs_wave))
    if n_sec <= 0:
        return np.zeros((0,), dtype=bool)

    map_1hz = map_1hz[:n_sec]
    hr_1hz = hr_1hz[:n_sec]
    spo2_1hz = spo2_1hz[:n_sec]
    abp = abp[: n_sec * fs_wave]

    valid = np.ones((n_sec,), dtype=bool)

    # Numeric NaN + physiological ranges
    valid &= np.isfinite(map_1hz) & (map_1hz >= map_range[0]) & (map_1hz <= map_range[1])
    valid &= np.isfinite(hr_1hz) & (hr_1hz >= hr_range[0]) & (hr_1hz <= hr_range[1])
    valid &= np.isfinite(spo2_1hz) & (spo2_1hz >= spo2_range[0]) & (spo2_1hz <= spo2_range[1])

    # ABP physiological range per second: any out-of-range sample invalidates the second
    abp2 = abp.reshape(n_sec, fs_wave)
    finite = np.isfinite(abp2)
    too_low = finite & (abp2 < abp_range[0])
    too_high = finite & (abp2 > abp_range[1])
    valid &= ~(too_low.any(axis=1) | too_high.any(axis=1))

    return valid


@dataclass(frozen=True)
class WindowQuality:
    passed: bool
    valid_ratio_1hz: float
    nan_ratio_abp: float
    nan_ratio_ecg: float
    nan_ratio_ppg: float
    max_nan_run_sec: float
    abp_amp: float
    abp_max_absdiff: float
    ecg_flatline_sec: float
    ppg_flatline_sec: float
    ecg_range: float
    ppg_range: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def check_window_quality(
    *,
    abp_win: np.ndarray,
    ecg_win: np.ndarray,
    ppg_win: np.ndarray,
    valid_sec_mask_win: np.ndarray,
    fs_wave: int = 100,
    max_nan_ratio_wave: float = 0.1,
    max_nan_run_sec: float = 1.0,
    min_valid_ratio_1hz: float = 0.9,
    min_abp_amp: float = 20.0,
    max_abp_absdiff: float = 20.0,
    abp_diff_smooth_n: int = 5,
    flatline_max_sec: float = 1.0,
    flatline_eps: float = 1e-4,
) -> WindowQuality:
    """
    Thresholds are heuristic (physiologic ranges + artifact screening), not tied to a single paper.
    Adjust per protocol or cite a target reference if required.
    Missingness gating uses 60s NaN ratio + max contiguous NaN length.
    ECG/PPG flatline gating uses long runs of near-constant values.
    """
    abp_win = np.asarray(abp_win, dtype=np.float32)
    ecg_win = np.asarray(ecg_win, dtype=np.float32)
    ppg_win = np.asarray(ppg_win, dtype=np.float32)
    v = np.asarray(valid_sec_mask_win, dtype=bool)

    valid_ratio = float(v.mean()) if v.size else 0.0
    nan_abp = _nan_ratio(abp_win)
    nan_ecg = _nan_ratio(ecg_win)
    nan_ppg = _nan_ratio(ppg_win)
    max_nan_run = max(
        _max_nan_run_sec(abp_win, fs_wave),
        _max_nan_run_sec(ecg_win, fs_wave),
        _max_nan_run_sec(ppg_win, fs_wave),
    )

    abp_amp = _range(abp_win)
    # diff across 100Hz samples after light smoothing
    abp_sm = _smooth_1d(_fill_nan_1d(abp_win), abp_diff_smooth_n)
    if abp_sm.size >= 2:
        abp_max_diff = float(np.max(np.abs(np.diff(abp_sm.astype(np.float64)))))
    else:
        abp_max_diff = float("nan")

    ecg_r = _range(ecg_win)
    ppg_r = _range(ppg_win)
    ecg_flat = _max_flatline_sec(ecg_win, fs_wave, flatline_eps)
    ppg_flat = _max_flatline_sec(ppg_win, fs_wave, flatline_eps)

    passed = True
    if valid_ratio < float(min_valid_ratio_1hz):
        passed = False
    if max(nan_abp, nan_ecg, nan_ppg) > float(max_nan_ratio_wave):
        passed = False
    if max_nan_run >= float(max_nan_run_sec):
        passed = False
    if not np.isfinite(abp_amp) or abp_amp < float(min_abp_amp):
        passed = False
    if np.isfinite(abp_max_diff) and abp_max_diff > float(max_abp_absdiff):
        passed = False
    if ecg_flat >= float(flatline_max_sec):
        passed = False
    if ppg_flat >= float(flatline_max_sec):
        passed = False

    return WindowQuality(
        passed=bool(passed),
        valid_ratio_1hz=float(valid_ratio),
        nan_ratio_abp=float(nan_abp),
        nan_ratio_ecg=float(nan_ecg),
        nan_ratio_ppg=float(nan_ppg),
        max_nan_run_sec=float(max_nan_run),
        abp_amp=float(abp_amp) if np.isfinite(abp_amp) else float("nan"),
        abp_max_absdiff=float(abp_max_diff) if np.isfinite(abp_max_diff) else float("nan"),
        ecg_flatline_sec=float(ecg_flat),
        ppg_flatline_sec=float(ppg_flat),
        ecg_range=float(ecg_r) if np.isfinite(ecg_r) else float("nan"),
        ppg_range=float(ppg_r) if np.isfinite(ppg_r) else float("nan"),
    )
