from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from common.io import ensure_dir, get_git_hash, write_json


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_run_dir(out_dir: str | Path, run_name: str | None, *, resume: bool = False) -> Path:
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    if run_name is None:
        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        run_name = f"run_{ts}"
    run_dir = out_dir / run_name
    if resume:
        if not run_dir.exists():
            raise FileNotFoundError(f"run_dir not found for resume: {run_dir}")
        return run_dir
    ensure_dir(run_dir)
    return run_dir


def save_pip_freeze(path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        path.write_text(out, encoding="utf-8")
    except Exception as e:
        path.write_text(f"pip freeze failed: {type(e).__name__}: {e}\n", encoding="utf-8")


def save_env_snapshot(run_dir: str | Path, config: Dict[str, Any]) -> None:
    run_dir = Path(run_dir)
    ensure_dir(run_dir)
    meta = {
        "git_hash": get_git_hash(run_dir),
        "python": sys.version,
        "executable": sys.executable,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(run_dir / "env.json", meta)
    write_json(run_dir / "config.json", config)
    save_pip_freeze(run_dir / "pip_freeze.txt")

