from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def atomic_torch_save(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def save_checkpoint(path: str | Path, state: Dict[str, Any]) -> None:
    atomic_torch_save(path, state)


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    weights_only: bool = False,
) -> Dict[str, Any]:
    """
    Torch 2.6 defaults weights_only=True, which blocks custom classes (e.g., BayesParams).
    For trusted local checkpoints, set weights_only=False (default here) to allow full pickle load.
    """
    return torch.load(Path(path), map_location=map_location, weights_only=bool(weights_only))


def capture_rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
