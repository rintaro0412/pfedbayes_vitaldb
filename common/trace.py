from __future__ import annotations

import hashlib
from typing import Dict

import numpy as np
import torch


def hash_state_dict(state: Dict[str, torch.Tensor]) -> str:
    """
    Compute sha256 over concatenated (name, dtype, shape, bytes) of tensors.
    """
    h = hashlib.sha256()
    for name in sorted(state.keys()):
        t = state[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def l2_diff_state_dict(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    """
    L2 norm of full state delta (sqrt(sum((a-b)^2))).
    """
    total = 0.0
    for k in sorted(a.keys()):
        ta = a[k].detach().cpu().float().view(-1)
        tb = b[k].detach().cpu().float().view(-1)
        diff = ta - tb
        total += float(torch.dot(diff, diff).item())
    return float(np.sqrt(total))
