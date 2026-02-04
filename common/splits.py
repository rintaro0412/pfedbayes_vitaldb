from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.8
    val: float = 0.0
    test: float = 0.2
    seed: int = 42

    def validate(self) -> None:
        s = float(self.train) + float(self.val) + float(self.test)
        if not np.isclose(s, 1.0):
            raise ValueError(f"split ratios must sum to 1.0, got {s}")
        if min(self.train, self.val, self.test) < 0:
            raise ValueError("split ratios must be non-negative")


def choose_group_key(available_cols: Sequence[str]) -> str:
    cols = {str(c).strip().lower() for c in available_cols}
    if "subjectid" in cols:
        return "subjectid"
    return "caseid"


def make_group_splits(
    case_rows: list[dict],
    *,
    group_key: str,
    cfg: SplitConfig,
) -> dict[int, SplitName]:
    """
    Return mapping: caseid -> split, splitting by `group_key` (e.g., subjectid).

    `case_rows` must have keys: caseid, and group_key (can be None).
    If group_key is None for a row, falls back to caseid as group.
    """
    cfg.validate()

    groups: dict[int, set[int]] = {}
    for r in case_rows:
        caseid = int(r["caseid"])
        g = r.get(group_key)
        try:
            gid = int(g) if g is not None else caseid
        except Exception:
            gid = caseid
        groups.setdefault(gid, set()).add(caseid)

    group_ids = np.array(sorted(groups.keys()), dtype=np.int64)
    rng = np.random.RandomState(int(cfg.seed))
    rng.shuffle(group_ids)

    n = int(group_ids.shape[0])
    n_train = int(round(n * float(cfg.train)))
    n_val = int(round(n * float(cfg.val)))
    n_train = max(0, min(n, n_train))
    n_val = max(0, min(n - n_train, n_val))
    n_test = n - n_train - n_val

    train_g = set(group_ids[:n_train].tolist())
    val_g = set(group_ids[n_train : n_train + n_val].tolist())
    test_g = set(group_ids[n_train + n_val :].tolist())
    assert len(train_g & val_g) == 0 and len(train_g & test_g) == 0 and len(val_g & test_g) == 0

    out: dict[int, SplitName] = {}
    for gid, caseids in groups.items():
        if gid in train_g:
            split: SplitName = "train"
        elif gid in val_g:
            split = "val"
        else:
            split = "test"
        for cid in caseids:
            out[int(cid)] = split

    # Any missing caseids (shouldn't happen) -> train
    for r in case_rows:
        cid = int(r["caseid"])
        out.setdefault(cid, "train")

    return out
