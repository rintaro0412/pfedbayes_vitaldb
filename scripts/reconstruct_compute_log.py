from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path
from typing import Dict, List

import numpy as np


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_run_config(run_dir: Path) -> Dict:
    for name in ["run_config.json", "config.json", "config_used.yaml"]:
        p = run_dir / name
        if p.exists() and p.suffix == ".json":
            return _read_json(p)
    raise FileNotFoundError(f"no run config found under {run_dir}")


def _count_train_windows(data_dir: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for client_dir in sorted(data_dir.iterdir()):
        if not client_dir.is_dir():
            continue
        train_dir = client_dir / "train"
        if not train_dir.exists():
            continue
        total = 0
        for npz in train_dir.glob("*.npz"):
            with np.load(npz, allow_pickle=False) as z:
                y = z.get("y")
                if y is None:
                    continue
                total += int(np.asarray(y).shape[0])
        out[client_dir.name] = total
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Reconstruct compute_log.json for a FedAvg run (estimated).")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-dir", default="federated_data")
    ap.add_argument("--out-name", default="compute_log.json")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    data_dir = Path(args.data_dir)
    cfg = _infer_run_config(run_dir)

    rounds = int(cfg.get("rounds", 0))
    local_cfg = cfg.get("local_cfg", {}) if isinstance(cfg, dict) else {}
    batch_size = int(local_cfg.get("batch_size", 256))
    local_epochs = int(local_cfg.get("epochs", 1))

    windows = _count_train_windows(data_dir)
    steps_per_round = 0
    for n in windows.values():
        steps_per_round += int(ceil(int(n) / max(batch_size, 1))) * int(local_epochs)

    compute_log: List[Dict] = []
    for rnd in range(1, rounds + 1):
        compute_log.append(
            {
                "round": int(rnd),
                "client_steps": int(steps_per_round),
                "distill_steps": 0,
                "total_steps": int(steps_per_round),
                "estimated": True,
            }
        )

    out_path = run_dir / args.out_name
    out_path.write_text(json.dumps(compute_log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"rounds={rounds} batch_size={batch_size} local_epochs={local_epochs} steps_per_round={steps_per_round}")


if __name__ == "__main__":
    main()
