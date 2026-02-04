from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def _python_exe() -> str:
    venv = Path("venv") / "bin" / "python"
    if venv.exists():
        return str(venv)
    return sys.executable


def _load_yaml(path: Path) -> Dict:
    import yaml  # type: ignore

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, cfg: Dict) -> None:
    import yaml  # type: ignore

    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _make_unlabeled(py_exe: str, data_dir: str, out_dir: str, frac: float, seed: int, mode: str) -> None:
    cmd = [
        py_exe,
        "scripts/make_server_unlabeled.py",
        "--federated-data-dir",
        data_dir,
        "--out-dir",
        out_dir,
        "--frac",
        str(frac),
        "--seed",
        str(seed),
        "--mode",
        str(mode),
    ]
    subprocess.check_call(cmd)


def _run_fedbe(py_exe: str, cfg_path: str, run_name: str) -> None:
    cmd = [py_exe, "fedbe_server.py", "--config", cfg_path, "--run-name", run_name, "--no-progress-bar"]
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run FedBE ablations (frac/temp).")
    ap.add_argument("--base-config", default="configs/fedbe.yaml")
    ap.add_argument("--data-dir", default="federated_data")
    ap.add_argument("--fractions", default="0.1,0.2,0.3")
    ap.add_argument("--temperatures", default="1,2,4")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="drop_y")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--out-config-dir", default="tmp/ablations")
    args = ap.parse_args()

    py_exe = _python_exe()
    base_cfg = _load_yaml(Path(args.base_config))
    out_cfg_dir = Path(args.out_config_dir)
    out_cfg_dir.mkdir(parents=True, exist_ok=True)

    fracs = [float(x) for x in str(args.fractions).split(",") if x.strip()]
    temps = [float(x) for x in str(args.temperatures).split(",") if x.strip()]

    # One-factor at a time: vary frac with temp fixed to base, and vary temp with frac fixed to base.
    base_temp = float(base_cfg.get("distill", {}).get("temperature", 2.0))
    base_frac = float(base_cfg.get("server_unlabeled", {}).get("frac", 0.2))

    # Ensure server_unlabeled dirs per frac
    for frac in fracs:
        out_dir = f"server_unlabeled_frac{str(frac).replace('.', '')}"
        if args.skip_existing and Path(out_dir).exists():
            continue
        _make_unlabeled(py_exe, args.data_dir, out_dir, frac, args.seed, args.mode)

    # Run frac ablations (temp fixed to base_temp)
    for frac in fracs:
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("server_unlabeled", {})
        cfg.setdefault("data", {})
        cfg.setdefault("train", {})
        cfg.setdefault("distill", {})
        cfg["server_unlabeled"]["frac"] = float(frac)
        cfg["data"]["server_unlabeled_dir"] = f"server_unlabeled_frac{str(frac).replace('.', '')}"
        cfg["train"]["rounds"] = int(args.rounds)
        cfg["distill"]["temperature"] = float(base_temp)
        cfg.setdefault("run", {})
        run_name = f"seed{args.seed}_frac{frac}_temp{base_temp}_r{args.rounds}"
        cfg["run"]["run_name"] = run_name
        cfg_path = out_cfg_dir / f"fedbe_frac{str(frac).replace('.', '')}.yaml"
        _write_yaml(cfg_path, cfg)
        if args.skip_existing and any(Path("runs").glob(f"*fedbe*{run_name}")):
            continue
        _run_fedbe(py_exe, str(cfg_path), run_name)

    # Run temperature ablations (frac fixed to base_frac)
    for temp in temps:
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("server_unlabeled", {})
        cfg.setdefault("data", {})
        cfg.setdefault("train", {})
        cfg.setdefault("distill", {})
        cfg["server_unlabeled"]["frac"] = float(base_frac)
        cfg["data"]["server_unlabeled_dir"] = f"server_unlabeled_frac{str(base_frac).replace('.', '')}"
        cfg["train"]["rounds"] = int(args.rounds)
        cfg["distill"]["temperature"] = float(temp)
        cfg.setdefault("run", {})
        run_name = f"seed{args.seed}_frac{base_frac}_temp{temp}_r{args.rounds}"
        cfg["run"]["run_name"] = run_name
        cfg_path = out_cfg_dir / f"fedbe_temp{str(temp).replace('.', '')}.yaml"
        _write_yaml(cfg_path, cfg)
        if args.skip_existing and any(Path("runs").glob(f"*fedbe*{run_name}")):
            continue
        _run_fedbe(py_exe, str(cfg_path), run_name)


if __name__ == "__main__":
    main()
