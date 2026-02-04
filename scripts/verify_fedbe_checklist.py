from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root in sys.path when executed as `python scripts/verify_fedbe_checklist.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.dataset import list_npz_files, parse_caseid_from_path
from common.io import now_utc_iso


@dataclass
class CheckResult:
    status: str
    details: Dict[str, Any]


def _read_json(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_yaml(path: Path) -> Dict[str, Any] | None:
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _python_exe() -> str:
    venv = PROJECT_ROOT / "venv" / "bin" / "python"
    if venv.exists():
        return str(venv)
    return sys.executable


def _safe_float(v: Any) -> float | None:
    try:
        return float(v)
    except Exception:
        return None


def _collect_caseids(paths: List[str]) -> List[int]:
    out: List[int] = []
    for p in paths:
        cid = parse_caseid_from_path(p)
        if cid is None:
            continue
        out.append(int(cid))
    return out


def _find_round_test_jsons(run_dir: Path) -> List[Path]:
    return sorted(run_dir.glob("round_*_test.json"))


def _load_round_metrics_from_json(run_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in _find_round_test_jsons(run_dir):
        m = re.search(r"round_(\d+)_test\.json", p.name)
        if not m:
            continue
        rnd = int(m.group(1))
        data = _read_json(p) or {}
        pre = data.get("metrics_pre", {}) if isinstance(data, dict) else {}
        thr = data.get("metrics_threshold", {}) if isinstance(data, dict) else {}
        rows.append(
            {
                "round": rnd,
                "algo": "unknown",
                "auroc": pre.get("auroc"),
                "auprc": pre.get("auprc"),
                "ece": pre.get("ece"),
                "nll": pre.get("nll"),
                "brier": pre.get("brier"),
                "threshold": data.get("threshold"),
                "acc": thr.get("accuracy"),
                "f1": thr.get("f1"),
                "precision": thr.get("ppv"),
                "recall": thr.get("sensitivity"),
            }
        )
    return sorted(rows, key=lambda r: int(r.get("round", 0)))


def _load_metrics_round_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as f:
        for r in csv.DictReader(f):
            row: Dict[str, Any] = {}
            for k, v in r.items():
                if k == "algo":
                    row[k] = v
                else:
                    row[k] = _safe_float(v) if v not in ("", None) else None
            rows.append(row)
    return rows


def _metrics_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    metrics = ["auroc", "auprc", "ece", "nll", "brier"]
    out: Dict[str, Any] = {"rounds": len(rows), "last_round": int(rows[-1].get("round", 0))}
    for m in metrics:
        vals = [r[m] for r in rows if isinstance(r.get(m), (int, float))]
        if len(vals) >= 3:
            out[f"{m}_first3_mean"] = float(sum(vals[:3]) / 3)
            out[f"{m}_last3_mean"] = float(sum(vals[-3:]) / 3)
    return out


def _teacher_metrics_summary(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "teacher_metrics_round.csv"
    if not path.exists():
        return {"status": "missing"}
    rows = _load_metrics_round_csv(path)
    summary = _metrics_summary(rows)
    summary["status"] = "pass"
    return summary


def _threshold_methods(run_dir: Path) -> Dict[str, Any]:
    methods: Dict[str, int] = {}
    for p in _find_round_test_jsons(run_dir):
        data = _read_json(p) or {}
        m = data.get("threshold_method")
        if m is None:
            continue
        methods[str(m)] = methods.get(str(m), 0) + 1
    return {"methods": methods, "unique": sorted(methods.keys())}


def _load_run_config(run_dir: Path) -> Dict[str, Any]:
    cfg = _read_yaml(run_dir / "config_used.yaml")
    if cfg:
        return cfg
    cfg = _read_json(run_dir / "run_config.json")
    if cfg:
        return cfg
    cfg = _read_json(run_dir / "config.json")
    if cfg:
        return cfg
    return {}


def _check_leakage(data_dir: str, server_unlabeled_dir: str, overlap_path: Path | None) -> CheckResult:
    details: Dict[str, Any] = {}
    if overlap_path and overlap_path.exists():
        overlap = _read_json(overlap_path)
        details["overlap_report"] = str(overlap_path)
        details["overlap"] = overlap
        if overlap:
            val_ok = int(overlap.get("overlap_val_count", -1)) == 0
            test_ok = int(overlap.get("overlap_test_count", -1)) == 0
        else:
            val_ok = test_ok = False
        status = "pass" if (val_ok and test_ok) else "fail"
    else:
        # fallback: compute overlaps directly
        unlabeled_files = sorted(Path(server_unlabeled_dir).glob("*.npz"))
        unlabeled_caseids = set(_collect_caseids([str(p) for p in unlabeled_files]))
        val_caseids = set(_collect_caseids(list_npz_files(data_dir, "val")))
        test_caseids = set(_collect_caseids(list_npz_files(data_dir, "test")))
        overlap_val = sorted(list(unlabeled_caseids & val_caseids))
        overlap_test = sorted(list(unlabeled_caseids & test_caseids))
        details["overlap_report"] = None
        details["overlap"] = {
            "overlap_val_count": len(overlap_val),
            "overlap_test_count": len(overlap_test),
            "overlap_val_sample": overlap_val[:10],
            "overlap_test_sample": overlap_test[:10],
        }
        status = "pass" if (len(overlap_val) == 0 and len(overlap_test) == 0) else "fail"

    # duplicate caseid check in server_unlabeled
    unlabeled_files = sorted(Path(server_unlabeled_dir).glob("*.npz"))
    caseids = _collect_caseids([str(p) for p in unlabeled_files])
    unique = len(set(caseids))
    dup = len(caseids) - unique
    details["server_unlabeled_files"] = len(caseids)
    details["server_unlabeled_unique_caseids"] = unique
    details["server_unlabeled_duplicate_caseids"] = dup
    if dup > 0:
        status = "fail"

    return CheckResult(status=status, details=details)


def _check_global_state_trace(model_trace: Path) -> CheckResult:
    details: Dict[str, Any] = {}
    if not model_trace.exists():
        return CheckResult(status="missing", details={"reason": "model_trace.jsonl not found"})
    rounds: Dict[int, Dict[str, Any]] = {}
    prev_after: Dict[int, str] = {}
    mismatches = 0
    client_init_mismatch = 0
    with model_trace.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rnd = int(rec.get("round", 0))
            ev = rec.get("event")
            rounds.setdefault(rnd, {})
            rounds[rnd][ev] = rec
            if ev == "round_end":
                prev_after[rnd] = rec.get("global_state_after_distill_hash") or rec.get("global_state_after_aggregate_hash")
            if ev == "client_init":
                if rec.get("match_count") != rec.get("total_clients"):
                    client_init_mismatch += 1

    for rnd, evs in rounds.items():
        if rnd <= 1:
            continue
        start = evs.get("round_start", {})
        prev_hash = prev_after.get(rnd - 1)
        if prev_hash is not None and start.get("global_state_in_hash") != prev_hash:
            mismatches += 1

    details["rounds"] = len(rounds)
    details["hash_mismatch_count"] = mismatches
    details["client_init_mismatch_count"] = client_init_mismatch
    status = "pass" if (mismatches == 0 and client_init_mismatch == 0) else "fail"
    return CheckResult(status=status, details=details)


def _check_compute_log(path: Path) -> CheckResult:
    if not path.exists():
        return CheckResult(status="missing", details={"reason": "compute_log.json not found"})
    data = _read_json(path) or []
    if not isinstance(data, list) or not data:
        return CheckResult(status="fail", details={"reason": "compute_log.json empty or invalid"})
    total = sum(int(d.get("total_steps", 0)) for d in data if isinstance(d, dict))
    client = sum(int(d.get("client_steps", 0)) for d in data if isinstance(d, dict))
    distill = sum(int(d.get("distill_steps", 0)) for d in data if isinstance(d, dict))
    return CheckResult(
        status="pass",
        details={
            "rounds": len(data),
            "total_steps": int(total),
            "client_steps": int(client),
            "distill_steps": int(distill),
        },
    )


def _teacher_proxy(model_trace: Path) -> CheckResult:
    if not model_trace.exists():
        return CheckResult(status="missing", details={"reason": "model_trace.jsonl not found"})
    vals: List[float] = []
    with model_trace.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("event") == "distill_start":
                v = rec.get("teacher_student_mae")
                if isinstance(v, (int, float)):
                    vals.append(float(v))
    if not vals:
        return CheckResult(status="missing", details={"reason": "teacher_student_mae not found"})
    out = {
        "count": len(vals),
        "min": float(min(vals)),
        "max": float(max(vals)),
    }
    if len(vals) >= 3:
        out["first3_mean"] = float(sum(vals[:3]) / 3)
        out["last3_mean"] = float(sum(vals[-3:]) / 3)
    return CheckResult(status="proxy", details=out)


def _distill_effect(history_path: Path, model_trace: Path) -> CheckResult:
    details: Dict[str, Any] = {}
    if history_path.exists():
        data = _read_json(history_path)
        if isinstance(data, list):
            losses = [d.get("distill_loss") for d in data if isinstance(d, dict) and isinstance(d.get("distill_loss"), (int, float))]
            updates = [d.get("distill_update_l2") for d in data if isinstance(d, dict) and isinstance(d.get("distill_update_l2"), (int, float))]
            if losses:
                details["distill_loss_last"] = float(losses[-1])
                if len(losses) >= 3:
                    details["distill_loss_first3_mean"] = float(sum(losses[:3]) / 3)
                    details["distill_loss_last3_mean"] = float(sum(losses[-3:]) / 3)
            if updates:
                details["distill_update_l2_min"] = float(min(updates))
                details["distill_update_l2_max"] = float(max(updates))
    if model_trace.exists():
        ups = []
        with model_trace.open() as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("event") == "round_end":
                    v = rec.get("global_update_l2")
                    if isinstance(v, (int, float)):
                        ups.append(float(v))
        if ups:
            details["global_update_l2_min"] = float(min(ups))
            details["global_update_l2_max"] = float(max(ups))
    status = "pass" if details else "missing"
    return CheckResult(status=status, details=details)


def _noniid_report(data_dir: str, report_path: Path, *, auto_generate: bool, py_exe: str) -> CheckResult:
    if not report_path.exists() and auto_generate:
        cmd = [py_exe, "scripts/check_noniid.py", "--data-dir", data_dir, "--out", str(report_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            return CheckResult(status="fail", details={"reason": "check_noniid failed", "stderr": proc.stderr})
    if not report_path.exists():
        return CheckResult(status="missing", details={"reason": "tmp_noniid_report.json not found"})
    data = _read_json(report_path) or {}
    pooled = data.get("per_client_pooled", {})
    pos_rates = []
    if isinstance(pooled, dict):
        for v in pooled.values():
            if isinstance(v, dict) and "pos_rate" in v:
                try:
                    pos_rates.append(float(v["pos_rate"]))
                except Exception:
                    pass
    chi2 = data.get("label_chi2_test", {})
    out = {
        "pos_rate_min": float(min(pos_rates)) if pos_rates else None,
        "pos_rate_max": float(max(pos_rates)) if pos_rates else None,
        "pos_rate_mean": float(sum(pos_rates) / len(pos_rates)) if pos_rates else None,
        "label_chi2_test": chi2,
    }
    return CheckResult(status="pass", details=out)


def _ablation_scan(runs_dir: Path, expected_frac: List[float], expected_temp: List[float]) -> CheckResult:
    found_frac = set()
    found_temp = set()
    scanned = 0
    for run in runs_dir.iterdir():
        if not run.is_dir():
            continue
        cfg = _read_yaml(run / "config_used.yaml")
        if not cfg:
            continue
        distill = cfg.get("distill", {}) if isinstance(cfg, dict) else {}
        server = cfg.get("server_unlabeled", {}) if isinstance(cfg, dict) else {}
        frac = server.get("frac")
        temp = distill.get("temperature")
        if frac is not None:
            try:
                found_frac.add(float(frac))
            except Exception:
                pass
        if temp is not None:
            try:
                found_temp.add(float(temp))
            except Exception:
                pass
        scanned += 1
    exp_frac = sorted(expected_frac)
    exp_temp = sorted(expected_temp)
    missing_frac = [v for v in exp_frac if v not in found_frac]
    missing_temp = [v for v in exp_temp if v not in found_temp]
    status = "pass" if (not missing_frac and not missing_temp) else "partial"
    return CheckResult(
        status=status,
        details={
            "scanned_runs": scanned,
            "found_frac": sorted(found_frac),
            "found_temperature": sorted(found_temp),
            "missing_frac": missing_frac,
            "missing_temperature": missing_temp,
        },
    )


def _bootstrap_compare(py_exe: str, data_dir: str, fedavg_run: str, fedbe_run: str, out_path: Path) -> CheckResult:
    cmd = [
        py_exe,
        "scripts/compare_significance.py",
        "--data-dir",
        data_dir,
        "--a-kind",
        "ioh",
        "--a-run-dir",
        fedavg_run,
        "--a-label",
        "FedAvg",
        "--b-kind",
        "ioh",
        "--b-run-dir",
        fedbe_run,
        "--b-label",
        "FedBE",
        "--variant",
        "pre",
        "--out",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return CheckResult(status="fail", details={"reason": "compare_significance failed", "stderr": proc.stderr})
    return CheckResult(status="pass", details={"out": str(out_path)})


def _plot_round_metrics(py_exe: str, fedavg_run: Path, fedbe_run: Path, out_dir: Path) -> CheckResult:
    cmd = [
        py_exe,
        "scripts/plot_round_metrics.py",
        "--fedavg-run",
        str(fedavg_run),
        "--fedbe-run",
        str(fedbe_run),
        "--out-dir",
        str(out_dir),
        "--title",
        "FedAvg vs FedBE Round Metrics",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return CheckResult(status="fail", details={"reason": "plot_round_metrics failed", "stderr": proc.stderr})
    return CheckResult(
        status="pass",
        details={"out_png": str(out_dir / "round_metrics.png"), "out_pdf": str(out_dir / "round_metrics.pdf")},
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify FedBE checklist (leakage/metrics/teacher/stats/ablations)")
    ap.add_argument("--data-dir", default="federated_data")
    ap.add_argument("--fedavg-run", default="runs/fedavg/seed0")
    ap.add_argument("--fedbe-run", default="runs/fedbe/seed0")
    ap.add_argument("--server-unlabeled-dir", default=None, help="Optional server_unlabeled dir override")
    ap.add_argument("--out-dir", default="outputs/verification")
    ap.add_argument("--noniid-report", default="tmp_noniid_report.json")
    ap.add_argument("--bootstrap", action="store_true", help="Run bootstrap significance test")
    ap.add_argument("--plot", action="store_true", help="Generate round-wise metric plot")
    ap.add_argument("--skip-heavy", action="store_true", help="Skip bootstrap/plot even if requested")
    ap.add_argument("--expected-frac", default="0.1,0.2,0.3")
    ap.add_argument("--expected-temperature", default="1,2,4")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    py_exe = _python_exe()

    fedavg_run = Path(args.fedavg_run)
    fedbe_run = Path(args.fedbe_run)
    data_dir = str(args.data_dir)

    report: Dict[str, Any] = {
        "started_utc": now_utc_iso(),
        "inputs": {
            "data_dir": data_dir,
            "fedavg_run": str(fedavg_run),
            "fedbe_run": str(fedbe_run),
        },
        "checks": {},
    }

    # Resolve server_unlabeled dir
    server_unlabeled_dir = args.server_unlabeled_dir
    if not server_unlabeled_dir:
        cfg = _read_yaml(fedbe_run / "config_used.yaml")
        if cfg and isinstance(cfg, dict):
            server_unlabeled_dir = cfg.get("data", {}).get("server_unlabeled_dir")
    if not server_unlabeled_dir:
        server_unlabeled_dir = "server_unlabeled"
    report["inputs"]["server_unlabeled_dir"] = server_unlabeled_dir

    # Leakage
    overlap_path = None
    if (fedbe_run / "overlap_report.json").exists():
        overlap_path = fedbe_run / "overlap_report.json"
    elif Path(server_unlabeled_dir).exists():
        p = Path(server_unlabeled_dir) / "overlap_report.json"
        if p.exists():
            overlap_path = p
    report["checks"]["leakage"] = _check_leakage(data_dir, server_unlabeled_dir, overlap_path).__dict__

    # Threshold method consistency
    report["checks"]["threshold_fedavg"] = _threshold_methods(fedavg_run)
    report["checks"]["threshold_fedbe"] = _threshold_methods(fedbe_run)

    # Same test split/data dir
    fa_cfg = _load_run_config(fedavg_run)
    fb_cfg = _load_run_config(fedbe_run)
    same_split = {
        "fedavg_test_split": (fa_cfg.get("splits", {}) if isinstance(fa_cfg, dict) else {}).get("test"),
        "fedbe_test_split": (fb_cfg.get("data", {}) if isinstance(fb_cfg, dict) else {}).get("test_split"),
        "fedavg_data_dir": fa_cfg.get("data_dir") if isinstance(fa_cfg, dict) else None,
        "fedbe_data_dir": (fb_cfg.get("data", {}) if isinstance(fb_cfg, dict) else {}).get("federated_dir"),
    }
    report["checks"]["same_test_split"] = same_split

    # Round metrics summaries
    fedbe_metrics = _load_metrics_round_csv(fedbe_run / "metrics_round.csv") if (fedbe_run / "metrics_round.csv").exists() else _load_round_metrics_from_json(fedbe_run)
    fedavg_metrics = _load_metrics_round_csv(fedavg_run / "metrics_round.csv") if (fedavg_run / "metrics_round.csv").exists() else _load_round_metrics_from_json(fedavg_run)
    report["checks"]["fedbe_metrics_summary"] = _metrics_summary(fedbe_metrics)
    report["checks"]["fedavg_metrics_summary"] = _metrics_summary(fedavg_metrics)

    # Global state trace
    report["checks"]["global_state_trace"] = _check_global_state_trace(fedbe_run / "model_trace.jsonl").__dict__

    # Compute log
    report["checks"]["compute_log_fedbe"] = _check_compute_log(fedbe_run / "compute_log.json").__dict__
    report["checks"]["compute_log_fedavg"] = _check_compute_log(fedavg_run / "compute_log.json").__dict__

    # Teacher proxy + distill effect
    report["checks"]["teacher_proxy"] = _teacher_proxy(fedbe_run / "model_trace.jsonl").__dict__
    report["checks"]["teacher_metrics_summary"] = _teacher_metrics_summary(fedbe_run)
    report["checks"]["distill_effect"] = _distill_effect(fedbe_run / "history.json", fedbe_run / "model_trace.jsonl").__dict__

    # Non-IID
    report["checks"]["noniid"] = _noniid_report(
        data_dir=data_dir,
        report_path=Path(args.noniid_report),
        auto_generate=True,
        py_exe=py_exe,
    ).__dict__

    # Ablation coverage
    exp_frac = [float(x) for x in str(args.expected_frac).split(",") if x.strip()]
    exp_temp = [float(x) for x in str(args.expected_temperature).split(",") if x.strip()]
    report["checks"]["ablation_scan"] = _ablation_scan(Path("runs"), exp_frac, exp_temp).__dict__

    # Optional heavy checks
    if args.bootstrap and not args.skip_heavy:
        report["checks"]["bootstrap"] = _bootstrap_compare(
            py_exe=py_exe,
            data_dir=data_dir,
            fedavg_run=str(fedavg_run),
            fedbe_run=str(fedbe_run),
            out_path=out_dir / "compare_significance.json",
        ).__dict__
    else:
        report["checks"]["bootstrap"] = CheckResult(status="skipped", details={}).__dict__

    if args.plot and not args.skip_heavy:
        report["checks"]["round_plot"] = _plot_round_metrics(
            py_exe=py_exe,
            fedavg_run=fedavg_run,
            fedbe_run=fedbe_run,
            out_dir=out_dir / "round_metrics_plot",
        ).__dict__
    else:
        report["checks"]["round_plot"] = CheckResult(status="skipped", details={}).__dict__

    report["finished_utc"] = now_utc_iso()

    out_json = out_dir / "verification_report.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # short text summary
    lines = [
        "FedBE verification report",
        f"- fedavg_run: {fedavg_run}",
        f"- fedbe_run: {fedbe_run}",
        f"- out: {out_json}",
        "",
        "Key checks:",
        f"- leakage: {report['checks']['leakage']['status']}",
        f"- threshold_fedavg: {report['checks']['threshold_fedavg']['unique']}",
        f"- threshold_fedbe: {report['checks']['threshold_fedbe']['unique']}",
        f"- global_state_trace: {report['checks']['global_state_trace']['status']}",
        f"- compute_log_fedbe: {report['checks']['compute_log_fedbe']['status']}",
        f"- noniid: {report['checks']['noniid']['status']}",
        f"- ablation_scan: {report['checks']['ablation_scan']['status']}",
    ]
    (out_dir / "verification_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
