#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_model_trace(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _summarize_compute_log(data: Any) -> Dict[str, Any]:
    if not isinstance(data, list) or not data:
        return {"status": "missing_or_invalid"}
    client_steps = [int(d.get("client_steps", 0)) for d in data if isinstance(d, dict)]
    distill_steps = [int(d.get("distill_steps", 0)) for d in data if isinstance(d, dict)]
    total_steps = [int(d.get("total_steps", 0)) for d in data if isinstance(d, dict)]
    rounds = len(data)
    return {
        "status": "ok",
        "rounds": rounds,
        "client_steps_total": int(sum(client_steps)),
        "distill_steps_total": int(sum(distill_steps)),
        "total_steps_total": int(sum(total_steps)),
        "client_steps_mean": float(mean(client_steps)) if client_steps else 0.0,
        "distill_steps_mean": float(mean(distill_steps)) if distill_steps else 0.0,
        "total_steps_mean": float(mean(total_steps)) if total_steps else 0.0,
    }


def _summarize_model_trace(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"status": "missing_or_invalid"}
    teacher_counts: List[int] = []
    teacher_types: set[str] = set()
    teacher_modes: set[str] = set()
    teacher_num_samples: set[int] = set()
    teacher_add_fedavg: set[bool] = set()
    teacher_include_clients: set[bool] = set()
    used_swa: List[bool] = []
    update_modes: set[str] = set()

    for rec in rows:
        if not isinstance(rec, dict):
            continue
        if rec.get("event") == "client_init":
            if "teacher_count" in rec:
                teacher_counts.append(int(rec.get("teacher_count", 0)))
            if "teacher_type" in rec:
                teacher_types.add(str(rec.get("teacher_type")))
            if "teacher_sample_mode" in rec:
                teacher_modes.add(str(rec.get("teacher_sample_mode")))
            if "teacher_num_sample" in rec:
                teacher_num_samples.add(int(rec.get("teacher_num_sample", 0)))
            if "teacher_add_fedavg" in rec:
                teacher_add_fedavg.add(bool(rec.get("teacher_add_fedavg")))
            if "teacher_include_clients" in rec:
                teacher_include_clients.add(bool(rec.get("teacher_include_clients")))
        if rec.get("event") == "round_end":
            if "used_swa" in rec:
                used_swa.append(bool(rec.get("used_swa")))
            if "update_mode" in rec:
                update_modes.add(str(rec.get("update_mode")))

    summary: Dict[str, Any] = {"status": "ok"}
    if teacher_counts:
        summary.update(
            {
                "teacher_count_min": int(min(teacher_counts)),
                "teacher_count_mean": float(mean(teacher_counts)),
                "teacher_count_max": int(max(teacher_counts)),
            }
        )
    summary["teacher_type_unique"] = sorted(teacher_types) if teacher_types else []
    summary["teacher_sample_mode_unique"] = sorted(teacher_modes) if teacher_modes else []
    summary["teacher_num_sample_unique"] = sorted(teacher_num_samples) if teacher_num_samples else []
    summary["teacher_add_fedavg_unique"] = sorted(teacher_add_fedavg) if teacher_add_fedavg else []
    summary["teacher_include_clients_unique"] = sorted(teacher_include_clients) if teacher_include_clients else []
    if used_swa:
        summary["used_swa_rounds"] = int(sum(1 for v in used_swa if v))
        summary["used_swa_ratio"] = float(sum(1 for v in used_swa if v) / max(len(used_swa), 1))
    summary["update_mode_unique"] = sorted(update_modes) if update_modes else []
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize compute_log.json and model_trace.jsonl for FedBE.")
    ap.add_argument("--run-dir", required=True, help="FedBE run directory (e.g., runs/fedbe/seed0)")
    ap.add_argument("--out-json", default=None, help="Write summary JSON to this path")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    compute_log = _read_json(run_dir / "compute_log.json")
    model_trace = _read_model_trace(run_dir / "model_trace.jsonl")

    summary = {
        "run_dir": str(run_dir),
        "compute_log": _summarize_compute_log(compute_log),
        "model_trace": _summarize_model_trace(model_trace),
    }

    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.out_json:
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
