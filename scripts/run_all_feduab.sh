#!/usr/bin/env bash
# Run Centralized + Local + FedAvg + FedUAB with shared data split and consistent run naming.
set -euo pipefail

# Tunables (override via env)
DATA_DIR="${DATA_DIR:-federated_data}"
SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-seed${SEED}}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
NO_PROGRESS_BAR="${NO_PROGRESS_BAR:-0}"

# Central/Local defaults
CENTRAL_OUT="${CENTRAL_OUT:-runs/centralized}"
LOCAL_OUT="${LOCAL_OUT:-runs/local}"
EPOCHS_CENTRAL="${EPOCHS_CENTRAL:-100}"
BATCH_TRAIN="${BATCH_TRAIN:-256}"
BATCH_CENTRAL="${BATCH_CENTRAL:-${BATCH_TRAIN}}"
BATCH_LOCAL="${BATCH_LOCAL:-${BATCH_TRAIN}}"

# FedAvg defaults
ROUNDS_FEDAVG="${ROUNDS_FEDAVG:-100}"
BATCH_FEDAVG="${BATCH_FEDAVG:-256}"
NUM_WORKERS="${NUM_WORKERS:-16}"
THRESHOLD="${THRESHOLD:-0.5}"
THRESHOLD_METHOD="${THRESHOLD_METHOD:-youden-val}"
PER_CLIENT_EVERY_ROUND="${PER_CLIENT_EVERY_ROUND:-1}"

# FedUAB defaults (paper-like)
ROUNDS_FEDUAB="${ROUNDS_FEDUAB:-${ROUNDS_FEDAVG}}"
LOCAL_EPOCHS_FEDUAB="${LOCAL_EPOCHS_FEDUAB:-1}"
BATCH_FEDUAB="${BATCH_FEDUAB:-256}"
MC_SAMPLES="${MC_SAMPLES:-25}"
MC_TRAIN="${MC_TRAIN:-5}"
KL_COEFF="${KL_COEFF:-1e-4}"
PARAM_TYPE="${PARAM_TYPE:-rho}"
VAR_REDUCTION_H="${VAR_REDUCTION_H:-1}"
FULL_BAYES="${FULL_BAYES:-1}"
TEST_EVERY_ROUND="${TEST_EVERY_ROUND:-1}"
EVAL_PERSONALIZED="${EVAL_PERSONALIZED:-1}"
EVAL_GLOBAL="${EVAL_GLOBAL:-1}"
EVAL_ENSEMBLE="${EVAL_ENSEMBLE:-1}"
BACKBONE_CHECKPOINT="${BACKBONE_CHECKPOINT:-}"
FEDUAB_CFG="${FEDUAB_CFG:-}"

# Post-processing
MAKE_EVAL_MODES="${MAKE_EVAL_MODES:-1}"
PLOT_ROUND_METRICS="${PLOT_ROUND_METRICS:-1}"
MAKE_TABLE1="${MAKE_TABLE1:-1}"
MAKE_NONIID="${MAKE_NONIID:-1}"
MAKE_CLIENT_COMPARE="${MAKE_CLIENT_COMPARE:-1}"
MAKE_CLIENT_DIRS="${MAKE_CLIENT_DIRS:-1}"
CLIENT_DIR_NAME="${CLIENT_DIR_NAME:-per_client}"
TABLE_CFG="${TABLE_CFG:-configs/feduab_tables.json}"
TABLE_OUT="${TABLE_OUT:-outputs_uab/feduab_tables}"
TABLE_METRICS="${TABLE_METRICS:-auroc,auprc,accuracy,ece,brier,nll}"

# Run directories (match run_all_fedbe.sh naming rule)
CENTRAL_RUN="${CENTRAL_OUT}/${RUN_TAG}"
LOCAL_RUN="${LOCAL_OUT}/${RUN_TAG}"
FEDAVG_RUN="runs/fedavg/${RUN_TAG}"
FEDUAB_RUN="runs/feduab/${RUN_TAG}"

echo "[INFO] seed=${SEED} tag=${RUN_TAG}"

# Centralized + Local (parallel)
central_pid=""
local_pid=""
central_status=0
local_status=0

if [[ "${SKIP_EXISTING}" == "1" && -d "${CENTRAL_RUN}" ]]; then
  echo "[SKIP] Centralized exists: ${CENTRAL_RUN}"
else
  (
    python centralized/train.py \
      --data-dir "${DATA_DIR}" \
      --out-dir "${CENTRAL_OUT}" \
      --run-name "${RUN_TAG}" \
      --epochs "${EPOCHS_CENTRAL}" \
      --batch-size "${BATCH_CENTRAL}" \
      --seed "${SEED}" \
      --num-workers "${NUM_WORKERS}" \
      --train-split train \
      --val-split val \
      --test-split test \
      --eval-threshold "${THRESHOLD}" \
      --test-every-epoch \
      --model-selection best \
      --selection-source val \
      --selection-metric auroc \
      $( [[ "${PER_CLIENT_EVERY_ROUND}" == "1" ]] && echo "--per-client-every-epoch" ) \
      $( [[ "${NO_PROGRESS_BAR}" == "1" ]] && echo "--no-progress-bar" )

    python centralized/eval.py \
      --data-dir "${DATA_DIR}" \
      --run-dir "${CENTRAL_RUN}" \
      --split test \
      --batch-size "${BATCH_CENTRAL}" \
      --num-workers "${NUM_WORKERS}" \
      --threshold "${THRESHOLD}" \
      --threshold-method "${THRESHOLD_METHOD}" \
      --val-split val \
      --per-client
  ) &
  central_pid="$!"
fi

if [[ "${SKIP_EXISTING}" == "1" && -d "${LOCAL_RUN}" ]]; then
  echo "[SKIP] Local exists: ${LOCAL_RUN}"
else
  (
    python scripts/train_local.py \
      --data-dir "${DATA_DIR}" \
      --out-dir "${LOCAL_OUT}" \
      --run-name "${RUN_TAG}" \
      --rounds "${EPOCHS_CENTRAL}" \
      --batch-size "${BATCH_LOCAL}" \
      --seed "${SEED}" \
      --num-workers "${NUM_WORKERS}" \
      --train-split train \
      --test-split test \
      --eval-threshold "${THRESHOLD}" \
      $( [[ "${NO_PROGRESS_BAR}" == "1" ]] && echo "--no-progress-bar" )
  ) &
  local_pid="$!"
fi

if [[ -n "${central_pid}" ]]; then
  if wait "${central_pid}"; then
    central_status=0
  else
    central_status=$?
  fi
fi
if [[ -n "${local_pid}" ]]; then
  if wait "${local_pid}"; then
    local_status=0
  else
    local_status=$?
  fi
fi
if [[ "${central_status}" -ne 0 || "${local_status}" -ne 0 ]]; then
  echo "[ERROR] Centralized/Local failed (central=${central_status}, local=${local_status})"
  exit 1
fi

# FedAvg + FedUAB (parallel)
fedavg_pid=""
feduab_pid=""
fedavg_status=0
feduab_status=0

if [[ "${SKIP_EXISTING}" == "1" && -d "${FEDAVG_RUN}" ]]; then
  echo "[SKIP] FedAvg exists: ${FEDAVG_RUN}"
else
  (
    LEGACY_RUN_DIR="${FEDAVG_RUN}" python federated/server.py \
      --data-dir "${DATA_DIR}" \
      --rounds "${ROUNDS_FEDAVG}" \
      --batch-size "${BATCH_FEDAVG}" \
      --seed "${SEED}" \
      --num-workers "${NUM_WORKERS}" \
      --train-split train \
      --test-split test \
      --test-every-round \
      --eval-threshold "${THRESHOLD}" \
      --threshold-method "${THRESHOLD_METHOD}" \
      --val-split val \
      --model-selection best \
      --selection-source val \
      --selection-metric auroc \
      --run-name "${RUN_TAG}" \
      $( [[ "${PER_CLIENT_EVERY_ROUND}" == "1" ]] && echo "--per-client-every-round" ) \
      $( [[ "${NO_PROGRESS_BAR}" == "1" ]] && echo "--no-progress-bar" )
  ) &
  fedavg_pid="$!"
fi

if [[ "${SKIP_EXISTING}" == "1" && -d "${FEDUAB_RUN}" ]]; then
  echo "[SKIP] FedUAB exists: ${FEDUAB_RUN}"
else
  (
    FEDUAB_ARGS=(
      --data-dir "${DATA_DIR}"
      --rounds "${ROUNDS_FEDUAB}"
      --local-epochs "${LOCAL_EPOCHS_FEDUAB}"
      --batch-size "${BATCH_FEDUAB}"
      --seed "${SEED}"
      --num-workers "${NUM_WORKERS}"
      --train-split train
      --test-split test
      --val-split val
      --mc-samples "${MC_SAMPLES}"
      --mc-train "${MC_TRAIN}"
      --kl-coeff "${KL_COEFF}"
      --param-type "${PARAM_TYPE}"
      --var-reduction-h "${VAR_REDUCTION_H}"
      --eval-threshold "${THRESHOLD}"
      --threshold-method "${THRESHOLD_METHOD}"
      --run-name "${RUN_TAG}"
    )
    if [[ "${FULL_BAYES}" == "0" ]]; then
      FEDUAB_ARGS+=(--no-full-bayes)
    fi
    if [[ "${TEST_EVERY_ROUND}" == "1" ]]; then
      FEDUAB_ARGS+=(--test-every-round)
    fi
    if [[ "${EVAL_PERSONALIZED}" == "0" ]]; then
      FEDUAB_ARGS+=(--no-eval-personalized)
    fi
    if [[ "${EVAL_GLOBAL}" == "0" ]]; then
      FEDUAB_ARGS+=(--no-eval-global)
    fi
    if [[ "${EVAL_ENSEMBLE}" == "0" ]]; then
      FEDUAB_ARGS+=(--no-eval-ensemble)
    fi
    if [[ "${PER_CLIENT_EVERY_ROUND}" == "1" ]]; then
      FEDUAB_ARGS+=(--per-client-every-round)
    fi
    if [[ -n "${BACKBONE_CHECKPOINT}" && "${BACKBONE_CHECKPOINT}" != "none" ]]; then
      FEDUAB_ARGS+=(--backbone-checkpoint "${BACKBONE_CHECKPOINT}")
    fi
    if [[ -n "${FEDUAB_CFG}" && "${FEDUAB_CFG}" != "none" ]]; then
      FEDUAB_ARGS+=(--config "${FEDUAB_CFG}")
    fi
    if [[ "${NO_PROGRESS_BAR}" == "1" ]]; then
      FEDUAB_ARGS+=(--no-progress-bar)
    fi

    LEGACY_RUN_DIR="${FEDUAB_RUN}" python bayes_federated/feduab_server.py "${FEDUAB_ARGS[@]}"
  ) &
  feduab_pid="$!"
fi

if [[ -n "${fedavg_pid}" ]]; then
  if wait "${fedavg_pid}"; then
    fedavg_status=0
  else
    fedavg_status=$?
  fi
fi
if [[ -n "${feduab_pid}" ]]; then
  if wait "${feduab_pid}"; then
    feduab_status=0
  else
    feduab_status=$?
  fi
fi
if [[ "${fedavg_status}" -ne 0 || "${feduab_status}" -ne 0 ]]; then
  echo "[ERROR] FedAvg/FedUAB failed (fedavg=${fedavg_status}, feduab=${feduab_status})"
  exit 1
fi

# Ensure FedAvg eval_modes output for consistent comparison (optional)
if [[ "${MAKE_EVAL_MODES}" == "1" ]]; then
  python - <<'PY' "${FEDAVG_RUN}"
import csv
import sys
from pathlib import Path
from common.eval_summary import METRIC_KEYS, build_mode_report, write_eval_outputs
from common.io import read_json

run_dir = Path(sys.argv[1])
out_path = run_dir / "eval_modes.json"
if out_path.exists():
    raise SystemExit(0)

test_report = run_dir / "test_report.json"
per_client_csv = run_dir / "test_report_per_client.csv"
if not test_report.exists() or not per_client_csv.exists():
    print(f"[WARN] missing FedAvg outputs for eval_modes: {run_dir}")
    raise SystemExit(0)

rep = read_json(test_report)
metrics_pre = rep.get("metrics_pre", {}) if isinstance(rep, dict) else {}
overall = {k: metrics_pre.get(k, float("nan")) for k in ("n", "n_pos", "n_neg", *METRIC_KEYS)}

per_client = {}
def _sf(v):
    try:
        return float(v)
    except Exception:
        return float("nan")

with per_client_csv.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get("status") not in (None, "", "ok"):
            continue
        cid = str(row.get("client_id", ""))
        if not cid:
            continue
        per_client[cid] = {
            "n": _sf(row.get("n", float("nan"))),
            "n_pos": _sf(row.get("n_pos", float("nan"))),
            "n_neg": _sf(row.get("n_neg", float("nan"))),
            "auroc": _sf(row.get("auroc_pre", float("nan"))),
            "auprc": _sf(row.get("auprc_pre", float("nan"))),
            "ece": _sf(row.get("ece_pre", float("nan"))),
            "nll": _sf(row.get("nll_pre", float("nan"))),
            "brier": _sf(row.get("brier_pre", float("nan"))),
        }

modes = {
    "global_idless": build_mode_report(overall=overall, per_client=per_client),
    "personalized_oracle": build_mode_report(overall=overall, per_client=per_client, note="FedAvg uses global model"),
    "ensemble_idless": build_mode_report(overall=overall, per_client=per_client, note="FedAvg uses global model"),
}
write_eval_outputs(run_dir=run_dir, algo="fedavg", modes=modes)
print(f"[INFO] wrote eval_modes for FedAvg: {run_dir}")
PY
fi

# Round-wise metrics plot (optional)
if [[ "${PLOT_ROUND_METRICS}" == "1" ]]; then
  if [[ -f "${FEDAVG_RUN}/metrics_round.csv" && -f "${FEDUAB_RUN}/metrics_round.csv" ]]; then
    python scripts/plot_round_metrics.py \
      --fedavg-run "${FEDAVG_RUN}" \
      --fedbe-run "${FEDUAB_RUN}" \
      --out-dir runs/compare_uab/round_metrics \
      --title "FedAvg vs FedUAB Round Metrics"
  else
    echo "[WARN] metrics_round.csv missing; skip plotting."
    echo "  FedAvg: ${FEDAVG_RUN}"
    echo "  FedUAB: ${FEDUAB_RUN}"
  fi
fi

# Non-IID report
if [[ "${MAKE_NONIID}" == "1" ]]; then
  python scripts/check_noniid.py \
    --data-dir "${DATA_DIR}" \
    --out tmp_noniid_report.json
fi

# Table1 (client summary)
if [[ "${MAKE_TABLE1}" == "1" ]]; then
  python scripts/make_table1_client_summary.py \
    --data-dir "${DATA_DIR}" \
    --summary-json "${DATA_DIR}/summary.json" \
    --out-csv outputs_uab/table1_client_summary.csv \
    --out-json outputs_uab/table1_client_summary.json
fi

# Client-level comparison (single seed; global_idless)
if [[ "${MAKE_CLIENT_COMPARE}" == "1" ]]; then
  python scripts/eval_compare_clients_feduab.py \
    --data-dir "${DATA_DIR}" \
    --fedavg-run "${FEDAVG_RUN}" \
    --feduab-run "${FEDUAB_RUN}" \
    --mode global_idless \
    --seed-label "${RUN_TAG}" \
    --out-json outputs_uab/feduab_vs_fedavg_client_comparison.json \
    --out-csv outputs_uab/feduab_vs_fedavg_client_comparison.csv
fi

# Per-client folders for test/eval outputs (optional)
if [[ "${MAKE_CLIENT_DIRS}" == "1" ]]; then
  python - <<'PY' "${CLIENT_DIR_NAME}" "${FEDAVG_RUN}" "${FEDUAB_RUN}"
import csv
import json
import sys
from pathlib import Path

client_dir_name = sys.argv[1]
run_dirs = [Path(p) for p in sys.argv[2:]]

def _split_csv(path: Path, out_base: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            if not fieldnames:
                return set()
            rows_by_client: dict[str, list[dict]] = {}
            for row in reader:
                cid = str(row.get("client_id", "")).strip()
                if not cid:
                    continue
                rows_by_client.setdefault(cid, []).append(row)
    except Exception:
        return set()
    for cid, rows in rows_by_client.items():
        client_dir = out_base / cid
        client_dir.mkdir(parents=True, exist_ok=True)
        out_path = client_dir / path.name
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return set(rows_by_client.keys())


def _split_json(path: Path, out_base: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    if not isinstance(data, dict):
        return set()
    out_clients: set[str] = set()
    if "clients" in data and isinstance(data.get("clients"), dict):
        base = {k: v for k, v in data.items() if k != "clients"}
        clients = data.get("clients", {})
        for cid, row in clients.items():
            cid_str = str(cid)
            client_dir = out_base / cid_str
            client_dir.mkdir(parents=True, exist_ok=True)
            out_data = dict(base)
            out_data["clients"] = {cid_str: row}
            (client_dir / path.name).write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
            out_clients.add(cid_str)
        return out_clients
    # Fallback: assume top-level dict is per-client
    for cid, row in data.items():
        cid_str = str(cid)
        client_dir = out_base / cid_str
        client_dir.mkdir(parents=True, exist_ok=True)
        (client_dir / path.name).write_text(json.dumps({cid_str: row}, ensure_ascii=False, indent=2), encoding="utf-8")
        out_clients.add(cid_str)
    return out_clients


def _process_run(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    out_base = run_dir / client_dir_name
    out_base.mkdir(parents=True, exist_ok=True)
    clients: set[str] = set()
    clients |= _split_csv(run_dir / "test_report_per_client.csv", out_base)
    clients |= _split_csv(run_dir / "eval_modes_per_client.csv", out_base)
    clients |= _split_csv(run_dir / "round_client_metrics.csv", out_base)
    clients |= _split_json(run_dir / "test_report_per_client.json", out_base)
    if clients:
        print(f"[INFO] per-client outputs: {out_base}")


for rd in run_dirs:
    _process_run(rd)
PY
fi

# Paired t-test across seeds (overall + client-macro)
# Round x client tables (Table 2-6 format; modes A/B)
if [[ -n "${TABLE_CFG}" && -f "${TABLE_CFG}" ]]; then
  if [[ -f "${FEDAVG_RUN}/round_client_metrics.csv" && -f "${FEDUAB_RUN}/round_client_metrics.csv" ]]; then
    TMP_CFG_DIR="${TMP_CFG_DIR:-/workspace/tmp}"
    mkdir -p "${TMP_CFG_DIR}"
    TMP_CFG="$(mktemp "${TMP_CFG_DIR}/feduab_tables_${SEED}_XXXX.json")"
    python - <<'PY' "${TABLE_CFG}" "${TMP_CFG}" "${FEDAVG_RUN}" "${FEDUAB_RUN}"
import json
import sys
from pathlib import Path

src, dst, fedavg_run, feduab_run = sys.argv[1:]
cfg = {}
try:
    text = Path(src).read_text(encoding="utf-8")
    if src.endswith((".yml", ".yaml")):
        try:
            import yaml  # type: ignore
            cfg = yaml.safe_load(text) or {}
        except Exception:
            cfg = {}
    else:
        cfg = json.loads(text) or {}
except Exception:
    cfg = {}

scenarios = cfg.get("scenarios", {}) if isinstance(cfg, dict) else {}
updated = False
if isinstance(scenarios, dict) and scenarios:
    for methods in scenarios.values():
        if not isinstance(methods, dict):
            continue
        if "FedAvg" in methods:
            methods["FedAvg"] = fedavg_run
            updated = True
        if "FedUAB" in methods:
            methods["FedUAB"] = feduab_run
            updated = True
if not updated:
    scenarios = {"A": {"FedAvg": fedavg_run, "FedUAB": feduab_run}}
cfg = cfg if isinstance(cfg, dict) else {}
cfg["scenarios"] = scenarios
Path(dst).write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[INFO] wrote table config: {dst}")
PY
    python scripts/aggregate_round_metrics.py \
      --config "${TMP_CFG}" \
      --out-dir "${TABLE_OUT}" \
      --metrics "${TABLE_METRICS}"
  else
    echo "[WARN] round_client_metrics.csv missing; skip table generation."
    echo "  FedAvg: ${FEDAVG_RUN}/round_client_metrics.csv"
    echo "  FedUAB: ${FEDUAB_RUN}/round_client_metrics.csv"
  fi
fi

echo "[DONE] FedUAB pipeline completed."
echo "  FedAvg: ${FEDAVG_RUN}"
echo "  FedUAB: ${FEDUAB_RUN}"
