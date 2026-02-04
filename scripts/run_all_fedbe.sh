#!/usr/bin/env bash
# Run FedAvg + FedBE with shared data split and produce round-wise metric plots.
set -euo pipefail

# Tunables (override via env)
DATA_DIR="${DATA_DIR:-federated_data}"
FEDBE_CFG="${FEDBE_CFG:-configs/fedbe.yaml}"
SEED="${SEED:-0}"
RUN_TAG="${RUN_TAG:-seed${SEED}}"
MAKE_UNLABELED="${MAKE_UNLABELED:-1}"
SKIP_UNLABELED_IF_EXISTS="${SKIP_UNLABELED_IF_EXISTS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
ROUNDS_FEDAVG="${ROUNDS_FEDAVG:-100}"
BATCH_FEDAVG="${BATCH_FEDAVG:-256}"
NUM_WORKERS="${NUM_WORKERS:-32}"
THRESHOLD="${THRESHOLD:-0.5}"
TABLE_CFG="${TABLE_CFG:-configs/fedbe_tables.json}"
RESOURCE_SUMMARY_OUT="${RESOURCE_SUMMARY_OUT:-outputs/fedbe_resource_summary.json}"

# Resolve server_unlabeled settings from config
read -r UNLABELED_OUT UNLABELED_FRAC UNLABELED_SEED UNLABELED_MODE <<EOF
$(python - <<'PY' "${FEDBE_CFG}"
import sys
from pathlib import Path
try:
    import yaml
except Exception as e:
    raise SystemExit("PyYAML required to parse fedbe config")
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
server = cfg.get("server_unlabeled", {})
data = cfg.get("data", {})
print(
    data.get("server_unlabeled_dir", "server_unlabeled"),
    server.get("frac", 0.1),
    server.get("seed", 42),
    server.get("mode", "drop_y"),
)
PY
)
EOF

echo "[INFO] server_unlabeled: dir=${UNLABELED_OUT} frac=${UNLABELED_FRAC} seed=${UNLABELED_SEED} mode=${UNLABELED_MODE}"

if [[ "${MAKE_UNLABELED}" == "1" ]]; then
  if [[ "${SKIP_UNLABELED_IF_EXISTS}" == "1" && -d "${UNLABELED_OUT}" && $(ls -1 "${UNLABELED_OUT}"/*.npz 2>/dev/null | wc -l) -gt 0 ]]; then
    echo "[SKIP] server_unlabeled exists: ${UNLABELED_OUT}"
  else
    python scripts/make_server_unlabeled.py \
      --federated-data-dir "${DATA_DIR}" \
      --out-dir "${UNLABELED_OUT}" \
      --frac "${UNLABELED_FRAC}" \
      --seed "${UNLABELED_SEED}" \
      --mode "${UNLABELED_MODE}"
  fi
fi

# FedAvg
FEDAVG_RUN=""
FEDAVG_LEGACY="runs/fedavg/${RUN_TAG}"
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  if [[ -d "${FEDAVG_LEGACY}" ]]; then
    FEDAVG_RUN="${FEDAVG_LEGACY}"
  fi
  if [[ -n "${FEDAVG_RUN}" ]]; then
    echo "[SKIP] FedAvg exists: ${FEDAVG_RUN}"
  fi
fi
if [[ -z "${FEDAVG_RUN}" ]]; then
  LEGACY_RUN_DIR="${FEDAVG_LEGACY}" python federated/server.py \
    --data-dir "${DATA_DIR}" \
    --rounds "${ROUNDS_FEDAVG}" \
    --batch-size "${BATCH_FEDAVG}" \
    --seed "${SEED}" \
    --num-workers "${NUM_WORKERS}" \
    --train-split train \
    --test-split test \
    --test-every-round \
    --eval-threshold "${THRESHOLD}" \
    --model-selection best \
    --selection-metric auroc \
    --run-name "${RUN_TAG}" \
    --no-progress-bar
  FEDAVG_RUN="${FEDAVG_LEGACY}"
fi
if [[ -z "${FEDAVG_RUN}" ]]; then
  FEDAVG_RUN="${FEDAVG_LEGACY}"
fi

# FedBE
FEDBE_RUN=""
FEDBE_LEGACY="runs/fedbe/${RUN_TAG}"
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  if [[ -d "${FEDBE_LEGACY}" ]]; then
    FEDBE_RUN="${FEDBE_LEGACY}"
  fi
  if [[ -n "${FEDBE_RUN}" ]]; then
    echo "[SKIP] FedBE exists: ${FEDBE_RUN}"
  fi
fi
if [[ -z "${FEDBE_RUN}" ]]; then
  LEGACY_RUN_DIR="${FEDBE_LEGACY}" python fedbe_server.py \
    --config "${FEDBE_CFG}" \
    --run-name "${RUN_TAG}" \
    --no-progress-bar
  FEDBE_RUN="${FEDBE_LEGACY}"
fi
if [[ -z "${FEDBE_RUN}" ]]; then
  FEDBE_RUN="${FEDBE_LEGACY}"
fi

# Round-wise metrics plot (only if metrics_round.csv exists for both)
if [[ -f "${FEDAVG_RUN}/metrics_round.csv" && -f "${FEDBE_RUN}/metrics_round.csv" ]]; then
  python scripts/plot_round_metrics.py \
    --fedavg-run "${FEDAVG_RUN}" \
    --fedbe-run "${FEDBE_RUN}" \
    --out-dir runs/compare/round_metrics \
    --title "FedAvg vs FedBE Round Metrics"
else
  echo "[WARN] metrics_round.csv missing; skip plotting."
  echo "  FedAvg: ${FEDAVG_RUN}"
  echo "  FedBE:  ${FEDBE_RUN}"
fi

# Non-IID report
python scripts/check_noniid.py \
  --data-dir "${DATA_DIR}" \
  --out tmp_noniid_report.json

# Table1 (client summary)
python scripts/make_table1_client_summary.py \
  --data-dir "${DATA_DIR}" \
  --summary-json "${DATA_DIR}/summary.json" \
  --out-csv outputs/table1_client_summary.csv \
  --out-json outputs/table1_client_summary.json

# Summary table for FedAvg vs FedBE
python scripts/summarize_fedbe_vs_fedavg.py \
  --fedavg-run "${FEDAVG_RUN}" \
  --fedbe-run "${FEDBE_RUN}" \
  --out-json outputs/fedbe_vs_fedavg_summary.json \
  --out-csv outputs/fedbe_vs_fedavg_summary.csv

# Resource summary (FedBE compute_log + model_trace)
python scripts/summarize_fedbe_resources.py \
  --run-dir "${FEDBE_RUN}" \
  --out-json "${RESOURCE_SUMMARY_OUT}"

# Client-level comparison (single seed)
python scripts/eval_compare_clients_fedbe.py \
  --data-dir "${DATA_DIR}" \
  --fedavg-runs "${FEDAVG_RUN}" \
  --fedbe-runs "${FEDBE_RUN}" \
  --seeds "${RUN_TAG}" \
  --out-json outputs/fedbe_vs_fedavg_client_comparison.json \
  --out-csv outputs/fedbe_vs_fedavg_client_comparison.csv

# Round x client tables (Table 2-6 format; modes A/B)
if [[ -n "${TABLE_CFG}" && -f "${TABLE_CFG}" ]]; then
  python scripts/aggregate_round_metrics.py \
    --config "${TABLE_CFG}" \
    --out-dir outputs/fedocw_tables \
    --metrics "auroc,auprc,accuracy,ece,brier,nll"
fi

# Compare significance (FedAvg vs FedBE)
python scripts/compare_significance.py \
  --data-dir "${DATA_DIR}" \
  --a-kind ioh --a-run-dir "${FEDAVG_RUN}" --a-label FedAvg \
  --b-kind ioh --b-run-dir "${FEDBE_RUN}" --b-label FedBE \
  --variant pre \
  --out runs/compare/fedbe_vs_fedavg_${RUN_TAG}_pre.json

# Paper tables/fig (FedAvg vs FedBE)
python scripts/make_paper_tables_fig3.py \
  --fedavg-run "${FEDAVG_RUN}" \
  --fedbe-run "${FEDBE_RUN}" \
  --compare-json runs/compare/fedbe_vs_fedavg_${RUN_TAG}_pre.json \
  --out-dir outputs \
  --allow-missing

# Round x client boxplots (requires round_*_test_per_client.csv)
python scripts/make_round_client_boxplots.py \
  --runs "${FEDAVG_RUN},${FEDBE_RUN}" \
  --labels "FedAvg,FedBE" \
  --out-dir outputs

echo "[DONE] FedBE pipeline completed."
echo "  FedAvg: ${FEDAVG_RUN}"
echo "  FedBE:  ${FEDBE_RUN}"
