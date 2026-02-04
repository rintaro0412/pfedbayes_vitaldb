#!/usr/bin/env bash
# Run centralized, FedAvg, and pFedBayes (BFL) training/evaluation on the same dataset split,
# then generate comparison artifacts for paper figures/tables.
set -euo pipefail

# Tunables (override via env)
DATA_DIR="${DATA_DIR:-federated_data}"
CENTRAL_OUT="${CENTRAL_OUT:-runs/centralized}"
FED_OUT="${FED_OUT:-runs/fedavg}"
PFB_OUT="${PFB_OUT:-runs/pfedbayes}"
LOCAL_OUT="${LOCAL_OUT:-runs/local}"
PFB_CFG="${PFB_CFG:-configs/pfedbayes.yaml}"

SEED="${SEED:-0}"
BATCH_TRAIN="${BATCH_TRAIN:-256}"      # 学習バッチを揃えるための共通値（必要なら個別上書き）
BATCH_CENTRAL="${BATCH_CENTRAL:-${BATCH_TRAIN}}"
BATCH_FED="${BATCH_FED:-${BATCH_TRAIN}}"
BATCH_PFB="${BATCH_PFB:-${BATCH_TRAIN}}"
EPOCHS_CENTRAL="${EPOCHS_CENTRAL:-100}"
ROUNDS_FED="${ROUNDS_FED:-100}"
NUM_WORKERS="${NUM_WORKERS:-64}"
MC_EVAL="${MC_EVAL:-25}" # pFedBayesの評価時MCサンプリング数
SAVE_TEST_PRED="${SAVE_TEST_PRED:-1}" # 1 to save per-sample predictions for F3
LOG_CLIENT_SIM="${LOG_CLIENT_SIM:-1}" # 1 to save client similarity (F5)
BACKBONE_CKPT="${BACKBONE_CKPT:-}"
PER_CLIENT_EVERY_ROUND="${PER_CLIENT_EVERY_ROUND:-1}" # 1でラウンド×クライアント評価を保存
SKIP_EXISTING="${SKIP_EXISTING:-1}" # 1で既存runをスキップ
THRESHOLD="${THRESHOLD:-0.5}" # 固定閾値
TABLE_CFG="${TABLE_CFG:-configs/fedocw_tables.json}"   # 例: configs/fedocw_tables.json

# Derived
FED_RUNS=()
PFB_RUNS=()
CENTRAL_RUNS=()
LOCAL_RUNS=()

RUN="seed${SEED}"
echo "Seed ${SEED}"

  # Centralized
  if [[ "${SKIP_EXISTING}" == "1" && -d "${CENTRAL_OUT}/${RUN}" ]]; then
    echo "[SKIP] Centralized ${CENTRAL_OUT}/${RUN} exists"
  else
    python centralized/train.py \
      --data-dir "${DATA_DIR}" \
      --out-dir "${CENTRAL_OUT}" \
      --run-name "${RUN}" \
      --epochs "${EPOCHS_CENTRAL}" \
      --batch-size "${BATCH_CENTRAL}" \
      --seed "${SEED}" \
      --num-workers "${NUM_WORKERS}" \
      --train-split train \
      --test-split test \
      --eval-threshold "${THRESHOLD}" \
      --test-every-epoch \
      --model-selection best \
      --selection-metric auroc \
      $( [[ "${PER_CLIENT_EVERY_ROUND}" == "1" ]] && echo "--per-client-every-epoch" )

    python centralized/eval.py \
      --data-dir "${DATA_DIR}" \
      --run-dir "${CENTRAL_OUT}/${RUN}" \
      --split test \
      --batch-size "${BATCH_CENTRAL}" \
      --num-workers "${NUM_WORKERS}" \
      --threshold "${THRESHOLD}" \
      --per-client
  fi

  CENTRAL_RUNS+=("${CENTRAL_OUT}/${RUN}")

  # Local
  if [[ "${SKIP_EXISTING}" == "1" && -d "${LOCAL_OUT}/${RUN}" ]]; then
    echo "[SKIP] Local ${LOCAL_OUT}/${RUN} exists"
  else
    python scripts/train_local.py \
      --data-dir "${DATA_DIR}" \
      --out-dir "${LOCAL_OUT}" \
      --run-name "${RUN}" \
      --rounds "${EPOCHS_CENTRAL}" \
      --batch-size "${BATCH_CENTRAL}" \
      --seed "${SEED}" \
      --num-workers "${NUM_WORKERS}" \
      --train-split train \
      --test-split test \
      --eval-threshold "${THRESHOLD}"
  fi

  LOCAL_RUNS+=("${LOCAL_OUT}/${RUN}")

  # FedAvg
  if [[ "${SKIP_EXISTING}" == "1" && -d "${FED_OUT}/${RUN}" ]]; then
    echo "[SKIP] FedAvg ${FED_OUT}/${RUN} exists"
  else
    python federated/server.py \
      --data-dir "${DATA_DIR}" \
      --out-dir "${FED_OUT}" \
      --run-name "${RUN}" \
      --rounds "${ROUNDS_FED}" \
      --batch-size "${BATCH_FED}" \
      --seed "${SEED}" \
      --num-workers "${NUM_WORKERS}" \
      --train-split train \
      --test-split test \
      --test-every-round \
      --eval-threshold "${THRESHOLD}" \
      --model-selection best \
      --selection-metric auroc \
      --save-test-pred-npz "${FED_OUT}/${RUN}/test_predictions.npz" \
      $( [[ "${PER_CLIENT_EVERY_ROUND}" == "1" ]] && echo "--per-client-every-round" ) \
      $( [[ "${LOG_CLIENT_SIM}" == "1" ]] && echo "--log-client-sim" )
  fi

  FED_RUNS+=("${FED_OUT}/${RUN}")

  # pFedBayes
  CKPT_OVERRIDE="${BACKBONE_CKPT}"
  if [[ "${CKPT_OVERRIDE}" == "none" ]]; then
    CKPT_OVERRIDE=""
  fi
  if [[ -z "${CKPT_OVERRIDE}" ]]; then
    CKPT_OVERRIDE="${FED_OUT}/${RUN}/checkpoints/model_best.pt"
  fi

  if [[ "${SKIP_EXISTING}" == "1" && -d "${PFB_OUT}/${RUN}" ]]; then
    echo "[SKIP] pFedBayes ${PFB_OUT}/${RUN} exists"
  else
    TMP_CFG_DIR="${TMP_CFG_DIR:-/workspace/tmp}"
    mkdir -p "${TMP_CFG_DIR}"
    TMP_CFG="$(mktemp "${TMP_CFG_DIR}/pfedbayes_cfg_${SEED}_XXXX.yaml")"
    python - <<'PY' "${PFB_CFG}" "${TMP_CFG}" "${PFB_OUT}" "${RUN}" "${SEED}" "${BATCH_PFB}" "${CKPT_OVERRIDE}" "${SAVE_TEST_PRED}" "${THRESHOLD}"
import sys
from pathlib import Path
try:
    import yaml
except Exception as e:
    raise SystemExit("PyYAML required to edit pfedbayes config") from e
src, dst, out_dir, run_name, seed, batch, ckpt_override, save_pred, fixed_thr = sys.argv[1:]
cfg = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
cfg.setdefault("run", {})
cfg["run"]["out_dir"] = out_dir
cfg["run"]["run_name"] = run_name
cfg["seed"] = int(seed)
cfg.setdefault("data", {})
cfg.setdefault("train", {})
cfg["train"]["batch_size"] = int(batch)
if ckpt_override:
    cfg.setdefault("backbone", {})
    cfg["backbone"]["checkpoint"] = ckpt_override
cfg.setdefault("eval", {})
cfg.setdefault("eval", {}).setdefault("threshold", {})
cfg["eval"]["threshold"]["fixed"] = float(fixed_thr)
cfg["eval"]["selection_use_post"] = False
if str(save_pred) == "1":
    cfg["eval"]["save_test_pred_npz"] = True
Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(f"[INFO] wrote pFedBayes cfg: {dst}")
PY

    python bayes_federated/pfedbayes_server.py \
      --config "${TMP_CFG}" \
      --run-name "${RUN}" \
      $( [[ "${LOG_CLIENT_SIM}" == "1" ]] && echo "--log-client-sim" )
  fi

PFB_RUNS+=("${PFB_OUT}/${RUN}")

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

# Client-level comparison (single seed)
fed_join=$(IFS=','; echo "${FED_RUNS[*]}")
pfb_join=$(IFS=','; echo "${PFB_RUNS[*]}")

python scripts/eval_compare_clients.py \
  --data-dir "${DATA_DIR}" \
  --fedavg-runs "${fed_join}" \
  --bfl-runs "${pfb_join}" \
  --seeds "${SEED}" \
  --mc-eval "${MC_EVAL}" \
  --out-json outputs/pfedbayes_vs_fedavg_client_comparison.json \
  --out-csv outputs/pfedbayes_vs_fedavg_client_comparison.csv

# Reliability (seed0)
python scripts/compare_significance.py \
  --data-dir "${DATA_DIR}" \
  --a-kind ioh --a-run-dir "${CENTRAL_OUT}/seed0" --a-label Central \
  --b-kind ioh --b-run-dir "${FED_OUT}/seed0" --b-label FedAvg \
  --variant pre \
  --out runs/compare/central_vs_fedavg_seed0_pre.json

python scripts/compare_significance.py \
  --data-dir "${DATA_DIR}" \
  --a-kind ioh --a-run-dir "${FED_OUT}/seed0" \
  --b-kind bfl --b-run-dir "${PFB_OUT}/seed0" \
  --variant pre \
  --out runs/compare/pfedbayes_vs_fedavg_seed0_pre.json

# Paper tables/fig (seed0)
python scripts/make_paper_tables_fig3.py \
  --central-run "${CENTRAL_OUT}/seed0" \
  --fedavg-run "${FED_OUT}/seed0" \
  --bfl-run "${PFB_OUT}/seed0" \
  --compare-json runs/compare/pfedbayes_vs_fedavg_seed0_pre.json \
  --out-dir outputs

# Round x client tables (Table 2-6 format; modes A/B)
if [[ -n "${TABLE_CFG}" && -f "${TABLE_CFG}" ]]; then
  python scripts/aggregate_round_metrics.py \
    --config "${TABLE_CFG}" \
    --out-dir outputs/fedocw_tables \
    --metrics "auroc,auprc,accuracy,ece,brier,nll"
fi

python scripts/make_round_client_boxplots.py \
  --runs "${FED_OUT}/seed${SEED},${PFB_OUT}/seed${SEED}" \
  --labels "FedAvg,pFedBayes" \
  --out-dir outputs

echo "[DONE] pFedBayes pipeline completed. Outputs are in outputs/ and runs/compare/."
