# リポジトリ監査レポート（現行仕様）

更新日: 2026-02-04
対象: コード/設定/README の現行内容に基づく仕様整理（実験結果の報告ではない）

## 0. 結論（要点）
- クライアント内の分割は train/val/test=70/10/20 を採用している。
- しきい値は基本は固定 0.5。Local は既定で youden-val（val が無い場合は 0.5）を用い、Centralized評価・FedAvg・FedUAB・pFedBayes は youden / youden-val に切り替え可能。
- 損失は Centralized/Local/FedAvg が BCEWithLogits、FedUAB/pFedBayes は bce を既定にし pos_weight を自動計算できる。
- round×client ログは Local とベイズ系（FedUAB/pFedBayes）は既定で出力し、Centralized/FedAvg はオプションで出力する。
- 論文表（mean (min–max)）の集計は `scripts/aggregate_round_metrics.py` が Mode A/B に対応する。

根拠:
- `scripts/build_dataset.py:17,59-66`（分割）
- `centralized/eval.py:55-164`（Centralized評価のしきい値）
- `scripts/train_local.py:108-367`（Localのしきい値）
- `federated/server.py:118-123,430-536`（FedAvg しきい値と round×client ログ）
- `bayes_federated/pfedbayes_server.py:270-290,600-649`（pFedBayes しきい値/ログ）
- `bayes_federated/feduab_server.py:812-938`（FedUAB しきい値/ログ）
- `centralized/train.py:240-427`、`scripts/train_local.py:268-339`、`federated/client.py:28-78`、`configs/pfedbayes.yaml:54-56`、`configs/feduab.yaml:24-26`（損失とpos_weight）
- `scripts/aggregate_round_metrics.py:22-139`（Mode A/B）

## 1. パイプライン入口
- データダウンロード: `scripts/data_download.py`
- データセット生成: `scripts/build_dataset.py`
- Centralized 学習: `centralized/train.py`
- Centralized 評価: `centralized/eval.py`
- Local 学習/評価: `scripts/train_local.py`
- FedAvg 学習/評価: `federated/server.py`
- FedUAB 学習/評価: `bayes_federated/feduab_server.py`（設定: `configs/feduab.yaml`）
- pFedBayes 学習/評価: `bayes_federated/pfedbayes_server.py`（設定: `configs/pfedbayes.yaml`、評価補助: `bayes_federated/eval.py`）

根拠:
- `README.md:92-306`（DL〜FedUAB/pFedBayesの入口）
- 実装ファイル: `centralized/train.py` / `centralized/eval.py` / `scripts/train_local.py` / `federated/server.py` / `bayes_federated/feduab_server.py` / `bayes_federated/pfedbayes_server.py`

## 2. データ仕様と分割
- 入力は ABP/ECG/PPG/ETCO2 と臨床情報を想定し、100Hz の波形を用いる。
- lead time は 5 分、イベント定義は MAP<=65 が 60秒以上継続した最初の秒、非低血圧区間から負例を抽出する。
- artifact 除外は MAP の外れ値と心拍周期の生理範囲で行う。
- クライアント内の分割は train/val/test=70/10/20 で行う。
- `scripts/build_dataset.py` は `federated_data/` と `federated_data/summary.json` を作成し、既存 `--out-dir` は削除して作り直す。

根拠:
- `README.md:7-15,68-135`（タスク定義・分割・出力）
- `scripts/build_dataset.py:5-17,59-87`（イベント定義・artifact・分割）
- `README.md:132-135`（出力の作成と再生成の注意）
- `scripts/build_dataset.py:41-49`（圧縮出力の既定）

## 3. 学習・評価仕様
- 損失は Centralized/Local/FedAvg が BCEWithLogits、FedUAB/pFedBayes は `loss_type=bce` を既定とし pos_weight を自動計算できる。
- しきい値は基本は固定 0.5。Local は既定で `youden-val`（val が無い場合は 0.5）を使い、Centralized評価・FedAvg・FedUAB・pFedBayes は `youden` / `youden-val` に切り替え可能。
- モデル選択は Centralized/FedAvg/pFedBayes で `last` / `best` を選べる。pFedBayes 設定の既定は `best`。
- ベイズ系評価は温度パラメータを受け取り `metrics_post` を出せるが、pFedBayes のサーバ実装では温度を渡していない。

根拠:
- `centralized/train.py:240-375`、`scripts/train_local.py:268-279`、`federated/client.py:28-78`（損失）
- `configs/pfedbayes.yaml:54-76`、`configs/feduab.yaml:24-57`（ベイズ系既定）
- `centralized/eval.py:70-164`、`scripts/train_local.py:108-367`、`federated/server.py:118-577`、`bayes_federated/pfedbayes_server.py:270-326,728-785`、`bayes_federated/feduab_server.py:812-909`（しきい値）
- `centralized/train.py:136-375`、`federated/server.py:121-475`、`configs/pfedbayes.yaml:58-66`、`bayes_federated/pfedbayes_server.py:280-302`（モデル選択）
- `bayes_federated/eval.py:37-211`、`bayes_federated/pfedbayes_server.py:309-325`（温度とmetrics_post）

## 4. 主要成果物
### 4.1 Centralized
- `run_config.json`
- `history.csv`
- `checkpoints/model_last.pt`（常時）と `checkpoints/model_best.pt`（`--model-selection best` の場合）
- `round_XXX_test.json`（`--save-round-json` の場合）
- `round_XXX_test_per_client.csv` と `round_client_metrics.csv/json`（`--per-client-every-epoch` の場合）
- `eval_<split>.json` / `eval_<split>_per_group.csv`（`centralized/eval.py`）

根拠:
- `centralized/train.py:246-447`、`centralized/eval.py:139-207`

### 4.2 Local
- `run_config.json`
- `round_XXX_test_per_client.csv`
- `round_client_metrics.csv/json`
- `test_report_per_client.json/csv`

根拠:
- `scripts/train_local.py:248-365`

### 4.3 FedAvg
- `meta.json`
- `history.csv`
- `metrics_round.csv`
- `round_XXX_test.json`（`--save-round-json` の場合）
- `round_XXX_test_per_client.csv` と `round_client_metrics.csv/json`（`--per-client-every-round` の場合）
- `test_report.json`
- `test_report_per_client.json/csv`
- `test_predictions.npz`（`--save-test-pred-npz` 指定時）

根拠:
- `federated/server.py:240-699`

### 4.4 FedUAB
- `meta.json`
- `history.csv`
- `metrics_round.csv`
- `round_XXX_test_per_client.csv` と `round_client_metrics.csv/json`（`per_client_every_round` の場合）
- `eval_modes.json` / `eval_modes_overall.csv` / `eval_modes_per_client.csv` / `eval_modes_client_stats.csv`
- `checkpoints/global_last.pt` と `checkpoints/clients/client_<id>.pt`
- `test_report.json` / `test_report_per_client.json`（global_idless の簡易互換）

根拠:
- `bayes_federated/feduab_server.py:800-1028`
- `common/eval_summary.py:83-139`

### 4.5 pFedBayes
- `summary.json`
- `history.csv`
- `round_XXX_test.json`（`test_every_round` の場合）
- `round_XXX_test_per_client.csv` と `round_client_metrics.csv/json`（`per_client_every_round` の場合）
- `test_report.json`
- `test_report_per_client.json/csv`
- `checkpoints/model_last.pt` と `checkpoints/model_best.pt`（best ありの場合）
- `test_predictions.npz`（`save_test_pred_npz` の場合）

根拠:
- `bayes_federated/pfedbayes_server.py:600-845,666-788`

## 5. Table 2〜6 集計仕様（mean (min–max)）
- Mode A: round ごとの「client 平均」→ round 平均の mean/min/max
- Mode B: round×client をフラット化 → mean/min/max
- 出力: `round_summary_all_modes.csv/json` と `table_<scenario>_modeA.csv` / `table_<scenario>_modeB.csv`

根拠:
- `scripts/aggregate_round_metrics.py:22-145`

## 6. 要確認（記述不一致）
- 入力窓長について、README とモデルコメントは 60 秒、`build_dataset` と `common/ioh_dataset.py` は 30 秒になっている。
- README は「閾値固定・温度スケーリングなし」と書かれているが、実装は youden / youden-val に対応し、pFedBayes/FedUAB の設定既定は youden-val。
- README の Centralized 出力記述（`val_report.json`、`model_best.pt` 常時）と実装が一致していない。
- README の FedAvg `history.csv` に val_* を記録という記述は実装と一致していない。
- README の pFedBayes 出力（`round_XXX_val_pre.json` など）と実装が一致していない。

根拠:
- `README.md:7-15,39-71,178-182,211-214,287-294`
- `scripts/build_dataset.py:64-66`
- `common/ioh_dataset.py:14-22`
- `common/ioh_model.py:54-58`
- `centralized/train.py:246-447`
- `federated/server.py:430-699`
- `bayes_federated/pfedbayes_server.py:600-845`

## 7. 参考文献（README準拠）
- Lee et al. (2022). VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients. Scientific Data.
- Shim et al. (2025). Machine Learning Methods for the Prediction of Intraoperative Hypotension with Biosignal Waveforms. Medicina.
- Choe et al. (2021). STEP-OP of Five-Minute Intraoperative Hypotension Using Hybrid Deep Learning. JMIR Med Inform.
- McMahan et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS.
- Zhang et al. (2022). Personalized Federated Learning via Variational Bayesian Inference. ICML.
- Hinton (2002). Training Products of Experts by Minimizing Contrastive Divergence. Neural Computation.
- Cho et al. (2014). Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. EMNLP.

根拠:
- `README.md:27-63`
