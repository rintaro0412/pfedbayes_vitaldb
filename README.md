# IOH (Intraoperative Hypotension) Prediction Pipeline (VitalDB)

本リポジトリには、VitalDB（`vitaldb` Pythonパッケージ）を用いた「麻酔中低血圧（IOH）予測」タスクの**再現可能**なデータ前処理＋学習パイプライン（中央学習 baseline と FedAvg 比較）を最小構成で実装しています。

## タスク定義（固定仕様）

- 入力: 60秒波形（100Hz）`ABP/ECG/PPG/ETCO2` + 臨床情報 → `shape=(B, 4, 6000)` (+ `x_clin`)
- 予測: `lead time=5分`（300秒）後のイベント発生（二値分類）
- 正例（イベント開始 `t0`）: 1秒平均MAP（100Hzの1秒平均）で `MAP<=65` が連続 `60秒` 以上成立した最初の秒
- 陰性: 1秒平均MAPで `MAP>65` が連続 `20分` 以上の区間（non-hypotensive segment）から、各区間につき 1 または 2 ウィンドウを抽出（総数が正例に近づくよう調整）
- artifact除外（`scripts/build_dataset.py`）:
  - `MAP < 20` または `MAP > 200` を含む入力ウィンドウを除外
  - ABPピーク間隔（心拍周期）が生理範囲（デフォルト 0.3–2.0 秒）外の拍が一定割合を超える入力ウィンドウを除外
- 主指標: AUPRC、補助: AUROC、校正: ECE/Brier/NLL
- 閾値は **固定（デフォルト0.5）**。温度スケーリングは使わない
- 損失関数: **BCEWithLogits**（必要に応じて pos_weight を使用）
  - **Centralized**: train 全体で `pos_weight = n_neg / n_pos`
  - **Local**: クライアントごとに `pos_weight = n_neg / n_pos`
  - **FedAvg**: クライアントごとに `pos_weight = n_neg / n_pos`（各クライアントの local 学習で使用）
  - **pFedBayes**: `loss.type = bce` の場合、`loss.pos_weight: auto` で **クライアント別に `n_neg / n_pos` を自動計算**。固定値にしたい場合は `loss.pos_weight` に数値を指定
- 連合学習の**クライアント重み付け**
  - **FedAvg**: サーバ集約は `n_examples` による加重平均（各クライアントの学習サンプル数）
- **pFedBayes**: `pfedbayes.weight_mode` に依存（`n_examples` or `uniform`）。**論文基準は `uniform`**（本リポジトリの既定も `uniform`）

## 参考論文と設計背景（処理別）

### 1) データソース（VitalDB）
- 参考論文: Lee et al., *VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients*（Scientific Data, 2022）
- 背景: VitalDB の公開データ（波形＋臨床情報）を使用し、再現性の高い公開データ基盤を採用。

### 2) IOHイベント定義／抽出・サンプリング
- 参考論文:  
  - Shim et al., *Machine Learning Methods for the Prediction of Intraoperative Hypotension with Biosignal Waveforms*（Medicina, 2025）  
  - Choe et al., *STEP-OP of Five-Minute Intraoperative Hypotension Using Hybrid Deep Learning*（JMIR Med Inform, 2021）  
- 背景:  
  - 5分先予測（lead time=5min）は Shim 論文に準拠。  
  - MAP<=65 の持続によるイベント定義は STEP-OP の定義を参考。  
  - 非低血圧区間からの negative 抽出は Shim 論文の方針を参考。  
  - 窓長（本実装は60秒）は、再現性と比較可能性を優先して固定。

### 3) 入力（波形＋臨床情報）
- 参考論文: Shim et al., Medicina 2025
- 背景: 4波形（ABP/ECG/PPG/ETCO2）と臨床情報の併用が前提。

### 4) モデル（CNN + GRU）
- 参考論文:  
  - Shim et al., Medicina 2025（hybrid CNN-RNN）  
  - Cho et al., *Learning Phrase Representations using RNN Encoder–Decoder*（EMNLP, 2014）  
  - 参考背景: STEP-OP（JMIR Med Inform, 2021）  
- 背景:  
  - Shim 論文の CNN-RNN 形を踏襲し、RNN を GRU に固定。  
  - GRU は Cho et al. で導入されたゲート付きRNNユニットに基づく。

### 5) フェデレーテッド学習（FedAvg）
- 参考論文: McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*（AISTATS, 2017）
- 背景: 標準的な FedAvg を比較基準として採用。

### 6) ベイズ化／パーソナライズ（pFedBayes）
- 参考論文:  
  - Zhang et al., *Personalized Federated Learning via Variational Bayesian Inference*（ICML, 2022）  
  - Hinton, *Training Products of Experts by Minimizing Contrastive Divergence*（Neural Computation, 2002）  
- 背景:  
  - pFedBayes の変分ベイズ枠組みを再現。  

### 7) 閾値（固定）
- 背景: 比較の公平性と実装の単純化を優先し、全手法で **固定閾値** を使用

### 8) クライアント内 80/20 分割（train/test）
- 参考論文: 直接の指定はなし（再現性と比較可能性のための実装判断）
- 背景: client 内で train/test = 80/20 の固定分割（val なし）。

## 使うトラック（固定）

- `Solar8000/ART_MBP`（MAP）
- `SNUADC/PLETH`（PPG）
- `SNUADC/ECG_II`（ECG）
- `SNUADC/ART`（ABP; 必須）
- `Primus/CO2`（ETCO2; downloader はこのトラックを保存）
- `Solar8000/HR`（品質用）
- `Solar8000/PLETH_SPO2`（品質用）

## 前提（Python環境）

必要に応じて venv を作成し、依存関係を入れてください（CPU/GPU で分離）。

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-venv.txt
```

## 実行手順（DL → データセット生成 → 中央学習 → 評価 → FedAvg）

### 1) ダウンロード

`vitaldb_data/case_{caseid}.csv.gz` を作ります。`clinical_data.csv` も保存します。

```bash
python scripts/data_download.py \
  --save-dir vitaldb_data \
  --clinical-save-path clinical_data.csv \
  --run-log download_run.json
```

任意のケース数に絞りたい場合:

```bash
python scripts/data_download.py --max-cases 100 --shuffle --seed 42
```

### 2) 学習/評価用データセット生成（build_dataset形式）

- 出力: `federated_data/<client_id>/<split>/case_*.npz`（split は train/test）
- 入力: `clinical_data.csv` と `vitaldb_data/`（`scripts/data_download.py` の出力）
- split は症例単位で分割し、正例イベント数の比率ができるだけ保たれるように調整します。

```bash
# 診療科内で術式優先→不足は手術カテゴリにフォールバックし、
# 小規模は診療科ごとのプールに統合する（デフォルト）
python scripts/build_dataset.py \
  --clinical-csv clinical_data.csv \
  --wave-dir vitaldb_data \
  --client-scheme opname_optype \
  --merge-strategy dept_pool \
  --opname-threshold 100 \
  --min-client-cases 100

# 診療科内の最小(>=min_client_cases)クライアントに吸収する場合
#（診療科内に吸収先が無ければ診療科別プールを残す）
# python scripts/build_dataset.py --merge-strategy dept_min

# 結合しない場合（min_client_cases 未満は除外）
# python scripts/build_dataset.py --merge-strategy none
```

`federated_data/summary.json` に、抽出ウィンドウ数などの集計メタを出力します。
`summary.json` には `client_scheme/merge_strategy/opname_threshold/min_client_cases/min_client_pos` と
最終クライアント数も含まれます。client_id は `Department__ClientRaw` 形式です。
さらに `splits_detail` / `clients_detail` に、split 別・クライアント別の症例数と正例イベント数、正例/負例ウィンドウ数を記録します。
`--out-dir` が既に存在する場合、build_dataset は中身を削除して作り直します。
`min_client_cases` 未満のクライアントは除外されます（client_scheme によらず適用）。
分割後、train/test の各 split で正例イベント数が `min_client_pos` 未満のクライアントも除外されます（0 で無効化）。

心拍周期の生理範囲（デフォルト 0.3–2.0 秒）を変更したい場合:

```bash
python scripts/build_dataset.py --cycle-min-sec 0.3 --cycle-max-sec 2.0
```

症例ごとのZ-score正規化（instance norm）を無効化したい場合:

```bash
python scripts/build_dataset.py --no-instance-norm
```

ETCO2 欠損症例も含めたい場合（欠損はゼロ埋め）:

```bash
python scripts/build_dataset.py --no-require-etco2
```

出力 `.npz` はデフォルトで圧縮されます（容量が非常に大きくなるのを防ぐため）。無圧縮にしたい場合:

```bash
BUILD_DATASET_COMPRESS=0 python scripts/build_dataset.py
```

すでに作成済みの `.npz` を後から圧縮したい場合（in-place）:

```bash
python scripts/recompress_npz.py --data-dir federated_data
```

### 3) 中央学習（baseline）

```bash
python centralized/train.py \
  --data-dir federated_data \
  --out-dir runs/centralized \
  --epochs 30 \
  --batch-size 64 \
  --eval-threshold 0.5
```

学習後、`runs/centralized/<run_name>/` に以下が保存されます（最終エポックのモデルを使用）:

- `checkpoints/model_best.pt`（`model_selection=best` の場合）
- `history.csv`（train_loss と test_*）
- `round_XXX_test.json`（`--save-round-json` が有効な場合）

### 4) test評価（固定閾値）

```bash
python centralized/eval.py \
  --data-dir federated_data \
  --run-dir runs/centralized/<run_name> \
  --split test \
  --threshold 0.5
```

### 5) FedAvg（比較用）

build_dataset 出力のクライアント分割（`federated_data/<client_id>/...`）をそのまま使います（依存を最小化するためFlowerは未使用の最小実装）。

```bash
python federated/server.py \
  --data-dir federated_data \
  --out-dir runs/fedavg \
  --rounds 5 \
  --local-epochs 1 \
  --batch-size 64 \
  --eval-threshold 0.5
```

進捗表示:
- ラウンド内のクライアント処理は progress bar を表示します（無効化: `--no-progress-bar`）
- 各クライアントのバッチ進捗も表示したい場合: `--client-progress-bar`
- `history.csv` には各ラウンドの `train_loss` と `test_*` を記録します

出力は `runs/fedavg/<run_name>/` に保存され、**最終ラウンドのモデル**で `test_report.json` と
クライアント別の test 指標（`test_report_per_client.json / test_report_per_client.csv`）を生成します。

中断復帰（同じ run を継続）:

```bash
python federated/server.py \
  --data-dir federated_data \
  --run-dir runs/fedavg/<run_name> \
  --rounds 100 \
  --resume
```

`--resume` は `checkpoints/model_last.pt` と `history.csv` を読み、次のラウンドから再開します。

### 5.5) FedUAB（ベイズ的重み分布の連合学習）

FedUAB は **重みが分布（平均・分散）** を持つ BNN をクライアントで学習し、サーバでガウス積（product of Gaussians）
として集約します。デフォルトは **全層分布化（論文設定）** で、軽量版（head のみ）は `--no-full-bayes` で切り替え可能です。
初期分散は層ごとに `1/H` で縮小し（`--var-reduction-h`, 既定 `1`）、分散パラメータは `rho`（softplus）を既定で使用します。
集約の係数は論文 Eq.(8) に合わせて **`n_k/n` の正規化** を必須とします（`client_weight_mode=samples` と `agg_beta_mode=normalized`）。

```bash
python bayes_federated/feduab_server.py \
  --data-dir federated_data \
  --out-dir runs/feduab \
  --rounds 50 \
  --local-epochs 1 \
  --batch-size 64 \
  --mc-samples 20 \
  --kl-coeff 1e-4 \
  --var-reduction-h 2

# YAML config 例（CLI が上書き）
python bayes_federated/feduab_server.py --config configs/feduab.yaml
```

**評価モード（同一の評価関数）**:
- `personalized_oracle`: 最終ラウンドの各クライアント事後で、そのクライアント test を評価（主結果）
- `global_idless`: サーバ側グローバル事後のみで全 test を評価（補助）
- `ensemble_idless`: 13 クライアント事後の平均予測で全 test を評価（補助, 計算コスト高）

**クライアント選択（論文アルゴリズム対応）**:
- `--clients-per-round` で各ラウンドの参加クライアント数を指定（0=全員）
- `--client-fraction` で割合指定（`clients-per-round=0` のとき有効）

出力は `runs/feduab/<run_name>/` に保存され、以下を生成します:
- `eval_modes.json / eval_modes_overall.csv / eval_modes_per_client.csv / eval_modes_client_stats.csv`
- `checkpoints/global_last.pt` と `checkpoints/clients/client_<id>.pt`
- NLL は MC で得た確率平均から計算（log(mean p)）で統一
- `eval_modes.json` には uncertainty の統計（`aleatoric/epistemic/total_var`）も含まれます

中断復帰（同じ run を継続）:

```bash
python bayes_federated/feduab_server.py \
  --run-dir runs/feduab/<run_name> \
  --rounds 100 \
  --resume
```

`--resume` は `checkpoints/global_last.pt` と `history.csv` を読み、次のラウンドから再開します。

`scripts/run_all_feduab.sh` を使う場合は、`runs/fedavg/<run_name>/per_client/<client_id>/` と
`runs/feduab/<run_name>/per_client/<client_id>/` に、クライアント別の
`test_report_per_client.csv` / `eval_modes_per_client.csv` / `round_client_metrics.csv`
（存在するもののみ）を分割して保存します。既存の集計ファイルは残します。
同スクリプト内で中央学習とローカル学習、FedAvg と FedUAB は並列で実行されます。

### 6) pFedBayes（Personalized VI）

各クライアントで **posterior q_i** と **localized global w_i** を交互更新し、サーバで
`v^{t+1} = (1-β) v^t + β * mean(v_w,i)` を実行します（完全版アルゴリズム）。

1. `configs/pfedbayes.yaml` の `backbone.checkpoint` を点推定モデルの checkpoint に合わせてください。

2. サーバ起動（round管理・評価・**最終ラウンドモデル**でtest評価まで実行）:

```bash
python bayes_federated/pfedbayes_server.py --config configs/pfedbayes.yaml
```

3. 中断復帰:

```bash
python bayes_federated/pfedbayes_server.py --config configs/pfedbayes.yaml --resume
```

4. test評価のみ再実行（best checkpoint を使う）:

```bash
python bayes_federated/eval.py \
  --data-dir federated_data \
  --checkpoint runs/pfedbayes/<run_name>/checkpoints/model_best.pt \
  --split test \
  --mc-eval 50
```

pFedBayes の出力は `runs/pfedbayes/<run_name>/` 以下に保存されます:
- `summary.json`
- `history.csv`
- `round_XXX_test.json`
- `round_XXX_clients.json`
- `clients/round_XXX/client_<id>.pt`（posterior / localized_global）
- `test_report.json`
- `checkpoints/model_best.pt`

### 6.5) FedAvg / FedUAB を同一条件・複数seedで実行

```bash
python scripts/train_federated.py \
  --algo fedavg,feduab \
  --seeds 0,1,2 \
  --data-dir federated_data \
  --rounds 50 \
  --local-epochs 1 \
  --batch-size 64
```

- 出力: `runs/<algo>/seed<seed>/<timestamp>/...`
- 集計: `runs/summary.csv`（mode×metric の平均/分散）

### 6.6) FedUAB パラメータ・スイープ（簡易グリッド）

```bash
# 例: KL係数のみをスイープ
python scripts/sweep_feduab.py \
  --config configs/feduab.yaml \
  --grid "train.kl_coeff=1e-5,1e-4,1e-3" \
  --seeds 0,1 \
  --mode personalized_oracle \
  --stat overall \
  --metric auprc
```

### 7) 有意性評価（paired bootstrap）

FedAvg と pFedBayes の test 指標差（例: AUPRC / ECE）について、caseid 単位の paired bootstrap で差の95%CIと p 値を出します。

```bash
python scripts/compare_significance.py \
  --data-dir federated_data \
  --split test \
  --a-kind ioh --a-run-dir runs/fedavg/<run_name> \
  --b-kind bfl --b-run-dir runs/pfedbayes/<run_name> \
  --variant pre \
  --bootstrap-n 2000 \
  --out runs/compare/pfedbayes_vs_fedavg_test_pre.json
```

### 8) クライアント単位マクロ平均での比較（min_client_cases 未満は除外）

同一分割・同一閾値でクライアント別指標を計算し、case数が `min_client_cases` 以上の「適格クライアント」だけでマクロ平均（各クライアント同一重み）した指標を **seed0のみ** で比較します。`min_client_cases` 未満のクライアントは一次解析から除外し、参考値として別出力します。

```bash
python scripts/eval_compare_clients.py \
  --data-dir federated_data \
  --fedavg-runs runs/fedavg/run_a \
  --bfl-runs runs/pfedbayes/run_a \
  --seeds 0 \
  --mc-eval 50
```

出力:
- `compare_clients.json`: seed0 の温度、クライアント別指標、適格クライアントのマクロ平均と差分、AUPRC差の sign-flip p値など。
- `compare_clients.csv`: method 行でマクロ平均指標を表形式で保存。

## 監査可能性（ログ/メタ）

- `scripts/data_download.py` は `vitaldb_data/` と `clinical_data.csv` と `download_run.json` を出力
- `scripts/build_dataset.py` は `federated_data/` と `federated_data/summary.json` を出力
- 学習は `run_config.json`（seed・split・pos_weight・git hash等）を保存

## 論文表（Table 2〜6）用の集計

ラウンド×クライアントのログから、論文の `mean (min–max)` 形式に整形するスクリプトを用意しています。

```bash
python scripts/aggregate_round_metrics.py \
  --config configs/fedocw_tables.json \
  --out-dir outputs/fedocw_tables \
  --metrics auprc,auroc,brier,nll,ece,accuracy,f1
```

出力:
- `outputs/fedocw_tables/table_A_modeA.csv`
- `outputs/fedocw_tables/table_A_modeB.csv`

## 参考文献

### 表記ルール
- 形式: 「著者（先頭＋et al.）(年). 英語題名. 会議/誌名. DOI(あれば)」
- 和文併記: 英語題名の直後に `【和訳】...` を付ける

1. Lee H-C, et al. (2022). *VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients*. Scientific Data. DOI: 10.1038/s41597-022-01411-5  
   【和訳】手術患者の高精度な多項目バイタルデータベース VitalDB
2. Shim J, et al. (2025). *Machine Learning Methods for the Prediction of Intraoperative Hypotension with Biosignal Waveforms*. Medicina. DOI: 10.3390/medicina61112039  
   【和訳】生体信号波形を用いた術中低血圧予測の機械学習手法
3. McMahan B, et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS.  
   【和訳】分散データからの通信効率的な深層学習
4. Guo C, et al. (2017). *On Calibration of Modern Neural Networks*. ICML.  
   【和訳】現代のニューラルネットの校正
5. Cho K, et al. (2014). *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation*. EMNLP. DOI: 10.3115/v1/D14-1179  
   【和訳】RNN Encoder–Decoder による句表現学習
6. Zhang Q, et al. (2022). *Personalized Federated Learning via Variational Bayesian Inference*. ICML.  
   【和訳】変分ベイズ推論によるパーソナライズドFederated Learning
7. Hinton GE. (2002). *Training Products of Experts by Minimizing Contrastive Divergence*. Neural Computation.  
   【和訳】コントラストダイバージェンス最小化によるPoE学習
8. Choe S, et al. (2021). *Short-Term Event Prediction in the Operating Room (STEP-OP) of Five-Minute Intraoperative Hypotension Using Hybrid Deep Learning*. JMIR Medical Informatics. DOI: 10.2196/31311  
   【和訳】ハイブリッド深層学習による術中低血圧（5分先）の短期予測

## TODO（最小構成のため）

- ECG/PPG欠損時の「ゼロ埋め＋欠損フラグ」をモデル入力に追加（現状は前処理で欠損が多い秒が落ちるため、実質的に多欠損症例が除外されやすい）
- より厳密な術中/導入期の抽出（現在は `--trim-start-sec/--trim-end-sec` による秒指定）
- より高度なSQI（PPG/ECG）と、将来的な特徴量追加
