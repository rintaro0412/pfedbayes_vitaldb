# IOH (Intraoperative Hypotension) Prediction Pipeline (VitalDB)

本リポジトリには、VitalDB（`vitaldb` Pythonパッケージ）を用いた「麻酔中低血圧（IOH）予測」タスクの**再現可能**なデータ前処理＋学習パイプライン（中央学習 baseline と FedAvg 比較）を最小構成で実装しています。

## タスク定義（固定仕様）

- 入力: 30秒波形（100Hz）`ABP/ECG/PPG/ETCO2` + 臨床情報 → `shape=(B, 4, 3000)` (+ `x_clin`)
- 予測: `lead time=5分`（300秒）後のイベント発生（二値分類）
- 正例（イベント開始 `t0`）: 1秒平均MAP（100Hzの1秒平均）で `MAP<=65` が連続 `60秒` 以上成立した最初の秒
- 陰性: 1秒平均MAPで `MAP>65` が連続 `20分` 以上の区間（non-hypotensive segment）から、各区間につき 1 または 2 ウィンドウを抽出（総数が正例に近づくよう調整）
- artifact除外（`scripts/build_dataset.py`）:
  - `MAP < 20` または `MAP > 200` を含む入力ウィンドウを除外
  - ABPピーク間隔（心拍周期）が生理範囲（デフォルト 0.3–2.0 秒）外の拍が一定割合を超える入力ウィンドウを除外
- 主指標: AUPRC、補助: AUROC、校正: ECE/Brier/NLL
- 閾値・温度（temperature scaling）は **valで決めて** test 固定

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
  - 窓長（本実装は30秒）は、再現性と比較可能性を優先して固定。

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

### 7) 校正（Temperature Scaling）と閾値（Youden）
- 参考論文:  
  - Guo et al., *On Calibration of Modern Neural Networks*（ICML, 2017）  
  - Youden, *Index for rating diagnostic tests*（Cancer, 1950）  
- 背景:  
- 温度スケーリングで確率校正（valで学習）  
- Youden 指標で閾値を選択（raw 確率, val）

### 8) クライアント内 70/10/20 分割（train/val/test）
- 参考論文: 直接の指定はなし（再現性と比較可能性のための実装判断）
- 背景: FL 比較を容易にするため、client 内で train/val/test = 70/10/20 の固定分割。

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

- 出力: `federated_data/<client_id>/<split>/case_*.npz`
- 入力: `clinical_data.csv` と `vitaldb_data/`（`scripts/data_download.py` の出力）

```bash
# 診療科内で術式優先→不足は手術カテゴリにフォールバックし、
# 小規模は診療科ごとのプールに統合する（デフォルト）
python scripts/build_dataset.py \
  --clinical-csv clinical_data.csv \
  --wave-dir vitaldb_data \
  --client-scheme opname_optype \
  --merge-strategy dept_pool \
  --opname-threshold 150 \
  --min-client-cases 150

# 診療科内の最小(>=min_client_cases)クライアントに吸収する場合
#（診療科内に吸収先が無ければ診療科別プールを残す）
# python scripts/build_dataset.py --merge-strategy dept_min
```

`federated_data/summary.json` に、抽出ウィンドウ数などの集計メタを出力します。
`summary.json` には `client_scheme/merge_strategy/opname_threshold/min_client_cases` と
最終クライアント数も含まれます。client_id は `Department__ClientRaw` 形式です。
`--out-dir` が既に存在する場合、build_dataset は中身を削除して作り直します。

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
  --batch-size 64
```

学習後、`runs/centralized/<run_name>/` に以下が保存されます:

- `checkpoints/model_best.pt`
- `temperature.json`（valで学習した温度）
- `threshold.json`（valで決めた閾値）
- `val_report.json`（校正前/後）
- `history.csv`

### 4) test評価（固定温度＋固定閾値）

```bash
python centralized/eval.py \
  --data-dir federated_data \
  --run-dir runs/centralized/<run_name> \
  --split test
```

### 5) FedAvg（比較用）

build_dataset 出力のクライアント分割（`federated_data/<client_id>/...`）をそのまま使います（依存を最小化するためFlowerは未使用の最小実装）。

```bash
python federated/server.py \
  --data-dir federated_data \
  --out-dir runs/fedavg \
  --rounds 5 \
  --local-epochs 1 \
  --batch-size 64
```

進捗表示:
- ラウンド内のクライアント処理は progress bar を表示します（無効化: `--no-progress-bar`）
- 各クライアントのバッチ進捗も表示したい場合: `--client-progress-bar`
- `history.csv` には各ラウンドの `train_loss` と `val_*` を記録します

出力は `runs/fedavg/<run_name>/` に保存され、中央学習と同様に `temperature.json / threshold.json / test_report.json` を持ちます。
加えて、クライアント別の test 指標を `test_report_per_client.json / test_report_per_client.csv` に保存します。

### 6) pFedBayes（Personalized VI）

各クライアントで **posterior q_i** と **localized global w_i** を交互更新し、サーバで
`v^{t+1} = (1-β) v^t + β * mean(v_w,i)` を実行します（完全版アルゴリズム）。

1. `configs/pfedbayes.yaml` の `backbone.checkpoint` を点推定モデルの checkpoint に合わせてください。

2. サーバ起動（round管理・val評価・best選択・test評価まで実行）:

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
- `round_XXX_val_pre.json` / `round_XXX_val_post.json`
- `round_XXX_clients.json`
- `clients/round_XXX/client_<id>.pt`（posterior / localized_global）
- `test_report.json`
- `checkpoints/model_best.pt`

### 7) 有意性評価（paired bootstrap）

FedAvg と pFedBayes の test 指標差（例: AUPRC / ECE）について、caseid 単位の paired bootstrap で差の95%CIと p 値を出します。

```bash
python scripts/compare_significance.py \
  --data-dir federated_data \
  --split test \
  --a-kind ioh --a-run-dir runs/fedavg/<run_name> \
  --b-kind bfl --b-run-dir runs/pfedbayes/<run_name> \
  --variant post \
  --bootstrap-n 2000 \
  --out runs/compare/pfedbayes_vs_fedavg_test_post.json
```

### 8) クライアント単位マクロ平均での比較（min_client_cases 未満は除外）

同一分割・同一温度学習（valのみ）でクライアント別指標を計算し、case数が `min_client_cases` 以上の「適格クライアント」だけでマクロ平均（各クライアント同一重み）した指標を seed ごとに比較します。`min_client_cases` 未満のクライアントは一次解析から除外し、参考値として別出力します。

```bash
python scripts/eval_compare_clients.py \
  --data-dir federated_data \
  --fedavg-runs runs/fedavg/run_a,runs/fedavg/run_b \
  --bfl-runs runs/pfedbayes/run_a,runs/pfedbayes/run_b \
  --seeds 0,1 \
  --mc-eval 50
```

出力:
- `compare_clients.json`: seedごとの温度、クライアント別指標、適格クライアントのマクロ平均と差分、AUPRC差の sign-flip p値など。
- `compare_clients.csv`: seed×method 行でマクロ平均指標を表形式で保存。

## 監査可能性（ログ/メタ）

- `scripts/data_download.py` は `vitaldb_data/` と `clinical_data.csv` と `download_run.json` を出力
- `scripts/build_dataset.py` は `federated_data/` と `federated_data/summary.json` を出力
- 学習は `run_config.json`（seed・split・pos_weight・git hash等）を保存

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
5. Youden WJ. (1950). *Index for rating diagnostic tests*. Cancer. DOI: 10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3  
   【和訳】診断検査の評価指標（Youden指数）
6. Cho K, et al. (2014). *Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation*. EMNLP. DOI: 10.3115/v1/D14-1179  
   【和訳】RNN Encoder–Decoder による句表現学習
7. Zhang Q, et al. (2022). *Personalized Federated Learning via Variational Bayesian Inference*. ICML.  
   【和訳】変分ベイズ推論によるパーソナライズドFederated Learning
8. Hinton GE. (2002). *Training Products of Experts by Minimizing Contrastive Divergence*. Neural Computation.  
   【和訳】コントラストダイバージェンス最小化によるPoE学習
9. Choe S, et al. (2021). *Short-Term Event Prediction in the Operating Room (STEP-OP) of Five-Minute Intraoperative Hypotension Using Hybrid Deep Learning*. JMIR Medical Informatics. DOI: 10.2196/31311  
   【和訳】ハイブリッド深層学習による術中低血圧（5分先）の短期予測

## TODO（最小構成のため）

- ECG/PPG欠損時の「ゼロ埋め＋欠損フラグ」をモデル入力に追加（現状は前処理で欠損が多い秒が落ちるため、実質的に多欠損症例が除外されやすい）
- より厳密な術中/導入期の抽出（現在は `--trim-start-sec/--trim-end-sec` による秒指定）
- より高度なSQI（PPG/ECG）と、将来的な特徴量追加
