# 監査結果（論文図表データの存在確認）

## 結論
- (1) A: 予測確率保存 … **NO**
- (2) B: ラウンド推移 … **NO**
- (3) 箱ひげ用のクライアント別スコア … **NO**
- (4) 非IID説明用のクライアント要約 … **YES**

## 参考論文と処理対応（背景つき）

### データソース
- 参考論文: Lee et al., Scientific Data 2022（VitalDB）
- 背景: 本リポジトリは VitalDB を前提にデータ取得・前処理を構築。

### IOHイベント定義／抽出・サンプリング
- 参考論文:  
  - Shim et al., Medicina 2025（IOH予測）  
  - Choe et al., JMIR Med Inform 2021（STEP-OP）  
- 背景:  
  - 5分先予測は Shim 論文に準拠。  
  - MAP<=65 の持続によるイベント定義は STEP-OP の定義を参考。  
  - negative 抽出方針は Shim 論文の記述を参照。  
  - 30秒窓は実装判断（再現性・比較可能性のため固定）。

### 入力（ETCO2＋臨床情報）
- 参考論文: Shim et al., Medicina 2025
- 背景:
  - ETCO2 はカプノグラフィ波形として、ABP/ECG/PPG と同列の 4 波形入力に含める。
  - 4 波形のいずれかが欠損する症例は除外。
  - 全波形は 100 Hz にリサンプルして使用（本実装も 100 Hz 前提）。
  - 低血圧イベント 5 分前の 30 秒波形を入力に使用。
- 実装対応:
  - `scripts/build_dataset.py` で ETCO2 波形の有無（全 NaN 含む）を検査し、欠損症例を除外（`--require-etco2` デフォルト有効）。

### モデル（CNN + GRU）
- 参考論文:  
  - Shim et al., Medicina 2025（CNN-RNN）  
  - Cho et al., EMNLP 2014（GRUの基礎）  
  - 参考背景: STEP-OP, JMIR Med Inform 2021  
- 背景: CNN-RNN構成を維持し、RNN は GRU に固定。

### FedAvg（フェデレーテッド基準）
- 参考論文: McMahan et al., AISTATS 2017
- 背景: 最も基本的な FedAvg を比較基準として採用。

### pFedBayes / BFL（ベイズ化）
- 参考論文:  
  - Zhang et al., ICML 2022（pFedBayes）  
  - Hinton, Neural Computation 2002（PoE概念）  
- 背景: pFedBayes の変分ベイズ枠組みを再現し、BFLでは PoE 的集約を利用。

### 校正・閾値
- 参考論文:  
  - Guo et al., ICML 2017（Temperature scaling）  
  - Youden, Cancer 1950（Youden 指標）  
- 背景: val（なければtrain）で温度・閾値を決めて test 固定。

### クライアント内 80/20 分割（valなし）
- 参考論文: 指定なし（実装判断）
- 背景: 比較可能性を優先し、client内で train/test=80/20 を固定。

## 根拠
### (1) A: 予測確率（サンプル単位）
- `runs/centralized_batch/seed0/eval_test_per_group.csv`
  - rows: 502
  - cols: `caseid`, `n`, `n_pos`, `prob_pre_mean`, `prob_pre_max`, `prob_post_mean`, `prob_post_max`
- `runs/centralized_batch/seed0/eval_test.json`
  - keys: `started_utc`, `run_dir`, `checkpoint`, `data_dir`, `split`, `n`, `n_pos`, `n_neg`, `metrics_pre`, `temperature`, `metrics_post`, `threshold`, `confusion_pre`, `confusion_post`, `finished_utc`
- `runs/fedavg_batch/seed0/test_report.json`
  - keys: `n`, `metrics_pre`, `metrics_post`, `threshold`
- `runs/pfedbayes/seed0/test_report.json`
  - keys: `n`, `n_pos`, `n_neg`, `metrics_pre`, `uncertainty`, `metrics_post`, `confusion_pre`, `confusion_post`, `bootstrap`
  - `uncertainty` keys: `prob_var_mean`, `prob_var_std`, `entropy_mean`, `entropy_std`
- `runs/outputs/results/logs/artifacts` 配下で `test_predictions.npz` / `*pred*.npz` / `*pred*.csv` を検索したが **該当ファイルなし**

### (2) B: ラウンド推移ログ
- `runs/bfl_batch/seed1/round_001_val_pre.json`
  - keys: `n`, `n_pos`, `n_neg`, `metrics_pre`, `uncertainty`, `threshold_selected`, `confusion_pre`
- `runs/bfl_batch/seed1/round_001_val_post.json`
  - keys: `n`, `n_pos`, `n_neg`, `metrics_pre`, `uncertainty`, `metrics_post`, `threshold_selected`, `confusion_pre`, `confusion_post`
- `runs/pfedbayes/seed0/round_001_val_pre.json`
  - keys: `n`, `n_pos`, `n_neg`, `metrics_pre`, `uncertainty`, `threshold_selected`, `confusion_pre`
- `runs/fedavg_batch/seed0/round_001_clients.json`
  - top keys: `clients`
  - clients_len: 12
  - first_client_keys: `client_id`, `status`, `n_examples`, `avg_loss`, `pos_weight`
- `runs/centralized_batch/seed0/val_report.json`
  - keys: `n`, `temperature`, `threshold`, `metrics_pre`, `metrics_post`

### (3) クライアント別スコア（箱ひげ）
- `runs/fedavg_batch/seed0/test_report_per_client.json`
  - top keys: `temperature`, `threshold`, `clients`
  - clients: dict (n_clients: 12)
  - sample client keys: `client_id`, `status`, `n`, `temperature`, `threshold`, `metrics_pre`, `metrics_post`, `confusion_pre`, `confusion_post`
- `runs/bfl_batch/seed1/test_report_per_client.json`
  - top keys: `temperature`, `threshold`, `clients`
  - clients: dict (n_clients: 12)
  - sample client keys: `n`, `n_pos`, `n_neg`, `metrics_pre`, `uncertainty`, `metrics_post`, `confusion_pre`, `confusion_post`, `client_id`, `status`
- `runs/` 配下で `*per_client*.json` を検索した結果、**pFedBayes / Centralized 用の同等ファイルは見つからず**

### (4) 非IID（opname_optype）説明用のクライアント要約
- `tmp_noniid_report.json`
  - top keys: `started_utc`, `data_dir`, `splits`, `clients`, `per_client_per_split`, `per_client_pooled`, `case_pos_rate`, `case_n_windows`, `clinical_mean`, `label_chi2_test`, `finished_utc`
  - `per_client_per_split`: dict (n_clients: 12)
    - sample client key: `General_surgery__Biliary_Pancreas`
    - split keys: `train`, `val`, `test`
    - per-split keys: `n_files`, `n_windows`, `n_pos`, `n_neg`
  - `per_client_pooled`: dict (n_clients: 12)
    - per-client keys: `n_files`, `n_windows`, `n_pos`, `n_neg`, `pos_rate`
  - `clinical_mean`: dict (n_keys: 12)
    - sample value keys: `n_cases`, `mean`

## 不足
### (1) A: 予測確率保存 … NO
- 欠けているキー/列:
  - **y_true**, **prob_mean**, **prob_var / entropy（サンプル単位）**, **case_id**
- 欠けている方式:
  - **Central / FedAvg / pFedBayes** すべてで `test_predictions.npz` もしくは同等のサンプル予測ファイルが未確認

### (2) B: ラウンド推移 … NO
- 欠けているキー/列:
  - **round ごとの評価指標（loss/AUROC/AUPRC 等）を含む val/test ログ** が **FedAvg/Central** に存在しない
- 欠けている方式:
  - **FedAvg / Central** に `round_*_val_pre.json`・`round_*_val_post.json` などの評価ログが未確認（`round_*_clients.json` はあるが評価指標なし）

### (3) 箱ひげ用のクライアント別スコア … NO
- 欠けているキー/列:
  - **client_id 付きの test metric** が **pFedBayes / Central** で未確認
- 欠けている方式:
- **pFedBayes / Central** に `test_report_per_client.json` 相当のファイルが未確認

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
