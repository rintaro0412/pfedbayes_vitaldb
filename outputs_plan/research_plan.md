# 研究計画書
タイトル「ベイズ連合学習を用いた医療時系列分類と不確実性評価」
VitalDB を用いた IOH（術中低血圧）予測パイプライン
中央学習・ローカル学習・FedAvg・FedUAB の比較

作成日: 2026-02-04
対象: /workspace のコードと設定

## 0. 前提
本書は report.md を参考にしたが、記述はコードと設定を優先して整理した。設定値は `configs/feduab.yaml` を基準に統一し、実行は一括処理スクリプトがあるが、手順上は分けて実行したものとして記述する。

根拠:
- `report.md`
- `configs/feduab.yaml`
- `scripts/run_all_feduab.sh`

## 1. 研究の背景、研究の意義、研究の構成

### 1.1 研究の背景
術中低血圧（IOH）は術中の合併症リスクと関連がある。VitalDB の高頻度波形と臨床情報を用いることで、IOH の事前予測が可能になる。本リポジトリは、VitalDB の波形と臨床情報を用いて IOH を予測し、中央学習と複数の連合学習手法を比較する。

根拠:
- `scripts/data_download.py`
- `scripts/build_dataset.py`
- `common/ioh_model.py`
- `README.md`

### 1.2 研究の意義
1. 連合学習の効果検証: 非IIDなクライアント分割下で、FedAvg と FedUAB の性能差を検証する。
2. 実運用を想定した評価: クライアントごとにデータ分布が異なる設定で、性能のばらつきと安定性を評価する。
3. ベイズ的手法の有用性検証: 不確実性を扱う FedUAB が、特に非IID条件で有利かを評価する。

根拠:
- `scripts/build_dataset.py`
- `federated/server.py`
- `bayes_federated/feduab_server.py`

### 1.3 研究の構成
1. データ取得と前処理
2. クライアント分割（非IID）
3. 学習と評価（中央、ローカル、FedAvg、FedUAB）
4. 結果整理と考察

根拠:
- `scripts/data_download.py`
- `scripts/build_dataset.py`
- `centralized/train.py`
- `scripts/train_local.py`
- `federated/server.py`
- `bayes_federated/feduab_server.py`

## 2. 実験設計

### 2.1 データダウンロード
- VitalDB API から症例波形を取得する。
- 対象は年齢 18 歳以上、全身麻酔（General）、移植と心臓外科を除外する。
- 取得トラックは MAP（ART_MBP）と 4 波形（ABP, ECG, PPG, ETCO2）で、100Hz で保存する。
- 必須トラックが欠ける症例は除外する。

根拠:
- `scripts/data_download.py`

### 2.2 前処理とラベル定義
- MAP は 1Hz 平均に変換する。
- 低血圧イベントは MAP <= 65 が 60 秒以上続く区間の開始時刻とする。
- 入力窓は 30 秒、予測リードタイムは 5 分とする。
- 正例はイベント 5 分前の 30 秒波形。
- 負例は MAP > 65 が 20 分以上続く区間から 1 もしくは 2 個抽出し、正例数に近づける。
- アーティファクト除外は以下を適用する。
1. MAP が 20 未満、または 200 超の区間は除外。
2. ABP のピーク間隔（心拍周期）が 0.3〜2.0 秒の範囲外となる割合が 10% を超える区間は除外。
- 波形は症例内でインスタンス正規化（平均 0、分散 1）する。
- 出力は `federated_data/<client>/<split>/case_*.npz` に保存し、`x_wave`（波形）、`x_clin`（臨床情報）、`y`、`t_event` を格納する。

根拠:
- `scripts/build_dataset.py`

### 2.3 クライアント分割（非IID性）
- クライアントは `opname` と `optype` を用いて決定し、部署（department）と結合してクライアントIDを作る。
- `opname` の症例数が閾値未満の場合は `optype` にフォールバックする。
- 症例数が少ないクライアントは部署内で `OtherSurgery` にまとめる（dept_pool）。
- クライアント内で train/val/test = 70/10/20 に分割し、正例数を考慮して層化する。
- 分割後に、各 split の正例イベント数が 10 未満のクライアントは除外する。
- これにより部署や術式に依存した分布差が生まれ、非IID性を確保する。

根拠:
- `scripts/build_dataset.py`

### 2.4 処理フローチャート

```
[ VitalDB API ]
        |
        v
[ scripts/data_download.py ]
        |
        v
[ vitaldb_data/case_*.csv.gz ]
        |
        v
[ scripts/build_dataset.py ]
        |
        v
[ federated_data/<client>/train|val|test/*.npz ]
        |
        +--> [ centralized/train.py ] -> [ centralized/eval.py ]
        |
        +--> [ scripts/train_local.py ]
        |
        +--> [ federated/server.py (FedAvg) ]
        |
        +--> [ bayes_federated/feduab_server.py (FedUAB) ]
        |
        v
[ runs/* と各種レポート出力 ]
```

根拠:
- `scripts/data_download.py`
- `scripts/build_dataset.py`
- `centralized/train.py`
- `centralized/eval.py`
- `scripts/train_local.py`
- `federated/server.py`
- `bayes_federated/feduab_server.py`

## 3. 実装

### 3.1 共通モデル（IOHNet）
- 1D CNN（3ブロック）+ GRU（任意）を用いる。
- 入力チャネルは ABP/ECG/PPG/ETCO2 の 4 波形。
- 臨床特徴量がある場合は全結合層で結合する。

根拠:
- `common/ioh_model.py`

### 3.2 Centralized
- すべてのクライアントの train データを結合して学習する。
- 損失は重み付き BCEWithLogits。正例の希少性を補正するため、`pos_weight = n_neg / n_pos` を用いる。

数式:

$$
L_{BCE} = -\{w y \log(\sigma(z)) + (1-y)\log(1-\sigma(z))\}
$$

- 評価は `centralized/eval.py` を用い、`youden-val`（val 由来のしきい値）または固定しきい値 0.5 で評価する。

根拠:
- `centralized/train.py`
- `centralized/eval.py`

### 3.3 Local
- クライアントごとに独立にモデルを学習し、集約しない。
- 損失はクライアントごとの重み付き BCEWithLogits。
- しきい値は既定で `youden-val`（val が無い場合は 0.5）。

根拠:
- `scripts/train_local.py`

### 3.4 FedAvg
- 各ラウンドで全クライアントを学習に参加させる。
- クライアントごとの更新をサンプル数で重み付け平均する。

数式:

$$
W^{(t+1)} = \sum_{k=1}^K \frac{n_k}{\sum_j n_j} W_k^{(t+1)}
$$

- 評価は `youden-val` を基本とし、val がない場合は固定しきい値 0.5 にフォールバックする。

根拠:
- `federated/server.py`
- `federated/client.py`

### 3.5 FedUAB
- パラメータをガウス分布で表現し、クライアントで変分推論により学習する。
- ローカル学習は ELBO 損失を用いる。

数式:

$$
L_{ELBO} = \mathrm{NLL}(y, f(x; w)) + \beta\, \mathrm{KL}(q(w)\Vert p(w))
$$

- 推論は MC サンプル平均で確率を推定する。

$$
\hat{p}(y=1|x) \approx \frac{1}{S} \sum_{s=1}^S \sigma(f(x; w_s))
$$

- 集約はガウス積（product of Gaussians）を用いる。クライアント重みはサンプル数で正規化する。

数式:

$$
\alpha_k = \frac{n_k}{\sum_j n_j},\quad \tau_k = 1/\sigma_k^2
$$
$$
\tau_{global} = \sum_k \alpha_k \tau_k,\quad \mu_{global} = \frac{\sum_k \alpha_k \tau_k \mu_k}{\tau_{global}}
$$

- 評価は global / personalized / ensemble を出力可能。

根拠:
- `bayes_federated/feduab_client.py`
- `bayes_federated/agg.py`
- `bayes_federated/feduab_server.py`

## 4. 実験設定（configs/feduab.yaml を基準）

### 4.1 共通データ設定
- `data_dir`: `federated_data`
- split 名: `train`, `val`, `test`

根拠:
- `configs/feduab.yaml`

### 4.2 共通学習設定
- rounds: 100
- local_epochs: 1
- batch_size: 256
- lr: 0.001
- weight_decay: 0.0001
- seed: 42
- num_workers: 32
- min_client_examples: 10
- eval_threshold: 0.5
- threshold_method: youden-val

根拠:
- `configs/feduab.yaml`

### 4.3 共通モデル設定
- base_channels: 32
- dropout: 0.1
- use_gru: true
- gru_hidden: 64

根拠:
- `configs/feduab.yaml`
- `common/ioh_model.py`

### 4.4 Centralized の設定
- epochs: 100（rounds に合わせる）
- batch_size: 256
- lr: 0.001
- weight_decay: 0.0001
- 評価: `centralized/eval.py` で `threshold_method=youden-val`

根拠:
- `centralized/train.py`
- `centralized/eval.py`
- `configs/feduab.yaml`

### 4.5 Local の設定
- rounds: 100
- batch_size: 256
- lr: 0.001
- weight_decay: 0.0001
- 評価: 固定しきい値 0.5

根拠:
- `scripts/train_local.py`
- `configs/feduab.yaml`

### 4.6 FedAvg の設定
- rounds: 100
- local_epochs: 1
- batch_size: 256
- lr: 0.001
- weight_decay: 0.0001
- 評価: `threshold_method=youden-val`、fallback は 0.5

根拠:
- `federated/server.py`
- `configs/feduab.yaml`

### 4.7 FedUAB の設定
- mc_train: 5
- mc_samples: 25
- kl_coeff: 0.0001
- full_bayes: true
- prior_sigma: 0.1
- logvar_min: -12.0
- logvar_max: 6.0
- agg.mode: feduab
- agg.beta_mode: normalized
- agg.beta_value: 1.0
- agg.precision_clamp: 1e-6
- eval_personalized/global/ensemble: true

根拠:
- `configs/feduab.yaml`
- `bayes_federated/feduab_server.py`
- `bayes_federated/agg.py`

### 4.8 実行手順（分けて実行）
1. データ取得
`python scripts/data_download.py`
2. データセット生成（`federated_data/` を削除して作り直すので注意）
`python scripts/build_dataset.py --out-dir federated_data`
3. Centralized
`python centralized/train.py --data-dir federated_data --out-dir runs/centralized --run-name seed42 --epochs 100 --batch-size 256 --lr 0.001 --weight-decay 0.0001 --seed 42 --num-workers 32 --train-split train --val-split val --test-split test --eval-threshold 0.5 --test-every-epoch --model-selection best --selection-source val --selection-metric auroc`
`python centralized/eval.py --data-dir federated_data --run-dir runs/centralized/seed42 --split test --batch-size 256 --num-workers 32 --threshold 0.5 --threshold-method youden-val --val-split val --per-client`
4. Local
`python scripts/train_local.py --data-dir federated_data --out-dir runs/local --run-name seed42 --rounds 100 --batch-size 256 --lr 0.001 --weight-decay 0.0001 --seed 42 --num-workers 32 --train-split train --test-split test --eval-threshold 0.5`
5. FedAvg
`python federated/server.py --data-dir federated_data --rounds 100 --local-epochs 1 --batch-size 256 --lr 0.001 --weight-decay 0.0001 --seed 42 --num-workers 32 --train-split train --test-split test --test-every-round --eval-threshold 0.5 --threshold-method youden-val --val-split val --model-selection best --selection-source val --selection-metric auroc --run-name seed42 --per-client-every-round`
6. FedUAB
`python bayes_federated/feduab_server.py --data-dir federated_data --rounds 100 --local-epochs 1 --batch-size 256 --lr 0.001 --weight-decay 0.0001 --seed 42 --num-workers 32 --train-split train --test-split test --val-split val --mc-samples 25 --mc-train 5 --kl-coeff 1e-4 --eval-threshold 0.5 --threshold-method youden-val --run-name seed42 --test-every-round --per-client-every-round`

根拠:
- `scripts/build_dataset.py`
- `centralized/train.py`
- `centralized/eval.py`
- `scripts/train_local.py`
- `federated/server.py`
- `bayes_federated/feduab_server.py`
- `scripts/run_all_feduab.sh`

## 5. 実験結果
現時点では本書に反映できる結果は用意できていない。結果が揃い次第、以下の表に追記する。

### 5.1 結果表（追記予定）

| 方法 | AUROC | AUPRC | ECE | NLL | Brier | Accuracy | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Centralized | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Local (macro) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| FedAvg | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| FedUAB (global) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| FedUAB (personalized) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| FedUAB (ensemble) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 5.2 期待される結果の傾向（推測です）
- Centralized は全体性能（AUROC/AUPRC）が最も高い可能性がある（推測です）。
- Local はクライアント間のばらつきが大きく、macro 平均は低下しやすい（推測です）。
- FedAvg は Local より安定するが、非IID性の影響で性能が頭打ちになる可能性がある（推測です）。
- FedUAB は FedAvg よりも較正（ECE, NLL）と少数クライアント性能で有利になる可能性がある（推測です）。

根拠:
- 実装上の学習・評価設定は `configs/feduab.yaml` と各トレーナの実装に依存するため、性能は実測で確定する。

## 6. 考察
1. 非IID性の影響で、中央学習と連合学習の差が明確になる可能性がある。特に症例構成が偏るクライアントでは性能低下が起こりやすい。
2. FedAvg は重み平均のみで分布差を吸収するため、クライアントごとの不確実性や分布の違いを扱いにくい。
3. FedUAB は分布パラメータの集約を行うため、少数データのクライアントに対してより安定した推定が期待されるが、MC サンプル数や KL 係数の影響で計算コストが増える。
4. しきい値選択（youden-val）により、評価値が固定しきい値 0.5 より改善する可能性がある。
5. データ生成時の窓長、負例抽出、クライアント削除条件が結果に大きく影響するため、再現性のために `summary.json` と設定の保存が重要である。

根拠:
- `scripts/build_dataset.py`
- `federated/server.py`
- `bayes_federated/feduab_server.py`
- `configs/feduab.yaml`

## 7. 主要根拠一覧
- データ取得: `scripts/data_download.py`
- データ生成と分割: `scripts/build_dataset.py`
- 中央学習: `centralized/train.py`
- 中央評価: `centralized/eval.py`
- ローカル学習: `scripts/train_local.py`
- FedAvg: `federated/server.py`, `federated/client.py`
- FedUAB: `bayes_federated/feduab_server.py`, `bayes_federated/feduab_client.py`, `bayes_federated/agg.py`
- 共通モデル: `common/ioh_model.py`
- 設定: `configs/feduab.yaml`
