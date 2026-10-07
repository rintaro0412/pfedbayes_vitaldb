# VitalDBを用いた術中低酸素イベント予測 — ICAISE 2026

論文 *Evaluating Bayesian Federated Learning for Medical Time-Series Classification and Uncertainty Assessment* に対応するコード、保存済みの実験設定、公開対象の集計結果をまとめたリポジトリである。

比較対象は Centralized、Local、FedAvg、FedProx、SCAFFOLD、FedNova、Per-FedAvg、pFedMe、pFedBayes の9手法である。既存の学習スクリプトを使用する。公開する過去のベンチマークは探索的な結果である。症例単位のデータ分割には患者の重複があり、前処理には将来の情報に依存する処理が含まれる。また、保存済みデータの正例窓の抽出条件は、厳密に300秒先を予測するための抽出条件と異なる。成績の解釈にあたっては、[実験の詳細](docs/icaise2026.md)を確認する。

## 最初に読む資料

- [実験一覧](configs/icaise2026/experiments.yaml)：9手法、実際の乱数種、実行段階の指定。
- [保存済み設定の出典](configs/icaise2026/provenance.json)：実験名、記録されたコミット、設定ファイルのハッシュ値。
- [過去のデータセット定義](configs/icaise2026/dataset.json)：確認済みの抽出条件とデータ分割。
- [スクリプト案内](scripts/README.md)：データ準備、学習、評価、図表作成、計算資源の計測。
- [公開成果物の一覧](artifacts/icaise2026/manifest.json)：元の実験と結果の対応、計測状況。

## 再実行の準備

既存の依存パッケージである PyYAML を利用できる環境で、リポジトリのルートから次のコマンドを実行する。

```bash
python scripts/run_experiments.py --manifest configs/icaise2026/experiments.yaml --stage all --dry-run
```

この `--dry-run` は実行予定を表示するだけで、出力ファイルの作成や学習は行わない。実際の再実行には、データセットと適切な学習環境が必要である。生データと学習済みモデルのチェックポイントは同梱していない。既存の依存関係ファイルと各段階の実行方法は、[環境と実行手順](docs/icaise2026.md#environment)に記載している。公開準備に用いた環境では、過去の成績の数値再現は確認していない。

## 結果の読み方

`model_scope` は評価するモデルの種類、`aggregation` は成績の集計方法を示す。Global の client-macro 成績は、同じグローバルモデルを各クライアントで評価し、クライアントごとの成績を平均したものである。個人化した事後分布に基づく成績とは区別する。Local はクライアントごとに異なるモデルを使用する。pFedBayes の個人化モデルを単独で評価した結果は、別のレポートで扱う。

保存済みの学習設定では、過去に `centralized/seed0` と命名された実験も含め、全9手法の乱数種は42である。pFedBayes の学習時間2160分は、37ラウンドの実測から推定した値であり、200ラウンド全体の実測値ではない。患者対応表、生の信号、チェックポイント、投稿に関する連絡、他の論文の資料、新しい研究は公開対象に含めていない。

## 引用とライセンス

このリポジトリの引用情報は [CITATION.cff](CITATION.cff) に記載している。この公開版には論文の DOI をまだ登録していない。ライセンスは未選定で、新たなオープンソースライセンスは付与していない。
