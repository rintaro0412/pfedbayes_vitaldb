# ICAISE スクリプトガイド

コマンドはリポジトリのルートから実行する。公開するファイルの一覧は `configs/icaise2026/publication_files.json` にある。開発用の作業ツリーには、公開対象外のスクリプトも残っている。

| 段階 | 実行スクリプト | 入力・出力 |
|---|---|---|
| データ準備 | `data_download.py`, `build_dataset.py`, `pack_contiguous_dataset.py` | VitalDBのダウンロード、入力窓の抽出、必要に応じた連続形式への変換 |
| 分割監査 | `audit_patient_split_leakage.py` | 臨床情報とファイル一覧を読み取り専用で確認し、分割間の重複件数を集計 |
| 学習 | `run_experiments.py` | 実験マニフェストに従い、Centralized、Local、6つの連合学習手法、pFedBayesを実行 |
| Local比較用モデル | `train_local.py` | クライアント別の学習と最終テスト評価レポート |
| 保存済み結果の集計 | `make_paper_ja_missing_assets.py --tables-only` | 保存済み実行設定とクライアント別CSVから、使用モデルを区別した設定表・クライアント平均の成績表を作成 |
| 共通モデルの論文用表 | `make_paper_tables_fig3.py --require-explicit-runs` | Centralized、FedAvg、pFedBayesの実験を明示し、全テスト窓をまとめた指標と確率予測の較正を出力 |
| 有意差の検証 | `compare_significance.py`, `compare_pfedbayes_vs_methods.py` | 明示した実験・チェックポイントを比較。全量推論やブートストラップには大きな計算コストがかかる場合あり |
| 不確実性 | `make_risk_coverage.py`, `make_bayesian_evaluation.py` | 保存済み予測から、coverage（予測を保持する割合）、較正、不確実性を診断 |
| データの説明 | `make_table1_client_summary.py`, `report_heterogeneity.py`, `make_client_split_flow_figure.py` | クライアントごとの件数・分布の要約と分割手順 |
| 学習曲線 | `make_icaise_training_curves.py` | 明示した手法一覧と保存済み学習履歴から作成 |
| 計算資源の計測 | `benchmark_icaise_inference.py`, `benchmark_icaise_training_resources.py` | 保存済みチェックポイントによる推論、または別途確認して実行する計測付き学習 |
| 任意の探索的データ解析（EDA） | `make_eda_figures.py`, `make_eda_supplement.py` | データセットの要約と抽出した信号の解析。全量集計・ブートストラップは別の作業として実施 |
| 公開用ファイルの書き出し | `prepare_public_release.py` | 公開対象リストのファイルだけを空の独立したディレクトリへ書き出し、ハッシュ値を記録 |

`run_all_pfedbayes.sh` はマニフェストに従う実験実行スクリプトを呼び出し、既定では実行計画の表示だけを行う。明示的に指定した場合の旧実行方式は、互換性のため残している。その乱数種・パスの規則は公開版の実験手順とは異なる。

学習の実装は、引き続き `centralized/train.py`、`federated/server.py` / `client.py`、`bayes_federated/pfedbayes_server.py` / `pfedbayes_client.py` にある。評価は `centralized/eval.py`、既存の学習スクリプト、`bayes_federated/eval.py` に実装している。共通のデータ読み込み、評価指標、実験記録は `common/` にある。

生成した同じ手法一覧を、表の作成、学習曲線の描画、推論の計測に渡せる。各スクリプトが求める入力形式が異なるため、`curve_kind` と `inference_kind` は別の項目である。論文用の成果物を作成するときは、自動検出に頼らず実験出力のパスを明示する。

コマンド、既知の制約、共通モデルのクライアント平均（client-macro）評価と個人化した事後分布による評価の違いは、[実験手順の詳細](../docs/icaise2026.md)を参照する。
