# リポジトリ監査レポート（BFL = pFedBayes）

## 0. 更新要点
- val分割必須化（pFedBayes/eval_compare で val 不在時はエラー、bfl.yaml も val_split=val）
- 非IIDレポートを `outputs/noniid_report.json` に保存し、参照先を統一
- `data_inventory.json` を更新（tmp エントリ削除＋新規成果物追加）

val 必須化の設計背景: 温度スケーリングによる校正や Youden 等の閾値選択、round/epoch のモデル選択を評価から独立させ、test への情報漏れを防ぐために train/val/test を分離する。非IID環境ではクライアント間の分布差が大きく、共通の val を固定しないと手法比較の公平性が崩れるため、val 不在時は実行を停止する仕様に統一した。

## 1. 実験パイプラインの入口（行番号付き）
- Central 学習: `centralized/train.py:111` に CLI 定義、`centralized/train.py:464` で `main()` 実行。
- Central 評価: `centralized/eval.py:57` に CLI 定義、`centralized/eval.py:327` で `main()` 実行。
- FL（FedAvg）学習/評価: `federated/server.py:89` に CLI 定義、`federated/server.py:592` で `main()` 実行。
- BFL（pFedBayes）学習: `bayes_federated/pfedbayes_server.py:122` に CLI 定義、`bayes_federated/pfedbayes_server.py:791` で `main()` 実行。
- BFL 評価: `bayes_federated/eval.py:237` に CLI 定義、`bayes_federated/eval.py:291` で `main()` 実行。
- 論文用成果物生成: `scripts/make_paper_tables_fig3.py:259` に CLI 定義、`scripts/make_paper_tables_fig3.py:496` で `main()` 実行。
- クライアント比較: `scripts/eval_compare_clients.py:254` に CLI 定義、`scripts/eval_compare_clients.py:407` で `main()` 実行。
- 有意差（paired bootstrap）: `scripts/compare_significance.py:107` に CLI 定義、`scripts/compare_significance.py:313` で `main()` 実行。

## 2. 非IIDの定量サマリー（`outputs/noniid_report.json`）
- ラベル分布: pos_rate = 0.362–0.631（min: Thoracic_surgery__Lung_wedge_resection, max: General_surgery__Exploratory_laparotomy）
- 数量分布: n_windows = 746–2402（min: General_surgery__Others, max: General_surgery__Biliary_Pancreas）
- 共変量平均のばらつき例（client平均）: age 48.1–64.6, emop 0.011–0.486, sex_M 0.000–0.744
- 参考: label chi2 は `chi2=433.5, df=12`（p_value は NaN のため有意性主張は避ける）

## 3. 見つかった成果物一覧（data_inventory.json から自動生成）
- `federated_data/Urology__OtherSurgery/val/case_5033.npz`（npz）キー: `x_wave, x_clin, y, t_event, is_pos`。shape例: x_wave:(3, 3, 3000), x_clin:(3, 12), y:(3,), t_event:(3,), is_pos:(3,)。
- `federated_data/summary.json`（json）キー: `eligible_cases, clients, splits, pass1_total_pos_events, pass1_total_norm_segments, assigned_total_neg_windows_est, written_case_files, written_pos_windows, written_neg_windows, exclusion, missing, window_drop_estimate, client_scheme, merge_strategy, opname_threshold, min_client_cases, notes`。
- `outputs/noniid_report.json`（json）キー: `case_n_windows, case_pos_rate, clients, clinical_mean, data_dir, finished_utc, label_chi2_test, per_client_per_split, per_client_pooled, splits, started_utc`。
- `outputs/noniid_summary.csv`（csv）14行。列: `metric, min_value, max_value, min_client, max_client, notes`。
- `outputs/pfedbayes_vs_fedavg_client_comparison_seed0.csv`（csv）2行。列: `seed, method, auprc, auroc, ece, brier, nll`。
- `outputs/pfedbayes_vs_fedavg_client_comparison_seed0.json`（json）キー: `started_utc, data_dir, min_client_cases, notes, seeds, pairs, summary, finished_utc`。
- `outputs/table2_global.csv`（csv）3行。列: `method, AUROC, AUPRC, Accuracy, ECE, Brier, NLL, T, thr, n_test, pos_rate`。
- `outputs/table3_client.csv`（csv）12行。列: `client_id, n_fedavg, n_pos_fedavg, auprc_fedavg, ece_fedavg, n_bfl, n_pos_bfl, auprc_bfl, ece_bfl`。
- `runs/centralized_batch/seed0/eval_test.json`（json）キー: `started_utc, run_dir, checkpoint, data_dir, split, n, n_pos, n_neg, metrics_pre, temperature, metrics_post, threshold, confusion_pre, confusion_post, finished_utc`。
- `runs/centralized_batch/seed0/eval_test_per_group.csv`（csv）502行。列: `caseid, n, n_pos, prob_pre_mean, prob_pre_max, prob_post_mean, prob_post_max`。
- `runs/centralized_batch/seed0/history.csv`（csv）30行。列: `epoch, train_loss, val_auprc, val_auroc, val_brier, val_nll, val_ece`。
- `runs/centralized_batch/seed0/run_config.json`（json）キー: `started_utc, git_hash, device, data_dir, splits, n_files, counts, seed, hyper, model, finished_utc, artifacts`。
- `runs/centralized_batch/seed0/temperature.json`（json）キー: `temperature, nll_before, nll_after`。
- `runs/centralized_batch/seed0/threshold.json`（json）キー: `threshold, method`。
- `runs/centralized_batch/seed0/val_report.json`（json）キー: `n, temperature, threshold, metrics_pre, metrics_post`。
- `runs/centralized_batch/seed0_focal_match/history.csv`（csv）30行。列: `epoch, train_loss, val_auprc, val_auroc, val_brier, val_nll, val_ece`。
- `runs/centralized_batch/seed0_focal_match/run_config.json`（json）キー: `started_utc, git_hash, device, data_dir, splits, n_files, counts, seed, hyper, model, finished_utc, artifacts`。
- `runs/centralized_batch/seed0_focal_match/temperature.json`（json）キー: `temperature, nll_before, nll_after`。
- `runs/centralized_batch/seed0_focal_match/threshold.json`（json）キー: `threshold, method`。
- `runs/centralized_batch/seed0_focal_match/val_report.json`（json）キー: `n, temperature, threshold, metrics_pre, metrics_post`。
- `runs/centralized_batch/seed1/eval_test.json`（json）キー: `started_utc, run_dir, checkpoint, data_dir, split, n, n_pos, n_neg, metrics_pre, temperature, metrics_post, threshold, confusion_pre, confusion_post, finished_utc`。
- `runs/centralized_batch/seed1/eval_test_per_group.csv`（csv）502行。列: `caseid, n, n_pos, prob_pre_mean, prob_pre_max, prob_post_mean, prob_post_max`。
- `runs/centralized_batch/seed1/history.csv`（csv）30行。列: `epoch, train_loss, val_auprc, val_auroc, val_brier, val_nll, val_ece`。
- `runs/centralized_batch/seed1/run_config.json`（json）キー: `started_utc, git_hash, device, data_dir, splits, n_files, counts, seed, hyper, model, finished_utc, artifacts`。
- `runs/centralized_batch/seed1/temperature.json`（json）キー: `temperature, nll_before, nll_after`。
- `runs/centralized_batch/seed1/threshold.json`（json）キー: `threshold, method`。
- `runs/centralized_batch/seed1/val_report.json`（json）キー: `n, temperature, threshold, metrics_pre, metrics_post`。
- `runs/centralized_batch/seed2/eval_test.json`（json）キー: `started_utc, run_dir, checkpoint, data_dir, split, n, n_pos, n_neg, metrics_pre, temperature, metrics_post, threshold, confusion_pre, confusion_post, finished_utc`。
- `runs/centralized_batch/seed2/eval_test_per_group.csv`（csv）502行。列: `caseid, n, n_pos, prob_pre_mean, prob_pre_max, prob_post_mean, prob_post_max`。
- `runs/centralized_batch/seed2/history.csv`（csv）30行。列: `epoch, train_loss, val_auprc, val_auroc, val_brier, val_nll, val_ece`。
- `runs/centralized_batch/seed2/run_config.json`（json）キー: `started_utc, git_hash, device, data_dir, splits, n_files, counts, seed, hyper, model, finished_utc, artifacts`。
- `runs/centralized_batch/seed2/temperature.json`（json）キー: `temperature, nll_before, nll_after`。
- `runs/centralized_batch/seed2/threshold.json`（json）キー: `threshold, method`。
- `runs/centralized_batch/seed2/val_report.json`（json）キー: `n, temperature, threshold, metrics_pre, metrics_post`。
- `runs/compare/bfl_vs_fedavg_seed0_feduab_focal_post.json`（json）キー: `started_utc, data_dir, split, group, n, n_pos, n_neg, bootstrap, variant, method_a, method_b, metrics_a, metrics_b, comparison, reliability, finished_utc`。
- `runs/compare/bfl_vs_fedavg_test_post.json`（json）キー: `started_utc, data_dir, split, group, n, n_pos, n_neg, bootstrap, variant, method_a, method_b, metrics_a, metrics_b, comparison, reliability, finished_utc`。
- `runs/compare/central_vs_fedavg_test_post.json`（json）キー: `started_utc, data_dir, split, group, n, n_pos, n_neg, bootstrap, variant, method_a, method_b, metrics_a, metrics_b, comparison, reliability, finished_utc`。
- `runs/compare/fullbfl_vs_fedavg_seed0_post.json`（json）キー: `started_utc, data_dir, split, group, n, n_pos, n_neg, bootstrap, variant, method_a, method_b, metrics_a, metrics_b, comparison, reliability, finished_utc`。
- `runs/compare/pfedbayes_vs_fedavg_seed0_post.json`（json）キー: `started_utc, data_dir, split, group, n, n_pos, n_neg, bootstrap, variant, method_a, method_b, metrics_a, metrics_b, comparison, reliability, finished_utc`。
- `runs/fedavg_batch/seed0/dataset_summary.json`（json）キー: `eligible_cases, clients, splits, pass1_total_pos_events, pass1_total_norm_segments, assigned_total_neg_windows_est, written_case_files, written_pos_windows, written_neg_windows, exclusion, missing, window_drop_estimate, client_scheme, merge_strategy, opname_threshold, min_client_cases, notes`。
- `runs/fedavg_batch/seed0/history.csv`（csv）30行。列: `round, train_loss, val_auprc, val_auroc, val_brier, val_nll, val_ece, n_clients_used, sum_examples`。
- `runs/fedavg_batch/seed0/round_001_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_002_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_003_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_004_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_005_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_006_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_007_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_008_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_009_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_010_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_011_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_012_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_013_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_014_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_015_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_016_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_017_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_018_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_019_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_020_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_021_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_022_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_023_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_024_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_025_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_026_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_027_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_028_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_029_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/round_030_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0/run_config.json`（json）キー: `started_utc, git_hash, device, data_dir, splits, seed, rounds, local_cfg, model, clients, dataset_summary, finished_utc, artifacts`。
- `runs/fedavg_batch/seed0/temperature.json`（json）キー: `temperature, nll_before, nll_after`。
- `runs/fedavg_batch/seed0/test_report.json`（json）キー: `n, metrics_pre, metrics_post, threshold`。
- `runs/fedavg_batch/seed0/test_report_per_client.csv`（csv）12行。列: `client_id, status, n, n_pos, n_neg, auprc_pre, auroc_pre, brier_pre, nll_pre, ece_pre, auprc_post, auroc_post, brier_post, nll_post, ece_post, temperature, threshold`。
- `runs/fedavg_batch/seed0/test_report_per_client.json`（json）キー: `temperature, threshold, clients`。
- `runs/fedavg_batch/seed0/threshold.json`（json）キー: `threshold, method`。
- `runs/fedavg_batch/seed0/val_report.json`（json）キー: `n, temperature, threshold, metrics_pre, metrics_post`。
- `runs/fedavg_batch/seed0_focal_match/dataset_summary.json`（json）キー: `eligible_cases, clients, splits, pass1_total_pos_events, pass1_total_norm_segments, assigned_total_neg_windows_est, written_case_files, written_pos_windows, written_neg_windows, exclusion, missing, window_drop_estimate, client_scheme, merge_strategy, opname_threshold, min_client_cases, notes`。
- `runs/fedavg_batch/seed0_focal_match/history.csv`（csv）30行。列: `round, train_loss, val_auprc, val_auroc, val_brier, val_nll, val_ece, n_clients_used, sum_examples`。
- `runs/fedavg_batch/seed0_focal_match/round_001_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_002_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_003_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_004_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_005_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_006_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_007_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_008_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_009_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_010_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_011_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_012_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_013_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_014_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_015_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_016_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_017_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_018_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_019_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_020_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_021_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_022_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_023_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_024_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_025_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_026_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_027_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_028_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_029_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/round_030_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed0_focal_match/run_config.json`（json）キー: `started_utc, git_hash, device, data_dir, splits, seed, rounds, local_cfg, model, clients, dataset_summary, finished_utc, artifacts`。
- `runs/fedavg_batch/seed0_focal_match/temperature.json`（json）キー: `temperature, nll_before, nll_after`。
- `runs/fedavg_batch/seed0_focal_match/test_report.json`（json）キー: `n, metrics_pre, metrics_post, threshold`。
- `runs/fedavg_batch/seed0_focal_match/test_report_per_client.csv`（csv）12行。列: `client_id, status, n, n_pos, n_neg, auprc_pre, auroc_pre, brier_pre, nll_pre, ece_pre, auprc_post, auroc_post, brier_post, nll_post, ece_post, temperature, threshold`。
- `runs/fedavg_batch/seed0_focal_match/test_report_per_client.json`（json）キー: `temperature, threshold, clients`。
- `runs/fedavg_batch/seed0_focal_match/threshold.json`（json）キー: `threshold, method`。
- `runs/fedavg_batch/seed0_focal_match/val_report.json`（json）キー: `n, temperature, threshold, metrics_pre, metrics_post`。
- `runs/fedavg_batch/seed1/dataset_summary.json`（json）キー: `eligible_cases, clients, splits, pass1_total_pos_events, pass1_total_norm_segments, assigned_total_neg_windows_est, written_case_files, written_pos_windows, written_neg_windows, exclusion, missing, window_drop_estimate, client_scheme, merge_strategy, opname_threshold, min_client_cases, notes`。
- `runs/fedavg_batch/seed1/history.csv`（csv）30行。列: `round, train_loss, val_auprc, val_auroc, val_brier, val_nll, val_ece, n_clients_used, sum_examples`。
- `runs/fedavg_batch/seed1/round_001_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_002_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_003_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_004_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_005_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_006_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_007_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_008_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_009_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_010_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_011_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_012_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_013_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_014_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_015_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_016_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_017_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_018_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_019_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_020_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_021_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_022_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_023_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_024_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_025_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_026_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_027_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_028_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_029_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/round_030_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed1/run_config.json`（json）キー: `started_utc, git_hash, device, data_dir, splits, seed, rounds, local_cfg, model, clients, dataset_summary, finished_utc, artifacts`。
- `runs/fedavg_batch/seed1/temperature.json`（json）キー: `temperature, nll_before, nll_after`。
- `runs/fedavg_batch/seed1/test_report.json`（json）キー: `n, metrics_pre, metrics_post, threshold`。
- `runs/fedavg_batch/seed1/test_report_per_client.csv`（csv）12行。列: `client_id, status, n, n_pos, n_neg, auprc_pre, auroc_pre, brier_pre, nll_pre, ece_pre, auprc_post, auroc_post, brier_post, nll_post, ece_post, temperature, threshold`。
- `runs/fedavg_batch/seed1/test_report_per_client.json`（json）キー: `temperature, threshold, clients`。
- `runs/fedavg_batch/seed1/threshold.json`（json）キー: `threshold, method`。
- `runs/fedavg_batch/seed1/val_report.json`（json）キー: `n, temperature, threshold, metrics_pre, metrics_post`。
- `runs/fedavg_batch/seed2/dataset_summary.json`（json）キー: `eligible_cases, clients, splits, pass1_total_pos_events, pass1_total_norm_segments, assigned_total_neg_windows_est, written_case_files, written_pos_windows, written_neg_windows, exclusion, missing, window_drop_estimate, client_scheme, merge_strategy, opname_threshold, min_client_cases, notes`。
- `runs/fedavg_batch/seed2/history.csv`（csv）30行。列: `round, train_loss, val_auprc, val_auroc, val_brier, val_nll, val_ece, n_clients_used, sum_examples`。
- `runs/fedavg_batch/seed2/round_001_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_002_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_003_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_004_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_005_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_006_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_007_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_008_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_009_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_010_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_011_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_012_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_013_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_014_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_015_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_016_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_017_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_018_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_019_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_020_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_021_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_022_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_023_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_024_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_025_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_026_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_027_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_028_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_029_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/round_030_clients.json`（json）キー: `clients`。
- `runs/fedavg_batch/seed2/run_config.json`（json）キー: `started_utc, git_hash, device, data_dir, splits, seed, rounds, local_cfg, model, clients, dataset_summary, finished_utc, artifacts`。
- `runs/fedavg_batch/seed2/temperature.json`（json）キー: `temperature, nll_before, nll_after`。
- `runs/fedavg_batch/seed2/test_report.json`（json）キー: `n, metrics_pre, metrics_post, threshold`。
- `runs/fedavg_batch/seed2/test_report_per_client.csv`（csv）12行。列: `client_id, status, n, n_pos, n_neg, auprc_pre, auroc_pre, brier_pre, nll_pre, ece_pre, auprc_post, auroc_post, brier_post, nll_post, ece_post, temperature, threshold`。
- `runs/fedavg_batch/seed2/test_report_per_client.json`（json）キー: `temperature, threshold, clients`。
- `runs/fedavg_batch/seed2/threshold.json`（json）キー: `threshold, method`。
- `runs/fedavg_batch/seed2/val_report.json`（json）キー: `n, temperature, threshold, metrics_pre, metrics_post`。
- `runs/pfedbayes/seed0/client_counts.json`（json）キー: `General_surgery__Biliary_Pancreas, General_surgery__Colorectal, General_surgery__Distal_gastrectomy, General_surgery__Hepatic, General_surgery__OtherSurgery, General_surgery__Stomach, Gynecology__OtherSurgery, Thoracic_surgery__Lung_lobectomy, Thoracic_surgery__Lung_wedge_resection, Thoracic_surgery__Minor_resection, Thoracic_surgery__Others, Urology__OtherSurgery`。
- `runs/pfedbayes/seed0/clients.json`（json）
- `runs/pfedbayes/seed0/config.json`（json）キー: `run, seed, device, data, clients, backbone, bayes, train, pfedbayes, loss, eval`。
- `runs/pfedbayes/seed0/data_counts.json`（json）キー: `train, val, test`。
- `runs/pfedbayes/seed0/dataset_summary.json`（json）キー: `client_scheme, merge_strategy, opname_threshold, min_client_cases, clients`。
- `runs/pfedbayes/seed0/env.json`（json）キー: `git_hash, python, executable, time_utc`。
- `runs/pfedbayes/seed0/history.csv`（csv）30行。列: `round, train_loss, val_auprc, temperature, threshold, server_beta, zeta`。
- `runs/pfedbayes/seed0/round_001_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_001_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_001_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_002_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_002_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_002_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_003_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_003_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_003_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_004_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_004_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_004_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_005_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_005_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_005_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_006_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_006_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_006_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_007_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_007_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_007_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_008_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_008_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_008_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_009_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_009_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_009_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_010_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_010_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_010_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_011_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_011_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_011_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_012_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_012_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_012_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_013_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_013_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_013_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_014_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_014_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_014_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_015_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_015_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_015_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_016_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_016_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_016_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_017_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_017_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_017_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_018_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_018_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_018_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_019_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_019_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_019_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_020_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_020_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_020_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_021_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_021_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_021_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_022_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_022_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_022_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_023_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_023_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_023_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_024_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_024_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_024_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_025_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_025_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_025_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_026_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_026_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_026_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_027_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_027_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_027_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_028_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_028_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_028_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_029_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_029_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_029_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/round_030_clients.json`（json）キー: `clients, selected`。
- `runs/pfedbayes/seed0/round_030_val_post.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, threshold_selected, confusion_pre, confusion_post`。
- `runs/pfedbayes/seed0/round_030_val_pre.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, threshold_selected, confusion_pre`。
- `runs/pfedbayes/seed0/summary.json`（json）キー: `best, rounds, used_point_init, train_backbone, run_dir`。
- `runs/pfedbayes/seed0/test_report.json`（json）キー: `n, n_pos, n_neg, metrics_pre, uncertainty, metrics_post, confusion_pre, confusion_post, bootstrap`。

## 4. Data Availability Matrix（T1/T2/F1/F2/F3/F4/F5/T3）
| ID | 必要データ | 根拠（ファイル・列/キー） | 判定 |
|---|---|---|---|
| T1 データ/クライアント要約 | client_id, n_samples, n_pos, pos_rate, train/val/test counts, exclusion_rate, missing_rate, split_unit | `outputs/noniid_report.json` の `per_client_per_split`（`n_files,n_windows,n_pos,n_neg`）と `per_client_pooled.pos_rate`、`federated_data/summary.json` の `splits/exclusion/missing`、`runs/pfedbayes/seed0/data_counts.json`（train/val/test の `n,n_pos,n_neg`） | **一部不足**（client別の missing/exclusion が無い。`split_unit` 相当の明示キーは `client_scheme` のみ） |
| T2 主結果（性能＋確率の質） | AUPRC/AUROC/ECE/Brier/NLL/Accuracy 等、seed繰り返し | Central: `runs/centralized_batch/seed{0,1,2}/eval_test.json` の `metrics_post`、FedAvg: `runs/fedavg_batch/seed{0,1,2}/test_report.json` の `metrics_post`、BFL: `runs/pfedbayes/seed0/test_report.json` の `metrics_post`。Accuracy は `outputs/table2_global.csv` に `Accuracy` 列あり | **一部不足**（BFL が seed0 のみで平均±分散が作れない） |
| F1 非IID可視化＋クライアント別性能分布 | client別の pos_rate, n_samples + client別評価指標 | `outputs/noniid_report.json`（client別 `n_files/n_windows/n_pos/n_neg/pos_rate`）、`runs/fedavg_batch/seed0/test_report_per_client.csv`（12 clients, AU*・ECE 等）、`outputs/pfedbayes_vs_fedavg_client_comparison_seed0.json`（client別 AU*・ECE 等） | **作れる（seed0のみ）** |
| F2 信頼度図（予測確率 vs 実測率） | bin集計（bin_conf, bin_acc, bin_count 等） | `runs/compare/pfedbayes_vs_fedavg_seed0_post.json` の `reliability`（15 bin, `confidence/accuracy/weight`） | **作れる**（`central_vs_fedavg_test_post.json` はラベル注意） |
| F3 不確かさが正しいことの図 | (a) 不確かさ順位×性能 or (b) 予測区間覆い率 | 既存ファイルに per-sample の `y_true/y_prob/uncert_score` や `pred_interval` が無い。`runs/pfedbayes/seed0/test_report.json` の `uncertainty` は集計のみ | **作れない（要再実行で保存）** |
| F4 学習の安定性（round/epoch推移） | round/epochごとの loss/指標 | `runs/centralized_batch/seed{0,1,2}/history.csv`（epoch, train_loss, val_*）、`runs/fedavg_batch/seed{0,1,2}/history.csv`（round, train_loss, val_*）、`runs/pfedbayes/seed0/history.csv`（round, train_loss, val_auprc, temperature, threshold） | **作れる** |
| F5 クライアント間影響（行列） | client×client 影響/類似度 | `round_*_client_similarity.json` が存在しない。FedAvg は `federated/server.py` に `--log-client-sim` 実装あり（`federated/server.py:107`）だが未出力。pFedBayes には同オプションが無い | **一部不足**（FedAvgは再実行、BFLは要ログ追加） |
| T3 統計的有意性 | 同一条件の繰り返しスコア集合 | Central/FedAvg は seed0-2 あり。BFL（pfedbayes）は seed0 のみ。`outputs/pfedbayes_vs_fedavg_client_comparison_seed0.json` に client差の sign-flip p (`diff_auprc_p_signflip`) はあるが seed 繰り返しではない | **一部不足** |

## 5. 結論（主張A/B）
- 主張A（予測確率が信頼できる）: T2 と F2 は作成可能だが、F3（不確かさが正しいことの図）は **作れない**。従って主張Aの完全立証には不足。
- 主張B（非IIDでも性能が落ちにくい／学習が安定）: F1 と F4 は作成可能。F5（クライアント間影響行列）は **作れない／一部不足**。

## 6. 不足データと解決策（算出可 / 要再実行 / 要ログ追加）
- **算出可**
  - T1 の `pos_rate` は `outputs/noniid_report.json` の `n_pos/n_neg` から計算可（client別）。
- **要再実行**
  - F3: per-sample の `y_true/y_prob/uncertainty` が未保存。`bayes_federated/eval.py --save-pred-npz` か `bayes_federated/pfedbayes_server.py --save-test-pred-npz` で再評価が必要。
  - F5（FedAvg側）: `federated/server.py --log-client-sim` を付けて再学習すると `round_*_client_similarity.json` が生成される。
  - T2/T3（BFLの統計）: BFL の seed0 のみ。seed1/2 の再実行が必要。
- **要ログ追加**
  - T1: client別 missing/exclusion rate が未記録。`scripts/build_dataset.py` などに client別ログ追加が必要。
  - F5（BFL側）: `bayes_federated/pfedbayes_server.py` に client similarity の保存機能が無い（オプション追加が必要）。
- **注意（ラベル）**
  - `runs/compare/central_vs_fedavg_test_post.json` の `reliability` キーが `FedAvg/BFL` 固定で、`method_a` は Central。図作成時にラベル補正が必要。

## 7. 次に実行すべき最小コマンド（実在スクリプトに基づく）
1) F3用の per-sample 予測保存（BFL/pFedBayes）
```
./venv/bin/python bayes_federated/eval.py \
  --data-dir federated_data \
  --checkpoint runs/pfedbayes/seed0/checkpoints/model_best.pt \
  --split test \
  --save-pred-npz outputs/pfedbayes_seed0_test_pred.npz
```
2) client別比較の再生成（pFedBayes vs FedAvg, seed0）
```
./venv/bin/python scripts/eval_compare_clients.py \
  --data-dir federated_data \
  --fedavg-runs runs/fedavg_batch/seed0 \
  --bfl-runs runs/pfedbayes/seed0 \
  --seeds 0 \
  --out-json outputs/pfedbayes_vs_fedavg_client_comparison_seed0.json \
  --out-csv outputs/pfedbayes_vs_fedavg_client_comparison_seed0.csv
```
3) 表2/表3・信頼度図の再作成（pFedBayes版の compare JSON を指定）
```
./venv/bin/python scripts/make_paper_tables_fig3.py \
  --central-run runs/centralized_batch/seed0 \
  --fedavg-run runs/fedavg_batch/seed0 \
  --bfl-run runs/pfedbayes/seed0 \
  --compare-json runs/compare/pfedbayes_vs_fedavg_seed0_post.json \
  --out-dir outputs
```
4) F5（FedAvg の client similarity 行列）
```
./venv/bin/python federated/server.py \
  --data-dir federated_data \
  --out-dir runs/fedavg_batch \
  --run-name seed0 \
  --log-client-sim
```
5) 有意差（paired bootstrap, seed0 の比較）
```
./venv/bin/python scripts/compare_significance.py \
  --data-dir federated_data \
  --a-kind ioh --a-run-dir runs/fedavg_batch/seed0 \
  --b-kind bfl --b-run-dir runs/pfedbayes/seed0 \
  --variant post \
  --out runs/compare/pfedbayes_vs_fedavg_seed0_post_signif.json
```

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
