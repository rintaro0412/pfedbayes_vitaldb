# ICAISE script guide

Run commands from the repository root. The public file list is `configs/icaise2026/publication_files.json`; other scripts may remain in the development worktree.

| Stage | Entry scripts | Inputs / outputs |
|---|---|---|
| Data preparation | `data_download.py`, `build_dataset.py`, `pack_contiguous_dataset.py` | VitalDB download, window extraction, optional contiguous packing |
| Split audit | `audit_patient_split_leakage.py` | Read-only clinical/file-inventory audit; aggregate overlap counts |
| Training | `run_experiments.py` | Dispatches Centralized, Local, six federated methods and pFedBayes from the experiment manifest |
| Local baseline | `train_local.py` | Client-specific training and final test reports |
| Saved-result aggregation | `make_paper_ja_missing_assets.py --tables-only` | Saved run configs / per-client CSVs; setting and client-macro tables with model scope |
| Global paper tables | `make_paper_tables_fig3.py --require-explicit-runs` | Explicit Centralized / FedAvg / pFedBayes runs; pooled metrics and reliability |
| Significance | `compare_significance.py`, `compare_pfedbayes_vs_methods.py` | Explicit run/checkpoint comparisons; full inference/bootstrap can be expensive |
| Uncertainty | `make_risk_coverage.py`, `make_bayesian_evaluation.py` | Saved predictions; coverage, calibration and uncertainty diagnostics |
| Data description | `make_table1_client_summary.py`, `report_heterogeneity.py`, `make_client_split_flow_figure.py` | Client counts, distribution summaries and split workflow |
| Learning curves | `make_icaise_training_curves.py` | Explicit method list and saved training histories |
| Resources | `benchmark_icaise_inference.py`, `benchmark_icaise_training_resources.py` | Saved-checkpoint inference or separately authorized instrumented training |
| Optional EDA | `make_eda_figures.py`, `make_eda_supplement.py` | Dataset summaries and sampled signals; full aggregation/bootstrap is separate work |
| Public export | `prepare_public_release.py` | Exact allowlist to an empty isolated directory with hashes |

`run_all_pfedbayes.sh` delegates to the manifest runner and defaults to planning. Its explicitly requested legacy path is retained for compatibility; its seed/path conventions are not the public protocol.

Training implementations remain in `centralized/train.py`, `federated/server.py` / `client.py` and `bayes_federated/pfedbayes_server.py` / `pfedbayes_client.py`. Evaluation is implemented in `centralized/eval.py`, the existing trainers and `bayes_federated/eval.py`. Shared loading, metrics and experiment records remain in `common/`.

The single generated method list can be passed to the table builder, learning-curve plotter and inference benchmark. `curve_kind` and `inference_kind` are separate because each consumer has a different input contract. Prefer explicit run paths over automatic discovery for paper artifacts.

See [the complete protocol](../docs/icaise2026.md) for commands, known limitations and the distinction between global client-macro and personalized-posterior evaluation.
