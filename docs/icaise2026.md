# ICAISE 2026 experiment release

This release packages the nine-method VitalDB hypoxemia study, its saved experimental settings, and selected aggregate results. The paper title recorded in the local study materials is *Evaluating Bayesian Federated Learning for Medical Time-Series Classification and Uncertainty Assessment*. Numerical equivalence between the current code and every historical run has not been established.

## Layout and publication scope

The existing training entrypoints remain in `centralized/`, `federated/`, `bayes_federated/` and `scripts/train_local.py`. Shared datasets, models, metrics and environment snapshots remain in `common/`. `configs/icaise2026/experiments.yaml` is the experiment list. `scripts/run_experiments.py` dispatches existing programs without implementing another training loop.

`configs/icaise2026/publication_files.json` is the exact public file list. `scripts/prepare_public_release.py` copies only those files into an isolated directory and records their hashes. Its README override allows the original development README and worktree to remain intact. Raw data, checkpoints, unselected outputs, other-paper PDFs, acceptance letters, original review forms, RO-MAN material, MOVER experiments, excluded methods, old hypotension research and new BFL/SSM research are outside this release.

## Historical conditions and limitations

The saved data summary uses `positive_mode=within_horizon`, stride 30 seconds, at most 8 positive windows per event, negative/positive target ratio 2, and at most 12 negative windows per stable segment. This differs from the builder's default `single` extraction exactly 300 seconds before event onset. The historical result is a sampled five-minute-horizon benchmark; it should not be relabeled as a strict 300-second-ahead onset experiment.

The input comprises 30 seconds of HR, SpO2, ETCO2 and FIO2 on a 100 Hz grid, plus clinical covariates. The grid does not imply a native measurement frequency of 100 Hz. Labels use one-second mean SpO2: at most 92 for at least 60 seconds for positive onset, and at least 95 for at least 20 minutes for stable negative segments. Clients are department/procedure groups from one dataset, not independent hospitals.

The existing split is case based. A saved read-only audit found 100 patients / 250 cases crossing splits, including 65 patients / 168 cases crossing train and test. Whole-case normalization and backfilling in the current preprocessing code also have future-dependent paths. Historical performance remains exploratory. Patient-independent splits and causal preprocessing require a separately approved dataset revision and new results; they are not silently applied to this archival profile.

The dataset-generation command, seed and commit have not been established. `dataset.json` records only confirmed definitions, aggregate audit counts and the saved summary hash. It includes no patient-to-case mapping. All nine saved training configurations record seed **42**, including `runs/centralized/seed0`. A run directory name is not evidence of its random seed.

`source_configs/` preserves the original saved configurations. The runnable method YAMLs are reconstructed from them, with resume disabled, a new output directory, and an explicit validation selection source. Original `resume=true` settings describe saved run state, not an instruction to resume a fresh public replay. The recorded historical commit is retained in `provenance.json`; unrecorded dirty changes cannot be reconstructed from that commit alone.

## Evaluation meanings

Model scope and aggregation are separate fields:

| Model scope | Aggregation | Meaning |
|---|---|---|
| `global` | `pooled` | One model on the pooled evaluation samples |
| `global` | `client_macro` | The same global model evaluated separately on clients, then averaged |
| `client_specific` | `client_macro` | Local models on their matching client splits |
| `personalized` | `client_macro` | Each pFedBayes posterior q_i on its matching client split |

Historical pFedBayes `test_report_per_client.csv` scores describe a **global** model. Their client average does not demonstrate personalized-posterior performance. The corrected aggregate table labels this scope explicitly. Local raw metric columns are accepted alongside the other methods' `_pre` columns; post-calibration scores are not silently substituted.

Checkpoint selection uses validation NLL and a fixed classification threshold of 0.5 in the saved profile. pFedBayes uses 5 training and 25 evaluation Monte Carlo draws. Its saved `loss_type=bce` is unweighted BCE even though `pos_weight=auto` is present. The saved `per_client_every_round=true` also requests client-wise test diagnostics; those historical diagnostics are recorded rather than treated as validation selection data.

The evaluator's standalone personalized report is separate from historical global client-macro tables. Bootstrap units and sampling assumptions must be stated for any new significance analysis; this release does not run a new full statistical comparison.

## Environment

Use Python 3.12 and a suitable PyTorch environment, then the existing dependencies. `requirements-venv.txt` records CUDA PyTorch 2.6.0+cu124; `requirements-gpu.txt` records a development CUDA PyTorch build. Despite their names, neither is a generic CPU-only installation recipe. The original PyTorch package index/build must be available for those exact freezes. A CPU port or newer environment may be useful for smoke checks, but is a different environment and must be recorded. No additional project dependency is introduced by the release tools.

## Planning and execution

Run from the repository root. These planning commands create no experiment outputs and do not launch training or inference:

```bash
python scripts/run_experiments.py --manifest configs/icaise2026/experiments.yaml --stage all --dry-run
python scripts/run_experiments.py --manifest configs/icaise2026/experiments.yaml --stage train --methods fedavg,pfedbayes --seeds 42 --dry-run
python scripts/run_experiments.py --manifest configs/icaise2026/experiments.yaml --stage aggregate --use-source-runs --dry-run
```

Outputs from a fresh replay go to `runs/icaise2026/icaise2026_paper_v1/<method>/seed42/`. Remove `--dry-run` only for an intentional execution with the required data and resources. Dataset download and regeneration are separate commands; the runner never invokes them. The builder currently replaces an existing output directory, so dataset regeneration requires prior confirmation in the development workflow.

Stages are `train`, `evaluate`, `aggregate`, `figures`, and `resources`. `all` covers the first four; resource benchmarks are optional. Training already creates the final reports for federated and Local methods. The evaluation stage adds deterministic pooled/client reports and separate Bayesian global and personalized reports. The aggregate stage reads saved run configurations and per-client CSVs only. `--use-source-runs` is allowed for aggregate, figures and resources, and rejects train/evaluate to keep historical runs read-only.

Training and individual evaluation accept multiple seeds. Combined table, curve and resource stages currently require one seed per method; run them separately with `--seeds <one>` to avoid merging distinct runs under the same method label.

Examples for aggregate-only regeneration from historical results and optional resources:

```bash
python scripts/run_experiments.py --manifest configs/icaise2026/experiments.yaml --stage aggregate --use-source-runs
python scripts/run_experiments.py --manifest configs/icaise2026/experiments.yaml --stage resources --use-source-runs --dry-run
python scripts/benchmark_icaise_training_resources.py --manifest configs/icaise2026/experiments.yaml --dry-run
```

Real resource benchmarking, full evaluation and multiple-epoch training require explicit compute authorization in the development workflow. They were not performed while preparing this release. The historical 2160-minute pFedBayes training time is a projection to 200 rounds from the first 37 completed rounds. It is not a measured full-run duration; other rows retain their documented measurement type.

Local inference per-client measurements are written alongside the requested benchmark CSV, not inside the source training run. This keeps historical checkpoints and reports unchanged when using `--use-source-runs`.

The runner records resolved settings and completion markers. An existing directory alone is never treated as successful completion. `--resume` explicitly requests continuation and requires the expected checkpoint/history; matching completed runs can be skipped only when their recorded plan and expected artifacts agree. `run_all_pfedbayes.sh` delegates to the new runner. Its retained old path is available only with `LEGACY_RUN_ALL=1` and is outside the ICAISE public reproduction protocol.

## Public artifacts and export

`artifacts/icaise2026/manifest.json` names each selected source. The settings and client-macro tables are rebuilt from existing small configuration/CSV files; previous files under `tables/`, `figures/` and `runs/` are unchanged. Learning-curve PNGs, inference summaries, risk-coverage aggregates and resource summaries are selected historical copies. Risk-coverage uses `prob_total_var` and keeps samples with the lowest uncertainty first.

For a fresh empty export directory:

```bash
python scripts/prepare_public_release.py --out-dir /tmp/icaise2026-public --dry-run
python scripts/prepare_public_release.py --out-dir /tmp/icaise2026-public
```

Existing nonempty exports are refused. The exporter does not stage files, create commits, rewrite history, or push to GitHub. Publish the reviewed export on a dedicated release branch; preserve the existing worktree and index. Add a version tag after the published commit has been verified. New research should use a subsequent branch and a different protocol ID.

The software citation contains confirmed repository metadata; no paper DOI or publication details are invented. License selection is pending until the repository author supplies it.
