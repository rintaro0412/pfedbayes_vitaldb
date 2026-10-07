# VitalDB hypoxemia prediction — ICAISE 2026

Code, saved experiment configurations, and selected aggregate artifacts for *Evaluating Bayesian Federated Learning for Medical Time-Series Classification and Uncertainty Assessment*.

The comparison covers Centralized, Local, FedAvg, FedProx, SCAFFOLD, FedNova, Per-FedAvg, pFedMe, and pFedBayes. Existing training entrypoints are preserved. The published historical benchmark is exploratory: its case-based split has patient overlap, preprocessing has documented future-dependent paths, and the saved positive-window sampling differs from strict 300-second-ahead extraction. See [experiment documentation](docs/icaise2026.md) before interpreting the scores.

## Start here

- [Experiment list](configs/icaise2026/experiments.yaml): nine methods, actual seeds, explicit stages.
- [Saved-config provenance](configs/icaise2026/provenance.json): run names, recorded commits and configuration hashes.
- [Historical dataset definition](configs/icaise2026/dataset.json): confirmed sampling and split information.
- [Script guide](scripts/README.md): data preparation, training, evaluation, figures and resource measurement.
- [Selected artifact manifest](artifacts/icaise2026/manifest.json): source/result correspondence and measurement status.

## Plan a replay

From the repository root, in an environment providing the existing PyYAML dependency:

```bash
python scripts/run_experiments.py --manifest configs/icaise2026/experiments.yaml --stage all --dry-run
```

Planning writes no outputs and starts no training. A real replay needs the dataset and a suitable training environment; raw data and model checkpoints are not bundled. [Environment and execution details](docs/icaise2026.md#environment) explain the existing requirements files and individual stages. Historical scores have not been reproduced numerically in the release preparation environment.

## Interpret the results

`model_scope` and `aggregation` describe different things. Global client-macro scores evaluate the same global model on each client; they are not personalized-posterior scores. Local uses client-specific models. Standalone pFedBayes personalized evaluation is a separate report.

All nine saved training configs record seed 42, including the historically named `centralized/seed0` run. The pFedBayes 2160-minute training duration is projected from 37 rounds, not measured over all 200 rounds. Patient maps, raw signals, checkpoints, submission correspondence, other-paper materials and new research are outside this export.

## Citation and license

Use [CITATION.cff](CITATION.cff) for the repository citation. A paper DOI is not yet recorded in this release. License selection is pending; no new open-source license is assumed.
