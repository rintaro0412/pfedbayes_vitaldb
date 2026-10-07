"""Small regression checks for ICAISE aggregate metrics and source provenance."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from scripts.make_paper_ja_missing_assets import (
    MethodSpec,
    _build_table2,
    _compute_table4,
    _normalize_metric_columns,
    _parse_method_specs,
)
from scripts.make_paper_tables_fig3 import _client_macro_ece_from_csv


class IcaiseAggregationTests(unittest.TestCase):
    def test_local_raw_metrics_and_global_client_macro_keep_distinct_scopes(self):
        local = pd.DataFrame({"client_id": ["a", "b"], "auprc": [0.4, 0.8], "nll": [0.2, 0.4]})
        global_model = pd.DataFrame({"client_id": ["a", "b"], "auprc_pre": [0.6, 0.9], "nll_pre": [0.1, 0.3]})
        summary, long = _compute_table4({"Local": local, "pFedBayes": global_model}, ["auprc", "nll"])
        rows = summary.set_index("method")
        self.assertAlmostEqual(rows.loc["Local", "macro_auprc"], 0.6)
        self.assertAlmostEqual(rows.loc["Local", "worst_client_auprc"], 0.4)
        self.assertEqual(rows.loc["Local", "n_clients"], 2)
        self.assertEqual(rows.loc["Local", "model_scope"], "client_specific")
        self.assertEqual(rows.loc["pFedBayes", "model_scope"], "global")
        self.assertEqual(set(long["aggregation"]), {"client_macro"})
        self.assertNotIn("auprc_pre", local.columns)

    def test_pre_metrics_take_precedence_and_post_is_not_aliased(self):
        frame = pd.DataFrame({"auprc": [0.1], "auprc_pre": [0.7], "nll_post": [0.2]})
        normalized = _normalize_metric_columns(frame)
        self.assertEqual(normalized.loc[0, "auprc_pre"], 0.7)
        self.assertNotIn("nll_pre", normalized.columns)

    def test_settings_use_saved_resolved_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs/example"
            run.mkdir(parents=True)
            saved = run / "config.json"
            saved.write_text(json.dumps({"args": {"epochs": 7, "batch_size": 11, "lr": 0.002, "model_selection": "best", "selection_metric": "nll"}}))
            method = MethodSpec("Centralized", "unused.yaml", "runs/example")
            table = _build_table2(root / "configs", [method], repo_root=root, require_saved=True)
            row = table.iloc[0]
            self.assertEqual(row["rounds_or_epochs"], 7)
            self.assertEqual(row["batch_size"], 11)
            self.assertEqual(row["selection_metric"], "nll")
            self.assertEqual(row["config_source"], "saved_run_config")
            self.assertEqual(row["config_path"], str(saved))

    def test_settings_support_nested_bayesian_run_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs/bayes"
            run.mkdir(parents=True)
            (run / "config.json").write_text(json.dumps({"train": {"rounds": 19, "lr_q": 0.001, "lr_w": 0.002, "mc_eval": 25}, "eval": {"selection_metric": "nll"}}))
            table = _build_table2(root / "configs", [MethodSpec("pFedBayes", "unused.yaml", "runs/bayes")], repo_root=root, require_saved=True)
            self.assertEqual(table.iloc[0]["rounds_or_epochs"], 19)
            self.assertEqual(table.iloc[0]["learning_rate"], "lr_q=0.001, lr_w=0.002")
            self.assertEqual(table.iloc[0]["mc_eval"], 25)

    def test_publication_requires_saved_config_instead_of_current_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs/example").mkdir(parents=True)
            with self.assertRaisesRegex(FileNotFoundError, "Saved run config"):
                _build_table2(root / "configs", [MethodSpec("FedAvg", "fedavg.yaml", "runs/example")], repo_root=root, require_saved=True)

    def test_runner_run_path_precedes_historical_manifest_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "methods.json"
            path.write_text(json.dumps({"methods": [{"key": "local", "label": "Local", "config": "configs/local.yaml", "run_dir": "runs/new", "source_run_dir": "runs/old", "model_scope": "client_specific"}]}))
            method, = _parse_method_specs(path, repo_root=root)
            self.assertEqual(method.run_dir, "runs/new")
            self.assertEqual(method.config_name, str(root / "configs/local.yaml"))

    def test_raw_ece_alias_preserves_status_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "per_client.csv"
            pd.DataFrame({"client_id": ["a", "b", "c"], "status": ["ok", "ok", "failed"], "ece": [0.1, 0.3, 0.9]}).to_csv(path, index=False)
            self.assertAlmostEqual(_client_macro_ece_from_csv(path), 0.2)

    def test_explicit_run_flag_fails_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            script = Path(__file__).resolve().parents[1] / "scripts/make_paper_tables_fig3.py"
            result = subprocess.run([sys.executable, str(script), "--require-explicit-runs", "--out-dir", str(output)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("--central-run", result.stderr)
            self.assertFalse(output.exists())

    def test_tables_only_uses_saved_csv_without_creating_figures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs/local"
            run.mkdir(parents=True)
            (run / "config.json").write_text(json.dumps({"args": {"rounds": 2, "batch_size": 4, "seed": 42}}))
            pd.DataFrame({"client_id": ["a", "b"], "auprc": [0.4, 0.8], "nll": [0.2, 0.4]}).to_csv(run / "test_report_per_client.csv", index=False)
            (root / "methods.json").write_text(json.dumps({"methods": [{"label": "Local", "config": "configs/local.yaml", "source_run_dir": "runs/local", "model_scope": "client_specific"}]}))
            script = Path(__file__).resolve().parents[1] / "scripts/make_paper_ja_missing_assets.py"
            result = subprocess.run([sys.executable, str(script), "--repo-root", str(root), "--manifest", "methods.json", "--tables-only", "--tables-dir", "out/tables", "--figures-dir", "out/figures"], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / "out/figures").exists())
            report = json.loads((root / "out/tables/paper_ja_missing_assets_manifest.json").read_text())
            self.assertEqual(set(report["generated"]), {"table2", "table4", "table4_long"})
            row = pd.read_csv(root / "out/tables/table4_client_macro.csv").iloc[0]
            self.assertAlmostEqual(row["macro_auprc"], 0.6)


if __name__ == "__main__":
    unittest.main()
