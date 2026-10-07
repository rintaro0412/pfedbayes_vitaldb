"""Filesystem and command-contract tests; no model imports or training."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_experiments as runner


class ExperimentRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for entry in runner.ENTRYPOINTS:
            self.put(entry, "# training placeholder\n")
        self.put("configs/base.json", json.dumps({
            "run": {"out_dir": "old", "run_name": "seed0"},
            "data": {"data_dir": "old"}, "train": {"seed": 42},
            "eval": {"model_selection": "best", "selection_source": "val"},
        }))
        self.manifest = {
            "schema_version": 1, "protocol_id": "icaise2026_test", "data": {"data_dir": "data"},
            "output_root": "runs/paper", "methods": [
                {"key": "centralized", "label": "Centralized", "entrypoint": "centralized/train.py", "config": "configs/base.json", "seeds": [42], "source_run_dir": "historical/seed0", "model_scope": "global"},
                {"key": "local", "label": "Local", "entrypoint": "scripts/train_local.py", "config": "configs/base.json", "seeds": [42], "source_run_dir": "historical/local", "model_scope": "client_specific"},
                {"key": "fedavg", "label": "FedAvg", "entrypoint": "federated/server.py", "config": "configs/base.json", "seeds": [42], "source_run_dir": "historical/fedavg", "model_scope": "global"},
                {"key": "pfedbayes", "label": "pFedBayes", "entrypoint": "bayes_federated/pfedbayes_server.py", "config": "configs/base.json", "seeds": [42], "source_run_dir": "historical/pfedbayes", "model_scope": "global"},
            ],
            "stages": {stage: [{"name": "make", "argv": ["{python}", "helper.py", "{methods_json}", "{output_root}/result with spaces.json"], "outputs": ["{output_root}/result with spaces.json"]}] for stage in runner.STAGES[1:]},
        }
        self.manifest_path = self.root / "configs/manifest.json"

    def put(self, relative, contents=""):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
        return path

    def args(self, *argv):
        return runner.parser().parse_args(list(argv))

    def plan(self, args):
        self.put("configs/manifest.json", json.dumps(self.manifest))
        return runner.plan(self.manifest_path, args, root=self.root)

    def data(self):
        for split in ("train", "val", "test"):
            self.put(f"data/client/{split}/case_1.npz", "placeholder")

    def completed_outputs(self, task):
        for pattern in task.outputs:
            path = Path(pattern.replace("*", "client"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("output", encoding="utf-8")

    def test_resolves_real_seed_data_and_explicit_federated_directory(self):
        args = self.args("--seeds", "7", "--data-dir", "alternative", "--dry-run")
        tasks = self.plan(args)
        for task in tasks:
            self.assertEqual(task.run.config["train"]["seed"], 7)
            self.assertEqual(task.run.config["run"]["run_name"], "seed7")
            self.assertEqual(task.run.config["data"]["data_dir"], str(self.root / "alternative"))
            self.assertIn("--config", task.argv)
            self.assertNotIn("--seed", task.argv)
        fed = next(task for task in tasks if task.name == "fedavg")
        self.assertEqual(fed.run.config["run"]["run_dir"], str(fed.run.run_dir))

    def test_dry_run_creates_nothing_and_starts_no_process(self):
        args = self.args("--dry-run", "--stage", "all")
        tasks = self.plan(args)
        before = sorted(str(p) for p in self.root.rglob("*"))
        with mock.patch.object(runner.subprocess, "run") as child, contextlib.redirect_stdout(io.StringIO()):
            runner.execute(tasks, args, self.root)
        child.assert_not_called()
        self.assertEqual(before, sorted(str(p) for p in self.root.rglob("*")))
        self.assertNotIn("resources", [task.stage for task in tasks])

    def test_missing_validation_stops_before_any_write_or_process(self):
        args = self.args("--methods", "centralized")
        tasks = self.plan(args)
        self.put("data/client/train/case_1.npz")
        self.put("data/client/test/case_1.npz")
        with mock.patch.object(runner.subprocess, "run") as child:
            with self.assertRaisesRegex(runner.ProtocolError, "no val data"):
                runner.execute(tasks, args, self.root)
        child.assert_not_called()
        self.assertFalse((self.root / "runs").exists())

    def test_matching_completed_run_skips_but_directory_alone_does_not(self):
        self.data()
        args = self.args("--methods", "centralized")
        task, = self.plan(args)
        self.completed_outputs(task)
        with self.assertRaisesRegex(runner.ProtocolError, "no matching runner record"):
            runner.execute([task], args, self.root)
        runner.write_record(task.complete_path, {"fingerprint": task.fingerprint})
        with mock.patch.object(runner.subprocess, "run") as child, contextlib.redirect_stdout(io.StringIO()):
            runner.execute([task], args, self.root)
        child.assert_not_called()

    def test_mismatched_configuration_never_overwrites_existing_run(self):
        self.data()
        args = self.args("--methods", "centralized", "--resume")
        task, = self.plan(args)
        existing = self.put(str((task.run.run_dir / "keep.txt").relative_to(self.root)), "user result")
        runner.write_record(task.started_path, {"fingerprint": "another-config"})
        with self.assertRaisesRegex(runner.ProtocolError, "no matching runner record"):
            runner.execute([task], args, self.root)
        self.assertEqual(existing.read_text(), "user result")
        self.assertFalse(task.methods_path.exists())

    def test_resume_requires_actual_server_state(self):
        self.data()
        args = self.args("--methods", "pfedbayes", "--resume")
        task, = self.plan(args)
        runner.write_record(task.started_path, {"fingerprint": task.fingerprint})
        self.put(str((task.run.run_dir / "checkpoints/model_last.pt").relative_to(self.root)))
        with self.assertRaisesRegex(runner.ProtocolError, "server_state.pt"):
            runner.execute([task], args, self.root)

    def test_success_requires_outputs_and_records_resolved_config(self):
        self.data()
        args = self.args("--methods", "centralized")
        task, = self.plan(args)
        with mock.patch.object(runner.subprocess, "run") as child, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(runner.ProtocolError, "required outputs are absent"):
                runner.execute([task], args, self.root)
        child.assert_called_once_with(task.argv, cwd=self.root, check=True)
        config = runner.read_mapping(task.run.run_dir / ".runner/resolved_config.json")
        self.assertEqual(config["train"]["seed"], 42)
        self.assertFalse(task.complete_path.exists())

    def test_successful_run_is_marked_and_subsequent_invocation_skips(self):
        self.data()
        args = self.args("--methods", "centralized")
        task, = self.plan(args)
        with mock.patch.object(runner.subprocess, "run", side_effect=lambda *a, **kw: self.completed_outputs(task)), contextlib.redirect_stdout(io.StringIO()):
            runner.execute([task], args, self.root)
        self.assertEqual(runner.read_marker(task.complete_path)["fingerprint"], task.fingerprint)
        next_task, = self.plan(args)
        with mock.patch.object(runner.subprocess, "run") as child, contextlib.redirect_stdout(io.StringIO()):
            runner.execute([next_task], args, self.root)
        child.assert_not_called()

    def test_source_figures_write_only_new_outputs_and_metadata(self):
        source_file = self.put("historical/seed0/history.csv", "existing history")
        args = self.args("--methods", "centralized", "--stage", "figures", "--use-source-runs")
        task, = self.plan(args)
        with mock.patch.object(runner.subprocess, "run", side_effect=lambda *a, **kw: self.completed_outputs(task)), contextlib.redirect_stdout(io.StringIO()):
            runner.execute([task], args, self.root)
        self.assertEqual(source_file.read_text(), "existing history")
        self.assertEqual(list(source_file.parent.iterdir()), [source_file])
        rows = json.loads(task.methods_path.read_text())
        self.assertEqual(rows[0]["seed"], 42)
        self.assertTrue(task.complete_path.exists())

    def test_shared_model_change_invalidates_training_fingerprint(self):
        args = self.args("--methods", "centralized", "--dry-run")
        original, = self.plan(args)
        self.put("common/ioh_model.py", "# changed model\n")
        changed, = self.plan(args)
        self.assertNotEqual(original.fingerprint, changed.fingerprint)

    def test_stage_skip_hash_changes_when_source_history_changes(self):
        history = self.put("historical/seed0/history.csv", "first history")
        args = self.args("--methods", "centralized", "--stage", "figures", "--use-source-runs")
        task, = self.plan(args)
        with mock.patch.object(runner.subprocess, "run", side_effect=lambda *a, **kw: self.completed_outputs(task)), contextlib.redirect_stdout(io.StringIO()):
            runner.execute([task], args, self.root)
        unchanged, = self.plan(args)
        with mock.patch.object(runner.subprocess, "run") as child, contextlib.redirect_stdout(io.StringIO()):
            runner.execute([unchanged], args, self.root)
        child.assert_not_called()
        history.write_text("changed history", encoding="utf-8")
        changed, = self.plan(args)
        with mock.patch.object(runner.subprocess, "run", side_effect=lambda *a, **kw: self.completed_outputs(changed)) as child, contextlib.redirect_stdout(io.StringIO()):
            runner.execute([changed], args, self.root)
        child.assert_called_once()

    def test_source_run_seed0_remains_seed42_in_methods_metadata(self):
        args = self.args("--methods", "centralized", "--stage", "figures", "--use-source-runs", "--dry-run")
        task, = self.plan(args)
        row, = task.methods_rows
        self.assertEqual(row["seed"], 42)
        self.assertEqual(row["run_dir"], str(self.root / "historical/seed0"))
        self.assertEqual(row["inference_kind"], "point")
        self.assertIn(str(task.methods_path), task.argv)
        self.assertIn("result with spaces.json", task.argv[-1])

    def test_source_mode_forbids_train_and_evaluate_and_seed_relabel(self):
        for stage in ("train", "evaluate", "all"):
            with self.subTest(stage=stage), self.assertRaisesRegex(runner.ProtocolError, "only for aggregate"):
                self.plan(self.args("--stage", stage, "--use-source-runs"))
        with self.assertRaisesRegex(runner.ProtocolError, "cannot relabel"):
            self.plan(self.args("--stage", "figures", "--use-source-runs", "--seeds", "7"))

    def test_source_mode_rejects_outputs_in_historical_run(self):
        self.manifest["output_root"] = "."
        self.manifest["stages"]["figures"][0].update(for_each_run=True, outputs=["{run_dir}/overwritten.json"])
        with self.assertRaisesRegex(runner.ProtocolError, "cannot overwrite historical"):
            self.plan(self.args("--stage", "figures", "--use-source-runs"))

    def test_all_nine_methods_can_share_existing_entrypoints(self):
        for key in ("fedprox", "scaffold", "fednova", "perfedavg", "pfedme"):
            method = dict(self.manifest["methods"][2], key=key, label=key)
            self.manifest["methods"].append(method)
        tasks = self.plan(self.args("--dry-run"))
        self.assertEqual(len(tasks), 9)
        self.assertEqual(len({task.run.run_dir for task in tasks}), 9)

    def test_unknown_method_and_unavailable_placeholders_fail(self):
        with self.assertRaisesRegex(runner.ProtocolError, "Unknown methods"):
            self.plan(self.args("--methods", "typo"))
        self.manifest["stages"]["figures"][0]["argv"].append("{seed}")
        with self.assertRaisesRegex(runner.ProtocolError, "unavailable placeholder"):
            self.plan(self.args("--stage", "figures"))

    def test_multiple_seeds_fail_before_combined_downstream_execution(self):
        self.put("configs/manifest.json", json.dumps(self.manifest))
        before = sorted(str(path) for path in self.root.rglob("*"))
        for stage in ("aggregate", "figures", "resources", "all"):
            with self.subTest(stage=stage), mock.patch.object(runner.subprocess, "run") as child:
                with self.assertRaisesRegex(runner.ProtocolError, "run downstream stages separately with --seeds <one>"):
                    runner.plan(self.manifest_path, self.args("--seeds", "1,2", "--stage", stage), self.root)
            child.assert_not_called()
        self.assertEqual(before, sorted(str(path) for path in self.root.rglob("*")))

    def test_multiple_seeds_are_allowed_for_training_and_per_run_evaluation(self):
        train = self.plan(self.args("--seeds", "1,2", "--stage", "train", "--dry-run"))
        self.assertEqual(len(train), 8)
        self.manifest["stages"]["evaluate"][0].update(
            for_each_run=True, outputs=["{run_dir}/eval_test.json"]
        )
        evaluate = self.plan(self.args("--seeds", "1,2", "--stage", "evaluate", "--dry-run"))
        self.assertEqual(len(evaluate), 8)
        self.assertEqual({task.run.seed for task in evaluate}, {1, 2})

    def test_evaluation_fingerprint_excludes_its_own_declared_reports(self):
        self.manifest["stages"]["evaluate"][0].update(
            for_each_run=True, methods=["centralized"], outputs=["{run_dir}/test_report_per_client.csv"]
        )
        args = self.args("--methods", "centralized", "--stage", "evaluate")
        task, = self.plan(args)
        with mock.patch.object(runner.subprocess, "run", side_effect=lambda *a, **kw: self.completed_outputs(task)), contextlib.redirect_stdout(io.StringIO()):
            runner.execute([task], args, self.root)
        unchanged, = self.plan(args)
        with mock.patch.object(runner.subprocess, "run") as child, contextlib.redirect_stdout(io.StringIO()):
            runner.execute([unchanged], args, self.root)
        child.assert_not_called()


if __name__ == "__main__":
    unittest.main()
