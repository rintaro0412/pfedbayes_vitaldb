import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch

import numpy as np
import torch

from scripts import benchmark_icaise_inference as benchmark
from scripts.make_icaise_training_curves import _parse_method_specs as curve_specs


class InferenceOutputTests(unittest.TestCase):
    def test_local_outputs_leave_source_run_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = root / "historical/local/checkpoints"
            checkpoints.mkdir(parents=True)
            (checkpoints / "client_a_best.pt").write_bytes(b"fixture")
            output = root / "new/benchmark_local_per_client.csv"
            spec = benchmark.MethodSpec("Local", "historical/local", "local_point")
            with patch.object(benchmark, "list_client_ids", return_value=["a"]), \
                 patch.object(benchmark, "list_npz_files", return_value=["fixture.npz"]), \
                 patch.object(benchmark, "_load_point_model", return_value=object()), \
                 patch.object(benchmark, "_make_loader", return_value=([], None)), \
                 patch.object(benchmark, "_point_once", return_value=(0.1, np.array([-1., 1.]), np.array([0, 1]))), \
                 patch.object(benchmark, "_reset_peak_memory"), \
                 patch.object(benchmark, "_ru_maxrss_mb", return_value=0.), \
                 patch.object(benchmark, "_peak_memory_mb", return_value={}):
                report = benchmark._benchmark_local_method(
                    spec=spec, repo_root=root, data_dir=root / "data", split="test",
                    batch_size=2, num_workers=0, repeats=1, warmup=0,
                    max_batches=1, device=torch.device("cpu"), per_client_csv=output,
                )
            self.assertEqual(report["per_client_csv"], str(output))
            self.assertTrue(output.is_file())
            self.assertEqual(list((root / "historical/local").iterdir()), [checkpoints])

    def test_shared_method_list_preserves_consumer_specific_kinds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "methods.json"
            path.write_text(json.dumps({"methods": [{
                "label": "Local", "source_run_dir": "runs/local/seed42",
                "config": "configs/icaise2026/local.yaml",
                "curve_kind": "local_round_client", "inference_kind": "local_point",
            }]}))
            curve, = curve_specs(path)
            inference, = benchmark._parse_method_specs(path)
            self.assertEqual(curve.kind, "local_round_client")
            self.assertEqual(inference.kind, "local_point")
            self.assertEqual(inference.run_dir, "runs/local/seed42")
            self.assertEqual(inference.config_path, "configs/icaise2026/local.yaml")

    def test_partial_benchmark_failure_cannot_be_marked_successful(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out_csv, out_json = root / "summary.csv", root / "summary.json"
            specs = [benchmark.MethodSpec("FedAvg", "unused", "point"),
                     benchmark.MethodSpec("Local", "unused", "local_point")]
            argv = ["benchmark", "--out-csv", str(out_csv), "--out-json", str(out_json), "--fail-on-error"]
            with patch.object(sys, "argv", argv), \
                 patch.object(benchmark, "_parse_method_specs", return_value=specs), \
                 patch.object(benchmark, "_device_from_arg", return_value=torch.device("cpu")), \
                 patch.object(benchmark.torch, "set_grad_enabled"), \
                 patch.object(benchmark, "_benchmark_point_method", return_value={
                     "method": "FedAvg", "time_s_mean": 0.1, "latency_ms_per_sample_mean": 0.1}), \
                 patch.object(benchmark, "_benchmark_local_method", side_effect=RuntimeError("missing checkpoint")):
                with self.assertRaises(SystemExit) as error:
                    benchmark.main()
            self.assertEqual(error.exception.code, 2)
            self.assertTrue(out_csv.is_file())
            self.assertEqual(len(json.loads(out_json.read_text())["warnings"]), 1)


if __name__ == "__main__":
    unittest.main()
