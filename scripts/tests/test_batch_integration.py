"""Exercise batch lifecycle wiring through the real simulation entry point and saver."""
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from torch import nn
from flwr.server import ServerApp
from flwr.server.strategy import FedAvg
from src.entrypoints.server import TrackedServerApp
from src.tier_cloud.cloud_aggregator_sim import CloudAggregator
from src.shared.network.model_payload import encode_model
from scripts.tests.test_batch_runs import cwd


class BatchIntegrationTests(unittest.TestCase):
    def test_entrypoint_records_success_and_failure_of_actual_server_call_boundary(self):
        for failure in (False, True):
            with self.subTest(failure=failure), TemporaryDirectory() as directory, cwd(directory):
                config = {"batch_attempt": "test", "run_name": "test"}
                app = TrackedServerApp(server_fn=lambda _: None)
                context = SimpleNamespace(run_config={"mode": "sim"})
                with patch("src.shared.utils.config_loader.get_merged_config", return_value=config), \
                     patch.object(ServerApp, "__call__", side_effect=RuntimeError("server failed") if failure else None) as run:
                    if failure:
                        with self.assertRaisesRegex(RuntimeError, "server failed"):
                            app(None, context)
                    else:
                        app(None, context)
                run.assert_called_once()
                status = json.loads(Path("results/test/batch_lifecycle.json").read_text())
                self.assertEqual(status["status"], "FAILED" if failure else "COMPLETED")

    def test_deployment_entrypoint_does_not_enable_batch_tracking(self):
        app = TrackedServerApp(server_fn=lambda _: None)
        with patch.object(ServerApp, "__call__"), \
             patch("src.shared.utils.simulation_lifecycle.track_simulation", side_effect=AssertionError("deployment must not track")):
            app(None, SimpleNamespace(run_config={"mode": "deploy"}))

    def test_round_marker_is_written_after_checkpoint_and_absent_on_save_failure(self):
        for failure in (False, True):
            with self.subTest(failure=failure), TemporaryDirectory() as directory, cwd(directory):
                strategy = CloudAggregator.__new__(CloudAggregator)
                strategy.logger, strategy.log_prefix = logging.getLogger("batch-test"), "CLOUD"
                strategy.snapshot_interval, strategy.snapshot_rounds = 0, []
                strategy.quantization_bits, strategy.global_model = 32, nn.Linear(2, 2)
                strategy.experiment_name = "test"
                strategy.run_metadata = {"batch_attempt": "test"}
                strategy.results_dict = {"metadata": {}, "performance": []}
                with patch.object(FedAvg, "evaluate", return_value=(.1, {"accuracy": .8})):
                    if failure:
                        with patch("torch.save", side_effect=OSError("disk full")), self.assertRaises(OSError):
                            strategy.evaluate(2, encode_model(strategy.global_model, 32))
                    else:
                        strategy.evaluate(2, encode_model(strategy.global_model, 32))
                progress = Path("results/test/batch_progress.json")
                self.assertEqual(progress.exists(), not failure)
                if not failure:
                    self.assertTrue(Path("results/test/global_model.pt").is_file())
                    self.assertEqual(json.loads(progress.read_text())["completed_round"], 2)
