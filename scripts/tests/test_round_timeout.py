"""Exercise YAML-to-Flower timeout wiring without running training or a network."""
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import yaml

from src.shared.utils.config_loader import get_round_timeout, load_yaml_configs


class RoundTimeoutTests(unittest.TestCase):
    def test_positive_seconds_are_preserved(self):
        for value in (0.5, 180, 3600., 7200.25, "9000"):
            with self.subTest(value=value):
                self.assertEqual(get_round_timeout({"round_timeout": value}), float(value))

    def test_invalid_or_missing_timeout_fails_explicitly(self):
        for config in ({}, *({"round_timeout": value} for value in
                           (None, False, True, 0, -1, "", "bad", float("inf"), float("nan")))):
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, "round_timeout.*finite positive"):
                    get_round_timeout(config)

    def test_yaml_reaches_simulation_and_both_deployed_server_tiers(self):
        from src.entrypoints import deploy_server, sim_server
        configs = load_yaml_configs()
        configs.update(num_fogs=2, uniform_edges_per_fog=3, custom_fog_topology="[]",
                       label_flip_ratio=0., backdoor_ratio=0., grad_manip_ratio=0.,
                       shap_aware_ratio=0., run_name="timeout-test", num_rounds=2,
                       socket_timeout=91.)
        cases = ((sim_server, "cloud", "src.entrypoints.sim_server.CloudAggregator"),
                 (deploy_server, "cloud", "src.tier_cloud.cloud_aggregator_deploy.CloudAggregator"),
                 (deploy_server, "fog", "src.tier_fog.fog_aggregator.FogAggregator"))
        with TemporaryDirectory() as directory:
            # Use the real configuration merge: node YAML must override the Flower bundle default.
            for value in (0.5, 7200.25):
                for module, tier, strategy_path in cases:
                    with self.subTest(value=value, module=module.__name__, tier=tier):
                        (Path(directory) / "network.yaml").write_text(yaml.safe_dump(
                            dict(configs, round_timeout=value)))
                        with ExitStack() as stack:
                            stack.enter_context(patch.dict("os.environ", {"CONFIG_PATH": directory}))
                            stack.enter_context(patch.object(module, "get_dataset", return_value=(
                                torch.zeros(100, 40), torch.arange(100) % 15, 15)))
                            logger = stack.enter_context(patch.object(module, "setup_logger", return_value=Mock())).return_value
                            stack.enter_context(patch.object(module, "GlobalEvaluator"))
                            stack.enter_context(patch.object(module, "get_device", return_value="cpu"))
                            strategy = stack.enter_context(patch(strategy_path))
                            result = module.server_fn(SimpleNamespace(run_config={
                                "tier": tier, "fog_id": "fog_1", "round_timeout": 180.}))
                        self.assertEqual(result.config.round_timeout, value)
                        self.assertEqual(result.config.num_rounds, 2)
                        self.assertEqual(strategy.call_args.kwargs["run_metadata"]["round_timeout"], value)
                        logger.info.assert_any_call("Flower client-response timeout: %s seconds", value)
                        if module is deploy_server:
                            self.assertEqual(strategy.call_args.kwargs["socket_timeout"], 91.)

    def test_invalid_timeout_is_rejected_before_loading_datasets(self):
        from src.entrypoints import deploy_server, sim_server
        for module in (sim_server, deploy_server):
            with self.subTest(module=module.__name__), \
                 patch("src.shared.utils.config_loader.get_merged_config", return_value=dict(
                     load_yaml_configs(), round_timeout=0)), \
                 patch.object(module, "get_dataset") as data:
                with self.assertRaisesRegex(ValueError, "round_timeout"):
                    module.server_fn(SimpleNamespace(run_config={}))
                data.assert_not_called()


if __name__ == "__main__":
    unittest.main()
