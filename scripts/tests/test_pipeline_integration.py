"""Exercise shared region processing and deployment/simulation wiring on CPU."""
from contextlib import contextmanager, nullcontext
import copy
import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn
from flwr.common import Code, FitIns, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from src.shared.models.cnn_lstm import CNNLSTMClassifier
from src.shared.network.model_payload import decode_model, encode_model, validate_region_result
from src.shared.utils.config_loader import load_yaml_configs
from src.shared.utils.metrics import compute_shap_stability
from src.shared.utils.security_metrics import calculate_security_rates
from src.tier_cloud.cloud_aggregator_deploy import CloudAggregator as DeployCloud
from src.tier_cloud.cloud_aggregator_sim import CloudAggregator as SimCloud
from src.tier_cloud.model_state import GlobalModelState
from src.tier_cloud.global_evaluator import GlobalEvaluator
from src.tier_fog.client_instructions import configure_edge_fit
from src.tier_fog.gatekeeper import ZeroTrustGatekeeper
from src.tier_fog.round_processing import process_region

LOGGER = logging.getLogger("pipeline-integration-tests")
LOGGER.addHandler(logging.NullHandler())
LOGGER.propagate = False
torch.set_num_threads(1)


@contextmanager
def working_directory(directory):
    previous = os.getcwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(previous)


class PipelineTests(unittest.TestCase):
    def test_fog_boot_does_not_construct_global_evaluator_or_load_test_data(self):
        from src.entrypoints import deploy_server
        config = {**load_yaml_configs(), "tier": "fog", "fog_id": "fog_1"}
        with patch("src.shared.utils.config_loader.get_merged_config", return_value=config), \
             patch.object(deploy_server, "setup_logger", return_value=LOGGER), \
             patch.object(deploy_server, "get_dataset", return_value=(torch.zeros(100, 40), torch.zeros(100, dtype=torch.long), 15)) as data, \
             patch.object(deploy_server, "GlobalEvaluator") as evaluator, \
             patch("src.tier_fog.fog_aggregator.FogAggregator") as strategy:
            deploy_server.server_fn(SimpleNamespace(run_config={}))
            evaluator.assert_not_called()
            self.assertEqual(data.call_count, 1)
            self.assertEqual(data.call_args.args[6], "val")
            self.assertIsNone(strategy.call_args.kwargs["evaluate_fn"])

    def test_both_cloud_strategies_persist_auc_including_undefined_values(self):
        for strategy_class in (DeployCloud, SimCloud):
            for metrics, expected in (({"accuracy": .8, "macro_f1": .7, "auc": .9}, .9),
                                      ({"accuracy": .8, "macro_f1": .7, "auc_defined": False}, None)):
                with self.subTest(strategy=strategy_class, auc=expected), TemporaryDirectory() as directory, working_directory(directory):
                    strategy = strategy_class.__new__(strategy_class)
                    strategy.logger, strategy.log_prefix = LOGGER, "CLOUD"
                    strategy.snapshot_interval, strategy.snapshot_rounds = 0, []
                    strategy.quantization_bits = 32
                    strategy.global_model = nn.Linear(2, 2)
                    strategy.experiment_name = "test"
                    strategy.results_dict = {"metadata": {}, "performance": []}
                    with patch.object(FedAvg, "evaluate", return_value=(.1, metrics)):
                        strategy.evaluate(1, encode_model(strategy.global_model, 32))
                    record = json.loads(Path("results/test/test.json").read_text())
                    self.assertEqual(record["performance"][0]["global_auc"], expected)
                    self.assertIsNone(record["performance"][0]["global_auc_details"])

    def test_both_cloud_strategies_persist_real_auc_and_coverage(self):
        for strategy_class in (DeployCloud, SimCloud):
            for labels, expected, status, counts in (([0, 2, 0], .5, "defined", [2, 0, 1]),
                                                     ([0, 0, 0], None, "insufficient_test_classes", [3, 0, 0])):
                with self.subTest(strategy=strategy_class, status=status), TemporaryDirectory() as directory, working_directory(directory):
                    model = nn.Linear(2, 3)
                    with torch.no_grad():
                        model.weight.zero_()
                        model.bias.zero_()
                    evaluator = GlobalEvaluator.__new__(GlobalEvaluator)
                    evaluator.model, evaluator.num_classes = copy.deepcopy(model), 3
                    evaluator.run_metadata = {"quantization_bits": 32}
                    evaluator.device, evaluator.tier, evaluator.logger = "cpu", "cloud", LOGGER
                    evaluator.criterion = nn.CrossEntropyLoss()
                    evaluator.test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
                        torch.zeros(3, 2), torch.tensor(labels)), batch_size=2)
                    strategy = strategy_class.__new__(strategy_class)
                    FedAvg.__init__(strategy, evaluate_fn=evaluator.evaluate)
                    strategy.logger, strategy.log_prefix = LOGGER, "CLOUD"
                    strategy.snapshot_interval, strategy.snapshot_rounds = 0, []
                    strategy.quantization_bits, strategy.global_model = 32, model
                    strategy.experiment_name = "test"
                    strategy.results_dict = {"metadata": {}, "performance": []}
                    for round_num in (0, 1):
                        strategy.evaluate(round_num, encode_model(model, 32))
                    record = json.loads(Path("results/test/test.json").read_text())
                    self.assertEqual(len(record["performance"]), 2)
                    for row in record["performance"]:
                        self.assertEqual(row["global_auc"], expected)
                        self.assertEqual(row["global_auc_defined"], expected is not None)
                        details = row["global_auc_details"]
                        self.assertEqual(details["status"], status)
                        self.assertEqual(details["class_counts"], counts)
                        self.assertEqual(details["evaluated_classes"], [0, 2] if expected is not None else [])
                        self.assertEqual(details["per_class_auc"], [.5, None, .5] if expected is not None else [None] * 3)

    def test_shared_region_processes_real_shap_and_records_correct_security_outcomes(self):
        for simulated, bits, fallback in ((s, b, f) for s in (False, True) for b in (32, 8) for f in (False, True)):
            score_context = (patch("src.tier_fog.shap_verifier.compute_shap_stability", return_value=-1.)
                             if fallback else nullcontext())
            with self.subTest(simulated=simulated, bits=bits, fallback=fallback), TemporaryDirectory() as directory, working_directory(directory), \
                 patch.dict(os.environ, {"ZTA_INSECURE_MODE": "false"}), score_context:
                root = Path(directory) / "tpm"
                ledger, truth = {}, {}
                for edge, role in ((1, "benign"), (2, "label_flip")):
                    folder = root / f"edge_1_{edge}"
                    folder.mkdir(parents=True)
                    (folder / "tpm_id.txt").write_text(f"identity-{edge}")
                    ledger[f"identity-{edge}"] = "pcr"
                    truth[f"[EDGE 1_{edge}]"] = {"role": role, "attestation_valid": True}
                (root / "pcr_ledger.json").write_text(json.dumps(ledger))
                metadata = {"experiment_name": "test", "tpm_freshness_window": 60,
                            "experiment_ground_truth": truth, "quantization_bits": bits}
                gatekeeper = ZeroTrustGatekeeper(LOGGER, "FOG", metadata, fog_num=1)
                gatekeeper.tpm_state_root = str(root)
                gatekeeper.tpm_engine.verify_attestation_token = Mock(return_value=True)
                model = nn.Linear(2, 2)
                with torch.no_grad():
                    model.weight.zero_()
                    model.bias.copy_(torch.tensor([1., 0.]))
                parameters = encode_model(model, bits)
                reference = copy.deepcopy(model)
                if fallback:
                    with torch.no_grad(): reference.weight.fill_(2.)
                proxies = []
                for edge in (1, 2):
                    metrics = {"log_prefix": f"[EDGE 1_{edge}]", "role": "benign",
                               "tpm_token_json": json.dumps({"IDi": f"identity-{edge}", "status": "attested"})}
                    proxy = SimpleNamespace(cid=f"client-{edge}", get_properties=Mock(return_value=SimpleNamespace(
                        status=SimpleNamespace(code=Code.OK), properties=metrics)))
                    proxies.append((proxy, metrics))
                nonces = {}
                configured = configure_edge_fit([(proxy, FitIns(parameters, {})) for proxy, _ in proxies],
                                                [gatekeeper], 1, nonces)
                self.assertTrue(all(not ins.config["attestation_only"] for _, ins in configured))
                malformed = ndarrays_to_parameters(parameters_to_ndarrays(parameters) + [np.zeros(1)])
                responses = [(proxy, SimpleNamespace(metrics=dict(metrics), num_examples=4,
                             parameters=parameters if i == 0 else malformed))
                             for i, (proxy, metrics) in enumerate(proxies)]
                history = {"metadata": {}, "rounds": {}}
                result, count, metrics = process_region(
                    gatekeeper=gatekeeper, reference=reference,
                    val_data=(torch.zeros(4, 2), torch.zeros(4, dtype=torch.long)),
                    results=responses, nonces=nonces, expected_clients=nonces,
                    metadata=metadata, fog_num=1, round_num=1, history=history,
                    n_explain=2, num_classes=2, simulated=simulated,
                )
                self.assertIsNotNone(result)
                self.assertEqual(count, 4)
                self.assertEqual(metrics["aggregation_status"], "aggregated")
                record = history["rounds"][1]
                self.assertEqual(record["shap_filter"]["valid_population_size"], 1)
                self.assertEqual(record["shap_filter"]["uniform_fallback"], fallback)
                self.assertNotIn("identity-2", record["raw_shap_scores"])
                self.assertEqual(record["node_ground_truths"]["identity-2"], "label_flip")
                self.assertTrue(gatekeeper.trust_db.is_quarantined("identity-2"))
                rates = calculate_security_rates("results/test")
                self.assertEqual(rates["attestation_gatekeeper"]["valid_attempts"], 4)
                self.assertIsNone(rates["attestation_gatekeeper"]["FAR_percentage"])
                self.assertEqual(rates["update_verification"]["poisoned_acceptance_percentage"], 0)
                if fallback:
                    self.assertEqual(metrics["total_regional_trust"], 1.)
                    self.assertEqual(record["shap_filter"]["raw_weight_total"], 0.)
                    self.assertEqual(validate_region_result(count, bool(result.tensors), metrics), "aggregated")
                    persisted = json.loads(Path("results/test/shap_scores/fog_1_state.json").read_text())
                    self.assertTrue(persisted["rounds"]["1"]["shap_filter"]["uniform_fallback"])
                    regional_model = decode_model(reference, result, bits)
                    torch.testing.assert_close(regional_model.weight, model.weight)
                    # A fallback model reaches Cloud validation and remains subject
                    # to the same rollback threshold as ordinary aggregates.
                    for accuracy, expected in ((1., "aggregated"), (.8, "rolled_back")):
                        cloud = GlobalModelState()
                        cloud.global_model = copy.deepcopy(reference)
                        cloud.quantization_bits = bits
                        cloud._broadcast_parameters = encode_model(reference, bits)
                        cloud.previous_val_acc = 1.
                        cloud.run_metadata = {"rollback_threshold": .8}
                        cloud.logger = LOGGER
                        cloud.validation_accuracy = Mock(return_value=accuracy)
                        previous = cloud._broadcast_parameters
                        broadcast, outcome = cloud.aggregate_regions(
                            [regional_model], [metrics["total_regional_trust"]], 1)
                        self.assertEqual(outcome["aggregation_status"], expected)
                        cloud.validation_accuracy.assert_called_once()
                        if expected == "rolled_back":
                            self.assertIs(broadcast, previous)
                        else:
                            torch.testing.assert_close(cloud.global_model.weight, model.weight)

    def test_preserved_cnn_lstm_supports_singleton_evaluation_and_real_attribution(self):
        torch.manual_seed(42)
        model = CNNLSTMClassifier().eval()
        X = torch.rand(4, 40)
        self.assertEqual(model(X[:1]).shape, (1, 15))
        score = compute_shap_stability(model, copy.deepcopy(model), X,
                                       torch.zeros(4, dtype=torch.long), n_explain=2)
        self.assertEqual(score, 1)


if __name__ == "__main__":
    unittest.main()
