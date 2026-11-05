"""Two in-process rounds through real node computation; TPM/RPC are test doubles."""
from contextlib import ExitStack
import copy
import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch.utils.data import DataLoader, TensorDataset
from flwr.common import Code, FitIns, ndarrays_to_parameters, parameters_to_ndarrays

from src.entrypoints.sim_client import Client
from src.shared.models.cnn_lstm import CNNLSTMClassifier
from src.shared.network.model_payload import decode_model, encode_model, load_model_arrays
from src.tier_cloud.model_state import GlobalModelState
from src.tier_fog.client_instructions import configure_edge_fit
from src.tier_fog.gatekeeper import ZeroTrustGatekeeper
from src.tier_fog.round_processing import process_region
from scripts.tests.test_pipeline_integration import working_directory


torch.set_num_threads(1)
LOGGER = logging.getLogger('node-flow-tests')
LOGGER.addHandler(logging.NullHandler())
LOGGER.propagate = False


class NodeFlowTests(unittest.TestCase):
    def test_two_rounds_train_encode_verify_aggregate_and_broadcast(self):
        self.run_two_rounds('label_flip')

    def test_shap_constrained_client_reaches_fog_cloud_and_persisted_telemetry(self):
        self.run_two_rounds('shap_aware')

    def run_two_rounds(self, second_role):
        # Keep validation predictions stable so this verifies a successful path,
        # not poisoning effectiveness. Separate tests cover rejection/rollback.
        with TemporaryDirectory() as directory, working_directory(directory), ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {'ZTA_INSECURE_MODE': 'false', 'ZTA_TIER': 'test'}))
            torch.manual_seed(42)
            reference = CNNLSTMClassifier(n_classes=2, dropout=0.)
            calibration = torch.randn(4, 40)
            # Calibrate BN on the fixture's repeated training batch. Uncalibrated
            # BN drift can yield negative stability and uniform fallback.
            with torch.no_grad():
                for _ in range(100): reference(calibration)
            reference.eval()
            with torch.no_grad(): reference.fc3.bias.copy_(torch.tensor([2., 0.]))
            X_val, y_val = torch.randn(16, 40), torch.zeros(16, dtype=torch.long)
            cloud = GlobalModelState()
            cloud.logger = LOGGER
            cloud.quantization_bits = 8
            cloud.run_metadata = {'rollback_threshold': .8}
            cloud.val_data = X_val, y_val
            cloud.global_model = copy.deepcopy(reference)
            cloud.sync_global_state(encode_model(reference, 8))
            cloud.previous_val_acc = cloud.validation_accuracy(cloud.global_model)
            self.assertEqual(cloud.previous_val_acc, 1.)
            initial = copy.deepcopy(cloud.global_model.state_dict())
            metadata = {'experiment_name': 'node-flow', 'random_seed': 42,
                        'quantization_bits': 8, 'tpm_freshness_window': 60,
                        'experiment_ground_truth': {
                            '[EDGE 1_1]': {'role': 'benign', 'attestation_valid': True},
                            '[EDGE 2_1]': {'role': second_role, 'attestation_valid': True}}}
            root = Path(directory) / 'tpm'
            root.mkdir()
            (root / 'pcr_ledger.json').write_text(json.dumps({'id-1': 'pcr', 'id-2': 'pcr'}))
            clients, proxies, keepers, clean_labels = [], [], [], []
            for fog, role in ((1, 'benign'), (2, second_role)):
                folder = root / f'edge_{fog}_1'
                folder.mkdir()
                (folder / 'tpm_id.txt').write_text(f'id-{fog}')
                keeper = ZeroTrustGatekeeper(LOGGER, f'FOG {fog}', metadata, fog_num=fog)
                keeper.tpm_state_root = str(root)
                # Only external quote verification is substituted. Challenge,
                # identity, enrollment, quarantine and admission logic are real.
                keeper.tpm_engine.verify_attestation_token = Mock(
                    side_effect=lambda **kw: kw['token']['nonce'] == kw['expected_nonce'])
                keepers.append(keeper)
                inputs = torch.cat([calibration, calibration, calibration[:1]])
                labels = torch.zeros(9, dtype=torch.long)
                loader = DataLoader(TensorDataset(inputs, labels), batch_size=4, shuffle=True)
                config = dict(random_seed=42, client_index=fog-1, learning_rate=.0001,
                              role=role, local_epochs=2, num_classes=2, p_flip=.2, clip_norm=1.,
                              quantization_bits=8, edge_inference_int8=True,
                              adv_ratio=.3, eps=.1, alpha=.01, n_iter=2,
                              robustness_eval_attack='pgd', clip_min=None, clip_max=None,
                              fog_num=fog, shap_aware_base_attack='label_flip', shap_aware_tau=.01,
                              shap_aware_constrained=True, shap_explain_count=3, shap_aware_max_backtracks=3)
                client = Client(LOGGER, f'[EDGE {fog}_1]', fog, 1, copy.deepcopy(reference),
                                loader, 'cpu', {}, config,
                                eval_loader=DataLoader(TensorDataset(X_val, y_val), batch_size=7),
                                shap_data=(X_val, y_val) if role == 'shap_aware' else None)
                stack.enter_context(patch.object(client.trainer.tpm_engine, 'generate_attestation_token',
                    side_effect=lambda nonce, prefix, round_num, identity=f'id-{fog}':
                    {'IDi': identity, 'status': 'attested', 'nonce': nonce}))
                # RPC proxy executes the real Client.get_properties method locally.
                proxy = SimpleNamespace(cid=f'client-{fog}')
                proxy.get_properties = Mock(side_effect=lambda ins, client=client, **kw:
                    SimpleNamespace(status=SimpleNamespace(code=Code.OK), properties=client.get_properties(ins.config)))
                clients.append(client)
                proxies.append(proxy)
                clean_labels.append(labels.clone())
            histories = [{'metadata': {}, 'rounds': {}} for _ in keepers]
            for round_num in (1, 2):
                nonces = {}
                instructions = configure_edge_fit(
                    [(p, FitIns(cloud._broadcast_parameters, {})) for p in proxies], keepers, round_num, nonces)
                regional_models, regional_weights = [], []
                for i, (proxy, instruction) in enumerate(instructions):
                    self.assertFalse(instruction.config['attestation_only'])
                    before_fit = decode_model(reference, instruction.parameters, 8)
                    arrays, count, metrics = clients[i].fit(parameters_to_ndarrays(instruction.parameters), instruction.config)
                    self.assertEqual(count, 9)
                    self.assertNotIn('status', metrics)  # A client crash must fail this test.
                    self.assertEqual(metrics['edge_inference_samples'], 16)
                    self.assertTrue(metrics['edge_inference_int8'])
                    self.assertGreater(metrics['latency_adv_training_sec'], 0)
                    self.assertLess(metrics['edge_int8_state_bytes'], metrics['edge_fp32_state_bytes'])
                    is_shap = clients[i].train_config['role'] == 'shap_aware'
                    self.assertEqual(int(clients[i].model.bn1.num_batches_tracked),
                                     int(before_fit.bn1.num_batches_tracked) + (0 if is_shap else 4))
                    if is_shap:
                        self.assertTrue(metrics['shap_attack_feasible'])
                        self.assertLess(metrics['shap_attack_distance'], .01)
                    self.assertTrue(torch.equal(clients[i].train_loader.dataset.tensors[1], clean_labels[i]))
                    response = SimpleNamespace(parameters=ndarrays_to_parameters(arrays), num_examples=count, metrics=metrics)
                    with self.assertLogs(LOGGER, level='INFO') as captured:
                        parameters, samples, fog_metrics = process_region(
                            gatekeeper=keepers[i], reference=copy.deepcopy(cloud.global_model),
                            val_data=(X_val, y_val), results=[(proxy, response)], nonces=nonces,
                            expected_clients=[proxy.cid], metadata=metadata, fog_num=i+1, round_num=round_num,
                            history=histories[i], n_explain=3, num_classes=2, simulated=True)
                    from scripts.ops.verify_pipeline import parse_round_state
                    node = f'[FOG {i+1} SERVER]'
                    state = parse_round_state([
                        dict(node=node, message=r.getMessage(), level=r.levelname)
                        for r in captured.records], {})
                    self.assertEqual(state['fog_servers_aggregated'], {node})
                    self.assertEqual(fog_metrics['aggregation_status'], 'aggregated', histories[i]['rounds'][round_num]['raw_shap_scores'])
                    self.assertEqual(samples, 9)
                    self.assertGreater(fog_metrics['total_regional_trust'], 0)
                    regional_models.append(decode_model(reference, parameters, 8))
                    regional_weights.append(fog_metrics['total_regional_trust'])
                    record = histories[i]['rounds'][round_num]
                    self.assertEqual(record['shap_filter']['valid_population_size'], 1)
                    self.assertTrue(record['update_outcomes'][f'id-{i+1}']['accepted'])
                    if is_shap:
                        self.assertEqual(record['node_shap_attack_metrics'][f'id-{i+1}']['shap_attack_distance'],
                                         metrics['shap_attack_distance'])
                    self.assertEqual(record['node_inference_metrics'][f'id-{i+1}']['edge_inference_samples'], 16)
                    self.assertEqual(record['node_update_array_bytes'][f'id-{i+1}'], metrics['update_array_bytes'])
                broadcast, cloud_metrics = cloud.aggregate_regions(regional_models, regional_weights, round_num)
                self.assertEqual(cloud_metrics['aggregation_status'], 'aggregated')
                # Cloud's stored model must equal what an Edge decodes next round.
                decoded = copy.deepcopy(reference)
                load_model_arrays(decoded, parameters_to_ndarrays(broadcast), 8)
                self.assertTrue(all(torch.equal(v, cloud.global_model.state_dict()[k]) for k, v in decoded.state_dict().items()))
            # Quantization can leave an individual tensor unchanged even when
            # aggregation accepted a changed model. Check the trainable state.
            self.assertTrue(any(not torch.equal(initial[name], value)
                                for name, value in cloud.global_model.named_parameters()))
            for fog in (1, 2):
                saved = json.loads(Path(f'results/node-flow/performance_metrics/fog_{fog}_state.json').read_text())
                self.assertEqual(set(saved['rounds']), {'1', '2'})
                self.assertEqual(saved['metadata']['telemetry_mode'], 'simulated')
                self.assertGreater(saved['rounds']['2']['node_update_tensor_bytes'][f'id-{fog}'], 0)
                if fog == 2 and second_role == 'shap_aware':
                    diagnostics = saved['rounds']['2']['node_shap_attack_metrics']['id-2']
                    self.assertTrue(diagnostics['shap_attack_feasible'])
                    self.assertLess(diagnostics['shap_attack_distance'], diagnostics['shap_attack_tau'])


if __name__ == '__main__':
    unittest.main()
