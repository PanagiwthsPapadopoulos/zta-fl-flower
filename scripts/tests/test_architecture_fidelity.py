"""Behavioral regressions for the coordinated PCA/attack/INT8 migration."""
import copy
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from flwr.common import parameters_to_ndarrays

from src.shared.data.artifact_manifest import PIPELINE_VERSION, validate_checkpoint_pipeline
from src.shared.models.cnn_lstm import CNNLSTMClassifier
from src.shared.network.model_payload import encode_model, load_model_arrays
from src.shared.security.adversarial_math import fgsm_attack, pgd_attack, label_flip_attack, local_train_byzantine
from src.shared.security.backdoor_math import poison_partition
from src.shared.utils.config_loader import load_yaml_configs
from src.shared.utils.metrics import compute_shap_attributions
from src.tier_edge.inference import quantize_model, evaluate_cpu_inference, evaluation_loader
from src.tier_edge.local_ids import EdgeTrainer

torch.set_num_threads(1)



class FidelityTests(unittest.TestCase):
    def test_label_replacement_can_keep_original_class_and_leaves_unselected_rows(self):
        labels = torch.tensor([0, 1, 2, 1])
        with patch('torch.rand', return_value=torch.tensor([.2, .8, .3, .1])), \
             patch('torch.randint', return_value=torch.tensor([2, 2, 1])):
            actual = label_flip_attack(labels, 3, .5)
        self.assertEqual(actual.tolist(), [2, 1, 2, 1])
        self.assertEqual(labels.tolist(), [0, 1, 2, 1])
        self.assertTrue(torch.equal(label_flip_attack(labels, 3, 0), labels))
        # Full replacement still allows unchanged labels, including a one-class dataset.
        self.assertTrue(torch.equal(label_flip_attack(torch.zeros(8, dtype=torch.long), 1, 1),
                                    torch.zeros(8, dtype=torch.long)))

    def test_backdoor_samples_all_rows_and_changes_only_trigger_features_and_labels(self):
        X, y = torch.zeros(30, 40), torch.zeros(30, dtype=torch.long)
        actual, labels = poison_partition(X, y, poison_fraction=.1, seed=42)
        changed = (actual != X).any(dim=1)
        # All original labels already equal the target: they must still be eligible.
        self.assertEqual(int(changed.sum()), 3)
        self.assertTrue(torch.equal(actual[changed, -3:], torch.full((3, 3), 1.5)))
        self.assertTrue(torch.equal(actual[:, :-3], X[:, :-3]))
        self.assertTrue(torch.equal(actual[~changed], X[~changed]))
        self.assertTrue(torch.equal(labels, y))
        self.assertTrue(torch.equal(actual, poison_partition(X, y, .1, seed=42)[0]))
        self.assertFalse(torch.equal(actual, poison_partition(X, y, .1, seed=43)[0]))
        # Fractional row budgets round down; poisoned non-target labels become target.
        full, targets = poison_partition(X, torch.ones(30, dtype=torch.long), 1.)
        self.assertTrue(torch.equal(targets, y))
        self.assertTrue(torch.equal(full[:, -3:], torch.full((30, 3), 1.5)))
        tiny, _ = poison_partition(X, y, .01)
        self.assertTrue(torch.equal(tiny, X))
        self.assertTrue(torch.equal(X, torch.zeros_like(X)))

    def test_fgsm_and_pgd_follow_analytic_gradient_and_epsilon_bounds(self):
        model = nn.Linear(2, 2)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1., -1.], [-1., 1.]]))
            model.bias.zero_()
        X, y = torch.tensor([[-3., 2.], [4., -.5]]), torch.tensor([0, 1])
        # For logits [x0-x1, -x0+x1], cross-entropy gradient signs are
        # [-,+] for label 0 and [+,-] for label 1, throughout this domain.
        expected = X + torch.tensor([[-.1, .1], [.1, -.1]])
        torch.testing.assert_close(fgsm_attack(model, X, y, epsilon=.1, adv_ratio=1), expected)
        for seed in (1, 42, 83):
            torch.manual_seed(seed)
            # Eight .03 steps reach the .1 boundary from any allowed random start.
            actual = pgd_attack(model, X, y, eps=.1, alpha=.03, n_iter=8, adv_ratio=1)
            torch.testing.assert_close(actual, expected)
            self.assertLessEqual(float((actual - X).abs().max()), .100001)

    def test_pgd_start_clips_to_attack_batch_range_but_steps_can_leave_it(self):
        model = nn.Linear(2, 2)
        X, y = torch.tensor([[-2., 3.], [3., -2.]]), torch.tensor([0, 1])
        torch.manual_seed(9)
        expected = (X + torch.empty_like(X).uniform_(-.5, .5)).clamp(X.min(), X.max())
        torch.manual_seed(9)
        actual = pgd_attack(model, X, y, eps=.5, alpha=0, n_iter=1, adv_ratio=1)
        self.assertTrue(torch.equal(actual, expected))
        # A fixed classifier gives a positive gradient for target class 1.
        # One full-budget step must exceed the initial batch maximum.
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1., 1.], [-1., -1.]]))
            model.bias.zero_()
        targets = torch.ones(2, dtype=torch.long)
        stepped = pgd_attack(model, X, targets, eps=.5, alpha=.5, n_iter=1, adv_ratio=1)
        self.assertGreater(float(stepped.max()), float(X.max()))
        self.assertLessEqual(float((stepped - X).abs().max()), .500001)

    def test_attack_generation_preserves_buffers_modes_and_parameter_gradients(self):
        model = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3), nn.Dropout(.5), nn.Linear(3, 2)).train()
        model[2].eval()
        for p in model.parameters(): p.grad = torch.ones_like(p)
        state = copy.deepcopy(model.state_dict())
        modes = [m.training for m in model.modules()]
        for attack in (fgsm_attack, pgd_attack):
            attack(model, torch.randn(4, 2), torch.tensor([0, 1, 0, 1]), adv_ratio=1)
            self.assertEqual(modes, [m.training for m in model.modules()])
            self.assertTrue(all(torch.equal(state[k], v) for k, v in model.state_dict().items()))
            self.assertTrue(all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters()))

    def test_label_flip_epoch_wiring_uses_selected_attack_and_freezes_labels(self):
        loader = DataLoader(TensorDataset(torch.randn(10, 2), torch.zeros(10, dtype=torch.long)), batch_size=5)
        trainer = EdgeTrainer(logging.getLogger('test'), 'edge', nn.Linear(2, 2), loader, 'cpu',
            dict(random_seed=42, client_index=0, learning_rate=.01, role='label_flip', local_epochs=2,
                 num_classes=2, p_flip=1., clip_norm=1., quantization_bits=8,
                 robustness_eval_attack='fgsm', adv_ratio=.4, eps=.07, alpha=.02,
                 n_iter=4, clip_min=None, clip_max=None), {})
        seen = []
        def train(model, active, optimizer, **kwargs):
            self.assertFalse(kwargs['use_pgd'])
            self.assertEqual(kwargs['adv_ratio'], .4)
            self.assertEqual((kwargs['eps'], kwargs['alpha'], kwargs['n_iter']), (.07, .02, 4))
            seen.append(active.dataset.tensors[1].clone())
            return .1
        with patch('src.shared.security.adversarial_math.adversarial_train_epoch', side_effect=train), \
             patch.object(trainer, 'attestation_metrics', return_value={}), \
             patch('src.tier_edge.local_ids.label_flip_loader', wraps=__import__(
                 'src.shared.security.adversarial_math', fromlist=['label_flip_loader']).label_flip_loader) as flip:
            trainer.execute_training(trainer.get_parameters(), 1, {})
            self.assertEqual(flip.call_count, 1)
        self.assertTrue(torch.equal(seen[0], seen[1]))
        self.assertTrue(torch.equal(loader.dataset.tensors[1], torch.zeros(10, dtype=torch.long)))

    def test_quantized_copy_has_int8_layers_and_preserves_training_model(self):
        torch.manual_seed(42)
        model = CNNLSTMClassifier().train()
        state = copy.deepcopy(model.state_dict())
        quantized = quantize_model(model)
        reference = copy.deepcopy(model).eval()
        self.assertIsInstance(quantized.lstm, torch.ao.nn.quantized.dynamic.LSTM)
        self.assertIsInstance(quantized.fc1, torch.ao.nn.quantized.dynamic.Linear)
        self.assertIsInstance(quantized.conv1, nn.Conv1d)
        for shape in [(3, 40), (3, 40, 8)]:
            x = torch.randn(*shape)
            with torch.no_grad():
                actual, expected = quantized(x), reference(x)
            self.assertEqual(actual.shape, (3, model.n_classes))
            self.assertTrue(torch.isfinite(actual).all())
            # Numerical sanity for this seeded fixture, not a universal accuracy bound.
            torch.testing.assert_close(actual, expected, rtol=.05, atol=.01)
        self.assertTrue(model.training)
        self.assertTrue(all(torch.equal(state[k], v) for k, v in model.state_dict().items()))
        model(torch.randn(2, 40)).sum().backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_eight_bit_flower_payload_roundtrip_and_error_bound(self):
        model = CNNLSTMClassifier().eval()
        arrays = parameters_to_ndarrays(encode_model(model, 8))
        decoded = copy.deepcopy(model)
        load_model_arrays(decoded, arrays, 8)
        self.assertTrue(all(a.dtype == np.uint8 for a in arrays[::2]))
        for original, restored, meta in zip(model.state_dict().values(), decoded.state_dict().values(), arrays[1::2]):
            if original.is_floating_point():
                self.assertLessEqual(float((original - restored).abs().max()), float(meta[1]) / 2 + 1e-6)
            else: self.assertTrue(torch.equal(original, restored))
        self.assertLess(sum(a.nbytes for a in arrays), sum(v.numel()*v.element_size() for v in model.state_dict().values()))

    def test_inference_reports_real_samples_size_and_latency_including_singleton(self):
        model = CNNLSTMClassifier()
        loader = DataLoader(TensorDataset(torch.randn(5, 40), torch.zeros(5, dtype=torch.long)), batch_size=2)
        loss, count, metrics = evaluate_cpu_inference(model, loader)
        self.assertEqual(count, 5)
        self.assertTrue(np.isfinite(loss))
        self.assertLess(metrics['edge_int8_state_bytes'], metrics['edge_fp32_state_bytes'])
        self.assertGreater(metrics['edge_int8_inference_ms_per_sample'], 0)
        self.assertTrue(model.training)

    def test_logit_bias_blind_spot_exists_before_attribution_reduction(self):
        model = nn.Linear(2, 2)
        changed = copy.deepcopy(model)
        with torch.no_grad(): changed.bias[0] += 100
        X, y = torch.randn(10, 2), torch.zeros(10, dtype=torch.long)
        self.assertTrue(torch.equal(compute_shap_attributions(model, X, y), compute_shap_attributions(changed, X, y)))
        self.assertFalse(torch.equal(model(X), changed(X)))

    def test_builder_fits_train_only_and_preserves_unscaled_pca_coordinates(self):
        from scripts.setup.build_artifacts import build_artifacts
        from src.shared.data.data_loader import MinMaxScaler, PCA
        from sklearn.model_selection import train_test_split
        config = load_yaml_configs()
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'raw' / 'data.csv'
            source.parent.mkdir()
            rng = np.random.default_rng(8)
            raw = rng.normal(size=(100, 42)).astype(np.float32)
            labels = np.arange(100) % 2
            source.write_text(','.join([f'f{i}' for i in range(42)] + ['label']) + '\n' +
                              '\n'.join(','.join(map(str, row)) + ',' + ('Normal' if y == 0 else 'DoS_TCP')
                                        for row, y in zip(raw, labels)))
            config.update(dataset_path=str(source), apply_smote=False, dataset_fraction=1.,
                          custom_fog_topology='[1]', n_classes_per=2)
            with patch('scripts.setup.build_artifacts.load_yaml_configs', return_value=config):
                build_artifacts()
            train, holdout, _, _ = train_test_split(raw, labels, test_size=.3, stratify=labels, random_state=42)
            scaler = MinMaxScaler().fit(train)
            pca = PCA(n_components=40).fit(scaler.transform(train))
            expected = pca.transform(scaler.transform(train)).astype(np.float32)
            path = Path(directory) / 'artifacts/edge_iiotset_leakage_False_smote_False_train.pt'
            actual, _, _ = torch.load(path, weights_only=True)
            np.testing.assert_allclose(actual.numpy(), expected, atol=1e-5)
            self.assertLess(float(actual.min()), 0)

    def test_legacy_artifact_manifest_is_rejected(self):
        import json
        from src.shared.data.artifact_manifest import validate_artifact, write_manifest
        with TemporaryDirectory() as directory:
            artifact, manifest = Path(directory) / 'x.pt', Path(directory) / 'manifest.json'
            artifact.write_bytes(b'old coordinates')
            write_manifest(manifest, {'pipeline': 'v2'}, {}, [artifact], {})
            with self.assertRaisesRegex(ValueError, 'provenance changed'):
                validate_artifact(artifact, manifest, {'pipeline': PIPELINE_VERSION})

    def test_gradient_manipulation_keeps_trained_buffers_when_scale_is_zero(self):
        model = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3), nn.Linear(3, 2))
        reference = copy.deepcopy(model)
        loader = DataLoader(TensorDataset(torch.ones(8, 2) * 4, torch.zeros(8, dtype=torch.long)), batch_size=4)
        local_train_byzantine(model, loader, 'gradient_manip', 2, scale=0, global_model=reference)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(model.parameters(), reference.parameters())))
        self.assertFalse(torch.equal(model[1].running_mean, reference[1].running_mean))

    def test_client_evaluate_paths_delegate_to_edge_trainer(self):
        from src.entrypoints.sim_client import Client as SimClient
        from src.entrypoints.deploy_client import Client as DeployClient
        from unittest.mock import Mock
        for client_type in (SimClient, DeployClient):
            client = client_type.__new__(client_type)
            client.trainer = Mock()
            client.trainer.evaluate.return_value = (.3, 5, {'accuracy': .6})
            self.assertEqual(client.evaluate(['parameters'], {}), (.3, 5, {'accuracy': .6}))
            client.trainer.evaluate.assert_called_once_with(['parameters'])

    def test_current_defaults_and_checkpoint_invalidation(self):
        config = load_yaml_configs()
        self.assertEqual(config['quantization_bits'], 8)
        self.assertTrue(config['edge_inference_int8'])
        self.assertIsNone(config['clip_min'])
        self.assertIsNone(config['clip_max'])
        self.assertEqual(config['backdoor_poison_fraction'], .1)
        validate_checkpoint_pipeline({'pipeline_version': PIPELINE_VERSION})
        for metadata in ({}, {'pipeline_version': 'whole-dataset-splits-minmax-pca40-minmax-validated-smote-v2'}):
            with self.assertRaisesRegex(ValueError, 'restart'): validate_checkpoint_pipeline(metadata)


if __name__ == '__main__': unittest.main()
