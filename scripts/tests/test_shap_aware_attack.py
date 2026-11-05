"""Constrained poisoning: actual GradientSHAP, wire feasibility and paired control."""
import copy
import logging
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from flwr.common import parameters_to_ndarrays

from src.shared.models.cnn_lstm import CNNLSTMClassifier
from src.shared.network.model_payload import encode_model, load_model_arrays
from src.shared.security.shap_aware import attack_validation_data, train_shap_constrained
from src.shared.utils.metrics import compute_shap_attributions
from src.shared.utils.reproducibility import derive_seed
from src.tier_edge.local_ids import EdgeTrainer
from scripts.tests import test_shap_background as background_tests

torch.set_num_threads(1)


class ShapAttackTests(unittest.TestCase):
    def fixture(self, bits=32):
        reference = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            reference.weight.copy_(torch.tensor([[1., 0.], [0., 0.]]))
        arrays = parameters_to_ndarrays(encode_model(reference, bits))
        load_model_arrays(reference, arrays, bits)
        model = copy.deepcopy(reference)
        loader = DataLoader(TensorDataset(torch.tensor([[1., 0.]] * 4), torch.zeros(4, dtype=torch.long)),
                            batch_size=4)
        data = (torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]]), torch.zeros(4, dtype=torch.long))
        return reference, model, loader, data, arrays

    def run_attack(self, constrained=True, tau=.005, bits=32, **overrides):
        reference, model, loader, data, arrays = self.fixture(bits)
        kwargs = dict(attack='label_flip', n_classes=2, constrained=constrained, tau=tau,
                      epochs=2, lr=.5, clip_norm=1., n_explain=4, seed=42,
                      quantization_bits=bits, max_backtracks=10, reference_arrays=arrays)
        kwargs.update(overrides)
        # Fix poisoned targets to make the objective analytically inspectable;
        # all optimization, attribution, encoding and feasibility checks are real.
        with patch('src.shared.security.shap_aware.label_flip_attack', return_value=torch.ones(4, dtype=torch.long)):
            loss, payload, stats = train_shap_constrained(model, reference, loader, data, **kwargs)
        return reference, model, data, payload, stats

    def distance(self, candidate, reference, data, seed=42):
        vectors = []
        for model in (candidate, reference):
            attrs = compute_shap_attributions(model, *data, 4, 'cpu', seed)
            vectors.append(attrs.abs().mean(0))
        return float((vectors[0] - vectors[1]).norm())

    def test_real_constrained_optimization_improves_loss_and_checks_transmitted_model(self):
        for bits in (8, 16, 32):
            with self.subTest(bits=bits):
                ref, model, data, payload, stats = self.run_attack(bits=bits)
                wire = copy.deepcopy(ref)
                load_model_arrays(wire, payload, bits)
                distance = self.distance(wire, ref, data)
                self.assertLess(distance, .005)
                self.assertAlmostEqual(stats['shap_attack_distance'], distance)
                self.assertTrue(stats['shap_attack_feasible'])
                self.assertLess(stats['shap_attack_final_loss'], stats['shap_attack_initial_loss'])
                self.assertGreater(stats['shap_attack_accepted_steps'], 0)
                self.assertGreater(stats['shap_attack_candidate_evaluations'], stats['shap_attack_steps'])
                torch.testing.assert_close(model.weight, wire.weight, rtol=0, atol=0)

    def test_unconstrained_control_uses_same_objective_seed_and_gradient_budget(self):
        _, constrained, _, _, c = self.run_attack()
        ref, unconstrained, data, _, u = self.run_attack(constrained=False, tau=None)
        for key in ('objective', 'seed', 'epochs', 'steps', 'initial_loss', 'learning_rate'):
            self.assertEqual(c['shap_attack_' + key], u['shap_attack_' + key])
        self.assertGreater(self.distance(unconstrained, ref, data), .005)
        self.assertLess(u['shap_attack_final_loss'], c['shap_attack_final_loss'])
        self.assertNotIn('shap_attack_feasible', u)
        self.assertFalse(torch.equal(constrained.weight, unconstrained.weight))

    def test_exhausted_budget_returns_exact_reference_and_reports_no_progress(self):
        ref, model, data, payload, stats = self.run_attack(tau=1e-12, max_backtracks=0, bits=8)
        expected = parameters_to_ndarrays(encode_model(ref, 8))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(payload, expected)))
        self.assertEqual(stats['shap_attack_status'], 'no_progress')
        self.assertEqual(stats['shap_attack_rejected_steps'], stats['shap_attack_steps'])
        self.assertEqual(stats['shap_attack_distance'], 0)
        self.assertTrue(stats['shap_attack_feasible'])

    def test_explicit_positive_tau_is_required_and_invalid_objective_fails(self):
        for value in (None, 0, -1, float('nan'), float('inf')):
            with self.subTest(tau=value), self.assertRaisesRegex(ValueError, 'shap_aware_tau'):
                self.run_attack(tau=value)
        with self.assertRaisesRegex(ValueError, 'objective'):
            self.run_attack(attack='unknown')

    def test_strict_constraint_rejects_equality_and_final_check_cannot_be_bypassed(self):
        # Control attribution distances to isolate the strict boundary. This
        # does not substitute the loss/gradient or feasibility decision itself.
        ref, model, loader, data, arrays = self.fixture()
        kwargs = dict(attack='label_flip', n_classes=2, constrained=True, tau=1.,
                      epochs=1, lr=.5, clip_norm=1., n_explain=4, seed=42,
                      quantization_bits=32, max_backtracks=0, reference_arrays=arrays)
        zero = torch.zeros(4, 2)
        boundary = zero.clone()
        boundary[:, 0] = 1.
        with patch('src.shared.security.shap_aware.label_flip_attack', return_value=torch.ones(4, dtype=torch.long)), \
             patch('src.shared.security.shap_aware.compute_shap_attributions', side_effect=[zero, boundary, zero]):
            _, _, stats = train_shap_constrained(model, ref, loader, data, **kwargs)
        self.assertEqual(stats['shap_attack_accepted_steps'], 0)
        with patch('src.shared.security.shap_aware.label_flip_attack', return_value=torch.ones(4, dtype=torch.long)), \
             patch('src.shared.security.shap_aware.compute_shap_attributions', side_effect=[zero, zero, boundary]), \
             self.assertRaisesRegex(ValueError, 'Final encoded'):
            train_shap_constrained(model, ref, loader, data, **kwargs)

    def test_untargeted_objective_increases_clean_loss(self):
        _, _, _, _, stats = self.run_attack(constrained=False, tau=None, attack='gradient_manipulation')
        self.assertEqual(stats['shap_attack_objective'], 'negative_clean_ce')
        self.assertLess(stats['shap_attack_final_loss'], stats['shap_attack_initial_loss'])

    def test_cnn_lstm_preserves_reference_and_buffers_and_handles_singletons(self):
        torch.manual_seed(4)
        reference = CNNLSTMClassifier(n_classes=2, dropout=0.)
        original = copy.deepcopy(reference.state_dict())
        model = copy.deepcopy(reference)
        data = torch.randn(3, 40), torch.zeros(3, dtype=torch.long)
        loader = DataLoader(TensorDataset(*data), batch_size=2)
        _, _, stats = train_shap_constrained(
            model, reference, loader, data, attack='gradient_manipulation', n_classes=2,
            constrained=True, tau=1., epochs=1, lr=.01, clip_norm=1., n_explain=2,
            seed=42, quantization_bits=32, max_backtracks=2)
        self.assertEqual(stats['shap_attack_steps'], 2)
        self.assertTrue(reference.training)
        self.assertTrue(model.training)
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        self.assertTrue(stats['shap_attack_feasible'])

    def test_attacker_validation_sample_matches_both_fog_adapters(self):
        existing = background_tests.BackgroundTests()
        X, y = torch.arange(250 * 40).reshape(250, 40).float(), torch.arange(250) % 15
        for fraction in (1., .1):
            config = dict(existing.config(), dataset_fraction=fraction)
            simulation = existing.simulation_data(config, X, y)
            for fog in (1, 2):
                with patch('src.shared.security.shap_aware.get_dataset', return_value=(X, y, 15)) as get_data:
                    actual = attack_validation_data(config, fog)
                expected = existing.deployment_data(dict(config, fog_id=f'fog_{fog}'), X, y)
                self.assertEqual(get_data.call_args.args[6], 'val')
                for a, b, c in zip(actual, expected, simulation['fog_val_data'][fog]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                    torch.testing.assert_close(a, c, rtol=0, atol=0)

    def test_trainer_returns_checked_payload_and_freezes_poison_labels_across_epochs(self):
        reference, model, loader, data, incoming = self.fixture(bits=8)
        config = dict(role='shap_aware', random_seed=42, client_index=0, fog_num=1,
                      learning_rate=.1, local_epochs=2, num_classes=2, p_flip=1.,
                      clip_norm=1., quantization_bits=8, shap_explain_count=4,
                      shap_aware_base_attack='label_flip', shap_aware_tau=.01,
                      shap_aware_constrained=True, shap_aware_max_backtracks=10)
        trainer = EdgeTrainer(logging.getLogger('shap-test'), 'edge', model, loader, 'cpu', config, {}, shap_data=data)
        with patch.object(trainer, 'attestation_metrics', return_value={}), \
             patch('src.shared.security.shap_aware.label_flip_attack', return_value=torch.ones(4, dtype=torch.long)) as labels:
            payload, count, stats = trainer.execute_training(incoming, 3, {})
        labels.assert_called_once()
        self.assertEqual(count, 4)
        self.assertEqual(stats['shap_attack_steps'], 2)
        load_model_arrays(model, payload, 8)
        distance = self.distance(model, reference, data, derive_seed(42, 'shap', 1, 3))
        self.assertAlmostEqual(stats['shap_attack_distance'], distance)
        self.assertLess(distance, .01)


if __name__ == '__main__':
    unittest.main()
