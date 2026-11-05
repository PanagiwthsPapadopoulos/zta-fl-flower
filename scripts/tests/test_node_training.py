"""Real CPU training and deterministic placement; no federation or TPM required."""
from collections import Counter
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.shared.models.cnn_lstm import CNNLSTMClassifier
from src.shared.security.adversarial_math import adversarial_train_epoch
from src.shared.security.threat_profiler import assign_edge_roles, experiment_ground_truth


torch.set_num_threads(1)


class LocalTrainingTests(unittest.TestCase):
    def test_batchnorm_observes_combined_batch_once(self):
        inputs = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 10
        labels = torch.arange(8) % 2
        for use_pgd in (False, True):
            for ratio in (0., .125, .3, 1.):
                with self.subTest(use_pgd=use_pgd, ratio=ratio):
                    torch.manual_seed(42)
                    model = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 2))
                    loader = DataLoader(TensorDataset(inputs, labels), batch_size=8)
                    combined = inputs.clone()
                    combined[:int(8 * ratio)] += 2
                    # Controlled attack output isolates the training pass and BN policy.
                    name = 'pgd_attack' if use_pgd else 'fgsm_attack'
                    with patch(f'src.shared.security.adversarial_math.{name}', return_value=combined):
                        adversarial_train_epoch(model, loader, torch.optim.Adam(model.parameters()),
                                                adv_ratio=ratio, use_pgd=use_pgd)
                    torch.testing.assert_close(model[0].running_mean, combined.mean(0) * .1)
                    torch.testing.assert_close(model[0].running_var, .9 + combined.var(0, unbiased=True) * .1)
                    self.assertEqual(int(model[0].num_batches_tracked), 1)

    def test_real_cnn_lstm_training_updates_weights_without_extra_bn_steps(self):
        for use_pgd in (False, True):
            for ratio in (0., .25, .3, 1.):
                with self.subTest(use_pgd=use_pgd, ratio=ratio):
                    torch.manual_seed(42)
                    model = CNNLSTMClassifier()
                    before = copy.deepcopy(model.state_dict())
                    loader = DataLoader(TensorDataset(torch.randn(9, 40), torch.arange(9) % 15), batch_size=4)
                    loss = adversarial_train_epoch(model, loader, torch.optim.Adam(model.parameters()),
                                                    adv_ratio=ratio, eps=.1, alpha=.03, n_iter=2,
                                                    use_pgd=use_pgd)
                    self.assertTrue(torch.isfinite(torch.tensor(loss)))
                    self.assertFalse(torch.equal(before['fc3.weight'], model.fc3.weight))
                    for layer in (model.bn1, model.bn2, model.bn3):
                        # Two train batches; final singleton skipped. Attack gradients
                        # must not update BN, even for a singleton adversarial subset.
                        self.assertEqual(int(layer.num_batches_tracked), 2)
                        self.assertTrue(torch.isfinite(layer.running_var).all())
                        self.assertTrue((layer.running_var >= 0).all())


class PlacementTests(unittest.TestCase):
    def config(self, **overrides):
        return dict(dict(num_fogs=2, uniform_edges_per_fog=10, custom_fog_topology='[]',
                         random_seed=42, backdoor_ratio=.1, label_flip_ratio=.1,
                         grad_manip_ratio=.1, shap_aware_ratio=0.), **overrides)

    def roles(self, config, count):
        return [assign_edge_roles(config, count, i, config['random_seed'], None) for i in range(count)]

    def test_counts_seed_and_server_ground_truth_agree(self):
        config = self.config()
        roles = self.roles(config, 20)
        self.assertEqual(Counter(roles), {'benign': 14, 'backdoor': 2, 'label_flip': 2, 'gradient_manip': 2})
        self.assertEqual(roles, self.roles(config, 20))
        self.assertNotEqual(roles, self.roles(self.config(random_seed=123), 20))
        truth = experiment_ground_truth(config)
        self.assertEqual(roles, [record['role'] for record in truth.values()])
        self.assertTrue(all(record['attestation_valid'] for record in truth.values()))
        self.assertEqual(roles, self.roles(self.config(custom_fog_topology='[5, 15]'), 20))

    def test_global_budget_allows_regional_malicious_majority(self):
        config = self.config(uniform_edges_per_fog=3, random_seed=1,
                             backdoor_ratio=0., label_flip_ratio=1/3, grad_manip_ratio=0.)
        roles = self.roles(config, 6)
        self.assertEqual(roles.count('label_flip'), 2)
        self.assertEqual(max(roles[:3].count('label_flip'), roles[3:].count('label_flip')), 2)
        self.assertEqual(roles, self.roles(dict(config, custom_fog_topology='[1, 1, 1, 1, 1, 1]'), 6))

    def test_invalid_requested_and_rounded_budgets_are_rejected(self):
        for ratio in (-.1, float('nan'), float('inf'), .5, .7):
            config = self.config(backdoor_ratio=0., label_flip_ratio=ratio, grad_manip_ratio=0.)
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                self.roles(config, 20)
        # Requested .45 < .5, but round(4*.225) twice gives 2/4 attackers.
        config = self.config(uniform_edges_per_fog=2, backdoor_ratio=.225,
                             label_flip_ratio=.225, grad_manip_ratio=0.)
        with self.assertRaisesRegex(ValueError, 'Rounded'):
            self.roles(config, 4)


if __name__ == '__main__':
    unittest.main()
