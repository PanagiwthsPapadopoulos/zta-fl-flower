"""Shared training settings and adapter forwarding, without services or external source trees."""
from contextlib import ExitStack
import copy
import logging
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.shared.security import adversarial_math as attacks
from src.shared.utils.config_loader import load_yaml_configs
from src.tier_edge.local_ids import EdgeTrainer

LOGGER = logging.getLogger('training-config-tests')
torch.set_num_threads(1)


class TrainingConfigurationTests(unittest.TestCase):
    def test_both_clients_forward_one_training_budget_without_overwriting_pgd_step(self):
        from src.entrypoints import sim_client, deploy_client
        config = dict(load_yaml_configs(), benign_adv_ratio=.4, benign_eps=.07,
                      benign_alpha=.02, benign_n_iter=4, robustness_eval_attack='pgd',
                      gradient_alpha=9., shap_aware_tau=.05, shap_aware_constrained=True,
                      shap_aware_max_backtracks=7, num_fogs=1, uniform_edges_per_fog=10,
                      custom_fog_topology='[]')
        for module, node in ((sim_client, {'partition-id': 0}),
                             (deploy_client, {'partition-id': 1, 'fog_num': 1})):
            for role in ('benign', 'label_flip', 'gradient_manip', 'shap_aware'):
                with self.subTest(adapter=module.__name__, role=role), ExitStack() as stack:
                    stack.enter_context(patch.dict(os.environ, {'ZTA_TIER': 'test'}))
                    stack.enter_context(patch('src.shared.utils.config_loader.get_merged_config', return_value=config))
                    stack.enter_context(patch('src.shared.security.threat_profiler.assign_edge_roles', return_value=role))
                    stack.enter_context(patch.object(module, 'setup_logger', return_value=LOGGER))
                    select_device = stack.enter_context(patch.object(module, 'get_device', return_value='cpu'))
                    stack.enter_context(patch.object(module, 'get_dataset', return_value=(torch.zeros(8, 40), torch.zeros(8, dtype=torch.long), 15)))
                    stack.enter_context(patch.object(module, 'evaluation_loader', return_value=None))
                    shap_data = (torch.zeros(4, 40), torch.zeros(4, dtype=torch.long))
                    attack_data = stack.enter_context(patch.object(module, 'attack_validation_data', return_value=shap_data))
                    stack.enter_context(patch.object(module, 'get_model', return_value=nn.Linear(40, 15)))
                    client = stack.enter_context(patch.object(module, 'Client'))
                    module.client_fn(SimpleNamespace(run_config={}, node_config=node))
                    if module is sim_client:
                        select_device.assert_called_once_with(tier='edge')
                    forwarded = client.call_args.kwargs['train_config']
                    self.assertEqual(forwarded['robustness_eval_attack'], 'pgd')
                    self.assertEqual([forwarded[k] for k in ('adv_ratio', 'eps', 'alpha', 'n_iter')], [.4, .07, .02, 4])
                    if role == 'gradient_manip':
                        self.assertEqual(forwarded['gradient_alpha'], 9.)
                    if role == 'shap_aware':
                        self.assertEqual(forwarded['shap_aware_tau'], .05)
                        self.assertTrue(forwarded['shap_aware_constrained'])
                        self.assertEqual(forwarded['shap_aware_max_backtracks'], 7)
                        self.assertEqual(forwarded['fog_num'], 1)
                        self.assertIs(client.call_args.kwargs['shap_data'], shap_data)
                        attack_data.assert_called_once_with(config, 1)
                    else:
                        attack_data.assert_not_called()

    def test_benign_and_label_flip_training_execute_the_selected_attack(self):
        for role in ('benign', 'label_flip'):
            for selector in ('fgsm', 'pgd'):
                with self.subTest(role=role, selector=selector):
                    model = nn.Linear(2, 2)
                    before = copy.deepcopy(model.state_dict())
                    loader = DataLoader(TensorDataset(torch.randn(8, 2), torch.arange(8) % 2), batch_size=4)
                    config = dict(role=role, random_seed=42, client_index=0, learning_rate=.01,
                                  local_epochs=1, num_classes=2, p_flip=1., clip_norm=1.,
                                  quantization_bits=32, robustness_eval_attack=selector,
                                  adv_ratio=.5, eps=.07, alpha=.02, n_iter=4,
                                  clip_min=None, clip_max=None)
                    trainer = EdgeTrainer(LOGGER, 'edge', model, loader, 'cpu', config, {})
                    with patch.object(trainer, 'attestation_metrics', return_value={}), \
                         patch.object(attacks, 'fgsm_attack', wraps=attacks.fgsm_attack) as fgsm, \
                         patch.object(attacks, 'pgd_attack', wraps=attacks.pgd_attack) as pgd:
                        trainer.execute_training(trainer.get_parameters(), 1, {})
                    chosen, other = (fgsm, pgd) if selector == 'fgsm' else (pgd, fgsm)
                    self.assertEqual(chosen.call_count, 2)
                    other.assert_not_called()
                    self.assertEqual(chosen.call_args.args[3:5], (.07, .5))
                    if selector == 'pgd':
                        self.assertEqual(chosen.call_args.args[5:7], (.02, 4))
                    self.assertFalse(torch.equal(before['weight'], model.weight))

    def test_zero_augmentation_label_flip_trains_on_supplied_labels_without_input_attacks(self):
        torch.manual_seed(9)
        model = nn.Linear(2, 2)
        expected = copy.deepcopy(model)
        loader = DataLoader(TensorDataset(torch.randn(8, 2), torch.ones(8, dtype=torch.long)), batch_size=4)
        with patch.object(attacks, 'fgsm_attack', side_effect=AssertionError('Unexpected FGSM')), \
             patch.object(attacks, 'pgd_attack', side_effect=AssertionError('Unexpected PGD')):
            loss = attacks.local_train_byzantine(model, loader, 'label_flip', 2,
                                                 labels_prepared=True, adv_ratio=0., lr=.01)
        expected_loss = attacks.local_train_honest(expected, loader, lr=.01)
        self.assertEqual(loss, expected_loss)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, expected.state_dict()[name], rtol=0, atol=0)



if __name__ == '__main__':
    unittest.main()
