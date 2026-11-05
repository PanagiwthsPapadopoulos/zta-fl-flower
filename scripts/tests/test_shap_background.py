"""Fog sample selection and real Captum background/explanation boundaries."""
import copy
import logging
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn
from captum.attr import GradientShap

from src.shared.utils.config_loader import load_yaml_configs
from src.shared.utils.metrics import compute_shap_attributions


torch.set_num_threads(1)
LOGGER = logging.getLogger('shap-background-tests')


class BackgroundTests(unittest.TestCase):
    def config(self):
        return dict(load_yaml_configs(), tier='fog', fog_id='fog_1', num_fogs=2,
                    uniform_edges_per_fog=3, dataset_fraction=1., shap_val_samples=100,
                    shap_explain_count=10, random_seed=42, run_name='test',
                    label_flip_ratio=0., backdoor_ratio=0., grad_manip_ratio=0., shap_aware_ratio=0.)

    def simulation_data(self, config, X, y):
        from src.entrypoints import sim_server
        with patch('src.shared.utils.config_loader.get_merged_config', return_value=config), \
             patch.object(sim_server, 'get_dataset', return_value=(X, y, 15)) as data, \
             patch.object(sim_server, 'setup_logger', return_value=LOGGER), \
             patch.object(sim_server, 'GlobalEvaluator'), \
             patch.object(sim_server, 'CloudAggregator') as strategy:
            sim_server.server_fn(SimpleNamespace(run_config={}))
        self.assertEqual(data.call_args.args[6], 'val')
        return strategy.call_args.kwargs

    def deployment_data(self, config, X, y):
        from src.entrypoints import deploy_server
        with patch('src.shared.utils.config_loader.get_merged_config', return_value=config), \
             patch.object(deploy_server, 'get_dataset', return_value=(X, y, 15)) as data, \
             patch.object(deploy_server, 'setup_logger', return_value=LOGGER), \
             patch.object(deploy_server, 'GlobalEvaluator') as evaluator, \
             patch('src.tier_fog.fog_aggregator.FogAggregator') as strategy:
            deploy_server.server_fn(SimpleNamespace(run_config={}))
        evaluator.assert_not_called()
        self.assertEqual(data.call_args.args[6], 'val')
        return strategy.call_args.kwargs['val_data']

    def test_100_seeded_validation_rows_match_between_deployment_and_simulation(self):
        X = torch.arange(250 * 40, dtype=torch.float32).reshape(250, 40)
        y = torch.arange(250) % 15
        config = self.config()
        simulation = self.simulation_data(config, X, y)
        self.assertTrue(torch.equal(simulation['val_data'][0], X))  # Cloud keeps full validation.
        for fog in (1, 2):
            chosen = simulation['fog_val_data'][fog]
            deployed = self.deployment_data(dict(config, fog_id=f'fog_{fog}'), X, y)
            self.assertEqual(len(chosen[0]), 100)
            self.assertEqual(len(torch.unique(chosen[0][:, 0])), 100)
            indices = (chosen[0][:, 0] / 40).long()
            self.assertTrue(torch.equal(chosen[1], y[indices]))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(chosen, deployed)))
        self.assertFalse(torch.equal(simulation['fog_val_data'][1][0], simulation['fog_val_data'][2][0]))
        torch.rand(13)  # Selection must not depend on incidental global Torch draws.
        repeated = self.simulation_data(config, X, y)
        self.assertTrue(torch.equal(simulation['fog_val_data'][1][0], repeated['fog_val_data'][1][0]))

    def test_fraction_limits_fog_pool_but_does_not_truncate_cloud_validation(self):
        X, y = torch.arange(250 * 40).reshape(250, 40).float(), torch.arange(250) % 15
        config = dict(self.config(), dataset_fraction=.1)
        selected = self.simulation_data(config, X, y)
        self.assertEqual(len(selected['val_data'][0]), 250)
        self.assertEqual(set(selected['fog_val_data'][1][0][:, 0].tolist()), set(X[:100, 0].tolist()))
        deployed = self.deployment_data(config, X, y)
        self.assertTrue(torch.equal(selected['fog_val_data'][1][0], deployed[0]))

    def test_small_validation_set_uses_available_rows_without_duplication(self):
        X, y = torch.randn(37, 40), torch.arange(37) % 15
        chosen = self.simulation_data(self.config(), X, y)['fog_val_data'][1]
        self.assertEqual(len(chosen[0]), 37)
        self.assertEqual(len(torch.unique(chosen[0], dim=0)), 37)
        # Explicit limitation: this smoke case does not satisfy the paper's 100 rows.

    def test_real_gradientshap_gets_all_100_background_rows_but_explains_10(self):
        torch.manual_seed(42)
        model = nn.Linear(4, 3).train()
        X, y = torch.randn(100, 4), torch.arange(100) % 3
        before = copy.deepcopy(model.state_dict())
        torch_state, numpy_state = torch.random.get_rng_state(), np.random.get_state()
        # Spy forwards to real Captum; no attribution values are substituted.
        original_attribute = GradientShap.attribute
        with patch.object(GradientShap, 'attribute', autospec=True, side_effect=original_attribute) as attribute:
            attrs = compute_shap_attributions(model, X, y, n_explain=10)
        self.assertEqual(attrs.shape, (10, 4))
        self.assertTrue(torch.isfinite(attrs).all())
        self.assertTrue(torch.equal(attribute.call_args.args[1], X[:10]))
        self.assertTrue(torch.equal(attribute.call_args.kwargs['baselines'], X))
        self.assertTrue(torch.equal(attribute.call_args.kwargs['target'], y[:10]))
        self.assertTrue(model.training)
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))
        self.assertTrue(torch.equal(torch_state, torch.random.get_rng_state()))
        after_numpy = np.random.get_state()
        np.testing.assert_array_equal(numpy_state[1], after_numpy[1])
        self.assertEqual(numpy_state[2:], after_numpy[2:])


if __name__ == '__main__':
    unittest.main()
