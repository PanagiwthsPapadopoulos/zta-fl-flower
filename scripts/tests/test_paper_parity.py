"""Numerical regressions for the paper baseline and evaluation boundaries."""
import copy
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from flwr.common import parameters_to_ndarrays

from src.shared.data.data_loader import load_edge_iiotset, load_cic_ids2017, load_unsw_nb15, resample_training_data
from src.shared.network.model_payload import encode_model, load_model_arrays
from src.shared.security.backdoor_math import compute_backdoor_asr
from src.shared.utils.metrics import compute_shap_stability, macro_f1
from src.tier_cloud.global_evaluator import GlobalEvaluator
from src.tier_cloud.model_state import GlobalModelState
from src.tier_fog.shap_verifier import shap_weighted_aggregate
from scripts.ops.plot_metrics import get_nested_metric, merge_offline_metrics

torch.set_num_threads(1)


def constant_model(classes=2):
    model = nn.Linear(2, classes)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
        model.bias[0] = 1
    return model


class PaperAggregationTests(unittest.TestCase):
    def aggregate(self, scores):
        model = constant_model()
        diagnostics = {}
        with patch("src.tier_fog.shap_verifier.compute_shap_attributions", return_value=torch.ones(3, 2)), \
             patch("src.tier_fog.shap_verifier.compute_shap_stability", side_effect=scores):
            result = shap_weighted_aggregate(
                [copy.deepcopy(model) for _ in scores], model,
                torch.zeros(3, 2), torch.zeros(3, dtype=torch.long), [10] * len(scores),
                diagnostics=diagnostics,
            )
        return result, diagnostics

    def test_above_mean_reward_differs_from_above_median(self):
        (_, _, passed, rewards, _), stats = self.aggregate([.1, .6, .9])
        self.assertEqual(passed, [True, True, True])
        self.assertEqual(rewards, [False, True, True])
        self.assertAlmostEqual(stats["mean"], 1.6 / 3)

    def test_small_population_limitation_is_preserved_and_visible(self):
        (_, _, passed, _, _), stats = self.aggregate([0, 1, 1, 1, 1])
        self.assertEqual(passed, [True] * 5)
        self.assertEqual(stats["valid_population_size"], 5)
        (_, _, passed, _, _), _ = self.aggregate([0, 1, 1, 1, 1, 1])
        self.assertEqual(passed, [False, True, True, True, True, True])

    def test_tied_and_singleton_scores_do_not_receive_rewards(self):
        for scores in ([.9], [.9, .9, .9], [.7, .7, .7], [.9] * 10):
            (_, total, passed, rewards, _), stats = self.aggregate(scores)
            self.assertEqual(passed, [True] * len(scores))
            self.assertEqual(rewards, [False] * len(scores))
            self.assertGreater(total, 0)
            self.assertEqual(stats["std"], 0)

    def test_negative_score_passes_filter_but_contributes_exactly_zero(self):
        reference = constant_model()
        models = [constant_model() for _ in range(3)]
        with torch.no_grad():
            for model, value in zip(models, (100., 2., 6.)):
                model.weight.fill_(value)  # Distinct values make each contribution observable.
            models[2].bias.copy_(torch.tensor([0., 1.]))
        with patch("src.tier_fog.shap_verifier.compute_shap_attributions", return_value=torch.ones(3, 2)), \
             patch("src.tier_fog.shap_verifier.compute_shap_stability", side_effect=[-.2, .4, .8]):
            aggregate, total, passed, _, scores = shap_weighted_aggregate(
                models, reference, torch.zeros(3, 2), torch.tensor([0, 0, 1]), [1, 4, 9])
        self.assertEqual(scores, [-.2, .4, .8])
        self.assertEqual(passed, [True, True, True])
        self.assertAlmostEqual(total, .4 * (2/3) * 2 + .8 * (1/3) * 3, places=6)
        # Accuracies 2/3 and 1/3 give normalized weights .4 and .6.
        # The first model passes but has no influence.
        torch.testing.assert_close(aggregate.weight, torch.full_like(aggregate.weight, 4.4))
        torch.testing.assert_close(aggregate.bias, torch.tensor([.4, .6]))

    def test_zero_total_weight_uniformly_averages_survivors_and_preserves_rewards(self):
        reference = constant_model()
        models = [constant_model(), constant_model()]
        with torch.no_grad():
            models[0].weight.fill_(2)
            models[1].weight.fill_(6)
        for scores, labels, expected_rewards in (([0., 0.], [0, 0, 0], [False, False]),
                                                  ([-2., -1.], [0, 0, 0], [False, True]),
                                                  ([.2, .8], [1, 1, 1], [False, True]),
                                                  ([1e-15, 2e-15], [0, 0, 0], [False, True])):
            diagnostics = {}
            with self.subTest(scores=scores, labels=labels), \
                 patch("src.tier_fog.shap_verifier.compute_shap_attributions", return_value=torch.ones(3, 2)), \
                 patch("src.tier_fog.shap_verifier.compute_shap_stability", side_effect=scores):
                aggregate, total, passed, rewards, _ = shap_weighted_aggregate(
                    models, reference, torch.zeros(3, 2), torch.tensor(labels), [4, 100],
                    diagnostics=diagnostics)
            self.assertEqual(total, 1)
            self.assertEqual(passed, [True, True])
            self.assertEqual(rewards, expected_rewards)
            self.assertTrue(diagnostics['uniform_fallback'])
            self.assertLess(diagnostics['raw_weight_total'], 1e-12)
            torch.testing.assert_close(aggregate.weight, torch.full_like(aggregate.weight, 4.))

    def test_uniform_fallback_excludes_invalid_and_shap_rejected_updates(self):
        reference = constant_model()
        models = [constant_model() for _ in range(7)]
        with torch.no_grad():
            models[0].weight.fill_(float('nan'))
            for model, value in zip(models[1:], (999., 2., 4., 6., 8., 10.)):
                model.weight.fill_(value)
        with patch("src.tier_fog.shap_verifier.compute_shap_attributions", return_value=torch.ones(3, 2)), \
             patch("src.tier_fog.shap_verifier.compute_shap_stability", side_effect=[-100.] + [-1.] * 5):
            aggregate, total, passed, rewards, _ = shap_weighted_aggregate(
                models, reference, torch.zeros(3, 2), torch.zeros(3, dtype=torch.long), [4] * 7)
        self.assertEqual(passed, [False, False] + [True] * 5)
        self.assertEqual(rewards, [False, False] + [True] * 5)
        self.assertEqual(total, 1)
        torch.testing.assert_close(aggregate.weight, torch.full_like(aggregate.weight, 6.))

    def test_no_eligible_updates_keeps_reference_without_fallback_or_rewards(self):
        reference, invalid = constant_model(), constant_model()
        with torch.no_grad(): invalid.weight.fill_(float('nan'))
        for models, sizes in (([], []), ([invalid], [4])):
            diagnostics = {}
            with patch("src.tier_fog.shap_verifier.compute_shap_attributions", return_value=torch.ones(3, 2)):
                aggregate, total, passed, rewards, _ = shap_weighted_aggregate(
                    models, reference, torch.zeros(3, 2), torch.zeros(3, dtype=torch.long), sizes,
                    diagnostics=diagnostics)
            self.assertEqual(total, 0)
            self.assertFalse(any(passed))
            self.assertFalse(any(rewards))
            self.assertFalse(diagnostics['uniform_fallback'])
            self.assertTrue(all(torch.equal(v, reference.state_dict()[k]) for k, v in aggregate.state_dict().items()))

    def test_stability_compares_one_feature_vector_without_clipping(self):
        model_attrs = torch.tensor([[6., 0.], [0., 6.]])
        ref_attrs = torch.tensor([[2., 0.], [0., 2.]])
        with patch("src.shared.utils.metrics.compute_shap_attributions", return_value=model_attrs):
            score = compute_shap_stability(nn.Identity(), nn.Identity(), torch.zeros(2, 2),
                                           torch.zeros(2), precomputed_attrs_ref=ref_attrs)
        self.assertAlmostEqual(score, 1 - (8 ** .5) / (2 ** .5 + 1e-6), places=5)
        self.assertLess(score, 0)

    def test_real_gradientshap_identical_model_has_unit_stability(self):
        torch.manual_seed(7)
        model = nn.Linear(2, 2)
        X, y = torch.randn(8, 2), torch.zeros(8, dtype=torch.long)
        state = torch.random.get_rng_state().clone()
        score = compute_shap_stability(model, copy.deepcopy(model), X, y, n_explain=4)
        self.assertEqual(score, 1)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))


class RollbackTests(unittest.TestCase):
    def state(self):
        state = GlobalModelState()
        state.global_model = constant_model()
        state.quantization_bits = 32
        state._broadcast_parameters = encode_model(state.global_model, 32)
        state.previous_val_acc = 1.0
        state.validation_accuracy = Mock()
        state.run_metadata = {"rollback_threshold": .8}
        state.logger = logging.getLogger("rollback-tests")
        return state

    def test_exactly_eighty_percent_rolls_back(self):
        state = self.state()
        previous = state._broadcast_parameters
        state.validation_accuracy.return_value = .8
        result, metrics = state.aggregate_regions([constant_model()], [1], 1)
        self.assertIs(result, previous)
        self.assertEqual(metrics["aggregation_status"], "rolled_back")

    def test_reference_tracks_previous_accepted_model_not_best_ever(self):
        state = self.state()
        for round_num, accuracy in enumerate((.9, .75), 1):
            state.validation_accuracy.return_value = accuracy
            _, metrics = state.aggregate_regions([constant_model()], [1], round_num)
            self.assertEqual(metrics["aggregation_status"], "aggregated")
        self.assertEqual(state.previous_val_acc, .75)

    def test_zero_regional_weight_keeps_exact_broadcast_and_skips_validation(self):
        state = self.state()
        previous = state._broadcast_parameters
        original = copy.deepcopy(state.global_model.state_dict())
        parameters, metrics = state.aggregate_regions([constant_model()], [0], 1)
        self.assertIs(parameters, previous)
        self.assertEqual(metrics['aggregation_status'], 'no_positive_trust')
        self.assertEqual(state.previous_val_acc, 1.)
        state.validation_accuracy.assert_not_called()
        self.assertTrue(all(torch.equal(original[k], v) for k, v in state.global_model.state_dict().items()))


class EvaluationTests(unittest.TestCase):
    def evaluator(self):
        evaluator = GlobalEvaluator.__new__(GlobalEvaluator)
        evaluator.model = constant_model(3)
        evaluator.run_metadata = {"quantization_bits": 32}
        evaluator.device, evaluator.num_classes, evaluator.tier = "cpu", 3, "cloud"
        evaluator.logger = logging.getLogger("evaluation-tests")
        evaluator.criterion = nn.CrossEntropyLoss()
        evaluator.test_loader = DataLoader(TensorDataset(torch.zeros(5, 2), torch.zeros(5, dtype=torch.long)), batch_size=2)
        return evaluator

    def test_single_test_class_has_valid_f1_but_undefined_auc(self):
        evaluator = self.evaluator()
        arrays = parameters_to_ndarrays(encode_model(evaluator.model, 32))
        _, metrics = evaluator.evaluate(1, arrays, {})
        self.assertEqual(metrics["accuracy"], 1)
        self.assertAlmostEqual(metrics["macro_f1"], 1)
        self.assertFalse(metrics["auc_defined"])
        self.assertNotIn("auc", metrics)

    def test_macro_f1_includes_prediction_only_and_missed_classes(self):
        # Class 0: F1=2/3; prediction-only class 1: F1=0; class 2 absent.
        for targets, predictions in (([0, 0], [0, 1]), ([0, 1], [0, 0])):
            for logits in (False, True):
                y_true, y_pred = torch.tensor(targets), torch.tensor(predictions)
                if logits:
                    y_pred = torch.nn.functional.one_hot(y_pred, 3).float()
                self.assertAlmostEqual(macro_f1(y_true, y_pred, 3), 1 / 3, places=6)
                self.assertAlmostEqual(macro_f1(y_true, y_pred), 1 / 3, places=6)
        self.assertEqual(macro_f1(torch.empty(0), torch.empty(0), 3), 0)

    def test_extra_payload_tensor_is_rejected_before_mutation(self):
        evaluator = self.evaluator()
        original = copy.deepcopy(evaluator.model.state_dict())
        arrays = parameters_to_ndarrays(encode_model(evaluator.model, 32))
        with self.assertRaisesRegex(ValueError, "payload arrays"):
            evaluator.evaluate(1, arrays + [np.zeros(1, dtype=np.float32)], {})
        self.assertTrue(all(torch.equal(original[k], v) for k, v in evaluator.model.state_dict().items()))

    def test_negative_batchnorm_variance_rejected_before_shap(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        arrays = parameters_to_ndarrays(encode_model(model, 32))
        index = list(model.state_dict()).index("1.running_var")
        arrays[index] = -np.ones(2, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Negative BatchNorm variance"):
            load_model_arrays(model, arrays, 32)
        self.assertTrue(torch.equal(model[1].running_var, torch.ones(2)))

    def test_backdoor_asr_includes_final_singleton(self):
        score = compute_backdoor_asr(constant_model(), torch.zeros(257, 2), torch.ones(257, dtype=torch.long),
                                     trigger_features=(0,), batch_size=256)
        self.assertEqual(score, 100)

    def test_backdoor_asr_without_eligible_examples_is_undefined(self):
        self.assertIsNone(compute_backdoor_asr(constant_model(), torch.zeros(1, 2), torch.zeros(1, dtype=torch.long)))

    def test_offline_metrics_are_only_attached_to_evaluated_round(self):
        rows = [{"round": 1}, {"round": 2}]
        result = merge_offline_metrics(rows, {"evaluated_round": 2, "metrics": {
            "backdoor_asr_percentage": 87.3,
            "adversarial_robustness": {"attack": "pgd", "clean_acc": .9, "adv_acc": .7},
        }})
        self.assertNotIn("global_asr", result[0])
        self.assertEqual(result[1]["global_asr"], 87.3)
        self.assertIsNone(get_nested_metric(result[1], "global_robustness_fgsm", "adv_acc"))
        self.assertNotIn("global_asr", rows[1])


class DatasetValidationTests(unittest.TestCase):
    def load(self, contents, loader=load_edge_iiotset):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            path.write_text(contents)
            return loader(str(path))

    def test_unknown_labels_and_invalid_features_do_not_become_benign_zeros(self):
        for contents in ("feature,label\n1,NewAttack\n", "feature,label\nwrong,Normal\n",
                         "feature,label\nnan,Normal\n", "feature,label\ninf,Normal\n",
                         "feature,label\n1\n", "feature,wrong_column\n1,Normal\n"):
            with self.subTest(contents=contents), self.assertRaises(ValueError):
                self.load(contents)

    def test_all_dataset_loaders_require_their_label_column(self):
        for loader in (load_edge_iiotset, load_cic_ids2017, load_unsw_nb15):
            with self.subTest(loader=loader), self.assertRaisesRegex(ValueError, "label column"):
                self.load("feature,wrong\n1,Normal\n", loader)

    def test_small_samples_may_omit_classes(self):
        X, y = self.load("feature,label\n1,Normal\n2,DoS_TCP\n")
        self.assertEqual(X.shape, (2, 1))
        self.assertEqual(y.tolist(), [0, 1])

    def test_smote_adapts_neighbors_to_observed_minority_size(self):
        X, y = np.arange(14, dtype=np.float32).reshape(7, 2), np.array([0] * 5 + [1] * 2)
        _, resampled, neighbors = resample_training_data(X, y, 42)
        self.assertEqual(neighbors, 1)
        self.assertEqual(np.bincount(resampled).tolist(), [5, 5])

    def test_smote_rejects_singleton_minority_class(self):
        with self.assertRaisesRegex(ValueError, "singleton"):
            resample_training_data(np.zeros((4, 2)), np.array([0, 0, 0, 1]), 42)


if __name__ == "__main__":
    unittest.main()
