"""Observed-class AUC: known rankings, coverage, and invalid-input boundaries."""
import json
import logging
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from flwr.common import parameters_to_ndarrays

from src.shared.network.model_payload import encode_model
from src.shared.utils.metrics import roc_auc_multiclass, roc_auc_multiclass_details
from src.tier_cloud.global_evaluator import GlobalEvaluator


class AUCTests(unittest.TestCase):
    def test_missing_noncontiguous_classes_use_original_scores_and_equal_weights(self):
        targets = torch.tensor([0, 0, 2, 2])
        # Dropping class 1 and renormalizing would incorrectly raise macro AUC to 1.
        probs = torch.tensor([[.2, .79, .01], [.3, .69, .01],
                              [.4, .1, .5], [.1, .1, .8]])
        auc, details = roc_auc_multiclass_details(targets, probs, 3)
        self.assertEqual(auc, .75)
        self.assertEqual(roc_auc_multiclass(targets, probs, 3), auc)
        self.assertEqual(details["evaluated_classes"], [0, 2])
        self.assertEqual(details["class_counts"], [2, 0, 2])
        self.assertEqual(details["per_class_auc"], [.5, None, 1.])
        self.assertEqual(details["status"], "defined")
        # Repeating one class changes prevalence but not these class rankings.
        idx = torch.tensor([0, 1, 0, 1, 2, 3])
        self.assertEqual(roc_auc_multiclass(targets[idx], probs[idx], 3), .75)

    def test_complete_coverage_has_known_perfect_tied_and_reversed_auc(self):
        targets = torch.tensor([0, 1, 2, 0, 1, 2])
        for scale, expected in ((1., 1.), (0., .5), (-1., 0.)):
            with self.subTest(scale=scale):
                probs = (scale * torch.nn.functional.one_hot(targets, 3).float()).softmax(1)
                auc, details = roc_auc_multiclass_details(targets, probs, 3)
                self.assertEqual(auc, expected)
                self.assertEqual(details["per_class_auc"], [expected] * 3)

    def test_empty_and_single_class_populations_remain_undefined(self):
        for labels, status in (([], "empty_test_split"), ([2, 2], "insufficient_test_classes")):
            with self.subTest(status=status):
                auc, details = roc_auc_multiclass_details(torch.tensor(labels), torch.full((len(labels), 3), 1/3), 3)
                self.assertIsNone(auc)
                self.assertEqual(details["status"], status)
                self.assertEqual(details["evaluated_classes"], [])
                self.assertEqual(details["per_class_auc"], [None] * 3)
                json.dumps(details, allow_nan=False)

    def test_invalid_inputs_raise_instead_of_silently_excluding_classes(self):
        valid = torch.tensor([[.6, .3, .1], [.2, .7, .1]])
        cases = [
            (torch.tensor([0, 1]), valid[:, :2]),
            (torch.tensor([0, 3]), valid),
            (torch.tensor([0., .5]), valid),
            (torch.tensor([0, 1]), valid * .5),
            (torch.tensor([0, 1]), torch.tensor([[float("nan"), .3, .1], [.2, .7, .1]])),
            (torch.tensor([0, 1]), torch.tensor([[1.2, -.3, .1], [.2, .7, .1]])),
        ]
        for targets, probs in cases:
            with self.subTest(targets=targets, probs=probs), self.assertRaises(ValueError):
                roc_auc_multiclass(targets, probs, 3)

    def test_calculation_failures_propagate(self):
        targets, probs = torch.tensor([0, 1]), torch.full((2, 3), 1/3)
        with patch("torchmetrics.functional.auroc", side_effect=RuntimeError("calculation failed")):
            with self.assertRaisesRegex(RuntimeError, "calculation failed"):
                roc_auc_multiclass(targets, probs, 3)
        with patch("torchmetrics.functional.auroc", return_value=torch.tensor(float("nan"))):
            with self.assertRaisesRegex(ValueError, "Nonfinite ROC-AUC"):
                roc_auc_multiclass(targets, probs, 3)

    def test_fifteen_output_evaluator_on_ten_classes_keeps_final_singleton(self):
        evaluator = GlobalEvaluator.__new__(GlobalEvaluator)
        evaluator.model = nn.Linear(15, 15)
        with torch.no_grad():
            evaluator.model.weight.copy_(torch.eye(15))
            evaluator.model.bias.zero_()
        evaluator.run_metadata = {"quantization_bits": 32}
        evaluator.device, evaluator.num_classes, evaluator.tier = "cpu", 15, "cloud"
        evaluator.logger = logging.getLogger("auc-tests")
        evaluator.criterion = nn.CrossEntropyLoss()
        targets = torch.arange(10)
        evaluator.test_loader = DataLoader(TensorDataset(
            torch.nn.functional.one_hot(targets, 15).float(), targets), batch_size=3)
        _, metrics = evaluator.evaluate(0, parameters_to_ndarrays(encode_model(evaluator.model, 32)), {})
        self.assertEqual(metrics["auc"], 1.)
        self.assertTrue(metrics["auc_defined"])
        details = json.loads(metrics["auc_details"])
        self.assertEqual(details["method"], "macro_ovr_observed_classes")
        self.assertEqual(details["configured_classes"], 15)
        self.assertEqual(details["evaluated_classes"], list(range(10)))
        self.assertEqual(details["class_counts"], [1] * 10 + [0] * 5)
        self.assertEqual(details["per_class_auc"], [1.] * 10 + [None] * 5)


if __name__ == "__main__":
    unittest.main()
