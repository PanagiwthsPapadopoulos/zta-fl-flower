"""Authentication errors must not be inferred from poisoning roles or quarantine."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.shared.utils.security_metrics import calculate_security_rates


class SecurityMetricsTests(unittest.TestCase):
    def calculate(self, rounds):
        with TemporaryDirectory() as directory:
            target = Path(directory) / "trustdb"
            target.mkdir()
            (target / "fog_1_state.json").write_text(json.dumps({"rounds": rounds}))
            return calculate_security_rates(directory)

    def test_valid_poisoner_and_quarantine_are_not_false_acceptance_or_rejection(self):
        result = self.calculate({"1": {
            "admission_outcomes": {"a": {"expected_valid": True, "authenticated": True,
                                          "accepted": True, "attempted": True}},
            "attestation_outcomes": {"b": {"expected_valid": True, "authenticated": True,
                                            "accepted": False, "attempted": True}},
            "update_outcomes": {"a": {"role": "label_flip", "accepted": True}},
            "trust_db": {"b": {"is_quarantined": True}},
        }})
        attestation = result["attestation_gatekeeper"]
        self.assertEqual(attestation["valid_attempts"], 2)
        self.assertEqual(attestation["FRR_percentage"], 0)
        self.assertIsNone(attestation["FAR_percentage"])
        self.assertEqual(result["update_verification"]["poisoned_acceptance_percentage"], 100)
        self.assertEqual(result["quarantine"]["quarantined_node_rounds"], 1)

    def test_unknown_missing_and_bypassed_attempts_do_not_fabricate_rates(self):
        result = self.calculate({"1": {
            "admission_outcomes": {"a": {"authenticated": False, "attempted": False}},
            "attestation_outcomes": {"b": {"authenticated": False}},
        }, "2": {"attestation_bypassed": True, "admission_outcomes": {
            "a": {"expected_valid": True, "authenticated": True}}}})
        metrics = result["attestation_gatekeeper"]
        self.assertEqual(metrics["missing_responses"], 1)
        self.assertEqual(metrics["unknown_attempts"], 1)
        self.assertEqual(metrics["bypassed_attempts"], 1)
        self.assertIsNone(metrics["FRR_percentage"])
        self.assertIsNone(metrics["FAR_percentage"])

    def test_invalid_attestation_and_model_rejections_have_separate_denominators(self):
        result = self.calculate({"1": {"admission_outcomes": {}, "attestation_outcomes": {
            "a": {"expected_valid": False, "authenticated": True},
            "b": {"expected_valid": False, "authenticated": False},
            "c": {"expected_valid": True, "authenticated": False},
        }, "update_outcomes": {"d": {"role": "benign", "accepted": False}}}})
        self.assertEqual(result["attestation_gatekeeper"]["FAR_percentage"], 50)
        self.assertEqual(result["attestation_gatekeeper"]["FRR_percentage"], 100)
        self.assertEqual(result["update_verification"]["honest_rejection_percentage"], 100)

    def test_legacy_roles_are_not_reinterpreted_as_attestation_truth(self):
        result = self.calculate({"1": {"attestation_accepted": ["a"],
                                         "node_ground_truths": {"a": "label_flip"}}})
        self.assertIsNone(result["attestation_gatekeeper"]["FAR_percentage"])
        self.assertEqual(result["attestation_gatekeeper"]["legacy_rounds_without_admission_records"], 1)


if __name__ == "__main__":
    unittest.main()
