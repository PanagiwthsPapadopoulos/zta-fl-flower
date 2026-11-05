"""Every round must authenticate before model release, including the first round."""
import json
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from flwr.common import Code, FitIns, ndarrays_to_parameters

from src.tier_fog.client_instructions import configure_edge_fit
from src.tier_fog.gatekeeper import ZeroTrustGatekeeper


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"ZTA_INSECURE_MODE": "false"})
        env.start()
        self.addCleanup(env.stop)
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.logger = logging.getLogger("admission-tests")
        self.logger.addHandler(logging.NullHandler())
        self.logger.propagate = False
        self.parameters = ndarrays_to_parameters([np.array([1.0], dtype=np.float32)])
        self.nonces = {}
        self.gatekeepers = [self.gatekeeper(1), self.gatekeeper(2)]
        self.proxy = self.client("client-a", 1)

    def gatekeeper(self, fog):
        root = Path(self.tmp.name)
        folder = root / f"edge_{fog}_1"
        folder.mkdir()
        (folder / "tpm_id.txt").write_text(f"identity-{fog}")
        (root / "pcr_ledger.json").write_text(json.dumps({"identity-1": "pcr", "identity-2": "pcr"}))
        gk = ZeroTrustGatekeeper(self.logger, f"FOG {fog}", {
            "tpm_freshness_window": 60,
            "experiment_ground_truth": {f"[EDGE {fog}_1]": {"role": "label_flip", "attestation_valid": True}},
        }, fog_num=fog)
        gk.tpm_state_root = str(root)
        gk.tpm_engine.verify_attestation_token = Mock(side_effect=lambda **kw: kw["token"]["nonce"] == kw["expected_nonce"])
        return gk

    def client(self, cid, fog):
        proxy = SimpleNamespace(cid=cid)
        proxy.get_properties = Mock(side_effect=lambda ins, **kwargs: SimpleNamespace(
            status=SimpleNamespace(code=Code.OK), properties=self.metrics(fog, ins.config["nonce"])))
        return proxy

    def metrics(self, fog, nonce):
        return {"log_prefix": f"[EDGE {fog}_1]", "role": "benign",
                "tpm_token_json": json.dumps({"IDi": f"identity-{fog}", "status": "attested", "nonce": nonce})}

    def configure(self, round_num, proxy=None):
        return configure_edge_fit([(proxy or self.proxy, FitIns(self.parameters, {}))],
                                  self.gatekeepers, round_num, self.nonces)[0][1]

    def test_authenticated_poisoner_receives_model_after_admission(self):
        fit = self.configure(1)
        self.proxy.get_properties.assert_called_once()
        self.assertIs(fit.parameters, self.parameters)
        self.assertFalse(fit.config["attestation_only"])
        self.assertEqual(self.gatekeepers[0].ground_truth("client-a")["role"], "label_flip")
        self.assertNotEqual(fit.config["nonce"], self.proxy.get_properties.call_args.args[0].config["nonce"])

    def test_invalid_first_quote_never_receives_parameters(self):
        self.gatekeepers[0].tpm_engine.verify_attestation_token.return_value = False
        self.gatekeepers[0].tpm_engine.verify_attestation_token.side_effect = None
        fit = self.configure(1)
        self.assertEqual(fit.parameters.tensors, [])
        self.assertTrue(fit.config["attestation_only"])
        self.assertEqual(self.gatekeepers[0].trust_db._db, {})

    def test_timeout_and_non_ok_response_fail_closed(self):
        for response in (TimeoutError("offline"), SimpleNamespace(status=SimpleNamespace(code=Code.GET_PROPERTIES_NOT_IMPLEMENTED))):
            with self.subTest(response=response):
                self.proxy.get_properties.side_effect = response if isinstance(response, Exception) else None
                self.proxy.get_properties.return_value = response
                fit = self.configure(1)
                self.assertEqual(fit.parameters.tensors, [])

    def test_existing_client_is_reauthenticated_each_round(self):
        self.configure(1)
        self.gatekeepers[0].tpm_engine.verify_attestation_token.side_effect = None
        self.gatekeepers[0].tpm_engine.verify_attestation_token.return_value = False
        fit = self.configure(2)
        self.assertEqual(self.proxy.get_properties.call_count, 2)
        self.assertEqual(fit.parameters.tensors, [])
        self.assertTrue(self.gatekeepers[0].trust_db.is_quarantined("identity-1"))

    def test_virtual_fog_selection_authenticates_with_correct_ledger(self):
        proxy = self.client("client-b", 2)
        fit = self.configure(1, proxy)
        self.assertIs(fit.parameters, self.parameters)
        self.assertNotIn(proxy.cid, self.gatekeepers[0].client_identities)
        self.assertIn(proxy.cid, self.gatekeepers[1].client_identities)

    def test_admission_quote_cannot_be_replayed_as_update_quote(self):
        self.configure(1)
        admission_nonce = self.proxy.get_properties.call_args.args[0].config["nonce"]
        result = SimpleNamespace(metrics=self.metrics(1, admission_nonce))
        self.assertEqual(self.gatekeepers[0].filter_node_updates(
            "fog", 1, [(self.proxy, result)], self.nonces), [])

    def test_two_quotes_per_round_do_not_accelerate_rehabilitation(self):
        self.configure(1)
        gk = self.gatekeepers[0]
        gk.trust_db.apply_behavioral_penalty("identity-1", "edge", 1)
        for round_num in range(2, 6):
            fit = self.configure(round_num)
            self.assertEqual(fit.parameters.tensors, [])
            response = SimpleNamespace(metrics=self.metrics(1, fit.config["nonce"]))
            gk.filter_node_updates("fog", round_num, [(self.proxy, response)], self.nonces)
            self.assertTrue(gk.trust_db.is_quarantined("identity-1"))
            self.assertEqual(gk.trust_db._db["identity-1"]["recovery_streak"], round_num - 1)
        fit = self.configure(6)
        self.assertIs(fit.parameters, self.parameters)
        self.assertEqual(gk.trust_db.get_score("identity-1"), .65)


if __name__ == "__main__":
    unittest.main()
