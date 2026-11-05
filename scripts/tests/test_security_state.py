"""Security-state regressions; runnable without the ML dependencies."""
import json
import base64
import hashlib
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.tier_fog.gatekeeper import ZeroTrustGatekeeper
from src.tier_fog.tpm_verifier import TPMVerifier
from src.tier_fog.trust_db import TrustDatabase

LOGGER = logging.getLogger('security-tests')
LOGGER.addHandler(logging.NullHandler())
LOGGER.propagate = False


class TrustTests(unittest.TestCase):
    def setUp(self):
        self.db = TrustDatabase(LOGGER)

    def quarantine(self):
        self.db.process_attestation('a', 'a', True, 1)
        self.db.apply_behavioral_penalty('a', 'a', 1)

    def test_invalid_first_claim_does_not_register(self):
        self.db.process_attestation('a', 'a', False, 1)
        self.assertNotIn('a', self.db._db)
        self.assertTrue(self.db.is_quarantined('a'))

    def test_readmission_on_fifth_valid_round(self):
        self.quarantine()
        for r in range(2, 6):
            self.db.process_attestation('a', 'a', True, r)
            self.assertTrue(self.db.is_quarantined('a'))
            self.assertEqual(self.db._db['a']['recovery_streak'], r - 1)
        self.db.process_attestation('a', 'a', True, 6)
        self.assertFalse(self.db.is_quarantined('a'))
        self.assertEqual(self.db.get_score('a'), 0.65)

    def test_duplicate_quote_does_not_advance_recovery(self):
        self.quarantine()
        for _ in range(8):
            self.db.process_attestation('a', 'a', True, 2)
        self.assertEqual(self.db._db['a']['recovery_streak'], 1)

    def test_missing_round_and_failure_break_streak(self):
        self.quarantine()
        self.db.process_attestation('a', 'a', True, 2)
        self.db.process_attestation('a', 'a', True, 4)
        self.assertEqual(self.db._db['a']['recovery_streak'], 1)
        self.db.process_attestation('a', 'a', False, 5)
        self.db.process_attestation('a', 'a', True, 5)
        self.assertEqual(self.db._db['a']['recovery_streak'], 0)
        self.assertFalse(self.db._db['a']['last_attestation_passed'])

    def test_penalty_once_per_round(self):
        self.quarantine()
        self.db.process_attestation('a', 'a', False, 1)
        self.assertEqual(self.db.get_score('a'), .35)

    def test_invalid_quote_after_readmission_restarts_quarantine(self):
        self.quarantine()
        for r in range(2, 7):
            self.db.process_attestation('a', 'a', True, r)
        self.db.process_attestation('a', 'a', False, 7)
        self.db.process_attestation('a', 'a', True, 8)
        self.assertTrue(self.db.is_quarantined('a'))
        self.assertEqual(self.db._db['a']['recovery_streak'], 1)


class GatekeeperTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'ZTA_INSECURE_MODE': 'false'})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        (root / 'edge_1_1').mkdir()
        (root / 'edge_1_1' / 'tpm_id.txt').write_text('a')
        (root / 'pcr_ledger.json').write_text(json.dumps({'a': 'known-pcr'}))
        self.gk = ZeroTrustGatekeeper(LOGGER, '[FOG 1]', {'tpm_freshness_window': 60}, fog_num=1)
        self.gk.tpm_state_root = str(root)

    def result(self, cid='client-a', token=None, label='[EDGE 1_1]'):
        token = {'IDi': 'a', 'status': 'attested'} if token is None else token
        return (SimpleNamespace(cid=cid), SimpleNamespace(metrics={
            'log_prefix': label, 'tpm_token_json': token if isinstance(token, str) else json.dumps(token)}))

    def authenticate(self):
        with patch.object(self.gk.tpm_engine, 'verify_attestation_token', return_value=True):
            self.assertEqual(len(self.gk.filter_node_updates('fog', 1, [self.result()], {'client-a': 'nonce'})), 1)

    def test_pipeline_monitor_recognizes_update_token_outcomes(self):
        from scripts.ops.verify_pipeline import parse_round_state

        for verified, round_num in ((True, 1), (False, 2)):
            with self.subTest(verified=verified):
                with patch.object(self.gk.tpm_engine, 'verify_attestation_token', return_value=verified):
                    with self.assertLogs(LOGGER, level='INFO') as captured:
                        self.gk.filter_node_updates('fog', round_num, [self.result()], {'client-a': 'nonce'})
                logs = [dict(message=r.getMessage(), level=r.levelname,
                             node='[FOG 1 SERVER]') for r in captured.records]
                state = parse_round_state(logs, {})
                if verified:
                    self.assertEqual(state['edge_tokens_verified'], {'[EDGE 1_1]'})
                    self.assertEqual(state['edge_tokens_rejected'], {})
                else:
                    self.assertEqual(state['edge_tokens_verified'], set())
                    self.assertIn('Attestation verification failed',
                                  state['edge_tokens_rejected']['[EDGE 1_1]'])
                self.assertTrue(all(r.round == round_num for r in captured.records
                                    if 'Cryptographic Result' in r.getMessage()
                                    or 'REJECTED: Attestation/PCR' in r.getMessage()))

    def test_client_cannot_request_insecure_bypass(self):
        result = self.result(token={'IDi': 'a', 'status': 'insecure_bypass'})
        with patch('src.tier_fog.tpm_verifier.subprocess.run') as command:
            self.assertEqual(self.gk.filter_node_updates('fog', 1, [result], {'client-a': 'nonce'}), [])
            command.assert_not_called()
        self.assertEqual(self.gk.trust_db._db, {})

    def test_bound_malformed_token_penalizes_real_identity(self):
        self.authenticate()
        self.gk.filter_node_updates('fog', 2, [self.result(token='not-json')], {'client-a': 'nonce'})
        self.assertEqual(self.gk.trust_db.get_score('a'), .35)
        self.assertTrue(self.gk.trust_db.is_quarantined('a'))

    def test_missing_response_resets_bound_recovery(self):
        self.authenticate()
        self.gk.trust_db.apply_behavioral_penalty('a', 'a', 1)
        with patch.object(self.gk.tpm_engine, 'verify_attestation_token', return_value=True):
            self.gk.filter_node_updates('fog', 2, [self.result()], {'client-a': 'nonce'})
        self.gk.filter_node_updates('fog', 3, [], {'client-a': 'new-nonce'})
        self.assertEqual(self.gk.trust_db._db['a']['recovery_streak'], 0)
        self.assertEqual(self.gk.last_outcomes['client-a']['reason'], 'missing_response')

    def test_unbound_claim_cannot_penalize_another_identity(self):
        self.authenticate()
        self.gk.filter_node_updates('fog', 2, [self.result(cid='attacker')], {'attacker': 'nonce'})
        self.assertEqual(self.gk.trust_db.get_score('a'), .7)
        self.assertNotIn('attacker', self.gk.client_identities)

    def test_changed_fog_label_cannot_rebind_identity(self):
        self.authenticate()
        self.gk.filter_node_updates('fog', 2, [self.result(label='[EDGE 2_1]')], {'client-a': 'nonce'})
        self.assertEqual(self.gk.client_identities['client-a'], ('a', '[EDGE 1_1]'))
        self.assertTrue(self.gk.trust_db.is_quarantined('a'))

    def test_missing_challenge_rejected_even_with_valid_mock_signature(self):
        with patch.object(self.gk.tpm_engine, 'verify_attestation_token', return_value=True) as verify:
            self.assertEqual(self.gk.filter_node_updates('fog', 1, [self.result()], {}), [])
            verify.assert_not_called()

    def test_duplicate_client_result_only_processed_once(self):
        self.authenticate()
        self.gk.trust_db.apply_behavioral_penalty('a', 'a', 1)
        with patch.object(self.gk.tpm_engine, 'verify_attestation_token', return_value=True) as verify:
            self.gk.filter_node_updates('fog', 2, [self.result(), self.result()], {'client-a': 'nonce'})
            verify.assert_called_once()
        self.assertEqual(self.gk.trust_db._db['a']['recovery_streak'], 1)

    def test_nonfinite_timestamp_rejected_before_crypto(self):
        verifier = TPMVerifier(LOGGER)
        with patch('src.tier_fog.tpm_verifier.subprocess.run') as command:
            self.assertFalse(verifier.verify_attestation_token(
                {'status': 'attested', 'timestamp': float('nan')}, 'nonce', 'key', 1, 'pcr'))
            command.assert_not_called()

    def test_client_role_cannot_override_server_ground_truth(self):
        self.gk.run_metadata["experiment_ground_truth"] = {
            "[EDGE 1_1]": {"role": "label_flip", "attestation_valid": True}}
        proxy, response = self.result()
        response.metrics["role"] = "benign"
        with patch.object(self.gk.tpm_engine, 'verify_attestation_token', return_value=True):
            self.gk.filter_node_updates('fog', 1, [(proxy, response)], {'client-a': 'nonce'})
        self.assertEqual(self.gk.ground_truth(proxy.cid)["role"], "label_flip")
        self.assertTrue(self.gk.last_outcomes[proxy.cid]["expected_valid"])

    def test_replay_signature_and_pcr_rejection_without_runtime_attack_manager(self):
        timestamp = 1000.0
        digest = hashlib.sha256(f"nonce:{timestamp:.6f}".encode()).hexdigest()
        token = {"status": "attested", "timestamp": timestamp, "IDi": "a",
                 "quote_msg": base64.b64encode(b"quote").decode(),
                 "signature": base64.b64encode(b"signature").decode()}
        for signature_ok, nonce, pcr, expected in [
            (True, digest, "aabb", True), (False, digest, "aabb", False),
            (True, "deadbeef", "aabb", False), (True, digest, "ccdd", False),
        ]:
            with self.subTest(signature_ok=signature_ok, nonce=nonce, pcr=pcr):
                replies = [SimpleNamespace(returncode=0 if signature_ok else 1, stderr="invalid"),
                           SimpleNamespace(returncode=0, stdout=f"extraData: {nonce}\npcrDigest: {pcr}")]
                with patch('src.tier_fog.tpm_verifier.time.time', return_value=timestamp), \
                     patch('src.tier_fog.tpm_verifier.subprocess.run', side_effect=replies):
                    self.assertEqual(self.gk.tpm_engine.verify_attestation_token(
                        token, "nonce", "key", 1, "aabb"), expected)
        with patch('src.tier_fog.tpm_verifier.time.time', return_value=timestamp + 61), \
             patch('src.tier_fog.tpm_verifier.subprocess.run') as command:
            self.assertFalse(self.gk.tpm_engine.verify_attestation_token(token, "nonce", "key", 1, "aabb"))
            command.assert_not_called()


if __name__ == '__main__':
    unittest.main()
