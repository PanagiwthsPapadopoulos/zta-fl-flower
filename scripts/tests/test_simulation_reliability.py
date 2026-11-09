"""Simulation scheduling, attestation timing, recovery, and scratch regressions."""
import base64
import hashlib
import json
import logging
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from scripts.ops import simulation_scratch as scratch
from src.shared.utils.simulation_resources import federation_config, simulation_config
from src.shared.utils.simulation_transport import TimedClientProxy
from src.tier_fog.tpm_verifier import TPMVerifier


class SimulationReliabilityTests(unittest.TestCase):
    def test_gpu_workers_reserve_resources_without_reducing_population(self):
        cfg = federation_config({}, 100, "cuda", 3)
        self.assertIn("num-supernodes=100", cfg)
        self.assertIn("client-resources-num-gpus=1.0", cfg)
        self.assertIn("init-args-num-cpus=3.0", cfg)
        self.assertIn("client-resources-num-gpus=0.0", federation_config({}, 100, "cpu", 1))
        manifest = {"tool": {"flwr": {"federations": {"local": {
            "options": {"backend": {"client-resources": {"num-gpus": 0}}}}}}}}
        with self.assertRaisesRegex(ValueError, "must reserve"):
            federation_config(manifest, 100, "cuda", 8)

    def test_simulation_freshness_override_preserves_deployment_configuration(self):
        original = {"tpm_freshness_window": 60}
        with patch("src.shared.utils.config_loader.get_merged_config", return_value=original.copy()):
            self.assertEqual(simulation_config({})["tpm_freshness_window"], 120)
        self.assertEqual(original["tpm_freshness_window"], 60)
        with patch("src.shared.utils.config_loader.get_merged_config", return_value={"tpm_freshness_window": 300}):
            self.assertEqual(simulation_config({})["tpm_freshness_window"], 300)

    def test_arrival_time_is_measured_by_coordinator_after_response(self):
        received = {}
        result = SimpleNamespace(metrics={"received_at": -100})
        proxy = SimpleNamespace(cid="edge", fit=Mock(return_value=result))
        with patch("src.shared.utils.simulation_transport.time.time", return_value=105):
            self.assertIs(TimedClientProxy(proxy, received).fit("instruction", timeout=30), result)
        self.assertEqual(received, {"edge": 105})
        proxy.fit.assert_called_once_with("instruction", timeout=30)

    def test_waiting_for_other_clients_does_not_expire_quote_but_stale_quotes_fail(self):
        nonce, token_time, pcr = "round-specific-nonce", 100., "abcd"
        digest = hashlib.sha256(f"{nonce}:{token_time:.6f}".encode()).hexdigest()
        token = {"status": "attested", "IDi": "edge", "timestamp": token_time,
                 "quote_msg": base64.b64encode(b"quote").decode(),
                 "signature": base64.b64encode(b"signature").decode()}
        verifier = TPMVerifier(logging.getLogger("test"))
        verifier.insecure_mode = False
        def verify(received, printed_digest=digest, check_result=0):
            calls = [SimpleNamespace(returncode=check_result, stderr="invalid"),
                     SimpleNamespace(returncode=0, stdout=f"pcrDigest: {pcr}\nextraData: {printed_digest}")]
            with patch("src.tier_fog.tpm_verifier.time.time", return_value=1000), \
                 patch("src.tier_fog.tpm_verifier.subprocess.run", side_effect=calls) as tool:
                accepted = verifier.verify_attestation_token(token, nonce, "/unused/ak.pub", 1,
                                                             pcr, 120, received_at=received)
                return accepted, tool.call_count
        self.assertEqual(verify(105), (True, 2))  # verified later, fresh on arrival
        self.assertEqual(verify(230), (False, 0))  # already stale on arrival
        self.assertEqual(verify(105, printed_digest="bad"), (False, 2))
        self.assertEqual(verify(105, check_result=1), (False, 1))
        self.assertEqual(verify(None), (False, 0))  # deployment still checks current time

    def client(self):
        from src.entrypoints.sim_client import Client
        client = Client.__new__(Client)
        client.logger = Mock()
        client.log_prefix = "[EDGE 1_1]"
        client.train_config = client.dataset_metadata = {}
        client.model, client.trainer = Mock(), Mock()
        client.device = "cuda"
        return client

    def test_cuda_oom_retries_original_update_on_cpu_with_normal_trainer(self):
        client = self.client()
        params, config = [1], {"server_round": 3, "nonce": "challenge"}
        client.trainer.execute_training.side_effect = [torch.OutOfMemoryError("OOM"),
                                                       ([], 7, {"tpm_token_json": "signed"})]
        with patch("torch.cuda.empty_cache"):
            _, count, metrics = client.fit(params, config)
        self.assertEqual(count, 7)
        self.assertTrue(metrics["simulation_cpu_retry"])
        self.assertEqual(metrics["tpm_token_json"], "signed")
        self.assertEqual(client.trainer.device, "cpu")
        for call in client.trainer.execute_training.call_args_list:
            self.assertEqual(call.args, (params, 3, config))

    def test_unrecoverable_client_exception_is_not_a_successful_empty_update(self):
        client = self.client()
        client.trainer.execute_training.side_effect = RuntimeError("disk error")
        with self.assertRaisesRegex(RuntimeError, "disk error"):
            client.fit([], {"server_round": 1})

    def test_partial_round_is_rejected_before_aggregation(self):
        from src.tier_cloud.cloud_aggregator_sim import CloudAggregator
        strategy = CloudAggregator.__new__(CloudAggregator)
        strategy.active_nonces = {"one": "a", "two": "b"}
        strategy.fail_round = Mock(side_effect=RuntimeError("incomplete"))
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            strategy.aggregate_fit(1, [(SimpleNamespace(cid="one"), SimpleNamespace(metrics={}))], [])
        self.assertEqual(strategy.fail_round.call_args.kwargs["missing_clients"], ["two"])

    def test_scratch_archive_preserves_state_and_removes_only_owned_directory(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, ZTA_SIM_SCRATCH_BASE=directory):
                local = scratch.prepare(root, "attempt")
            self.addCleanup(lambda: __import__('shutil').rmtree(local, ignore_errors=True))
            db = local / "flower/local-superlink/state.db"
            db.parent.mkdir(parents=True)
            with sqlite3.connect(db) as con:
                con.execute("create table task(sub_status,details,finished_at,pending_at)")
                con.execute("insert into task values ('failed','No heartbeat received from the task','now','then')")
            self.assertEqual(scratch.flower_status(root, "attempt")["status"], "failed")
            with self.assertRaisesRegex(RuntimeError, "ownership mismatch"):
                scratch.archive_scratch(root, root / "archive", "foreign")
            self.assertTrue(db.exists())
            scratch.archive_scratch(root, root / "archive", "attempt")
            self.assertFalse(local.exists())
            self.assertTrue((root / "archive/local_scratch/flower/local-superlink/state.db").exists())


if __name__ == "__main__":
    unittest.main()
