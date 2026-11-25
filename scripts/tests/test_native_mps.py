"""Native runtime regression checks; live Docker testing is separate."""
import copy
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from src.shared.models.cnn_lstm import CNNLSTMClassifier
from src.shared.models.factory import get_model
from src.shared.security.tpm_transport import edge_identity, request_tpm
from src.tier_edge.tpm_attestation import TPMAttestation
from src.tier_fog.gatekeeper import ZeroTrustGatekeeper
from src.tier_fog.tpm_verifier import TPMVerifier

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("native_launcher", ROOT / "scripts/simulation/run_native_mps.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)
torch.set_num_threads(1)


class NativePlacementTests(unittest.TestCase):
    def test_default_model_is_unmodified_and_opt_in_preserves_math_and_state(self):
        with patch.dict(os.environ, ZTA_NATIVE_MPS="0"):
            baseline = get_model("cnnlstm", 40, 15).eval()
        self.assertEqual(len(baseline.adaptive_pool._forward_hooks), 0)
        with patch.dict(os.environ, ZTA_NATIVE_MPS="1"):
            native = get_model("cnnlstm", 40, 15).eval()
        self.assertIs(type(native), CNNLSTMClassifier)
        self.assertIs(type(native.adaptive_pool), torch.nn.AdaptiveAvgPool1d)
        native.load_state_dict(baseline.state_dict())
        self.assertEqual(list(native.state_dict()), list(baseline.state_dict()))
        x = torch.randn(8, 40, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        a, b = baseline(x), native(y)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.sum().backward()
        b.sum().backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
        duplicate = copy.deepcopy(native)
        torch.testing.assert_close(duplicate(x), a, rtol=0, atol=0)
        torch.testing.assert_close(native.get_feature_vector(x), baseline.get_feature_vector(x), rtol=0, atol=0)

    @unittest.skipUnless(torch.backends.mps.is_available(), "requires native MPS")
    def test_mps_gradients_shap_and_int8_copy(self):
        from src.shared.security.adversarial_math import local_train_honest, pgd_attack
        from src.shared.utils.metrics import compute_shap_attributions
        from src.tier_edge.inference import quantize_model
        from torch.utils.data import DataLoader, TensorDataset
        with patch.dict(os.environ, ZTA_NATIVE_MPS="1"):
            model = get_model("cnnlstm", 40, 15).to("mps")
        x, y = torch.randn(16, 40), torch.arange(16) % 15
        before = model.fc3.weight.detach().clone()
        loss = local_train_honest(model, DataLoader(TensorDataset(x, y), batch_size=16), device="mps")
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(torch.equal(before, model.fc3.weight))
        self.assertTrue(pgd_attack(model, x.to("mps"), y.to("mps"), n_iter=3).isfinite().all())
        self.assertTrue(compute_shap_attributions(model, x, y, 4, "mps").isfinite().all())
        self.assertEqual(quantize_model(model)(x).device.type, "cpu")
        self.assertEqual(model(x.to("mps")).device.type, "mps")


class TPMTransportTests(unittest.TestCase):
    def test_route_and_json_transport_do_not_invoke_a_shell(self):
        result = subprocess.CompletedProcess([], 0, stdout=json.dumps({"result": {"status": "attested"}}))
        with patch.dict(os.environ, ZTA_TPM_CONTAINER_PREFIX="test-tpm-"), \
             patch("src.shared.security.tpm_transport.subprocess.run", return_value=result) as run:
            token = request_tpm("quote", "[EDGE 2_3]", nonce='quoted " nonce', round_num=4)
        self.assertEqual(token["status"], "attested")
        command = run.call_args.args[0]
        self.assertIn("test-tpm-2-3", command)
        self.assertIn("ZTA_TPM_CONTAINER_PREFIX=", command)
        self.assertIn("ZTA_INSECURE_MODE=false", command)
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertEqual(json.loads(run.call_args.kwargs["input"])["nonce"], 'quoted " nonce')
        self.assertEqual(run.call_args.kwargs["timeout"], 60)
        for value in ("edge_0_1", "[EDGE 1_2];echo bad", "../edge_1_1", "edge_1_2]"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                edge_identity(value)

    def test_native_attestation_does_not_provision_on_host(self):
        with patch.dict(os.environ, ZTA_TPM_CONTAINER_PREFIX="test-tpm-", ZTA_INSECURE_MODE="false"), \
             patch.object(TPMAttestation, "_provision_ak") as provision, \
             patch("src.tier_edge.tpm_attestation.request_tpm", return_value={"status": "attested"}) as request:
            engine = TPMAttestation(logging.getLogger("test"), tcti="swtpm:host=tpm-1-2,port=2321")
            self.assertEqual(engine.generate_attestation_token("nonce", "[EDGE 1_2]", 1)["status"], "attested")
            provision.assert_not_called()
            request.assert_called_once_with("quote", "[EDGE 1_2]", nonce="nonce", round_num=1)

    def test_verifier_fails_closed_on_transport_failure(self):
        with patch.dict(os.environ, ZTA_TPM_CONTAINER_PREFIX="test-tpm-", ZTA_INSECURE_MODE="false"), \
             patch("src.tier_fog.tpm_verifier.request_tpm", side_effect=subprocess.TimeoutExpired("docker", 60)):
            self.assertFalse(TPMVerifier(logging.getLogger("test")).verify_attestation_token(
                {}, "nonce", "/native/runtime/tpm_state/edge_1_1/ak.pub", 1, "pcr"))

    def test_state_root_keeps_container_default(self):
        for root in ("", "/native/state"):
            env = {**os.environ}
            env.pop("ZTA_TPM_STATE_ROOT", None)
            if root:
                env["ZTA_TPM_STATE_ROOT"] = root
            with patch.dict(os.environ, env, clear=True):
                gatekeeper = ZeroTrustGatekeeper(logging.getLogger("test"), "fog", {})
                self.assertEqual(gatekeeper.tpm_state_root, root or "/app/runtime/tpm_state")


class NativeLauncherTests(unittest.TestCase):
    def test_config_artifacts_and_containers_are_isolated(self):
        before = {p: p.read_bytes() for p in (ROOT / "config").glob("*.yaml")}
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "native_test"
            run.mkdir()
            configs, topology = launcher.prepare_run(run, ROOT / "config", True)
            self.assertEqual(topology, [2])
            self.assertEqual(configs["training"]["num_rounds"], 2)
            self.assertTrue(Path(configs["training"]["dataset_path"]).is_symlink())
            self.assertTrue(Path(configs["training"]["dataset_path"]).is_relative_to(run))
            compose = launcher.tpm_compose(run, topology, "test-native", "test-image")
            self.assertEqual(set(compose["services"]), {"tpm-1-1", "tpm-1-2"})
            for name, service in compose["services"].items():
                self.assertEqual(service["container_name"], "test-native-" + name)
                self.assertNotIn("ports", service)
                self.assertNotIn("deploy", service)
                self.assertEqual(service["environment"]["ZTA_INSECURE_MODE"], "false")
                self.assertIn("--bootstrap", service["command"][0])
        self.assertEqual(before, {p: p.read_bytes() for p in before})


if __name__ == "__main__":
    unittest.main()
