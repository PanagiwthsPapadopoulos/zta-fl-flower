"""TPM routing and Docker GPU policy regressions, without live containers or GPUs."""
from contextlib import ExitStack
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import yaml

from src.shared.utils import hardware
from src.tier_edge.tpm_attestation import TPMAttestation

ROOT = Path(__file__).resolve().parents[2]


class TPMRoutingTests(unittest.TestCase):
    def test_explicit_node_environment_routes_each_edge_without_leaking_state(self):
        source = (ROOT / 'scripts/simulation/deploy_code_sim_apptainer.sh').read_text()
        patch_source = source.split('cat << \'EOF\' > "$PATCH_FILE"\n', 1)[1].split('\nEOF', 1)[0]
        recorder = Mock(return_value=subprocess.CompletedProcess([], 0, stdout=''))
        inherited = 'swtpm:host=tpm-9-9,port=2321'
        with patch.dict(os.environ, TPM2TOOLS_TCTI=inherited, ZTA_INSECURE_MODE='false'), \
             patch('subprocess.run', recorder):
            namespace = {}
            exec(compile(patch_source, 'tpm_socket_patch.py', 'exec'), namespace)
            run = namespace['_patched_run']
            for fog, edge in ((1, 2), (3, 4), (1, 2)):
                tcti = f'swtpm:host=tpm-{fog}-{edge},port=2321'
                engine = TPMAttestation(logging.getLogger('tpm-test'), tcti=tcti)
                expected = f'swtpm:path=/app/runtime/tpm_state/edge_{fog}_{edge}/swtpm.sock'
                self.assertEqual(recorder.call_args.kwargs['env']['TPM2TOOLS_TCTI'], expected)
                env = engine._get_tpm_env()
                env['KEEP_ME'] = 'value'
                run(['/usr/bin/tpm2_quote'], env=env, timeout=12, check=True)
                self.assertEqual(recorder.call_args.kwargs['env']['TPM2TOOLS_TCTI'], expected)
                self.assertEqual(recorder.call_args.kwargs['env']['KEEP_ME'], 'value')
                self.assertEqual(recorder.call_args.kwargs['timeout'], 12)
                self.assertTrue(recorder.call_args.kwargs['check'])
                self.assertEqual(env['TPM2TOOLS_TCTI'], tcti)
                self.assertEqual(os.environ['TPM2TOOLS_TCTI'], inherited)
            run(['tpm2_quote'])
            self.assertEqual(recorder.call_args.kwargs['env']['TPM2TOOLS_TCTI'],
                             'swtpm:path=/app/runtime/tpm_state/edge_9_9/swtpm.sock')
            # Explicit empty env must not fall back to the parent's routing.
            for env in ({}, {'TPM2TOOLS_TCTI': 'device:/dev/tpmrm0'},
                        {'TPM2TOOLS_TCTI': 'swtpm:path=/existing.sock'}):
                run(['tpm2_quote'], env=env)
                self.assertIs(recorder.call_args.kwargs['env'], env)
            env = {'TPM2TOOLS_TCTI': inherited}
            run(['python3'], env=env)
            self.assertIs(recorder.call_args.kwargs['env'], env)


class GPUSelectionTests(unittest.TestCase):
    def test_tier_policy_and_cpu_fallback(self):
        cases = [
            # allowed tiers, host, CUDA, MPS, explicit tier, expected
            (('cloud',), 'true', True, False, None, 'cuda'),
            (('edge',), 'true', True, False, None, 'cpu'),
            (('edge',), 'true', True, False, 'edge', 'cuda'),
            (('cloud',), 'true', True, False, 'edge', 'cpu'),
            (('edge',), 'false', True, False, 'edge', 'cpu'),
            (('edge',), 'true', False, False, 'edge', 'cpu'),
            (('edge',), 'true', False, True, 'edge', 'mps'),
        ]
        for tiers, host, cuda, mps, tier, expected in cases:
            with self.subTest(case=(tiers, host, cuda, mps, tier)), ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, ZTA_TIER='cloud', ZTA_GPU_AVAILABLE=host))
                stack.enter_context(patch.object(hardware, '_gpu_enabled_tiers', return_value=tiers))
                stack.enter_context(patch.object(hardware.torch.cuda, 'is_available', return_value=cuda))
                stack.enter_context(patch.object(hardware.torch.backends.mps, 'is_available', return_value=mps))
                self.assertEqual(hardware.get_device(tier=tier), expected)
                self.assertEqual(hardware.get_dataloader_kwargs(device=expected)['pin_memory'], expected != 'cpu')

    def test_compose_assigns_tiers_and_gpu_flags_matching_reservations(self):
        for mode in ('deployment', 'simulation'):
            for gpu in (False, True):
                with self.subTest(mode=mode, gpu=gpu), TemporaryDirectory() as directory:
                    root = Path(directory)
                    bin_dir = root / 'bin'
                    bin_dir.mkdir()
                    # Isolate detection from whatever GPU tools the test host has.
                    for name in ('mkdir', 'seq', 'cat'):
                        (bin_dir / name).symlink_to(shutil.which(name))
                    if gpu:
                        tool = bin_dir / 'nvidia-smi'
                        tool.write_text('#!/bin/sh\nexit 0\n')
                        tool.chmod(0o755)
                    compose = root / 'compose.yml'
                    env = {**os.environ, 'PATH': str(bin_dir), 'PROJECT_ROOT': str(root),
                           'COMPOSE_FILE': str(compose), 'NUM_FOGS': '1', 'CLOUD_SA': '9001',
                           'CLOUD_FL': '9002', 'CLOUD_CTRL': '9003', 'FOG_SA_BASE': '9100',
                           'FOG_FL_BASE': '9200', 'FOG_CTRL_BASE': '9300',
                           'FOG_CIO_BASE': '9400', 'EDGE_CIO_BASE': '10000'}
                    if mode == 'deployment':
                        script, args = 'generate_compose.sh', ['true', 'UTC', '1']
                        tiers = {'cloud-serverapp': 'cloud', 'fog-1-serverapp': 'fog',
                                 'fog-1-clientapp': 'fog', 'edge-1-1-clientapp': 'edge'}
                    else:
                        script, args = 'generate_compose_sim.sh', ['UTC', '1']
                        tiers = {'cloud': 'cloud'}
                    subprocess.run(['/bin/bash', str(ROOT / 'scripts' / mode / script), *args],
                                   env=env, check=True, capture_output=True, timeout=10)
                    services = yaml.safe_load(compose.read_text())['services']
                    for name, tier in tiers.items():
                        service = services[name]
                        environment = dict(value.split('=', 1) for value in service['environment'])
                        self.assertEqual(environment['ZTA_TIER'], tier)
                        reserved = 'deploy' in service
                        self.assertEqual(environment['ZTA_GPU_AVAILABLE'], str(reserved).lower())
                        self.assertEqual(reserved, gpu and name != 'fog-1-clientapp')

    def test_docker_dispatch_passes_gpu_allocation_without_changing_node_count(self):
        source = (ROOT / 'scripts/simulation/deploy_code_sim_docker.sh').read_text()
        dispatch = source[source.index('docker exec ${EXEC_ARGS'):source.index('\necho ""', source.index('docker exec ${EXEC_ARGS'))]
        for device, expected in (('cuda', '1.0'), ('cpu', '0.0'), ('mps', '0.0')):
            with self.subTest(device=device), TemporaryDirectory() as directory:
                root = Path(directory)
                # Execute the real nested shell command with local CLI substitutes.
                # Its Python expression still runs, with only the device probe mocked.
                python = root / 'python3'
                python.write_text('#!/bin/sh\nexec "' + sys.executable + '" -B -c \'import sys; from unittest.mock import patch; from src.shared.utils import hardware\nwith patch.object(hardware, "get_device", return_value="' + device + '"):\n exec(sys.argv[1])\' "$2"\n')
                python.chmod(0o755)
                script = root / 'dispatch.sh'
                script.write_text('''set -euo pipefail
docker() { while [ "$1" != bash ]; do shift; done; "$@"; }
EXEC_ARGS=()
BIN_FLWR=printf
FEDERATION=local
TOTAL_EDGES=100
''' + dispatch.replace('/app/logs/system/run_cloud.log', str(root / 'run.log')).replace('$BIN_FLWR run', '$BIN_FLWR \'%s\\n\' run'))
                env = {**os.environ, 'PATH': str(root) + os.pathsep + os.environ['PATH'],
                       'PYTHONPATH': str(ROOT), 'N_SIM': '100', 'RUN_CFG': 'mode="sim" tier="cloud" min_clients=100'}
                subprocess.run(['/bin/bash', str(script)], env=env, check=True, capture_output=True, timeout=30)
                args = (root / 'run.log').read_text().splitlines()
                self.assertIn(f'num-supernodes=100 client-resources-num-gpus={expected}', args)
                self.assertIn(env['RUN_CFG'], args)


if __name__ == '__main__':
    unittest.main()
