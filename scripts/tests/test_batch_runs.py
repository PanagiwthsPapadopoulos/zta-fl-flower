"""Batch control tests use temporary workspaces and fake container backends only."""
from contextlib import redirect_stdout, contextmanager
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml
from scripts.ops import batch_runs as batch
from src.shared.utils.simulation_lifecycle import track_simulation, record_saved_round, atomic_json


LIMITS = dict(boot_timeout=30, prepare_timeout=20, stall_timeout=10, run_timeout=100,
              shutdown_timeout=5, poll_interval=1)


@contextmanager
def cwd(path):
    before = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(before)


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "config").mkdir()
        self.baseline = {
            "network.yaml": dict(num_fogs=1, uniform_edges_per_fog=2, custom_fog_topology="[]", quantization_bits=8, round_timeout=3600.),
            "training.yaml": dict(run_name="", num_rounds=2, learning_rate=.1, dataset_path="data/example/raw/input.csv"),
            "security.yaml": dict(label_flip_ratio=0),
        }
        for name, value in self.baseline.items():
            (self.root / "config" / name).write_text("# baseline comment\n" + yaml.safe_dump(value))
        self.queue = self.root / "queue"
        self.queue.mkdir()
        self.add_run("01", "training.yaml", {"learning_rate": .01})
        self.add_run("02", "security.yaml", {"label_flip_ratio": .2})
        self.folder = self.root / "results" / "test"
        self.folder.mkdir(parents=True)
        (self.root / "runtime").mkdir()

    def add_run(self, name, config, values):
        folder = self.queue / name
        folder.mkdir(exist_ok=True)
        (folder / config).write_text(yaml.safe_dump(values))


    def test_overrides_start_from_same_baseline_and_preserve_omitted_keys(self):
        baseline, entries = batch.plan_queue(self.root, self.queue)
        self.assertEqual(entries[0]["configs"]["training.yaml"]["learning_rate"], .01)
        self.assertEqual(entries[1]["configs"]["training.yaml"]["learning_rate"], .1)
        self.assertEqual(entries[0]["configs"]["training.yaml"]["num_rounds"], 2)
        self.assertEqual(baseline, self.baseline)


    def test_host_launches_boot_and_worker_once_without_process_inventory(self):
        session = self.root / 'results/batch_archives/session'
        state = dict(runtime='apptainer', infrastructure_id='infra', limits=LIMITS)
        marker = self.root / 'runtime/infra/.network_ready'
        boot, worker = Mock(pid=101), Mock(pid=102, returncode=0)
        worker.poll.return_value = 0
        def launch(args, **kwargs):
            if 'boot_network_sim_' in args[1]:
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()
                return boot
            return worker
        with patch.object(batch.subprocess, 'Popen', side_effect=launch) as popen, patch.object(batch, 'stop_process') as stop, redirect_stdout(io.StringIO()):
            self.assertEqual(batch.supervise(self.root, session, state), 0)
        self.assertEqual(popen.call_count, 2)
        self.assertEqual([c.args[0] for c in stop.call_args_list], [worker, boot])
        self.assertFalse(marker.exists())

    def test_boot_timeout_stops_boot_without_launching_worker(self):
        session = self.root / 'results/batch_archives/session'
        state = dict(runtime='apptainer', infrastructure_id='infra', limits=LIMITS)
        boot = Mock(pid=101)
        with patch.object(batch.subprocess, 'Popen', return_value=boot) as popen, patch.object(batch, 'stop_process') as stop, patch.object(batch.time, 'monotonic', side_effect=[0, 31]):
            with self.assertRaisesRegex(batch.BatchError, 'boot timed out'):
                batch.supervise(self.root, session, state)
        popen.assert_called_once()
        stop.assert_called_once_with(boot, LIMITS['shutdown_timeout'])

    def test_timeout_override_is_independent_for_each_run(self):
        self.add_run("01", "network.yaml", {"round_timeout": 7200.5})
        _, entries = batch.plan_queue(self.root, self.queue)
        self.assertEqual(entries[0]["configs"]["network.yaml"]["round_timeout"], 7200.5)
        self.assertEqual(entries[1]["configs"]["network.yaml"]["round_timeout"], 3600.)
        self.assertEqual(entries[0]["topology"], entries[1]["topology"])


    def test_partial_or_unrelated_json_never_indicates_progress(self):
        (self.folder / "batch_progress.json").write_text('{"completed_round":')
        atomic_json(self.root / "results/latest.json", {"round": 2})
        self.assertEqual(batch.progress(self.root, "test"), -1)


    def test_lifecycle_records_real_resolved_config_success_and_exceptions(self):
        config = {"batch_attempt": "test", "run_name": "test", "learning_rate": .01}
        with cwd(self.root), track_simulation(config, {"mode": "sim", "min_clients": 2}):
            self.assertEqual(batch.read_json(self.folder / "batch_lifecycle.json")["status"], "STARTED")
            record_saved_round({"batch_attempt": "test"}, 2)
        self.assertEqual(batch.read_json(self.folder / "batch_lifecycle.json")["status"], "COMPLETED")
        runtime = batch.read_json(self.folder / "resolved_runtime.json")
        self.assertEqual(runtime["resolved_config"], config)
        self.assertEqual(runtime["flower_run_config"]["min_clients"], 2)
        with self.assertRaisesRegex(RuntimeError, "training broke"), cwd(self.root), track_simulation(config, {}):
            raise RuntimeError("training broke")
        status = batch.read_json(self.folder / "batch_lifecycle.json")
        self.assertEqual(status["status"], "FAILED")
        self.assertIn("training broke", status["traceback"])

    def test_nonbatch_lifecycle_does_not_create_files(self):
        with cwd(self.root), track_simulation({"run_name": "ordinary"}, {}):
            pass
        self.assertFalse((self.root / "results/ordinary").exists())


    def test_dry_run_changes_nothing_and_needs_no_container_runtime(self):
        before = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        with patch.object(batch, "ROOT", self.root), patch.object(batch, "supervise", side_effect=AssertionError("must not boot")), redirect_stdout(io.StringIO()):
            self.assertEqual(batch.main(["--runtime", "docker", "--queue", str(self.queue), "--dry-run"]), 0)
        after = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)


    def test_simulation_launchers_propagate_foreground_failure_without_global_cleanup(self):
        # Run the actual shell launchers against recording CLI stubs. No images or network.
        source_root = Path(batch.__file__).resolve().parents[2]
        for backend in ("docker", "apptainer"):
            with self.subTest(backend=backend), TemporaryDirectory() as directory:
                root = Path(directory)
                for relative in (f"scripts/simulation/deploy_code_sim_{backend}.sh", "scripts/setup/parse_topology.py", "pyproject.toml"):
                    target = root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_root / relative, target)
                shutil.copytree(self.root / "config", root / "config")
                for relative in ("runtime/infra", "logs/system", "logs/nodes/cloud", "data", "images", "docker", "bin"):
                    (root / relative).mkdir(parents=True, exist_ok=True)
                (root / "runtime/infra/.network_ready").touch()
                (root / "logs/nodes/cloud/existing.log").write_text("must survive managed launch")
                stub = root / "bin" / backend
                stub.write_text("#!" + sys.executable + "\n" + '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["CALL_RECORD"], "a") as f:
    f.write(json.dumps(args) + "\\n")
with open(os.environ["CALL_RECORD"] + ".env", "a") as f:
    f.write(json.dumps({"pythonpath": os.environ.get("APPTAINERENV_PYTHONPATH")}) + "\\n")
if "compose" in args and "ps" in args:
    print("container-id")
elif args[:1] == ["ps"]:
    print("sim-tpm-1-1")
elif "--writable-tmpfs" in args or (args[:1] == ["exec"] and "RUN_CFG=mode=\\"sim\\" tier=\\"cloud\\" min_clients=1" in args):
    sys.exit(7)
''')
                stub.chmod(0o755)
                pgrep = root / "bin/pgrep"
                pgrep.write_text("#!/bin/sh\nexit 0\n")
                pgrep.chmod(0o755)
                env = {**os.environ, "PATH": str(root / "bin") + os.pathsep + os.environ["PATH"],
                       "CALL_RECORD": str(root / "calls.jsonl"), "ZTA_BATCH_MANAGED": "1",
                       "ZTA_BATCH_ATTEMPT": "test", "ZTA_BATCH_PHASE_FILE": str(root / "runtime/phase"), "USER": "test"}
                result = subprocess.run(["bash", str(root / f"scripts/simulation/deploy_code_sim_{backend}.sh")], env=env, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
                self.assertEqual((root / "runtime/phase").read_text().strip(), "training")
                self.assertTrue((root / "logs/nodes/cloud/existing.log").exists())
                calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
                self.assertFalse(any("pkill" in call for call in calls))
                if backend == "docker":
                    build = next(call for call in calls if call[:1] == ["run"])
                    self.assertIn("zta.batch_attempt=test", build)
                    self.assertIn("PYTHONPATH=/app", build)
                    self.assertIn(str(root / "src") + ":/app/src:ro", build)
                    self.assertIn(str(root / "scripts") + ":/app/scripts:ro", build)
                    self.assertFalse(any(call[:1] == ["cp"] for call in calls))
                    dispatch = next(call for call in calls if call[:1] == ["exec"])
                    self.assertNotIn("-d", dispatch)
                else:
                    environments = [json.loads(line) for line in (root / "calls.jsonl.env").read_text().splitlines()]
                    self.assertTrue(all(item["pythonpath"] == "/app" for item in environments))
                    build = next(call for call in calls if "/app/scripts/setup/build_artifacts.py" in call)
                    self.assertIn(str(root / "src") + ":/app/src:ro", build)
                    self.assertIn(str(root / "scripts") + ":/app/scripts:ro", build)
                    launch = next(call for call in calls if "--writable-tmpfs" in call)
                    self.assertIn("PYTHONPATH=/tmp:/app", launch)
                    self.assertIn(str(root) + ":/app", launch)
                    patch_source = (root / "runtime/batch/tpm_socket_patch.py").read_text()
                    # Execute the generated compatibility patch with a fake Flower module:
                    # it must route TPM sockets while leaving a caller's timeout untouched.
                    class ServerConfig:
                        def __init__(self, num_rounds=1, round_timeout=None):
                            self.round_timeout = round_timeout
                    fake_server = SimpleNamespace(ServerConfig=ServerConfig)
                    original_init = ServerConfig.__init__
                    fake_run = Mock(return_value="tpm-result")
                    with patch.dict(sys.modules, {"flwr.server": fake_server}), \
                         patch("subprocess.run", fake_run), \
                         patch.dict(os.environ, {"TPM2TOOLS_TCTI": "swtpm:host=sim-tpm-1-2,port=2321"}):
                        namespace = {}
                        exec(compile(patch_source, "tpm_socket_patch.py", "exec"), namespace)
                        self.assertIs(ServerConfig.__init__, original_init)
                        self.assertEqual(ServerConfig(round_timeout=.5).round_timeout, .5)
                        self.assertEqual(namespace["_patched_run"](["tpm2_quote"]), "tpm-result")
                        self.assertEqual(fake_run.call_args.kwargs["env"]["TPM2TOOLS_TCTI"], "swtpm:path=/app/runtime/tpm_state/edge_1_2/swtpm.sock")
                        self.assertEqual(os.environ["TPM2TOOLS_TCTI"], "swtpm:host=sim-tpm-1-2,port=2321")

    def test_generated_compose_labels_and_runtime_paths_are_bound_to_attempt(self):
        root = self.root
        env = {**os.environ, "PROJECT_ROOT": str(root), "COMPOSE_FILE": str(root / "compose.yml"),
               "NUM_FOGS": "1", "ZTA_BATCH_MANAGED": "1", "ZTA_BATCH_ATTEMPT": "attempt_test"}
        source = Path(batch.__file__).resolve().parents[2] / "scripts/simulation/generate_compose_sim.sh"
        subprocess.run(["bash", str(source), "UTC", "2"], env=env, check=True, capture_output=True)
        compose = yaml.safe_load((root / "compose.yml").read_text())
        self.assertEqual(set(compose["services"]), {"cloud", "tpm-1-1", "tpm-1-2"})
        for service in compose["services"].values():
            self.assertEqual(service["labels"]["zta.batch_attempt"], "attempt_test")
        cloud = compose["services"]["cloud"]
        self.assertIn("RAY_TMPDIR=/app/runtime/batch/ray", cloud["environment"])
        self.assertIn("./runtime/batch:/app/runtime/batch", cloud["volumes"])


if __name__ == "__main__":
    unittest.main()
