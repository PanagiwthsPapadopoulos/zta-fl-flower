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

    def completed(self, rounds=2):
        atomic_json(self.folder / "batch_lifecycle.json", {"attempt": "test", "status": "COMPLETED"})
        atomic_json(self.folder / "batch_progress.json", {"attempt": "test", "completed_round": rounds})
        atomic_json(self.folder / "test.json", {"metadata": {"experiment_name": "test", "num_rounds": rounds},
                    "performance": [{"round": i} for i in range(rounds + 1)]})
        (self.folder / "global_model.pt").write_bytes(b"checkpoint stand-in")


    def test_overrides_start_from_same_baseline_and_preserve_omitted_keys(self):
        baseline, entries = batch.plan_queue(self.root, self.queue)
        self.assertEqual(entries[0]["configs"]["training.yaml"]["learning_rate"], .01)
        self.assertEqual(entries[1]["configs"]["training.yaml"]["learning_rate"], .1)
        self.assertEqual(entries[0]["configs"]["training.yaml"]["num_rounds"], 2)
        self.assertEqual(baseline, self.baseline)

    def tpm_process(self, edge, pid):
        base = f"/app/runtime/tpm_state/{edge}"
        argv = ["/usr/bin/swtpm", "socket", "--tpmstate", f"dir={base}", "--tpm2",
                "--server", f"type=unixio,path={base}/swtpm.sock",
                "--ctrl", f"type=unixio,path={base}/swtpm.sock.ctrl", "--flags", "startup-clear"]
        return {"pid": pid, "argv": argv, "command": " ".join(argv)}

    def test_apptainer_health_accepts_100_owned_children_with_different_launcher_pids(self):
        infra = self.root / "runtime/infra"
        infra.mkdir()
        (infra / "batch_tpm_pids").write_text("\n".join(str(i) for i in range(100)))
        processes = [self.tpm_process(f"edge_{i}_{j}", 1000 + i * 10 + j)
                     for i in range(1, 11) for j in range(1, 11)]
        # The captured HPC diagnostic format did not yet include argv.
        for process in processes:
            del process["argv"]
        processes.append({"pid": 2000, "command": "bash boot_network_sim_apptainer.sh"})
        runtime = batch.Runtime(self.root, "apptainer", "this-attempt", [10] * 10)
        with patch.object(batch, "owned_processes", return_value=processes) as owned, \
             patch.object(Path, "is_socket", return_value=True):
            self.assertTrue(runtime.healthy())
        owned.assert_called_once_with("this-attempt")

    def test_apptainer_health_rejects_missing_duplicate_and_wrong_endpoint_services(self):
        runtime = batch.Runtime(self.root, "apptainer", "this-attempt", [2])
        first, second = self.tpm_process("edge_1_1", 100), self.tpm_process("edge_1_2", 101)
        malformed = {"pid": 102, "argv": ["swtpm", "socket", "--tpmstate"], "command": ""}
        wrong = self.tpm_process("edge_1_2", 103)
        wrong["argv"][6] = "type=unixio,path=/other/swtpm.sock"
        for processes in ([first], [first, first], [first, second, second],
                          [first, malformed], [first, wrong],
                          [first, self.tpm_process("edge_9_9", 104)]):
            with self.subTest(processes=processes), \
                 patch.object(batch, "owned_processes", return_value=processes), \
                 patch.object(Path, "is_socket", return_value=True):
                self.assertFalse(runtime.healthy())

    def test_apptainer_health_requires_both_socket_files_and_owned_services(self):
        runtime = batch.Runtime(self.root, "apptainer", "this-attempt", [1])
        process = self.tpm_process("edge_1_1", 100)
        for missing in ("swtpm.sock", "swtpm.sock.ctrl"):
            with self.subTest(missing=missing), \
                 patch.object(batch, "owned_processes", return_value=[process]), \
                 patch.object(Path, "is_socket", autospec=True, side_effect=lambda p: p.name != missing):
                self.assertFalse(runtime.healthy())
        # Socket files alone cannot make an unowned service healthy.
        with patch.object(batch, "owned_processes", return_value=[]), \
             patch.object(Path, "is_socket", return_value=True):
            self.assertFalse(runtime.healthy())

    def test_unknown_keys_cross_file_keys_rounds_and_topology_are_rejected(self):
        for filename, override in (("training.yaml", {"learning_raet": .2}),
                                   ("security.yaml", {"num_rounds": 2}),
                                   ("training.yaml", {"num_rounds": "2"}),
                                   ("network.yaml", {"uniform_edges_per_fog": 3}),
                                   ("network.yaml", {"round_timeout": 0}),
                                   ("network.yaml", {"round_timeout": None}),
                                   ("network.yaml", {"round_timeout": True})):
            with self.subTest(override=override):
                folder = self.queue / "bad"
                folder.mkdir()
                (folder / filename).write_text(yaml.safe_dump(override))
                with self.assertRaises(batch.BatchError):
                    batch.plan_queue(self.root, self.queue)
                shutil.rmtree(folder)

    def test_timeout_override_is_independent_for_each_run(self):
        self.add_run("01", "network.yaml", {"round_timeout": 7200.5})
        _, entries = batch.plan_queue(self.root, self.queue)
        self.assertEqual(entries[0]["configs"]["network.yaml"]["round_timeout"], 7200.5)
        self.assertEqual(entries[1]["configs"]["network.yaml"]["round_timeout"], 3600.)
        self.assertEqual(entries[0]["topology"], entries[1]["topology"])

    def test_success_requires_exact_identity_all_rounds_checkpoint_and_normal_return(self):
        self.completed()
        self.assertTrue(batch.complete(self.root, "test", 2))
        for filename, data in (("batch_lifecycle.json", {"attempt": "test", "status": "FAILED"}),
                               ("batch_progress.json", {"attempt": "other", "completed_round": 2}),
                               ("test.json", {"metadata": {"experiment_name": "test", "num_rounds": 2},
                                              "performance": [{"round": 20}]})):
            with self.subTest(filename=filename):
                self.completed()
                atomic_json(self.folder / filename, data)
                self.assertFalse(batch.complete(self.root, "test", 2))
        self.completed()
        (self.folder / "global_model.pt").unlink()
        self.assertFalse(batch.complete(self.root, "test", 2))


    def test_partial_or_unrelated_json_never_indicates_progress(self):
        (self.folder / "batch_progress.json").write_text('{"completed_round":')
        atomic_json(self.root / "results/latest.json", {"round": 2})
        self.assertEqual(batch.progress(self.root, "test"), -1)
        self.assertFalse(batch.complete(self.root, "test", 2))


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

    def test_readiness_requires_complete_unique_tpm_ledger(self):
        folder = self.root / "runtime/tpm_state"
        atomic_json(folder / "tpm_identities.json", {"edge_1_1": "a"})
        atomic_json(folder / "pcr_ledger.json", {"a": "pcr"})
        self.assertFalse(batch.ledger_ready(self.root, [2]))
        atomic_json(folder / "tpm_identities.json", {"edge_1_1": "a", "edge_1_2": "a"})
        self.assertFalse(batch.ledger_ready(self.root, [2]))
        atomic_json(folder / "tpm_identities.json", {"edge_1_1": "a", "edge_1_2": "b"})
        atomic_json(folder / "pcr_ledger.json", {"a": "pcr", "b": "pcr"})
        self.assertTrue(batch.ledger_ready(self.root, [2]))


    def test_backup_collision_is_fatal_instead_of_overwriting_diagnostics(self):
        destination = self.root / "archive"
        (destination / "runtime").mkdir(parents=True)
        with self.assertRaisesRegex(batch.BatchError, "refusing to overwrite"):
            batch.move_diagnostics(self.root, destination)


    def test_dry_run_changes_nothing_and_needs_no_container_runtime(self):
        before = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        with patch.object(batch, "ROOT", self.root), patch.object(batch, "preflight", side_effect=AssertionError("must not boot")), redirect_stdout(io.StringIO()):
            self.assertEqual(batch.main(["--runtime", "docker", "--queue", str(self.queue), "--dry-run"]), 0)
        after = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)


    def test_second_supervisor_cannot_acquire_workspace_lock(self):
        with batch.batch_lock(self.root / "lock"):
            with self.assertRaisesRegex(batch.BatchError, "Another batch"):
                with batch.batch_lock(self.root / "lock"):
                    self.fail("lock should be exclusive")


    def test_owned_cleanup_refuses_foreign_docker_containers(self):
        runtime = batch.Runtime(self.root, "docker", "owned", [2])
        with patch.object(runtime, "containers", side_effect=lambda project=False: ["foreign"] if project else ["owned"]), \
             patch.object(batch.os, "kill") as kill:
            with self.assertRaisesRegex(batch.BatchError, "Foreign"):
                runtime.stop(.01)
            kill.assert_not_called()

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
