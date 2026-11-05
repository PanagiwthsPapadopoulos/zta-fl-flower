"""Check local-file selection without starting real containers or training."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]


class RuntimeMountTests(unittest.TestCase):
    def test_generated_compose_uses_checkout_for_every_project_service(self):
        for mode in ("deployment", "simulation"):
            for insecure in ("false", "true"):
                with self.subTest(mode=mode, insecure=insecure), TemporaryDirectory() as directory:
                    root = Path(directory)
                    compose = root / "compose.yml"
                    env = {**os.environ, "PROJECT_ROOT": str(root), "COMPOSE_FILE": str(compose),
                           "NUM_FOGS": "2", "CLOUD_SA": "9001", "CLOUD_FL": "9002",
                           "CLOUD_CTRL": "9003", "FOG_SA_BASE": "9100", "FOG_FL_BASE": "9200",
                           "FOG_CTRL_BASE": "9300", "FOG_CIO_BASE": "9400", "EDGE_CIO_BASE": "10000",
                           "NGINX_CONF": str(root / "nginx.conf"), "CERTS_DIR": str(root / "certs"),
                           "ZTA_BATCH_MANAGED": "1", "ZTA_BATCH_ATTEMPT": "mount-test"}
                    if mode == "deployment":
                        script, args = "generate_compose.sh", [insecure, "UTC", "0 2"]
                    else:
                        script, args = "generate_compose_sim.sh", ["UTC", "0 2"]
                    result = subprocess.run(["bash", str(ROOT / "scripts" / mode / script), *args],
                                            env=env, capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    services = yaml.safe_load(compose.read_text())["services"]
                    if mode == "simulation":
                        self.assertIn("./.gitignore:/app/.gitignore:ro", services["cloud"]["volumes"])
                    apps = {name: service for name, service in services.items()
                            if service["image"].startswith("panagiotispapadopoulos/")}
                    self.assertEqual(len(apps), 7 if mode == "deployment" else 3)
                    for name, app in apps.items():
                        with self.subTest(service=name):
                            mounts = app["volumes"]
                            self.assertIn("./src:/app/src:ro", mounts)
                            self.assertIn("./scripts:/app/scripts:ro", mounts)
                            self.assertIn("./config:/app/config:ro", mounts)
                            self.assertIn("PYTHONPATH=/app", app["environment"])
                            self.assertEqual(app["working_dir"], "/app")
                            if name.startswith("edge-"):
                                self.assertEqual(app["entrypoint"], ["/bin/bash", "/app/scripts/ops/edge_entrypoint.sh"])
                    if mode == "simulation":
                        self.assertIn("./pyproject.toml:/app/pyproject.toml:ro", apps["cloud"]["volumes"])
                        self.assertIn("FLWR_HOME=/app/runtime/batch/flower", apps["cloud"]["environment"])

    def prepare_runtime(self, root):
        for folder in ("scripts/setup", "scripts/deployment", "scripts/simulation", "bin", "config",
                       "images", "data", "runtime/infra", "logs/nodes", "logs/system"):
            (root / folder).mkdir(parents=True, exist_ok=True)
        for name in ("parse_topology.py", "setup_tpm.sh"):
            shutil.copy2(ROOT / "scripts/setup" / name, root / "scripts/setup" / name)
        (root / "scripts/setup/collect_ledgers.py").write_text("# Provisioning is replaced in this launcher test.\n")
        (root / "config/network.yaml").write_text("num_fogs: 1\nuniform_edges_per_fog: 1\n")
        (root / "config/training.yaml").write_text("num_rounds: 1\n")
        for name in ("cloud_node.sif", "edge_node.sif", "nginx.sif"):
            (root / "images" / name).touch()
        (root / "runtime/infra/.network_ready").touch()
        # Record calls and materialize a readiness marker without opening a socket.
        stub = "#!" + sys.executable + "\n" + '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["CALL_RECORD"], "a") as output:
    output.write(json.dumps({"args": args, "pythonpath": os.environ.get("APPTAINERENV_PYTHONPATH")}) + "\\n")
if "compose" in args and "ps" in args:
    print("test-container")
if "swtpm" in args and "socket" in args:
    server = args[args.index("--server") + 1]
    path = server.split("path=", 1)[1].replace("/app/", os.environ["PROJECT_ROOT"] + "/", 1)
    Path(path).touch()
'''
        for name in ("apptainer", "docker", "flwr"):
            p = root / "bin" / name
            p.write_text(stub)
            p.chmod(0o755)
        for name in ("pgrep", "pkill", "nc", "sleep"):
            p = root / "bin" / name
            p.write_text("#!/bin/sh\nexit 0\n")
            p.chmod(0o755)
        env = {**os.environ, "PROJECT_ROOT": str(root), "ZTA_BATCH_MANAGED": "0",
               "PATH": str(root / "bin") + os.pathsep + os.environ["PATH"],
               "CALL_RECORD": str(root / "calls.jsonl")}
        return env

    def assert_apptainer_source(self, root, call):
        args = call["args"]
        mounts = [mount for i, value in enumerate(args) if value == "--bind"
                  for mount in args[i + 1].split(",")]
        if f"{root}:/app" not in mounts:
            self.assertIn(f"{root}/src:/app/src:ro", mounts)
            self.assertIn(f"{root}/scripts:/app/scripts:ro", mounts)
        self.assertEqual(call["pythonpath"], "/app")

    def test_apptainer_boot_passes_local_source_to_all_application_processes(self):
        for mode, name in (("deployment", "boot_network_apptainer.sh"),
                           ("simulation", "boot_network_sim_apptainer.sh")):
            with self.subTest(mode=mode), TemporaryDirectory(prefix="zta-", dir="/tmp") as directory:
                root = Path(directory)
                env = self.prepare_runtime(root)
                text = (ROOT / "scripts" / mode / name).read_text()
                # Substitute only socket readiness; no real TPM or socket is started.
                text = text.replace('[ ! -S "$EDGE_TPM_DIR/swtpm.sock" ]', '[ ! -e "$EDGE_TPM_DIR/swtpm.sock" ]')
                # Stop before the deployment boot's unrelated global Flower config write.
                # Everything up to that point is the real shell orchestration, with CLI stubs.
                if mode == "deployment":
                    text = text.split('FLWR_GLOBAL_DIR="$HOME/.flwr"', 1)[0] + "\nwait\n"
                script = root / "scripts" / mode / name
                script.write_text(text)
                result = subprocess.run(["bash", str(script), *(["--insecure"] if mode == "deployment" else [])],
                                        env=env, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
                apps = [call for call in calls if "flower-superexec" in call["args"]
                        or any("TPMAttestation" in arg for arg in call["args"])]
                self.assertEqual(len(apps), 5 if mode == "deployment" else 1)
                for call in apps:
                    self.assert_apptainer_source(root, call)

    def test_deployment_builders_and_dispatch_use_local_files(self):
        for backend in ("docker", "apptainer"):
            with self.subTest(backend=backend), TemporaryDirectory() as directory:
                root = Path(directory)
                env = self.prepare_runtime(root)
                name = f"deploy_code_{backend}.sh"
                script = root / "scripts/deployment" / name
                script.write_text((ROOT / "scripts/deployment" / name).read_text() + "\nwait\n")
                result = subprocess.run(["bash", str(script)], cwd=root, env=env,
                                        capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                calls = [json.loads(line) for line in (root / "calls.jsonl").read_text().splitlines()]
                builder = next(call for call in calls if "/app/scripts/setup/build_artifacts.py" in call["args"])
                args = builder["args"]
                self.assertIn(f"{root}/src:/app/src:ro", args)
                self.assertIn(f"{root}/scripts:/app/scripts:ro", args)
                self.assertIn("/app", args)
                if backend == "docker":
                    self.assertIn("PYTHONPATH=/app", args)
                else:
                    self.assert_apptainer_source(root, builder)
                    dispatches = [call for call in calls if "/python/venv/bin/flwr" in call["args"]]
                    self.assertEqual(len(dispatches), 2)
                    for call in dispatches:
                        self.assert_apptainer_source(root, call)


if __name__ == "__main__":
    unittest.main()
