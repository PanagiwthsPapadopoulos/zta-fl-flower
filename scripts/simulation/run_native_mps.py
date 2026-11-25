"""Run native Flower/MPS training with isolated Docker TPM sidecars.

Usage: .venv/bin/python scripts/simulation/run_native_mps.py [--smoke-test]
The existing Docker/Apptainer launchers and source configuration are untouched.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml


def prepare_run(run_dir, config_dir, smoke_test=False):
    """Copy configuration and isolate derived data without copying large raw inputs."""
    from src.shared.data.artifact_manifest import partition_provenance
    configs = {name: yaml.safe_load((config_dir / f"{name}.yaml").read_text())
               for name in ("network", "training", "security")}
    if smoke_test:
        configs["network"].update(num_fogs=1, uniform_edges_per_fog=2, custom_fog_topology="[]")
        configs["training"].update(num_rounds=2, local_epochs=1, dataset_fraction=0.1)
        configs["security"].update(label_flip_ratio=0., grad_manip_ratio=0., backdoor_ratio=0.,
                                   shap_aware_ratio=0., benign_adv_ratio=0.3,
                                   robustness_eval_attack="pgd", benign_n_iter=3)
    merged = {key: value for config in configs.values() for key, value in config.items()}
    topology = partition_provenance(merged)["topology"]
    configs["network"]["num_fogs"] = len(topology)
    if set(merged.get("gpu_enabled_tiers", [])) != {"cloud", "fog", "edge"}:
        raise ValueError("Native MPS requires gpu_enabled_tiers: [cloud, fog, edge] in the selected config")
    source = Path(merged["dataset_path"]).expanduser()
    source = (ROOT / source).resolve() if not source.is_absolute() else source.resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    raw = run_dir / "data" / str(merged["dataset"]) / "raw" / source.name
    raw.parent.mkdir(parents=True)
    raw.symlink_to(source, target_is_directory=source.is_dir())
    name = run_dir.name
    configs["training"].update(dataset_path=str(raw), run_name=name, batch_attempt=name)
    config_target = run_dir / "config"
    config_target.mkdir()
    for name, config in configs.items():
        (config_target / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return configs, topology


def tpm_compose(run_dir, topology, project, image):
    """Only TPM services; no published ports, native training uses docker exec."""
    services = {}
    for fog, count in enumerate(topology, 1):
        for edge in range(1, count + 1):
            state = run_dir / "runtime" / "tpm_state" / f"edge_{fog}_{edge}"
            state.mkdir(parents=True)
            target = f"/app/runtime/tpm_state/edge_{fog}_{edge}"
            command = (
                f"swtpm socket --tpmstate dir={target} --tpm2 "
                "--server type=tcp,port=2321,bindaddr=127.0.0.1 "
                "--ctrl type=tcp,port=2322,bindaddr=127.0.0.1 --flags startup-clear --daemon; "
                "i=0; until tpm2_startup -c; do i=$$((i+1)); "
                "[ $$i -lt 30 ] || exit 1; sleep 1; done; "
                f"python3 -m src.shared.security.tpm_transport {fog} {edge} --bootstrap; "
                "exec sleep infinity"
            )
            services[f"tpm-{fog}-{edge}"] = {
                "image": image, "container_name": f"{project}-tpm-{fog}-{edge}",
                "working_dir": "/app", "init": True, "user": "0:0",
                "environment": {"PYTHONPATH": "/app", "PYTHONDONTWRITEBYTECODE": "1",
                                "TPM2TOOLS_TCTI": "swtpm:host=127.0.0.1,port=2321",
                                "ZTA_INSECURE_MODE": "false"},
                "volumes": [f"{ROOT / 'src'}:/app/src:ro", f"{state}:{target}"],
                "entrypoint": ["/bin/sh", "-ec"], "command": [command],
                "healthcheck": {
                    "test": ["CMD-SHELL", f"test -s {target}/ak.pub && test -s {target}/tpm_id.txt "
                             f"&& test -s {target}/clean_pcr.bin && tpm2_readpublic -c 0x81010002 >/dev/null"],
                    "interval": "2s", "timeout": "5s", "retries": 60,
                },
            }
    return {"name": project, "services": services}


def stop_process(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def wait_superlink(process, address):
    import grpc
    from flwr.proto.control_pb2 import ListFederationsRequest
    from flwr.proto.control_pb2_grpc import ControlStub
    with grpc.insecure_channel(address) as channel:
        for _ in range(60):
            if process.poll() is not None:
                raise RuntimeError("Native SuperLink exited; inspect superlink.log")
            try:
                ControlStub(channel).ListFederations(ListFederationsRequest(), timeout=1)
                return
            except grpc.RpcError:
                time.sleep(0.5)
    raise TimeoutError("Native SuperLink did not become ready")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Two rounds, one Fog/two Edges, PGD training; changes only the run's config copy")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent Ray client workers (default: 1)")
    parser.add_argument("--tpm-image", default="panagiotispapadopoulos/zta-edge-node:latest")
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Native run interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    if args.workers < 1:
        parser.error("workers must be positive")
    if sys.platform != "darwin":
        parser.error("This launcher requires native macOS")
    import torch
    if not torch.backends.mps.is_available():
        parser.error("MPS is unavailable in this Python environment")
    bin_dir = Path(sys.executable).parent
    for executable in ("flwr", "flower-superlink"):
        if not (bin_dir / executable).exists():
            parser.error(f"Missing {executable} beside the selected Python interpreter")
    subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["docker", "image", "inspect", args.tpm_image], check=True, stdout=subprocess.DEVNULL)
    run_name = "native_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    run_dir = ROOT / "runtime" / "native-mps" / run_name
    run_dir.mkdir(parents=True)
    configs, topology = prepare_run(run_dir, args.config_dir.resolve(), args.smoke_test)
    total = sum(topology)
    project = "zta-mps-" + run_name.rsplit("_", 1)[1]
    compose_file = run_dir / "compose.yaml"
    compose_file.write_text(yaml.safe_dump(tpm_compose(run_dir, topology, project, args.tpm_image), sort_keys=False))
    compose = ["docker", "compose", "-p", project, "-f", str(compose_file)]
    ray_tmp = tempfile.mkdtemp(prefix="zta-ray-", dir="/tmp")
    env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
           "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1",
           "CONFIG_PATH": str(run_dir / "config"), "PROJECT_ROOT": str(run_dir),
           "TOTAL_EDGES": str(total), "ZTA_TIER": "cloud", "ZTA_GPU_AVAILABLE": "true",
           "ZTA_NATIVE_MPS": "1", "ZTA_INSECURE_MODE": "false",
           "ZTA_TPM_CONTAINER_PREFIX": project + "-tpm-",
           "ZTA_TPM_STATE_ROOT": str(run_dir / "runtime" / "tpm_state"),
           "FLWR_HOME": str(run_dir / "flower"), "RAY_TMPDIR": ray_tmp,
           "FLWR_DISABLE_UPDATE_CHECK": "1", "FLWR_TELEMETRY_ENABLED": "0",
           "RAY_USAGE_STATS_ENABLED": "0", "PYTORCH_ENABLE_MPS_FALLBACK": "0",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    # Use an explicit, private SuperLink address: :local: can reuse a service
    # started earlier with another run's environment, even with a new FLWR_HOME.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        address = f"127.0.0.1:{sock.getsockname()[1]}"
    flower_home = Path(env["FLWR_HOME"])
    flower_home.mkdir()
    (flower_home / "config.toml").write_text(
        f'[superlink]\ndefault = "native"\n[superlink.native]\naddress = "{address}"\ninsecure = true\n')
    print(f"Native run directory: {run_dir}", flush=True)
    print(f"Training on MPS with unchanged pooling on CPU; {total} clients, {args.workers} worker(s).", flush=True)
    processes = []
    status = {"status": "STARTED", "run_dir": str(run_dir), "ray_logs": ray_tmp,
              "num_clients": total, "workers": args.workers, "model": "unchanged",
              "torch": torch.__version__, "tpm_project": project}
    try:
        with (run_dir / "prepare.log").open("w") as log:
            subprocess.run([sys.executable, str(ROOT / "scripts/setup/build_artifacts.py")],
                           cwd=run_dir, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(compose + ["up", "-d", "--wait", "--wait-timeout", "120"], check=True)
        subprocess.run([sys.executable, str(ROOT / "scripts/setup/collect_ledgers.py")], cwd=run_dir, env=env, check=True)
        ledger_root = Path(env["ZTA_TPM_STATE_ROOT"])
        for filename in ("pcr_ledger.json", "tpm_identities.json"):
            if len(json.loads((ledger_root / filename).read_text())) != total:
                raise RuntimeError("TPM enrollment is incomplete")
        with (run_dir / "superlink.log").open("w") as server_log, (run_dir / "flower.log").open("w") as run_log:
            server = subprocess.Popen([str(bin_dir / "flower-superlink"), "--insecure", "--simulation",
                                       "--isolation", "subprocess", "--control-api-address", address,
                                       "--serverappio-api-address", "127.0.0.1:0"],
                                      cwd=run_dir, env=env, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(server)
            wait_superlink(server, address)
            resources = (f"num-supernodes={total} client-resources-num-gpus=0 "
                         f"client-resources-num-cpus=2 init-args-num-cpus={2 * args.workers} init-args-num-gpus=0")
            run = subprocess.Popen([str(bin_dir / "flwr"), "run", str(ROOT), "native", "--stream",
                                    "--federation-config", resources,
                                    "--run-config", f'mode="sim" tier="cloud" min_clients={total}'],
                                   cwd=run_dir, env=env, stdout=run_log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(run)
            print(f"Flower started. Progress: {run_dir / 'flower.log'}", flush=True)
            if run.wait() != 0:
                raise RuntimeError("Flower failed; inspect flower.log and superlink.log")
        evidence = run_dir / "results" / run_name
        lifecycle = json.loads((evidence / "batch_lifecycle.json").read_text())
        progress = json.loads((evidence / "batch_progress.json").read_text())
        if lifecycle["status"] != "COMPLETED" or progress["completed_round"] != configs["training"]["num_rounds"]:
            raise RuntimeError("Flower did not complete every configured round")
        status["status"] = "COMPLETED"
        print(f"Completed. Results: {evidence}", flush=True)
    except BaseException as exc:
        status.update(status="FAILED", error=str(exc))
        raise
    finally:
        for process in reversed(processes):
            stop_process(process)
        with (run_dir / "tpm.log").open("w") as log:
            subprocess.run(compose + ["logs", "--no-color"], stdout=log, stderr=subprocess.STDOUT, timeout=30)
        subprocess.run(compose + ["down"], check=False, timeout=60)
        (run_dir / "native_status.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
