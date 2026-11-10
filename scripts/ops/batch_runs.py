"""Sequential simulations on one persistent Docker/Apptainer environment.

Host dependencies: Python and PyYAML. ML runs in the existing simulation images.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.shared.utils.simulation_lifecycle import atomic_json, stamp
from src.shared.utils.config_loader import get_round_timeout
from scripts.ops.apptainer_lifecycle import (
    LifecycleError, ProcessInventory, is_runtime, signal_payload,
)

CONFIG_FILES = ("network.yaml", "training.yaml", "security.yaml")


class BatchError(RuntimeError):
    pass


class Cancelled(BatchError):
    pass


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def read_yaml(path):
    value = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise BatchError(f"{path}: expected a mapping with string keys")
    json.dumps(value, allow_nan=False)
    return value


def merged(configs):
    result = {}
    for name in CONFIG_FILES:
        result.update(configs[name])
    return result


def topology(config):
    n, uniform = config["num_fogs"], config["uniform_edges_per_fog"]
    custom = config.get("custom_fog_topology", "[]")
    if isinstance(custom, str):
        custom = ast.literal_eval(custom)
    if type(n) is not int or n <= 0 or type(uniform) is not int or uniform <= 0:
        raise BatchError("Topology counts must be positive integers")
    if not isinstance(custom, list) or (custom and len(custom) != n):
        raise BatchError("custom_fog_topology must be empty or contain one count per Fog")
    edges = custom or [uniform] * n
    if any(type(e) is not int or e <= 0 for e in edges):
        raise BatchError("Each Fog must have a positive integer Edge count")
    return edges


def plan_queue(root, queue):
    baseline = {name: read_yaml(root / "config" / name) for name in CONFIG_FILES}
    base_topology = topology(merged(baseline))
    owners = {}
    for name, values in baseline.items():
        for key in values:
            if key in owners:
                raise BatchError(f"Ambiguous baseline key {key!r} in multiple config files")
            owners[key] = name
    runs = []
    for folder in sorted(queue.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        if not re.fullmatch(r"[A-Za-z0-9_-]+", folder.name) or folder.is_symlink():
            raise BatchError(f"Unsafe queue directory: {folder}")
        configs = json.loads(json.dumps(baseline))
        overrides = {}
        for path in sorted(folder.iterdir()):
            if path.name.startswith("."):
                continue
            if path.name not in CONFIG_FILES:
                raise BatchError(f"Unexpected queue file: {path}; use only {CONFIG_FILES}")
            if path.is_symlink() or not path.is_file():
                raise BatchError(f"Expected regular configuration file: {path}")
            patch = read_yaml(path)
            for key in patch:
                if owners.get(key) != path.name:
                    raise BatchError(f"{path}: unknown or misplaced key {key!r}")
            configs[path.name].update(patch)
            overrides[path.name] = path.read_text()
        if not overrides:
            raise BatchError(f"{folder}: no configuration overrides")
        config = merged(configs)
        if topology(config) != base_topology:
            raise BatchError(f"{folder}: topology changes are not supported within a batch")
        if type(config.get("num_rounds")) is not int or config["num_rounds"] < 1:
            raise BatchError(f"{folder}: num_rounds must be a positive integer")
        try:
            get_round_timeout(config)
        except ValueError as exc:
            raise BatchError(f"{folder}: {exc}") from exc
        runs.append({"name": folder.name, "configs": configs, "overrides": overrides,
                     "target_rounds": config["num_rounds"], "topology": base_topology, "status": "PENDING"})
    if not runs:
        raise BatchError(f"No runs found in {queue}")
    return baseline, runs


def source_files(root):
    paths = []
    for directory in ("src", "scripts", "docs"):
        paths.extend(p for p in (root / directory).rglob("*") if p.suffix in {".py", ".sh", ".md"})
    paths.extend((root / "docker").glob("*.Dockerfile"))
    paths.extend(root / name for name in ("pyproject.toml", "requirements.txt", "README.md", ".dockerignore", ".gitignore"))
    return sorted(p for p in paths if p.is_file())


def fingerprint(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files(root)}


def command(args, timeout=30):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {"command": args, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": args, "returncode": None, "stdout": "", "stderr": str(exc)}


def require(result):
    if result["returncode"] != 0:
        raise BatchError(f"Command failed: {result['command']}: {result['stderr']}")
    return result["stdout"]


def owned_processes(attempt):
    """Find inherited attempt markers, including orphaned Ray workers; never match names."""
    marker = f"ZTA_BATCH_ATTEMPT={attempt}"
    records = []
    if Path("/proc").is_dir():
        for path in Path("/proc").iterdir():
            if not path.name.isdigit():
                continue
            try:
                if path.stat().st_uid != os.getuid():
                    continue
                before = (path / "stat").read_text().rsplit(")", 1)[1].split()
                env = (path / "environ").read_bytes().split(b"\0")
                after = (path / "stat").read_text().rsplit(")", 1)[1].split()
                if marker.encode() in env and after[0] != "Z" and before[19] == after[19]:
                    argv = [arg.decode(errors="replace") for arg in (path / "cmdline").read_bytes().split(b"\0") if arg]
                    records.append({"pid": int(path.name), "start_time": int(after[19]),
                                    "command": shlex.join(argv), "argv": argv})
            except (OSError, IndexError):
                continue
    else:
        result = command(["ps", "eww", "-axo", "pid=,stat=,command="])
        for line in require(result).splitlines():
            parts = line.split(None, 2)
            if len(parts) == 3 and marker in parts[2].split() and not parts[1].startswith("Z"):
                pid = int(parts[0])
                description = command(["ps", "-p", str(pid), "-o", "command="])["stdout"].strip()
                records.append({"pid": pid, "command": description})
    return records


class Runtime:
    def __init__(self, root, backend, attempt, edges):
        self.root, self.backend, self.attempt, self.edges = root, backend, attempt, edges
        self.children = []
        self.inventory = None

    def observe(self):
        marked = owned_processes(self.attempt)
        if self.inventory is not None:
            self.inventory.scan(marked)
        return marked

    def containers(self, project=False):
        selector = "com.docker.compose.project=zta-fl" if project else f"zta.batch_attempt={self.attempt}"
        return require(command(["docker", "ps", "-aq", "--filter", f"label={selector}"])).split()

    def launch(self, script, log):
        if self.backend == "apptainer" and self.inventory is None:
            self.inventory = ProcessInventory(
                self.root / "results" / self.attempt / "process_inventory.json", self.attempt)
            self.inventory.begin()
        env = {**os.environ, "ZTA_BATCH_MANAGED": "1", "ZTA_BATCH_ATTEMPT": self.attempt,
               "APPTAINERENV_ZTA_BATCH_ATTEMPT": self.attempt,
               "ZTA_BATCH_SESSION": "/app/results/batch_archives/" + self.session.name if hasattr(self, "session") else "",
               "ZTA_BATCH_PHASE_FILE": str(self.root / "runtime" / "batch_phase"),
               "PROJECT_ROOT": str(self.root), "CONFIG_PATH": str(self.root / "config")}
        env["APPTAINERENV_CONFIG_PATH"] = "/app/config"
        with log.open("wb") as handle:
            child = subprocess.Popen(["bash", str(script)], cwd=self.root, env=env,
                                     stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        self.children.append(child)
        if self.inventory is not None:
            self.inventory.remember(child.pid)
        return child

    def healthy(self):
        if self.backend == "docker":
            ids = self.containers()
            if not ids:
                return False
            info = json.loads(require(command(["docker", "inspect", *ids])))
            expected = {"sim-cloud"} | {f"sim-tpm-{i}-{j}" for i, e in enumerate(self.edges, 1) for j in range(1, e + 1)}
            nodes = [item for item in info if item["Name"].lstrip("/") in expected]
            return (len(nodes) == len(expected) and all(
                item["State"].get("Running") and not item["State"].get("OOMKilled")
                and item.get("RestartCount", 0) == 0
                and item["State"].get("Health", {}).get("Status", "healthy") == "healthy" for item in nodes))
        # $! identifies the Apptainer launcher, which need not be the marked
        # swtpm child. Validate owned services by their exact per-Edge endpoints.
        expected = {f"edge_{i}_{j}" for i, count in enumerate(self.edges, 1)
                    for j in range(1, count + 1)}
        found = {edge: 0 for edge in expected}
        for process in self.observe():
            try:
                argv = process.get("argv") or shlex.split(process["command"])
                if len(argv) < 2 or Path(argv[0]).name != "swtpm" or argv[1] != "socket":
                    continue
                state = argv[argv.index("--tpmstate") + 1]
                server = argv[argv.index("--server") + 1]
                control = argv[argv.index("--ctrl") + 1]
            except (KeyError, ValueError, IndexError):
                continue
            edge = state.rsplit("/", 1)[-1]
            base = f"/app/runtime/tpm_state/{edge}"
            if (edge in expected and state == f"dir={base}"
                    and server == f"type=unixio,path={base}/swtpm.sock"
                    and control == f"type=unixio,path={base}/swtpm.sock.ctrl"):
                found[edge] += 1
        return bool(expected) and all(
            found[edge] == 1
            and (self.root / "runtime/tpm_state" / edge / "swtpm.sock").is_socket()
            and (self.root / "runtime/tpm_state" / edge / "swtpm.sock.ctrl").is_socket()
            for edge in expected
        )

    def diagnostics(self, folder):
        atomic_json(folder / "processes.json", self.observe())
        records = {"version": command([self.backend, "--version"]),
                   "disk": command(["df", "-h", str(self.root)])}
        if shutil.which("nvidia-smi"):
            records["gpu"] = command(["nvidia-smi"])
        if self.backend == "docker":
            ids = self.containers()
            records["inspect"] = command(["docker", "inspect", *ids]) if ids else {}
            for container in ids:
                result = command(["docker", "logs", "--timestamps", container])
                (folder / f"docker-{container}.log").write_text(result["stdout"] + result["stderr"])
        else:
            records["image"] = command(["apptainer", "inspect", "--json", str(self.root / "images/cloud_node.sif")])
        atomic_json(folder / "runtime_diagnostics.json", records)

    def stop(self, grace):
        if self.backend == "apptainer":
            try:
                return self.stop_apptainer(grace)
            except (LifecycleError, OSError, ValueError, KeyError) as exc:
                raise BatchError(str(exc)) from exc
        if self.backend == "docker" and set(self.containers(project=True)) - set(self.containers()):
            raise BatchError("Foreign containers in Compose project; refusing destructive cleanup")
        for record in owned_processes(self.attempt):
            try:
                os.kill(record["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            for child in self.children:
                child.poll()
            if not owned_processes(self.attempt):
                break
            time.sleep(.2)
        for record in owned_processes(self.attempt):
            try:
                os.kill(record["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self.backend == "docker":
            ids = self.containers()
            if ids:
                require(command(["docker", "rm", "-f", *ids]))
            if self.containers(project=True):
                raise BatchError("Foreign containers remain in the simulation Compose project")
            compose = self.root / "docker/docker-compose.yml"
            if compose.exists():
                require(command(["docker", "compose", "-f", str(compose), "--project-directory", str(self.root), "down", "--remove-orphans"]))
        for child in self.children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired as exc:
                raise BatchError("A supervised process did not exit") from exc
        deadline = time.monotonic() + 5
        while owned_processes(self.attempt) and time.monotonic() < deadline:
            time.sleep(.2)
        if owned_processes(self.attempt):
            raise BatchError("Owned processes remain alive; refusing to start another run")

    def stop_apptainer(self, grace):
        if self.inventory is None:
            self.inventory = ProcessInventory(
                self.root / "results" / self.attempt / "process_inventory.json", self.attempt)
        if not self.inventory.initialized:
            # Recovery from old batches cannot manufacture a trustworthy baseline.
            # No signals are sent on this path, even if a marker is still present.
            self.inventory.assert_clean(owned_processes(self.attempt))
            return
        # Stop application processes first. Killing a starter/runtime parent can
        # bypass Apptainer's unmount and FUSE-driver shutdown, even on success.
        # Each phase discovers new children; helpers are NEVER directly signalled.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            sent = set()
            deadline = time.monotonic() + grace
            while True:
                for child in self.children:
                    child.poll()
                marked = owned_processes(self.attempt)
                try:
                    owned, unknown, unreadable = self.inventory.scan(marked)
                except LifecycleError:
                    # A /proc read can race process exit. Retry within this phase's
                    # existing deadline, but never declare an unreadable process gone.
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.2)
                    continue
                atomic_json(self.inventory.path.parent / "shutdown_processes.json", {
                    "owned": owned, "unknown_helpers": unknown,
                    "unreadable_pids": sorted(unreadable), "signal_phase": sig.name})
                if not owned:
                    break
                for record in owned:
                    key = (record["pid"], record["start_time"])
                    if key not in sent and not is_runtime(record):
                        signal_payload(record, sig)
                        sent.add(key)
                if time.monotonic() >= deadline:
                    break
                time.sleep(.2)
            if not owned:
                break
        self.inventory.assert_clean(owned_processes(self.attempt))
        for child in self.children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired as exc:
                raise BatchError("A supervised process did not exit") from exc


def ledger_ready(root, edges):
    folder = root / "runtime/tpm_state"
    identities, ledger = read_json(folder / "tpm_identities.json"), read_json(folder / "pcr_ledger.json")
    expected = {f"edge_{i}_{j}" for i, n in enumerate(edges, 1) for j in range(1, n + 1)}
    return (set(identities) == expected and len(set(identities.values())) == len(expected)
            and all(identity in ledger and ledger[identity] for identity in identities.values()))


def progress(root, attempt):
    value = read_json(root / "results" / attempt / "batch_progress.json")
    number = value.get("completed_round", -1)
    return number if value.get("attempt") == attempt and type(number) is int else -1


def complete(root, attempt, rounds):
    folder = root / "results" / attempt
    lifecycle, metrics = read_json(folder / "batch_lifecycle.json"), read_json(folder / f"{attempt}.json")
    saved = {r.get("round") for r in metrics.get("performance", []) if isinstance(r, dict) and type(r.get("round")) is int}
    return (lifecycle.get("attempt") == attempt and lifecycle.get("status") == "COMPLETED"
            and metrics.get("metadata", {}).get("experiment_name") == attempt
            and metrics.get("metadata", {}).get("num_rounds") == rounds
            and set(range(1, rounds + 1)) <= saved and progress(root, attempt) == rounds
            and (folder / "global_model.pt").is_file() and (folder / "global_model.pt").stat().st_size > 0)


def move_diagnostics(root, destination):
    """Rename after processes stop, preserving root-owned files and socket metadata."""
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("logs", "runtime", ".flwr"):
        source = root / name
        if source.is_symlink():
            raise BatchError(f"Refusing to move symlink: {source}")
        if source.exists():
            target = destination / name
            if target.exists():
                raise BatchError(f"Backup already exists: {target}; refusing to overwrite it")
            source.rename(target)


@contextmanager
def batch_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BatchError("Another batch supervisor holds this workspace lock") from exc
        yield


def preflight(root, backend):
    if not shutil.which(backend):
        raise BatchError(f"{backend} is not installed")
    if (root / "runtime/infra/.network_ready").exists():
        raise BatchError("A network is already marked active; stop it before starting a batch")
    if backend == "docker":
        require(command(["docker", "info"]))
        if require(command(["docker", "ps", "-aq", "--filter", "label=com.docker.compose.project=zta-fl"])).strip():
            raise BatchError("The zta-fl Compose project already contains containers; refusing to take it over")
    else:
        if not Path("/proc").is_dir():
            raise BatchError("Apptainer batch supervision requires Linux /proc for owned-process cleanup")
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise BatchError("Apptainer batch supervision requires Python/Linux pidfd support")
        # Check kernel support before starting any workload, not during teardown.
        os.close(os.pidfd_open(os.getpid()))
        processes = require(command(["ps", "-u", str(os.getuid()), "-o", "args="]))
        if any(re.search(r"\b(swtpm|flwr-simulation|flwr run)\b", line) for line in processes.splitlines()):
            raise BatchError("Existing TPM/Flower processes detected; stop the standalone simulation first")


def supervise(root, session, state):
    """Own infrastructure once; the in-container worker serializes all experiments."""
    limits = state["limits"]
    runtime = Runtime(root, state["runtime"], state["infrastructure_id"], state["runs"][0]["topology"])
    runtime.session = session
    infrastructure = session / state["infrastructure_id"]
    infrastructure.mkdir(exist_ok=True)
    interrupted = False
    worker = None
    previous = {}
    def cancel(*_):
        nonlocal interrupted
        interrupted = True
        (session / "STOP").touch()
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, cancel)
    state["infrastructure_active"] = True
    atomic_json(session / "batch.json", state)
    code = 2
    stopped = False
    try:
        boot = runtime.launch(root / f"scripts/simulation/boot_network_sim_{state['runtime']}.sh", infrastructure / "boot.log")
        deadline = time.monotonic() + limits["boot_timeout"]
        while not (root / "runtime/infra/.network_ready").exists():
            if interrupted:
                return 130
            if boot.poll() is not None:
                raise BatchError(f"Network boot exited {boot.returncode}; see infrastructure/boot.log")
            if time.monotonic() >= deadline:
                raise BatchError("Network boot timed out")
            runtime.observe()
            time.sleep(limits["poll_interval"])
        if not ledger_ready(root, runtime.edges) or not runtime.healthy():
            raise BatchError("Incomplete TPM provisioning or unhealthy infrastructure")
        worker = runtime.launch(root / f"scripts/simulation/deploy_code_sim_{state['runtime']}.sh", infrastructure / "worker.log")
        print(f"Training progress: {infrastructure / 'worker.log'}", flush=True)
        last_report = None
        cancel_deadline = None
        while worker.poll() is None:
            current = read_json(session / "batch.json")
            active = current.get("active")
            if active is not None:
                entry = current["runs"][active]
                report = (entry["name"], entry["phase"], entry.get("completed_round", -1))
                if report != last_report:
                    print(f"[{stamp()}] {report[0]}: {report[1]}, saved round {report[2]}", flush=True)
                    last_report = report
            if boot.poll() is not None:
                (session / "STOP").touch()
                raise BatchError("Infrastructure exited while training")
            if interrupted:
                if cancel_deadline is None:
                    cancel_deadline = time.monotonic() + 4 * limits["shutdown_timeout"] + 150
                if time.monotonic() >= cancel_deadline:
                    raise BatchError("Worker did not finish cancellation; stopping owned infrastructure")
            time.sleep(limits["poll_interval"])
        code = worker.returncode
        if code not in (0, 1, 130):
            raise BatchError("Training worker failed; see infrastructure/worker.log and per-attempt status")
        final_runs = read_json(session / "batch.json").get("runs", [])
        succeeded = sum(entry["status"] == "SUCCESS" for entry in final_runs)
        pending = sum(entry["status"] == "PENDING" for entry in final_runs)
        print(f"Batch finished: {succeeded}/{len(final_runs)} successful; {pending} pending", flush=True)
        return 130 if interrupted else code
    finally:
        # No per-experiment container/FUSE teardown. This runs once per invocation.
        try:
            runtime.diagnostics(infrastructure)
        except Exception as exc:
            (infrastructure / "diagnostics_error.txt").write_text(str(exc))
        try:
            runtime.stop(limits["shutdown_timeout"])
            stopped = True
            move_diagnostics(root, infrastructure)
        finally:
            latest = read_json(session / "batch.json") or state
            latest["infrastructure_active"] = not stopped
            if stopped and latest.get("active") is not None:
                entry = latest["runs"][latest["active"]]
                if entry["status"] == "RUNNING":
                    entry.update(status="CANCELLED" if interrupted else "INTERRUPTED", finished_at=stamp())
                latest["active"] = None
            atomic_json(session / "batch.json", latest)
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def positive(value):
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("Must be a positive finite number of seconds")
    return number


def retry_interrupted(state):
    """Requeue interrupted/cancelled entries while retaining their attempt history."""
    plan_keys = {"name", "configs", "overrides", "target_rounds", "topology"}
    for entry in state["runs"]:
        if entry["status"] not in {"INTERRUPTED", "CANCELLED"}:
            continue
        history = entry.get("previous_attempts", []) + [
            {key: value for key, value in entry.items()
             if key not in plan_keys and key != "previous_attempts"}
        ]
        plan = {key: entry[key] for key in plan_keys}
        entry.clear()
        entry.update(plan, status="PENDING", previous_attempts=history)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("docker", "apptainer"))
    parser.add_argument("--queue", type=Path, default=ROOT / "experiments_queue")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan without starting or changing anything")
    parser.add_argument("--resume", type=Path, help="Resume pending entries; completed attempts are never rerun")
    recovery_policy = parser.add_mutually_exclusive_group()
    recovery_policy.add_argument("--retry-interrupted", action="store_true",
                                 help="With --resume, restart INTERRUPTED/CANCELLED entries from round 1 as new attempts")
    recovery_policy.add_argument("--skip-interrupted", action="store_true",
                                 help="With --resume, skip INTERRUPTED/CANCELLED entries (default)")
    for flag, default in (("boot-timeout", 1800), ("prepare-timeout", 7200), ("stall-timeout", 3600),
                          ("run-timeout", 25200), ("shutdown-timeout", 120), ("poll-interval", 5)):
        parser.add_argument("--" + flag, type=positive, default=default)
    args = parser.parse_args(argv)
    if (args.retry_interrupted or args.skip_interrupted) and not args.resume:
        parser.error("--retry-interrupted and --skip-interrupted require --resume")
    archives = (ROOT / "results/batch_archives").resolve()
    try:
        if args.resume:
            session = args.resume.resolve()
            if session.parent != archives or not (session / "batch.json").is_file():
                raise BatchError("--resume must name an existing batch under results/batch_archives")
            state = read_json(session / "batch.json")
            if args.runtime and args.runtime != state["runtime"]:
                raise BatchError("Cannot change runtime when resuming")
        else:
            if not args.runtime:
                parser.error("--runtime docker|apptainer is required for a new batch")
            baseline, runs = plan_queue(ROOT, args.queue.resolve())
            state = {"version": 2, "infrastructure_active": False, "runtime": args.runtime, "created_at": stamp(), "active": None,
                     "queue": str(args.queue.resolve()), "runs": runs,
                     "limits": {key: getattr(args, key) for key in ("boot_timeout", "prepare_timeout", "stall_timeout", "run_timeout", "shutdown_timeout", "poll_interval")},
                     "source_fingerprint": fingerprint(ROOT)}
            session = archives / f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
        if not args.resume:
            state["infrastructure_id"] = "infra_" + session.name
        if args.dry_run:
            if args.resume and args.retry_interrupted:
                # Preview recovery without touching the journal or live processes.
                if state["active"] is not None:
                    state["runs"][state["active"]]["status"] = "INTERRUPTED"
                    state["active"] = None
                retry_interrupted(state)
            print(json.dumps(state, indent=2))
            return 0
        with batch_lock(archives / ".supervisor.lock"):
            pointer = read_json(archives / "active_batch.json")
            if not args.resume and pointer:
                previous = read_json(Path(pointer["session"]) / "batch.json")
                if previous.get("active") is not None or previous.get("infrastructure_active"):
                    raise BatchError(f"An unfinished attempt needs recovery first: --resume {pointer['session']}")
            if args.resume:
                if state.get("version") != 2:
                    if state.get("active") is not None:
                        entry = state["runs"][state["active"]]
                        recovery = session / ("recovery_" + uuid.uuid4().hex[:8])
                        recovery.mkdir()
                        runtime = Runtime(ROOT, state["runtime"], entry["attempt"], entry["topology"])
                        runtime.stop(state["limits"]["shutdown_timeout"])
                        move_diagnostics(ROOT, recovery)
                        for name in CONFIG_FILES:
                            shutil.copy2(session / "baseline_config" / name, ROOT / "config" / name)
                        entry.update(status="INTERRUPTED", finished_at=stamp())
                        state["active"] = None
                        atomic_json(session / "batch.json", state)
                    raise BatchError("Legacy batch recovery completed. Start a new batch to use the persistent runner and current queue settings.")
                if state.get("infrastructure_active"):
                    recovery = session / ("recovery_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
                    recovery.mkdir()
                    runtime = Runtime(ROOT, state["runtime"], state["infrastructure_id"], state["runs"][0]["topology"])
                    try:
                        runtime.diagnostics(recovery)
                    except Exception as exc:
                        (recovery / "diagnostics_error.txt").write_text(str(exc))
                    runtime.stop(state["limits"]["shutdown_timeout"])
                    move_diagnostics(ROOT, recovery)
                    state["infrastructure_active"] = False
                if state["active"] is not None:
                    state["runs"][state["active"]].update(status="INTERRUPTED", finished_at=stamp())
                    state["active"] = None
                atomic_json(session / "batch.json", state)
                if fingerprint(ROOT) != state["source_fingerprint"]:
                    raise BatchError("Source differs from saved batch; recovery completed, start a new batch.")
                (session / "STOP").unlink(missing_ok=True)
            preflight(ROOT, state["runtime"])
            if args.resume and args.retry_interrupted:
                # Only requeue after recovery, source validation and preflight succeed.
                retry_interrupted(state)
                atomic_json(session / "batch.json", state)
            if not args.resume:
                session.mkdir(parents=True)
                shutil.copytree(ROOT / "config", session / "baseline_config")
                for relative in state["source_fingerprint"]:
                    target = session / "source" / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ROOT / relative, target)
                atomic_json(session / "git_state.json", {
                    "head": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
                    "status": command(["git", "-C", str(ROOT), "status", "--short"]),
                    "diff": command(["git", "-C", str(ROOT), "diff", "--binary"]),
                })
                move_diagnostics(ROOT, session / "preexisting")
                atomic_json(session / "batch.json", state)
            if args.resume:
                state["infrastructure_id"] = "infra_" + session.name + "_" + uuid.uuid4().hex[:8]
                atomic_json(session / "batch.json", state)
            atomic_json(archives / "active_batch.json", {"session": str(session)})
            print(f"Batch archive: {session}", flush=True)
            return supervise(ROOT, session, state)
    except (BatchError, OSError, ValueError, KeyError) as exc:
        print(f"Batch stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
