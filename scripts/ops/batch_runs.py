"""Sequential Docker/Apptainer simulations with durable per-attempt diagnostics.

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
                env = (path / "environ").read_bytes().split(b"\0")
                status = (path / "stat").read_text().rsplit(")", 1)[1].split()[0]
                if marker.encode() in env and status != "Z":
                    records.append({"pid": int(path.name), "command": (path / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")})
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

    def containers(self, project=False):
        selector = "com.docker.compose.project=zta-fl" if project else f"zta.batch_attempt={self.attempt}"
        return require(command(["docker", "ps", "-aq", "--filter", f"label={selector}"])).split()

    def launch(self, script, log):
        env = {**os.environ, "ZTA_BATCH_MANAGED": "1", "ZTA_BATCH_ATTEMPT": self.attempt,
               "APPTAINERENV_ZTA_BATCH_ATTEMPT": self.attempt,
               "ZTA_BATCH_PHASE_FILE": str(self.root / "runtime" / "batch_phase"),
               "PROJECT_ROOT": str(self.root), "CONFIG_PATH": str(self.root / "config")}
        env["APPTAINERENV_CONFIG_PATH"] = "/app/config"
        with log.open("wb") as handle:
            child = subprocess.Popen(["bash", str(script)], cwd=self.root, env=env,
                                     stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        self.children.append(child)
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
        try:
            pids = {int(p) for p in (self.root / "runtime/infra/batch_tpm_pids").read_text().split()}
        except (OSError, ValueError):
            return False
        alive = {p["pid"] for p in owned_processes(self.attempt)}
        return len(pids) == sum(self.edges) and pids <= alive

    def diagnostics(self, folder):
        atomic_json(folder / "processes.json", owned_processes(self.attempt))
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


def watch(root, runtime, boot, launch, entry, limits, start, update, sleep=time.sleep, now=time.monotonic):
    phase, phase_start, last_progress, best = "preparing", now(), now(), -1
    finished_at = None
    while True:
        current = now()
        if current - start >= limits["run_timeout"]:
            return "TIMEOUT", "Total run time limit exceeded"
        if boot.poll() is not None or not runtime.healthy():
            return "CRASHED", "Simulation infrastructure exited or became unhealthy"
        try:
            observed = (root / "runtime/batch_phase").read_text().strip()
        except OSError:
            observed = ""
        if observed == "training" and phase != "training":
            phase, phase_start, last_progress = "training", current, current
            update(phase=phase)
        lifecycle = read_json(root / "results" / runtime.attempt / "batch_lifecycle.json")
        if lifecycle.get("attempt") == runtime.attempt and lifecycle.get("status") == "FAILED":
            return "CRASHED", "ServerApp recorded an exception; see batch_lifecycle.json"
        code = launch.poll()
        if code is not None:
            if code == 0 and complete(root, runtime.attempt, entry["target_rounds"]):
                return "SUCCESS", "ServerApp completed, all requested rounds and checkpoint saved, launcher exited 0"
            return "CRASHED" if code else "INCOMPLETE", f"Launcher exited {code}; completion requirements were not met"
        number = progress(root, runtime.attempt)
        if number > best:
            best, last_progress = number, current
            update(completed_round=number)
        if lifecycle.get("status") == "COMPLETED":
            finished_at = current if finished_at is None else finished_at
            if current - finished_at >= limits["shutdown_timeout"]:
                return "SHUTDOWN_TIMEOUT", "Server finished but the simulation launcher did not exit"
        if phase == "preparing" and current - phase_start >= limits["prepare_timeout"]:
            return "PREPARATION_TIMEOUT", "Artifact preparation/dispatch time limit exceeded"
        if phase == "training" and current - last_progress >= limits["stall_timeout"]:
            return "STALLED", "No newly saved round within the progress time limit (suspected deadlock)"
        sleep(limits["poll_interval"])


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


def restore_config(root, baseline_folder):
    for name in CONFIG_FILES:
        target = root / "config" / name
        temporary = target.with_name(name + ".batch-restore")
        shutil.copy2(baseline_folder / name, temporary)
        temporary.replace(target)


def archive(root, folder, attempt, configs):
    output = root / "results" / attempt
    if output.exists():
        shutil.copytree(output, folder / "results", dirs_exist_ok=True)
    raw_path = Path(merged(configs)["dataset_path"])
    if str(raw_path).startswith("/app/"):
        raw_path = root / raw_path.relative_to("/app")
    elif not raw_path.is_absolute():
        raw_path = root / raw_path
    for path in (raw_path.parent.parent / "artifacts").glob("*_manifest.json"):
        target = folder / "artifact_manifests" / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)
    report = raw_path.parent / "preprocessing_report.json"
    if report.exists():
        shutil.copy2(report, folder / report.name)
    if raw_path.exists():
        atomic_json(folder / "dataset_reference.json", {
            "path": str(raw_path), "bytes": raw_path.stat().st_size,
            "note": "Dataset and training tensors remain in place; content hashes are in artifact manifests.",
        })
    compose = root / "docker/docker-compose.yml"
    if compose.exists():
        shutil.copy2(compose, folder / "docker-compose.yml")
    if not (folder / "config_at_exit").exists():
        shutil.copytree(root / "config", folder / "config_at_exit")
    move_diagnostics(root, folder)


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
        processes = require(command(["ps", "-u", str(os.getuid()), "-o", "args="]))
        if any(re.search(r"\b(swtpm|flwr-simulation|flwr run)\b", line) for line in processes.splitlines()):
            raise BatchError("Existing TPM/Flower processes detected; stop the standalone simulation first")


def supervise(root, session, state):
    limits = state["limits"]
    interrupted = False

    def save():
        atomic_json(session / "batch.json", state)

    def cancel(signum, frame):
        raise Cancelled(f"Received signal {signum}")

    previous_handlers = {sig: signal.signal(sig, cancel) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for index, entry in enumerate(state["runs"]):
            if entry["status"] != "PENDING":
                continue
            if fingerprint(root) != state["source_fingerprint"]:
                raise BatchError("Source files changed during the batch; start a new batch for the new revision")
            attempt = f"batch_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{entry['name']}_{uuid.uuid4().hex[:8]}"
            folder = session / "attempts" / attempt
            folder.mkdir(parents=True)
            (root / "results" / attempt).mkdir(parents=True, exist_ok=False)
            configs = json.loads(json.dumps(entry["configs"]))
            configs["training.yaml"].update(run_name=attempt, batch_attempt=attempt)
            for dirname in ("executed_config", "overrides"):
                (folder / dirname).mkdir()
            for name in CONFIG_FILES:
                (folder / "executed_config" / name).write_text(yaml.safe_dump(configs[name], sort_keys=False))
            for name, content in entry["overrides"].items():
                (folder / "overrides" / name).write_text(content)
            atomic_json(folder / "effective_config.json", merged(configs))
            entry.update(status="RUNNING", attempt=attempt, started_at=stamp(), phase="booting", completed_round=-1)
            state["active"] = index
            save()

            def update(**values):
                entry.update(values)
                with (folder / "events.jsonl").open("a") as handle:
                    handle.write(json.dumps({"timestamp": stamp(), **values}) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                atomic_json(folder / "status.json", entry)
                save()

            runtime = Runtime(root, state["runtime"], attempt, entry["topology"])
            start = time.monotonic()
            print(f"[{stamp()}] Starting {entry['name']} -> {attempt}", flush=True)
            outcome, reason = "CRASHED", "Supervisor did not reach dispatch"
            cleanup_error = None
            try:
                for name in CONFIG_FILES:
                    shutil.copy2(folder / "executed_config" / name, root / "config" / name)
                (root / "runtime").mkdir(exist_ok=True)
                boot = runtime.launch(root / f"scripts/simulation/boot_network_sim_{state['runtime']}.sh", folder / "boot.log")
                update(boot_pid=boot.pid)
                while True:
                    if boot.poll() is not None:
                        raise BatchError(f"Network boot exited {boot.returncode}")
                    elapsed = time.monotonic() - start
                    if elapsed >= min(limits["boot_timeout"], limits["run_timeout"]):
                        outcome, reason = "BOOT_TIMEOUT", "Network readiness time limit exceeded"
                        break
                    if (root / "runtime/infra/.network_ready").exists():
                        if not ledger_ready(root, entry["topology"]) or not runtime.healthy():
                            raise BatchError("Boot signalled readiness with incomplete TPM provisioning or unhealthy processes")
                        update(phase="preparing")
                        launch = runtime.launch(root / f"scripts/simulation/deploy_code_sim_{state['runtime']}.sh", folder / "dispatch.log")
                        update(launcher_pid=launch.pid)
                        outcome, reason = watch(root, runtime, boot, launch, entry, limits, start, update)
                        break
                    time.sleep(limits["poll_interval"])
            except Cancelled as exc:
                outcome, reason, interrupted = "CANCELLED", str(exc), True
            except Exception as exc:
                outcome, reason = "CRASHED", f"{type(exc).__name__}: {exc}"
            finally:
                for sig in previous_handlers:
                    signal.signal(sig, signal.SIG_IGN)
                diagnostic_error = None
                exit_codes = {str(child.pid): child.poll() for child in runtime.children}
                try:
                    runtime.diagnostics(folder)
                except Exception as exc:
                    diagnostic_error = str(exc)
                stopped = False
                try:
                    runtime.stop(limits["shutdown_timeout"])
                    stopped = True
                    archive(root, folder, attempt, configs)
                    if diagnostic_error:
                        raise BatchError(f"Diagnostic collection failed: {diagnostic_error}")
                except Exception as exc:
                    cleanup_error = str(exc)
                finally:
                    if stopped:
                        try:
                            restore_config(root, session / "baseline_config")
                        except OSError as exc:
                            cleanup_error = f"Configuration restoration failed: {exc}"
                try:
                    update(status=outcome, reason=reason, finished_at=stamp(),
                           process_exit_codes_before_cleanup=exit_codes,
                           elapsed_seconds=time.monotonic() - start, cleanup_error=cleanup_error)
                except OSError as exc:
                    cleanup_error = f"Could not persist final status: {exc}"
                if not cleanup_error:
                    state["active"] = None
                    save()
                for sig in previous_handlers:
                    signal.signal(sig, cancel)
            print(f"[{stamp()}] {entry['name']}: {outcome} — {reason}", flush=True)
            if cleanup_error:
                raise BatchError(f"Batch halted to preserve diagnostics/avoid overlapping runs: {cleanup_error}")
            if interrupted:
                return 130
        return 0 if all(entry["status"] == "SUCCESS" for entry in state["runs"]) else 1
    finally:
        for sig, handler in previous_handlers.items():
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
            state = {"version": 1, "runtime": args.runtime, "created_at": stamp(), "active": None,
                     "queue": str(args.queue.resolve()), "runs": runs,
                     "limits": {key: getattr(args, key) for key in ("boot_timeout", "prepare_timeout", "stall_timeout", "run_timeout", "shutdown_timeout", "poll_interval")},
                     "source_fingerprint": fingerprint(ROOT)}
            session = archives / f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
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
                if previous.get("active") is not None:
                    raise BatchError(f"An unfinished attempt needs recovery first: --resume {pointer['session']}")
            if args.resume:
                if state["active"] is not None:
                    entry = state["runs"][state["active"]]
                    attempt = entry["attempt"]
                    folder = session / "attempts" / attempt
                    runtime = Runtime(ROOT, state["runtime"], attempt, entry["topology"])
                    recovery = folder / ("recovery_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
                    runtime.diagnostics(recovery)
                    runtime.stop(state["limits"]["shutdown_timeout"])
                    archive(ROOT, folder, attempt, entry["configs"])
                    restore_config(ROOT, session / "baseline_config")
                    entry.update(status="INTERRUPTED", reason="Previous supervisor exited before finalizing the attempt", finished_at=stamp())
                    atomic_json(folder / "status.json", entry)
                    state["active"] = None
                    atomic_json(session / "batch.json", state)
                if fingerprint(ROOT) != state["source_fingerprint"]:
                    raise BatchError("Source differs from the saved batch. Recovery is complete; start a new batch for the new revision.")
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
            atomic_json(archives / "active_batch.json", {"session": str(session)})
            print(f"Batch archive: {session}", flush=True)
            return supervise(ROOT, session, state)
    except (BatchError, OSError, ValueError, KeyError) as exc:
        print(f"Batch stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
