"""Submit simulation experiments sequentially on one persistent environment."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.shared.utils.simulation_lifecycle import atomic_json, stamp

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
    return value


def merged(configs):
    result = {}
    for name in CONFIG_FILES:
        result.update(configs[name])
    return result


def topology(config):
    custom = config.get("custom_fog_topology", "[]")
    if isinstance(custom, str):
        custom = ast.literal_eval(custom)
    return custom or [config["uniform_edges_per_fog"]] * config["num_fogs"]


def plan_queue(root, queue):
    baseline = {name: read_yaml(root / "config" / name) for name in CONFIG_FILES}
    runs = []
    for folder in sorted(queue.iterdir()):
        if not folder.is_dir() or folder.name.startswith("."):
            continue
        configs = json.loads(json.dumps(baseline))
        overrides = {}
        for name in CONFIG_FILES:
            path = folder / name
            if path.is_file():
                configs[name].update(read_yaml(path))
                overrides[name] = path.read_text()
        if not overrides:
            continue
        config = merged(configs)
        runs.append({"name": folder.name, "configs": configs, "overrides": overrides,
                     "target_rounds": config["num_rounds"], "topology": topology(config), "status": "PENDING"})
    return baseline, runs


def progress(root, attempt):
    value = read_json(root / "results" / attempt / "batch_progress.json")
    number = value.get("completed_round", -1)
    return number if value.get("attempt") == attempt and type(number) is int else -1


def stop_process(child, grace):
    """Signal the process group we launched; no system-wide process inspection."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=5)


def supervise(root, session, state):
    limits = state["limits"]
    folder = session / state["infrastructure_id"]
    folder.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "ZTA_BATCH_MANAGED": "1", "ZTA_BATCH_ATTEMPT": state["infrastructure_id"],
           "APPTAINERENV_ZTA_BATCH_ATTEMPT": state["infrastructure_id"],
           "ZTA_BATCH_SESSION": "/app/results/batch_archives/" + session.name,
           "ZTA_BATCH_PHASE_FILE": str(root / "runtime/batch_phase"),
           "PROJECT_ROOT": str(root), "CONFIG_PATH": str(root / "config"),
           "APPTAINERENV_CONFIG_PATH": "/app/config"}
    children = []
    interrupted = False
    def cancel(*_):
        nonlocal interrupted
        interrupted = True
        (session / "STOP").touch()
    handlers = {sig: signal.signal(sig, cancel) for sig in (signal.SIGINT, signal.SIGTERM)}
    def launch(name, log):
        with (folder / log).open("wb") as output:
            child = subprocess.Popen(["bash", str(root / f"scripts/simulation/{name}_{state['runtime']}.sh")],
                                     cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        children.append(child)
        return child
    marker = root / "runtime/infra/.network_ready"
    marker.unlink(missing_ok=True)
    try:
        launch("boot_network_sim", "boot.log")
        deadline = time.monotonic() + limits["boot_timeout"]
        while not marker.exists():
            if interrupted:
                return 130
            if time.monotonic() >= deadline:
                raise BatchError("Network boot timed out; see boot.log")
            time.sleep(limits["poll_interval"])
        worker = launch("deploy_code_sim", "worker.log")
        print(f"Training progress: {folder / 'worker.log'}", flush=True)
        deadline = None
        while worker.poll() is None:
            if interrupted:
                deadline = deadline or time.monotonic() + limits["shutdown_timeout"]
                if time.monotonic() >= deadline:
                    return 130
            time.sleep(limits["poll_interval"])
        return 130 if interrupted else worker.returncode
    finally:
        for child in reversed(children):
            try:
                stop_process(child, limits["shutdown_timeout"])
            except (OSError, subprocess.TimeoutExpired) as exc:
                print(f"Shutdown: {exc}", file=sys.stderr)
        if state["runtime"] == "docker":
            subprocess.run(["docker", "compose", "-f", str(root / "docker/docker-compose.yml"),
                            "--project-directory", str(root), "down"], timeout=limits["shutdown_timeout"])
        marker.unlink(missing_ok=True)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


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
    parser.add_argument("--dry-run", action="store_true", help="Print the merged queue without launching anything")
    parser.add_argument("--resume", type=Path, help="Continue pending entries in an existing batch")
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--retry-interrupted", action="store_true")
    policy.add_argument("--skip-interrupted", action="store_true")
    for flag, default in (("boot-timeout", 1800), ("prepare-timeout", 7200), ("stall-timeout", 3600),
                          ("run-timeout", 25200), ("shutdown-timeout", 120), ("poll-interval", 5)):
        parser.add_argument("--" + flag, type=float, default=default)
    args = parser.parse_args(argv)
    try:
        if args.resume:
            session = args.resume.resolve()
            state = read_json(session / "batch.json")
            for entry in state["runs"]:
                if entry["status"] == "RUNNING":
                    entry["status"] = "INTERRUPTED"
            state["active"] = None
            if args.retry_interrupted:
                retry_interrupted(state)
        else:
            if not args.runtime:
                parser.error("--runtime docker|apptainer is required")
            baseline, runs = plan_queue(ROOT, args.queue.resolve())
            session = ROOT / "results/batch_archives" / f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:8]}"
            state = {"version": 3, "runtime": args.runtime, "created_at": stamp(), "active": None,
                     "queue": str(args.queue.resolve()), "runs": runs,
                     "limits": {key: getattr(args, key) for key in ("boot_timeout", "prepare_timeout", "stall_timeout", "run_timeout", "shutdown_timeout", "poll_interval")}}
        if args.dry_run:
            print(json.dumps(state, indent=2))
            return 0
        if not any(r["status"] == "PENDING" for r in state["runs"]):
            print("No pending experiments.")
            return 0
        session.mkdir(parents=True, exist_ok=True)
        (session / "STOP").unlink(missing_ok=True)
        state["infrastructure_id"] = "infra_" + session.name + "_" + uuid.uuid4().hex[:8]
        atomic_json(session / "batch.json", state)
        atomic_json(ROOT / "results/batch_archives/active_batch.json", {"session": str(session)})
        print(f"Batch archive: {session}", flush=True)
        return supervise(ROOT, session, state)
    except (BatchError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        print(f"Batch stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
