"""Sequential native Flower/MPS experiments, each with its own Docker TPMs."""
from __future__ import annotations

import argparse
import fcntl
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
from scripts.ops.batch_runs import BatchError, completion_evidence, plan_queue, read_json
from src.shared.utils.simulation_lifecycle import atomic_json, stamp

RETRYABLE = {"FAILED", "INCOMPLETE", "DEGRADED", "TIMEOUT", "CANCELLED", "INTERRUPTED"}


def stop_launcher(child, grace):
    """Let the native launcher clean up its independent Flower groups and TPMs."""
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
        raise BatchError("Native launcher did not finish cleanup; inspect its processes and compose.yaml before resuming")


def run_one(root, folder, entry, settings, cancelled, update):
    config_dir = folder / "config"
    config_dir.mkdir(parents=True)
    for name, config in entry["configs"].items():
        (config_dir / name).write_text(yaml.safe_dump(config, sort_keys=False))
    run_dir = Path(entry["run_dir"])
    command = [sys.executable, "-u", "-B", str(root / "scripts/simulation/run_native_mps.py"),
               "--config-dir", str(config_dir), "--run-dir", str(run_dir),
               "--workers", str(settings["workers"]), "--tpm-image", settings["tpm_image"]]
    if settings["smoke_test"]:
        command.append("--smoke-test")
    deadline = time.monotonic() + settings["run_timeout"]
    interrupted = None
    with (folder / "launcher.log").open("wb") as log:
        child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        update(launcher_pid=child.pid)
        try:
            best = -1
            while child.poll() is None:
                if cancelled():
                    interrupted = "CANCELLED"
                    break
                if time.monotonic() >= deadline:
                    interrupted = "TIMEOUT"
                    break
                progress = read_json(run_dir / "results" / run_dir.name / "batch_progress.json")
                number = progress.get("completed_round", -1)
                if isinstance(number, int) and number > best:
                    best = number
                    update(completed_round=number)
                time.sleep(1)
        finally:
            stop_launcher(child, settings["shutdown_timeout"])
    status = read_json(run_dir / "native_status.json")
    if not status:
        raise BatchError(f"Native launcher exited {child.returncode} without native_status.json; "
                         f"startup may have failed. Inspect {folder / 'launcher.log'}. Cleanup is unverified.")
    if "cleanup_errors" not in status or status["cleanup_errors"]:
        raise BatchError(f"Cleanup not confirmed: {status.get('cleanup_errors', 'missing cleanup evidence')}; "
                         f"inspect {run_dir / 'native_status.json'}")
    if interrupted:
        return interrupted, "Native launcher stopped and cleanup completed"
    if child.returncode != 0 or status.get("status") != "COMPLETED":
        return "FAILED", status.get("error", f"Native launcher exited {child.returncode}")
    target = 2 if settings["smoke_test"] else entry["target_rounds"]
    outcome, reason = completion_evidence(run_dir, run_dir.name, target)
    if outcome in {"SUCCESS", "DEGRADED"}:
        update(completed_round=target)
    return outcome, reason


def execute(root, session, state):
    cancelled_flag = False
    def cancel(*_):
        nonlocal cancelled_flag
        cancelled_flag = True
    handlers = {sig: signal.signal(sig, cancel) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for index, entry in enumerate(state["runs"]):
            if cancelled_flag:
                return 130
            if entry["status"] != "PENDING":
                continue
            attempt = f"native_{session.name}_{index + 1:03d}_{uuid.uuid4().hex[:8]}"
            folder = session / "attempts" / attempt
            folder.mkdir(parents=True)
            entry.update(status="RUNNING", attempt=attempt, started_at=stamp(), completed_round=-1,
                         run_dir=str(root / "runtime/native-mps" / attempt))
            def update(**values):
                entry.update(values)
                atomic_json(session / "batch.json", state)
                atomic_json(folder / "status.json", entry)
            update()
            print(f"Starting {entry['name']}: {entry['run_dir']}", flush=True)
            try:
                outcome, reason = run_one(root, folder, entry, state["settings"], lambda: cancelled_flag, update)
            except Exception as exc:
                update(status="FAILED", reason=str(exc), finished_at=stamp())
                raise
            update(status=outcome, reason=reason, finished_at=stamp())
            print(f"{entry['name']}: {outcome} — {reason}", flush=True)
            if outcome != "SUCCESS":
                return 130 if outcome == "CANCELLED" else 1
        return 0 if all(e["status"] == "SUCCESS" for e in state["runs"]) else 1
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def resume_state(session, retry_failed):
    state = read_json(session / "batch.json")
    if state.get("runtime") != "native-mps" or state.get("version") != 1:
        raise BatchError("Not a native MPS batch archive")
    for entry in state["runs"]:
        if entry.get("run_dir") and entry["status"] != "SUCCESS":
            previous = read_json(Path(entry["run_dir"]) / "native_status.json")
            if not previous or "cleanup_errors" not in previous or previous["cleanup_errors"]:
                raise BatchError("Previous attempt has no confirmed cleanup; inspect it before resuming")
        if entry["status"] == "RUNNING":
            status = read_json(Path(entry["run_dir"]) / "native_status.json")
            if not status or "cleanup_errors" not in status or status["cleanup_errors"]:
                raise BatchError("Previous running attempt has no confirmed cleanup; inspect it before resuming")
            if status.get("status") == "COMPLETED":
                run_dir = Path(entry["run_dir"])
                target = 2 if state["settings"]["smoke_test"] else entry["target_rounds"]
                entry["status"], entry["reason"] = completion_evidence(run_dir, run_dir.name, target)
            else:
                entry["status"] = "INTERRUPTED"
        if retry_failed and entry["status"] in RETRYABLE:
            # Never retry an attempt whose launcher might still own resources.
            if entry.get("run_dir"):
                status = read_json(Path(entry["run_dir"]) / "native_status.json")
                if not status or "cleanup_errors" not in status or status["cleanup_errors"]:
                    raise BatchError("Retry requires confirmed cleanup of the previous attempt")
            history = entry.get("previous_attempts", []) + [{
                k: v for k, v in entry.items() if k not in {"configs", "overrides", "previous_attempts"}}]
            for key in ("attempt", "run_dir", "launcher_pid", "started_at", "finished_at", "completed_round", "reason"):
                entry.pop(key, None)
            entry.update(status="PENDING", previous_attempts=history)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=ROOT / "experiments_queue")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tpm-image", default="panagiotispapadopoulos/zta-edge-node:latest")
    parser.add_argument("--run-timeout", type=float, default=172800, help="Seconds per experiment including preparation (default: 48 hours)")
    parser.add_argument("--shutdown-timeout", type=float, default=180)
    parser.add_argument("--smoke-test", action="store_true", help="Explicitly run each entry as two rounds/two clients using the native smoke profile")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--retry-failed", action="store_true", help="On resume, restart unsuccessful entries from round one")
    args = parser.parse_args(argv)
    if not args.tpm_image.strip():
        parser.error("--tpm-image must not be empty; supply an image name directly or omit the option")
    if not 1 <= args.workers <= (os.cpu_count() or 1) or not 0 < args.run_timeout < float("inf") or not 0 < args.shutdown_timeout < float("inf"):
        parser.error("workers must fit this host; timeouts must be finite and positive")
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")
    try:
        if args.resume:
            session = args.resume.resolve()
        else:
            _, runs = plan_queue(ROOT, args.queue.resolve(), fixed_topology=False)
            if not runs:
                raise BatchError("Queue contains no experiment configurations")
            settings = {key: getattr(args, key) for key in ("workers", "tpm_image", "run_timeout", "shutdown_timeout", "smoke_test")}
            state = {"version":1, "runtime":"native-mps", "created_at":stamp(), "queue":str(args.queue.resolve()),
                     "settings":settings, "runs":runs}
            session = ROOT / "results/native_batch_archives" / (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
        if args.dry_run:
            if args.resume:
                state = resume_state(session, args.retry_failed)
            print(json.dumps(state, indent=2))
            return 0
        session.mkdir(parents=True, exist_ok=True)
        with (session / ".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise BatchError("This batch is already running") from None
            if args.resume:
                state = resume_state(session, args.retry_failed)
            atomic_json(session / "batch.json", state)
            print(f"Native batch archive: {session}", flush=True)
            return execute(ROOT, session, state)
    except (BatchError, OSError, ValueError, KeyError) as exc:
        print(f"Native batch stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
