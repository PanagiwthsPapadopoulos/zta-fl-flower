"""Track Apptainer descendants without relying on their sanitized environments.

Names classify processes we must leave to Apptainer; they never confer ownership.
Unknown FUSE helpers block completion but are never signalled.
"""
import json
import os
from pathlib import Path
import signal

from src.shared.utils.simulation_lifecycle import atomic_json


class LifecycleError(RuntimeError):
    pass


def process_record(pid, proc=Path("/proc")):
    path = proc / str(pid)
    if path.stat().st_uid != os.getuid():
        return None
    raw = (path / "stat").read_text()
    fields = raw.rsplit(")", 1)[1].split()
    return {"pid": int(pid), "start_time": int(fields[19]),
            "ppid": int(fields[1]), "pgid": int(fields[2]), "sid": int(fields[3]),
            "state": fields[0], "name": raw.split("(", 1)[1].rsplit(")", 1)[0]}


def snapshot(proc=Path("/proc")):
    records, unreadable = {}, set()
    for path in proc.iterdir():
        if not path.name.isdigit():
            continue
        try:
            record = process_record(int(path.name), proc)
            if record is not None and record["state"] != "Z":
                records[record["pid"]] = record
        except (FileNotFoundError, ProcessLookupError):
            pass
        except (OSError, ValueError, IndexError):
            unreadable.add(int(path.name))
    return records, unreadable


def identity(record):
    return record["pid"], record["start_time"]


def is_helper(record):
    return record["name"].lower().startswith(
        ("squashfuse", "fuse-overlayfs", "fuse2fs", "gocryptfs"))


def is_runtime(record):
    return is_helper(record) or record["name"].lower().startswith(
        ("apptainer", "singularity", "starter"))


def signal_payload(record, sig):
    """Pin the PID before rechecking its identity; never kill a recycled PID."""
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise LifecycleError("Safe Apptainer shutdown requires Python/Linux pidfd support")
    try:
        fd = os.pidfd_open(record["pid"])
    except ProcessLookupError:
        return
    try:
        try:
            current = process_record(record["pid"])
        except (FileNotFoundError, ProcessLookupError):
            return
        if current and identity(current) == identity(record) and not is_runtime(current):
            signal.pidfd_send_signal(fd, sig)
    except ProcessLookupError:
        pass
    finally:
        os.close(fd)


class ProcessInventory:
    def __init__(self, path, attempt):
        self.path, self.attempt = Path(path), attempt
        self.known = {}
        self.baseline_helpers = set()
        self.initialized = False
        if self.path.exists():
            state = json.loads(self.path.read_text())
            if state["attempt"] != attempt:
                raise LifecycleError("Process inventory belongs to another attempt")
            self.known = {r["pid"]: r for r in state["processes"]}
            self.baseline_helpers = {tuple(key) for key in state["baseline_helpers"]}
            self.initialized = True

    def save(self):
        atomic_json(self.path, {"attempt": self.attempt,
                              "baseline_helpers": sorted(self.baseline_helpers),
                              "processes": list(self.known.values())})

    def begin(self):
        """Persist the baseline BEFORE the first launcher can create helpers."""
        if not self.initialized:
            records, unreadable = snapshot()
            if unreadable:
                raise LifecycleError("Cannot establish a complete process baseline")
            self.baseline_helpers = {identity(r) for r in records.values() if is_helper(r)}
            self.initialized = True
            self.save()

    def remember(self, pid):
        try:
            record = process_record(pid)
        except (FileNotFoundError, ProcessLookupError):
            return
        if record is not None:
            self.known[pid] = record
            self.save()

    def scan(self, marked):
        records, unreadable = snapshot()
        if unreadable & self.known.keys():
            raise LifecycleError("Cannot inspect a previously owned process; shutdown is unconfirmed")
        owned = {pid for pid, record in records.items()
                 if pid in self.known and identity(record) == identity(self.known[pid])}
        # Markers are read afresh by the caller. Recheck birth identity when supplied.
        for record in marked:
            current = records.get(record["pid"])
            if current and record.get("start_time") == current["start_time"]:
                owned.add(record["pid"])
        # A session is trusted only while it has an independently identified member.
        # A helper may change process group without leaving its launcher's session.
        sessions = {records[pid]["sid"] for pid in owned}
        changed = True
        while changed:
            additions = {pid for pid, r in records.items() if pid not in owned
                         and (r["ppid"] in owned or r["sid"] in sessions)}
            changed = bool(additions)
            owned.update(additions)
        updated = {pid: records[pid] for pid in owned}
        if any(self.known.get(pid) != record for pid, record in updated.items()):
            self.known.update(updated)
            self.save()
        # This is an uncertainty guard, NOT authority to signal another workload.
        unknown_helpers = [r for pid, r in records.items() if is_helper(r)
                           and pid not in owned and identity(r) not in self.baseline_helpers]
        return list(updated.values()), unknown_helpers, unreadable

    def assert_clean(self, marked):
        owned, unknown, unreadable = self.scan(marked)
        if owned or unknown or unreadable:
            raise LifecycleError(
                f"Apptainer shutdown unconfirmed: {len(owned)} owned processes, "
                f"{len(unknown)} unexplained new FUSE helpers, {len(unreadable)} unreadable processes; "
                "preserving outputs and refusing the next experiment")
