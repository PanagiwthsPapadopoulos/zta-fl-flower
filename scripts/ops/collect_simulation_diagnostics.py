"""Collect simulation logs and this user's HPC allocation; never stop processes."""
import argparse
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {"command": args, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": args, "error": str(exc)}


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = (args.output or ROOT / f"results/hpc_diagnostics_{stamp}.tar.gz").resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    commands = [["hostname"], ["lscpu"], ["free", "-h"], ["nvidia-smi"],
                ["nvidia-smi", "-L"], ["df", "-h", str(ROOT), "/tmp"],
                ["findmnt", "-T", str(ROOT)], ["findmnt", "-T", "/tmp"],
                ["squeue", "-u", getpass.getuser(), "-o", "%.18i %.9T %.20N %.40j"],
                ["ps", "-u", str(os.getuid()), "-o", "pid,ppid,stat,etime,comm"]]
    report = {"host": socket.gethostname(), "collected_utc": stamp,
              "environment": {k: v for k, v in os.environ.items()
                              if k in {"SLURM_JOB_ID", "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE",
                                       "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_JOB_GPUS",
                                       "SLURM_STEP_GPUS", "CUDA_VISIBLE_DEVICES", "SLURM_TMPDIR",
                                       "TMPDIR", "ZTA_SIM_SCRATCH_BASE"}},
              "commands": [command(args) for args in commands]}
    jobs = command(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i"])
    ids = set(jobs.get("stdout", "").split()) | {os.environ.get("SLURM_JOB_ID", "")}
    for job in sorted(j for j in ids if re.fullmatch(r"\d+(?:_\d+)?", j))[:20]:
        report["commands"].append(command(["scontrol", "show", "job", "-dd", job]))
        report["commands"].append(command(["sacct", "-j", job, "--units=M", "--format",
                                          "JobID,State,AllocCPUS,ReqMem,MaxRSS,Elapsed,ExitCode,AllocTRES"]))
    report["proc"] = {}
    for name in ("status", "limits", "cgroup"):
        path = Path("/proc/self") / name
        if path.exists():
            report["proc"][name] = path.read_text()
    sources = [(ROOT / "logs", "current/logs"), (ROOT / "runtime/batch", "current/runtime/batch")]
    marker = read_json(ROOT / "runtime/batch/local_runtime.json")
    report["scratch_marker"] = marker
    if marker.get("host") == socket.gethostname() and marker.get("path"):
        path = Path(marker["path"])
        if (path.is_dir() and not path.is_symlink() and path.name.startswith("zta-sim-")
                and path.stat().st_uid == os.getuid()
                and read_json(path / ".zta-owner.json") == marker):
            sources.append((path, "current/local_scratch"))
    # The last failed attempt often has only pointers: its logs remain under
    # current/logs. Also include any logs already moved by successful cleanup.
    attempts = sorted((ROOT / "results/batch_archives").glob("*/attempts/batch_*"),
                      key=lambda p: p.name)[-4:]
    for attempt in attempts:
        prefix = f"archived/{attempt.name}"
        for name in ("logs", "runtime/batch", "local_scratch", "boot.log", "dispatch.log",
                     "status.json", "events.jsonl", "preserved_paths.json", "runtime_diagnostics.json"):
            sources.append((attempt / name, f"{prefix}/{name}"))
        result = ROOT / "results" / attempt.name
        for name in ("batch_lifecycle.json", "batch_progress.json", "resolved_runtime.json",
                     "shutdown_processes.json", "process_inspection_error.json", "simulation_failure.json",
                     "flower_status.json"):
            sources.append((result / name, f"{prefix}/results/{name}"))
    report["found"] = [str(p) for p, _ in sources if p.exists()]
    report["missing"] = [str(p) for p, _ in sources if not p.exists()]
    report["note"] = "Read-only collection. Files from still-running processes may be changing; no processes were stopped."
    report["collection_errors"] = []
    with tempfile.TemporaryDirectory(prefix="zta-diagnostics-") as directory:
        with tarfile.open(output, "w:gz", dereference=False) as archive:
            for source, label in sources:
                if not source.exists() or source.is_symlink():
                    continue
                paths = [source] if source.is_file() else source.rglob("*")
                for path in paths:
                    if not path.is_file() or path.is_symlink():
                        continue
                    name = label if path == source else f"{label}/{path.relative_to(source)}"
                    try:
                        archive.add(path, arcname=name, recursive=False)
                    except OSError as exc:
                        report["collection_errors"].append(f"{path}: {exc}")
            details = Path(directory) / "allocation_and_locations.json"
            details.write_text(json.dumps(report, indent=2))
            archive.add(details, arcname=details.name)
    print(f"Collected logs and allocation details: {output}")
    print("Found locations:")
    for path in report["found"]:
        print(f"  {path}")
    if report["collection_errors"]:
        print("Some files could not be read; details are included in the archive.")
    print("Upload this .tar.gz file. No jobs or processes were stopped.")


if __name__ == "__main__":
    main()
