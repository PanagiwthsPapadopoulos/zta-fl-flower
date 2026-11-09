"""Own node-local runtime storage, archiving it only after confirmed shutdown."""
import json
import os
from pathlib import Path
import shutil
import socket
import stat
import subprocess
import tempfile
import sqlite3

from src.shared.utils.simulation_lifecycle import atomic_json


def prepare(root, attempt):
    marker = Path(root) / "runtime/batch/local_runtime.json"
    if marker.exists():
        raise RuntimeError("Previous simulation scratch marker remains; recover that attempt first")
    candidates = [os.environ.get(k) for k in ("ZTA_SIM_SCRATCH_BASE", "SLURM_TMPDIR", "TMPDIR")]
    candidates.append("/tmp")
    errors = []
    for candidate in dict.fromkeys(p for p in candidates if p):
        base = Path(candidate).resolve()
        try:
            if not base.is_dir():
                raise ValueError("directory does not exist")
            if Path("/proc/self/mountinfo").exists():
                # Do not silently put a WAL database back on shared home/scratch.
                fs = subprocess.run(["findmnt", "-n", "-o", "FSTYPE", "-T", str(base)],
                                    check=True, capture_output=True, text=True, timeout=5).stdout.strip()
                if fs not in {"tmpfs", "ext2", "ext3", "ext4", "xfs", "btrfs", "overlay", "zfs"}:
                    raise ValueError(f"filesystem {fs!r} is not verified node-local storage")
            path = Path(tempfile.mkdtemp(prefix="zta-sim-", dir=base))
            owner = {"attempt": attempt, "root": str(Path(root).resolve()),
                     "path": str(path), "host": socket.gethostname(), "uid": os.getuid()}
            atomic_json(path / ".zta-owner.json", owner)
            atomic_json(marker, owner)
            return path
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            errors.append(f"{base}: {exc}")
    raise RuntimeError("No writable node-local simulation scratch: " + "; ".join(errors))


def owned_path(root, attempt):
    marker = Path(root) / "runtime/batch/local_runtime.json"
    if not marker.exists():
        return None
    owner = json.loads(marker.read_text())
    path = Path(owner["path"])
    if (owner["attempt"] != attempt or owner["root"] != str(Path(root).resolve())
            or owner["host"] != socket.gethostname() or owner["uid"] != os.getuid()
            or not path.name.startswith("zta-sim-") or path.is_symlink()
            or path.resolve() != path or path.stat().st_uid != os.getuid()
            or json.loads((path / ".zta-owner.json").read_text()) != owner):
        raise RuntimeError("Simulation scratch ownership mismatch; preserving files")
    return path


def archive_scratch(root, destination, attempt):
    path = owned_path(root, attempt)
    if path is None:
        return
    def ignore_special(directory, names):
        return [name for name in names if not (stat.S_ISREG((Path(directory) / name).lstat().st_mode)
                or stat.S_ISDIR((Path(directory) / name).lstat().st_mode)
                or stat.S_ISLNK((Path(directory) / name).lstat().st_mode))]
    # Preserve symlinks without following Ray's session_latest or external links.
    shutil.copytree(path, Path(destination) / "local_scratch", symlinks=True,
                    ignore=ignore_special, dirs_exist_ok=True)
    shutil.rmtree(path)
    (Path(root) / "runtime/batch/local_runtime.json").unlink()


def flower_status(root, attempt):
    """Read terminal task evidence without exporting authentication fields."""
    local = owned_path(root, attempt)
    home = local / "flower" if local else Path(root) / "runtime/batch/flower"
    path = home / "local-superlink/state.db"
    if not path.exists():
        return {}
    try:
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=1) as connection:
            rows = connection.execute("SELECT sub_status, details, finished_at FROM task ORDER BY pending_at DESC LIMIT 1").fetchall()
        return dict(zip(("status", "details", "finished_at"), rows[0])) if rows else {}
    except sqlite3.Error as exc:
        return {"inspection_error": str(exc)}
