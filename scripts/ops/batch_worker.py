"""Run the frozen queue inside one container, with one owned Flower SuperLink.

This is an internal entry point. Start batches through batch_runs.py on the host.
"""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
import uuid

try:
    import psutil
except ImportError:
    from ray.thirdparty_files import psutil
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.ops.batch_runs import BatchError, Cancelled, CONFIG_FILES, complete, merged, read_json, progress
from src.shared.utils.simulation_lifecycle import atomic_json, stamp


def federation_config(manifest, count):
    local = manifest.get('tool', {}).get('flwr', {}).get('federations', {}).get('local', {})
    opts = local.get('options', local)
    backend = opts.get('backend', {})
    values = {'num-supernodes': count}
    for section in ('client-resources', 'init-args'):
        for key, value in backend.get(section, opts.get(section, {})).items():
            if key in ('num-cpus', 'num-gpus'):
                values[f'{section}-{key}'] = value
    if 'verbose' in opts:
        values['verbose'] = opts['verbose']
    return ' '.join(f'{key}={str(value).lower()}' for key, value in values.items())


class Flower:
    """One coordinator; only its run processes are drained between experiments."""
    def __init__(self, root, session, grace):
        self.root, self.session, self.grace = root, session, grace
        self.link = self.stream = None
        self.services = {}
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        self.scratch = Path(tempfile.mkdtemp(prefix='zta-batch-', dir='/tmp'))
        # Orphaned simulation/Ray processes stay children of this supervisor on Linux.
        if sys.platform == 'linux':
            if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
                raise OSError(ctypes.get_errno(), 'Cannot enable child subreaper')
        os.environ['PATH'] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get('PATH', '')
        os.environ.update(FLWR_HOME=str(self.scratch / 'flower'),
                          RAY_TMPDIR=str(self.scratch / 'ray'),
                          FLWR_DISABLE_UPDATE_CHECK='1',
                          FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION='1',
                          CONFIG_PATH=str(root / 'runtime/batch/config'))
        Path(os.environ['CONFIG_PATH']).mkdir(parents=True, exist_ok=True)
        home = Path(os.environ['FLWR_HOME'])
        home.mkdir()
        (home / 'config.toml').write_text(
            f'[superlink]\ndefault = "local"\n[superlink.local]\n'
            f'address = "127.0.0.1:{port}"\ninsecure = true\n')
        self.output = (session / 'superlink-console.log').open('ab', buffering=0)
        self.link = subprocess.Popen([
            'flower-superlink', '--insecure', '--simulation', '--isolation', 'subprocess',
            '--control-api-address', f'127.0.0.1:{port}',
            '--serverappio-api-address', '127.0.0.1:0',
            '--database', str(self.scratch / 'state.db'),
        ], cwd=root, stdout=self.output, stderr=subprocess.STDOUT)
        atomic_json(session / 'flower_runtime.json', {'pid': self.link.pid, 'scratch': str(self.scratch),
                    'control_address': f'127.0.0.1:{port}', 'started_at': stamp()})

    def cli(self, *args, timeout=30):
        if self.link.poll() is not None:
            raise BatchError('Flower SuperLink exited; see superlink-console.log')
        for pid, born in self.services.items():
            try:
                service = psutil.Process(pid)
                if service.create_time() != born or service.status() == psutil.STATUS_ZOMBIE:
                    raise BatchError('Flower SuperExec exited')
            except psutil.NoSuchProcess as exc:
                raise BatchError('Flower SuperExec exited') from exc
        result = subprocess.run(['flwr', *args, '--format', 'json'], cwd=self.root,
                                capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise BatchError(f'Flower command failed: {result.stderr[-2000:]} {result.stdout[-2000:]}')
        data = json.loads(result.stdout)
        if not isinstance(data, dict) or data.get('success') is not True:
            raise BatchError(f'Flower command did not succeed: {data}')
        return data

    def ready(self):
        deadline = time.monotonic() + 90
        while True:
            try:
                data = self.cli('list', 'local', timeout=5)
                if data['runs']:
                    raise BatchError('New private SuperLink unexpectedly contains existing runs')
                children = psutil.Process(self.link.pid).children()
                executors = [p for p in children if any(Path(arg).name == 'flower-superexec' for arg in p.cmdline()[:2])]
                if not executors:
                    raise BatchError('Waiting for Flower SuperExec')
                self.services = {p.pid: p.create_time() for p in executors}
                return
            except (BatchError, subprocess.TimeoutExpired):
                if self.link.poll() is not None or time.monotonic() >= deadline:
                    raise
                time.sleep(1)

    def remaining(self):
        excluded = {self.link.pid} if self.link else set()
        if self.stream:
            excluded.add(self.stream.pid)
        live = []
        for process in psutil.Process().children(recursive=True):
            if process.pid in excluded:
                continue
            try:
                if self.services.get(process.pid) == process.create_time():
                    continue
                if process.status() == psutil.STATUS_ZOMBIE:
                    try:
                        os.waitpid(process.pid, os.WNOHANG)
                    except ChildProcessError:
                        pass
                else:
                    live.append(process)
            except psutil.NoSuchProcess:
                pass
        return live

    def drain(self):
        """Allow normal exit, then terminate only this worker's run descendants."""
        for action in (None, 'terminate', 'kill'):
            deadline = time.monotonic() + self.grace
            sent = set()
            while True:
                live = self.remaining()
                if not live:
                    return
                for process in live:
                    if action and process.pid not in sent:
                        try:
                            getattr(process, action)()
                        except psutil.NoSuchProcess:
                            pass
                        sent.add(process.pid)
                if time.monotonic() >= deadline:
                    break
                time.sleep(.2)
        raise BatchError('Training processes remain alive; refusing to change experiment configuration')

    def stop_stream(self):
        if self.stream:
            if self.stream.poll() is None:
                self.stream.terminate()
                try:
                    self.stream.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.stream.kill()
            self.stream.wait()
            self.stream = None

    def archive_ray(self, folder):
        ray = self.scratch / 'ray'
        if ray.exists():
            for source in ray.glob('ray/session_*/logs/*'):
                if not source.parent.parent.is_symlink() and source.is_file() and source.suffix in ('.log', '.out', '.err'):
                    target = folder / 'ray' / source.parent.parent.name / source.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
            shutil.rmtree(ray)

    def close(self):
        self.stop_stream()
        self.drain()
        if self.link:
            self.link.terminate()
            try:
                self.link.wait(timeout=self.grace)
            except subprocess.TimeoutExpired:
                self.link.kill()
                self.link.wait(timeout=5)
        self.output.close()
        self.services.clear()
        self.drain()
        database = self.scratch / 'state.db'
        if database.exists():
            shutil.copy2(database, self.session / 'flower-state.db')
        shutil.rmtree(self.scratch)


def save_logs(root, folder):
    # Keep mounted directory in place; only rotate files after all run processes exit.
    for source in (root / 'logs').glob('*.jsonl'):
        target = folder / 'logs' / source.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
        source.unlink()


def save_artifact_metadata(root, folder, configs):
    raw = Path(merged(configs)["dataset_path"])
    if not raw.is_absolute():
        raw = root / raw
    for source in (raw.parent.parent / 'artifacts').glob('*_manifest.json'):
        target = folder / 'artifact_manifests' / source.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
    if raw.exists():
        atomic_json(folder / 'dataset_reference.json', {'path': str(raw), 'bytes': raw.stat().st_size})


def run_experiment(root, folder, entry, limits, flower, update, cancelled):
    start = time.monotonic()
    run_id = None
    prepared = False
    status, reason = 'CRASHED', 'Run was not submitted'
    log_path = root / 'logs/system/run_cloud.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f'Preparing dataset; see {folder / "prepare.log"}\n')
    try:
        with (folder / 'prepare.log').open('wb') as log:
            builder = subprocess.Popen([sys.executable, str(root / 'scripts/setup/build_artifacts.py')],
                                       cwd=root, stdout=log, stderr=subprocess.STDOUT)
            while builder.poll() is None:
                if cancelled():
                    raise Cancelled('Batch interrupted')
                if time.monotonic() - start >= min(limits['prepare_timeout'], limits['run_timeout']):
                    return 'PREPARATION_TIMEOUT', 'Dataset preparation exceeded its time limit'
                time.sleep(min(limits['poll_interval'], 1))
            if builder.returncode:
                return 'CRASHED', 'Dataset preparation failed; see prepare.log'
        prepared = True
        save_artifact_metadata(root, folder, entry['configs'])
        update(phase='training')
        manifest = tomllib.loads((root / 'pyproject.toml').read_text())
        submitted = flower.cli('run', str(root), 'local', '--run-config',
            f'mode="sim" tier="cloud" min_clients={sum(entry["topology"])}',
            '--federation-config', federation_config(manifest, sum(entry['topology'])), timeout=120)
        run_id = submitted['run-id']
        update(flower_run_id=run_id)
        atomic_json(folder / 'flower_submission.json', submitted)
        with log_path.open('wb') as log:
            flower.stream = subprocess.Popen(['flwr', 'log', str(run_id), 'local', '--stream'],
                                             cwd=root, stdout=log, stderr=subprocess.STDOUT)
        last_progress, best = time.monotonic(), -1
        while True:
            if cancelled():
                raise Cancelled('Batch interrupted')
            number = progress(root, entry['attempt'])
            if number > best:
                best, last_progress = number, time.monotonic()
                update(completed_round=number)
            rows = flower.cli('list', 'local', '--run-id', str(run_id))['runs']
            if len(rows) != 1 or str(rows[0]['run-id']) != str(run_id):
                raise BatchError('Flower returned no matching run status')
            row = rows[0]
            atomic_json(folder / 'flower_status.json', row)
            if row['status'].startswith('finished:'):
                if row['status'] == 'finished:completed' and complete(root, entry['attempt'], entry['target_rounds']):
                    status, reason = 'SUCCESS', 'Flower completed; all requested rounds and final checkpoint saved'
                else:
                    status, reason = 'INCOMPLETE', f"Flower {row['status']}: {row.get('status-details', '')}"
                return status, reason
            if time.monotonic() - start >= limits['run_timeout']:
                return 'TIMEOUT', 'Total experiment time limit exceeded'
            if time.monotonic() - last_progress >= limits['stall_timeout']:
                return 'STALLED', 'No newly saved round within the progress time limit'
            time.sleep(limits['poll_interval'])
    finally:
        # Even a failed/expired Flower task can still have live workers.
        if run_id:
            try:
                flower.cli('stop', str(run_id), 'local')
            except (BatchError, subprocess.TimeoutExpired, ValueError) as exc:
                (folder / 'stop_error.txt').write_text(str(exc))
        flower.stop_stream()
        flower.drain()
        if prepared and log_path.exists():
            shutil.copy2(log_path, folder / 'run_cloud.log')
        save_logs(root, folder)
        flower.archive_ray(folder)


def work(root, session):
    state = read_json(session / 'batch.json')
    limits = state['limits']
    stop_file = session / 'STOP'
    cancelled_flag = False
    def cancel(*_):
        nonlocal cancelled_flag
        cancelled_flag = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, cancel)
    def cancelled():
        return cancelled_flag or stop_file.exists()
    flower = None
    try:
        runtime_folder = session / state['infrastructure_id']
        runtime_folder.mkdir(exist_ok=True)
        flower = Flower(root, runtime_folder, limits['shutdown_timeout'])
        flower.ready()
        for index, entry in enumerate(state['runs']):
            if cancelled():
                return 130
            if entry['status'] != 'PENDING':
                continue
            attempt = f"batch_{session.name}_{index + 1:02d}_{uuid.uuid4().hex[:8]}"
            folder = session / 'attempts' / attempt
            folder.mkdir(parents=True)
            entry.update(attempt=attempt, status='RUNNING', phase='preparing', started_at=stamp(), completed_round=-1)
            state['active'] = index
            def update(**values):
                entry.update(values)
                atomic_json(folder / 'status.json', entry)
                atomic_json(session / 'batch.json', state)
                with (folder / 'events.jsonl').open('a') as handle:
                    handle.write(json.dumps({'timestamp': stamp(), **values}) + '\n')
            configs = json.loads(json.dumps(entry['configs']))
            configs['training.yaml'].update(run_name=attempt, batch_attempt=attempt)
            for directory in (folder / 'executed_config', Path(os.environ['CONFIG_PATH'])):
                directory.mkdir(parents=True, exist_ok=True)
                for name, config in configs.items():
                    (directory / name).write_text(yaml.safe_dump(config, sort_keys=False))
            atomic_json(folder / 'effective_config.json', merged(configs))
            update()
            print(f"[{stamp()}] Starting {entry['name']} -> {attempt}", flush=True)
            started = time.monotonic()
            try:
                status, reason = run_experiment(root, folder, entry, limits, flower, update, cancelled)
            except Cancelled as exc:
                status, reason = 'CANCELLED', str(exc)
            except Exception as exc:
                # Unknown worker/control failures halt; never replace config under a live app.
                update(status='CRASHED', reason=str(exc), finished_at=stamp())
                raise
            state['active'] = None
            update(status=status, reason=reason, finished_at=stamp(), elapsed_seconds=time.monotonic() - started)
            print(f"[{stamp()}] {entry['name']}: {status} — {reason}", flush=True)
            if status == 'CANCELLED':
                return 130
        return 0 if all(r['status'] == 'SUCCESS' for r in state['runs']) else 1
    finally:
        if flower:
            flower.close()


if __name__ == '__main__':
    try:
        raise SystemExit(work(Path('/app'), Path(sys.argv[1])))
    except Exception as exc:
        print(f'Batch worker stopped: {exc}', file=sys.stderr, flush=True)
        raise
