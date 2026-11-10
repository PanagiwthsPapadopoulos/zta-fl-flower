"""Run the frozen queue inside one container, with one owned Flower SuperLink.

This is an internal entry point. Start batches through batch_runs.py on the host.
"""
from __future__ import annotations

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

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.ops.batch_runs import BatchError, Cancelled, stop_process, merged, read_json, progress
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
    """One coordinator; Flower run status controls bundle handover."""
    def __init__(self, root, session, grace):
        self.root, self.session, self.grace = root, session, grace
        self.link = self.stream = None
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        self.scratch = Path(tempfile.mkdtemp(prefix='zta-batch-', dir='/tmp'))
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
        ], cwd=root, stdout=self.output, stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(session / 'flower_runtime.json', {'pid': self.link.pid, 'scratch': str(self.scratch),
                    'control_address': f'127.0.0.1:{port}', 'started_at': stamp()})

    def cli(self, *args, timeout=30):
        result = subprocess.run(['flwr', *args, '--format', 'json'], cwd=self.root,
                                capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise BatchError(f'Flower command failed: {result.stderr[-2000:]} {result.stdout[-2000:]}')
        data = json.loads(result.stdout)
        if not isinstance(data, dict) or data.get('success') is not True:
            raise BatchError(f'Flower command did not succeed: {data}')
        return data

    def ready(self, timeout):
        # Only ask Flower whether its API is accepting bundle/status requests.
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.cli('list', 'local', timeout=5)
                return
            except (BatchError, subprocess.TimeoutExpired, ValueError, OSError):
                if time.monotonic() >= deadline:
                    raise BatchError('Flower API startup timed out; see superlink-console.log')
                time.sleep(1)

    def wait_finished(self, run_id):
        deadline = time.monotonic() + self.grace
        while time.monotonic() < deadline:
            try:
                rows = self.cli('list', 'local', '--run-id', str(run_id), timeout=5)['runs']
                if any(str(row['run-id']) == str(run_id) and row['status'].startswith('finished:') for row in rows):
                    return
            except (BatchError, subprocess.TimeoutExpired, ValueError, OSError):
                pass
            time.sleep(1)
        raise BatchError('Flower stop timed out; next bundle was not submitted')

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
        if self.link:
            stop_process(self.link, self.grace)
        self.output.close()
        database = self.scratch / 'state.db'
        if database.exists():
            shutil.copy2(database, self.session / 'flower-state.db')
        shutil.rmtree(self.scratch, ignore_errors=True)


def save_logs(root, folder):
    # Rotate after Flower reports the run finished.
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
    finished = False
    builder = None
    status, reason = 'CRASHED', 'Run was not submitted'
    log_path = root / 'logs/system/run_cloud.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f'Preparing dataset; see {folder / "prepare.log"}\n')
    try:
        with (folder / 'prepare.log').open('wb') as log:
            builder = subprocess.Popen([sys.executable, str(root / 'scripts/setup/build_artifacts.py')],
                                       cwd=root, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while builder.poll() is None:
                if cancelled():
                    raise Cancelled('Batch interrupted')
                if time.monotonic() - start >= min(limits['prepare_timeout'], limits['run_timeout']):
                    return 'PREPARATION_TIMEOUT', 'Dataset preparation exceeded its time limit'
                time.sleep(min(limits['poll_interval'], 1))
            if builder.returncode:
                return 'CRASHED', 'Dataset preparation failed; see prepare.log'
        prepared = True
        try:
            save_artifact_metadata(root, folder, entry['configs'])
        except OSError as exc:
            print(f'Artifact metadata: {exc}', file=sys.stderr, flush=True)
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
            try:
                rows = flower.cli('list', 'local', '--run-id', str(run_id))['runs']
                row = next((r for r in rows if str(r['run-id']) == str(run_id)), None)
                if row:
                    atomic_json(folder / 'flower_status.json', row)
                    if row['status'].startswith('finished:'):
                        finished = True
                        status = 'SUCCESS' if row['status'] == 'finished:completed' else 'INCOMPLETE'
                        return status, f"Flower {row['status']}: {row.get('status-details', '')}"
            except (BatchError, subprocess.TimeoutExpired, ValueError, OSError) as exc:
                (folder / 'status_error.txt').write_text(str(exc))
            if time.monotonic() - start >= limits['run_timeout']:
                return 'TIMEOUT', 'Total experiment time limit exceeded'
            if time.monotonic() - last_progress >= limits['stall_timeout']:
                return 'STALLED', 'No newly saved round within the progress time limit'
            time.sleep(limits['poll_interval'])
    finally:
        if builder is not None and builder.poll() is None:
            stop_process(builder, limits["shutdown_timeout"])
        if run_id and not finished:
            try:
                flower.cli('stop', str(run_id), 'local')
            except (BatchError, subprocess.TimeoutExpired, ValueError) as exc:
                (folder / 'stop_error.txt').write_text(str(exc))
            flower.wait_finished(run_id)
        flower.stop_stream()
        try:
            if prepared and log_path.exists():
                shutil.copy2(log_path, folder / 'run_cloud.log')
            save_logs(root, folder)
            flower.archive_ray(folder)
        except OSError as exc:
            print(f'Log archive: {exc}', file=sys.stderr, flush=True)


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
        flower.ready(limits["boot_timeout"])
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
