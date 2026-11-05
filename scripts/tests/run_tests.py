"""Run the CPU unittest suite and optionally save reproducible evidence as JSON."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]


def source_fingerprint():
    paths = [ROOT / name for name in ('pyproject.toml', 'requirements.txt', 'README.md', '.dockerignore', '.gitignore')]
    for folder, pattern in (('src', '*.py'),
                            ('scripts', '*.py'), ('scripts', '*.sh'), ('docs', '*.md'),
                            ('docker', '*.Dockerfile'), ('config', '*.yaml')):
        paths.extend((ROOT / folder).rglob(pattern))
    files = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(paths)}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return digest, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pattern', default='test_*.py', help='unittest filename pattern')
    parser.add_argument('--report', type=Path, help='optional JSON evidence file (relative to repository root)')
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    # Synthetic/test fixtures select CPU; no TPM bypass is enabled.
    os.environ['ZTA_TIER'] = 'test'
    os.environ['ZTA_INSECURE_MODE'] = 'false'
    os.environ['CONFIG_PATH'] = str(ROOT / 'config')
    before_hash, files = source_fingerprint()
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'scripts/tests'), pattern=args.pattern)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.testsRun == 0:
        print('No tests matched; this is not a successful verification run.', file=sys.stderr)
    after_hash, _ = source_fingerprint()
    unchanged = before_hash == after_hash
    if not unchanged:
        print('Source changed during this run; evidence is invalid. Rerun after edits finish.', file=sys.stderr)
    versions = {}
    for package in ('torch', 'flwr', 'captum', 'numpy', 'scikit-learn', 'imbalanced-learn', 'torchmetrics'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    payload = {
        'started_at_utc': started_at, 'duration_seconds': time.perf_counter() - start,
        'pattern': args.pattern, 'python': sys.version, 'platform': platform.platform(),
        'dependencies': versions, 'source_sha256': before_hash, 'files_sha256': files,
        'source_unchanged_during_run': unchanged, 'tests_run': result.testsRun,
        'successful': result.wasSuccessful() and unchanged and result.testsRun > 0,
        'failures': [{'test': str(test), 'traceback': trace} for test, trace in result.failures],
        'errors': [{'test': str(test), 'traceback': trace} for test, trace in result.errors],
        'skipped': [{'test': str(test), 'reason': reason} for test, reason in result.skipped],
        'expected_failures': [str(test) for test, _ in result.expectedFailures],
        'unexpected_successes': [str(test) for test in result.unexpectedSuccesses],
        'scope': 'CPU source/component regression tests; TPM commands and RPC are replaced where documented in docs/testing.md. No live federation or performance reproduction.',
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
        print(f'Test evidence: {args.report.resolve()}')
    return 0 if payload['successful'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
