"""Direct builder execution must prefer mounted source over an old installation."""
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class BuilderImportTests(unittest.TestCase):
    def test_direct_execution_uses_checkout_from_unrelated_working_directory(self):
        root = Path(__file__).resolve().parents[2]
        with TemporaryDirectory() as temporary:
            stale = Path(temporary) / 'site-packages'
            package = stale / 'src/shared/data'
            package.mkdir(parents=True)
            (package / 'data_loader.py').write_text(
                "raise RuntimeError('stale installed data_loader imported')\n")
            config = Path(temporary) / 'empty-config'
            config.mkdir()
            env = dict(os.environ)
            env['PYTHONPATH'] = os.pathsep.join(
                [str(stale)] + [p for p in env.get('PYTHONPATH', '').split(os.pathsep) if p])
            env['CONFIG_PATH'] = str(config)
            result = subprocess.run(
                [sys.executable, '-B', str(root / 'scripts/setup/build_artifacts.py')],
                cwd=temporary, env=env, capture_output=True, text=True, timeout=120)
            # Missing configuration deliberately stops before any dataset writes.
            # Reaching this error proves all real builder imports succeeded.
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("KeyError: 'dataset'", result.stderr)
            self.assertNotIn('stale installed data_loader imported', result.stderr)
            self.assertNotIn('ImportError', result.stderr)


if __name__ == '__main__':
    unittest.main()
