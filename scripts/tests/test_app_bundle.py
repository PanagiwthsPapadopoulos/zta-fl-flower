"""Real Flower packaging must tolerate retained runtime and batch diagnostics."""
import io
from pathlib import Path
import unittest
import zipfile

from flwr.cli.build import build_fab_from_files


ROOT = Path(__file__).resolve().parents[2]


class AppBundleTests(unittest.TestCase):
    def files(self):
        files = {name: ROOT / name for name in (
            "pyproject.toml", "src/entrypoints/server.py", "config/training.yaml")}
        files["results/batch_archives/batch/attempts/attempt/runtime/batch/flower/apps/publisher/app/config/training.yaml"] = b"num_rounds: 99\n"
        return files

    def test_missing_ignore_file_reproduces_archived_configuration_depth_failure(self):
        with self.assertRaisesRegex(ValueError, "maximum directory depth"):
            build_fab_from_files(self.files())

    def test_local_ignore_rules_exclude_archives_and_keep_active_configuration(self):
        files = self.files()
        files[".gitignore"] = ROOT / ".gitignore"
        payload, _ = build_fab_from_files(files)
        with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
            names = bundle.namelist()
            self.assertIn("src/entrypoints/server.py", names)
            self.assertEqual(bundle.read("config/training.yaml"), (ROOT / "config/training.yaml").read_bytes())
            self.assertFalse(any(name.startswith("results/") for name in names))


if __name__ == "__main__":
    unittest.main()
