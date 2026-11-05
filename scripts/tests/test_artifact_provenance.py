"""Dataset provenance checks without requiring Torch or source datasets."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.shared.data.artifact_manifest import (
    base_provenance, partition_provenance, split_fractions, validate_artifact, write_manifest,
)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.source = root / 'source.csv'
        self.source.write_text('feature,label\n1,Normal\n')
        self.artifact = root / 'train.pt'
        self.artifact.write_bytes(b'compiled-tensor-stand-in')
        self.manifest = root / 'manifest.json'
        self.base = self.provenance()
        self.partition = partition_provenance(dict(custom_fog_topology='[2, 3]', dataset_fraction=1,
                                                   n_classes_per=3, power_law_a=.4))
        write_manifest(self.manifest, self.base, self.partition, [self.artifact], dict(train=70, val=15, test=15))

    def provenance(self, **changes):
        args = dict(dataset_name='edge', dataset_path=self.source, num_classes=15, random_seed=42,
                    simulate_global_leakage=False, apply_smote=True, test_split=.15, val_split=.15)
        args.update(changes)
        return base_provenance(**args)

    def test_whole_dataset_split_semantics(self):
        holdout, test_of_holdout = split_fractions(.15, .15)
        self.assertAlmostEqual(1 - holdout, .7)
        self.assertAlmostEqual(holdout * test_of_holdout, .15)
        holdout, test_of_holdout = split_fractions(.2, .1)
        self.assertAlmostEqual(holdout * test_of_holdout, .2)
        for test, val in [(0, .2), (.8, .2), (float('nan'), .1), (-.1, .2)]:
            with self.assertRaises(ValueError):
                split_fractions(test, val)

    def test_current_artifact_accepted(self):
        validate_artifact(self.artifact, self.manifest, self.base, self.partition)

    def test_seed_split_and_preprocessing_changes_rejected(self):
        for changes in [dict(random_seed=7), dict(val_split=.2), dict(apply_smote=False),
                        dict(simulate_global_leakage=True), dict(num_classes=10)]:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, 'offline'):
                validate_artifact(self.artifact, self.manifest, self.provenance(**changes))

    def test_raw_source_change_rejected(self):
        self.source.write_text('feature,label\n12,Normal\n')
        with self.assertRaisesRegex(ValueError, 'provenance changed'):
            validate_artifact(self.artifact, self.manifest, self.provenance())

    def test_missing_manifest_fails_with_rebuild_instruction(self):
        self.manifest.unlink()
        with self.assertRaisesRegex(ValueError, 'build_artifacts.py'):
            validate_artifact(self.artifact, self.manifest, self.base)

    def test_artifact_corruption_rejected(self):
        self.artifact.write_bytes(b'changed tensor contents')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            validate_artifact(self.artifact, self.manifest, self.base)

    def test_partition_change_rejected_but_base_remains_usable(self):
        validate_artifact(self.artifact, self.manifest, self.base)
        with self.assertRaisesRegex(ValueError, 'partition configuration changed'):
            validate_artifact(self.artifact, self.manifest, self.base, {**self.partition, 'topology': [5]})


if __name__ == '__main__':
    unittest.main()
