"""Small full-schema fixtures only: no network or large dataset reads."""
import csv
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.ops import download_datasets as downloader
from scripts.ops import preprocess_datasets as cleaner
from src.shared.data.dataset_schema import EDGE_FULL_FEATURES, EDGE_FULL_LABELS


class DatasetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def row(self, dataset, number=1, label=None):
        columns = cleaner.SOURCE_COLUMNS[dataset]
        row = [str(number)] * len(columns)
        if dataset == "edge":
            row[columns.index("ip.src_host")] = f"host-{number}"
            row[columns.index("Attack_type")] = label or "Normal"
            row[columns.index("Attack_label")] = "0" if label in (None, "Normal") else "1"
        elif dataset == "cic":
            row[columns.index("Label")] = label or "BENIGN"
        else:
            row[columns.index("srcip")] = f"host-{number}"
            row[columns.index("attack_cat")] = label or ""
            row[columns.index("Label")] = "0" if label in (None, "Normal") else "1"
        return row

    def write(self, dataset, groups=None):
        folder = cleaner.download_directory(self.root, dataset)
        folder.mkdir(parents=True, exist_ok=True)
        files = cleaner.SOURCE_FILES[dataset]
        groups = groups if groups is not None else [[self.row(dataset, i + 1)] for i in range(len(files))]
        for name, rows in zip(files, groups):
            with (folder / name).open("w", newline="", encoding="utf-8", errors="surrogateescape") as handle:
                writer = csv.writer(handle)
                if dataset != "unsw":
                    # CIC has header padding in its actual source files.
                    writer.writerow([" " + c if dataset == "cic" else c for c in cleaner.SOURCE_COLUMNS[dataset]])
                writer.writerows(rows)
        return folder

    def clean(self, dataset):
        return cleaner.preprocess_dataset(self.root, dataset, log=lambda _: None)

    def output_rows(self, report):
        with Path(report["output"]).open(newline="", encoding="utf-8") as handle:
            return list(csv.reader(handle))

    def test_full_record_deduplication_precedes_projection_and_counts_invalid_rows(self):
        columns = cleaner.SOURCE_COLUMNS["edge"]
        valid = self.row("edge")
        distinct_event = valid.copy()
        distinct_event[columns.index("ip.src_host")] = "another-host"
        invalid = []
        for index, value in enumerate(("", "text", "NaN", "Infinity", "-inf", "3.5e38")):
            row = self.row("edge", index + 10, "Backdoor")
            row[columns.index("tcp.len")] = value
            invalid.append(row)
        self.write("edge", [[valid] + invalid, [valid, distinct_event, invalid[0]]])
        report = self.clean("edge")
        self.assertEqual((report["input_rows"], report["duplicate_rows"], report["invalid_numeric_rows"], report["kept_rows"]), (10, 2, 6, 2))
        rows = self.output_rows(report)
        # Different original events can have identical numeric projections.
        self.assertEqual(rows[1], rows[2])
        self.assertEqual(rows[0], list(EDGE_FULL_FEATURES) + ["label"])
        self.assertEqual(report["sources"][0]["invalid_fields"], {"tcp.len": 6})
        self.assertEqual(report["sources"][0]["invalid_class_counts"], {"Backdoor": 6})
        self.assertEqual(report["class_counts"]["Normal"], 2)
        self.assertEqual(report["class_counts"]["Backdoor"], 0)

    def test_all_full_schema_classes_load_with_expected_dimensions(self):
        from src.shared.data.data_loader import (
            DATASET_METADATA, load_edge_iiotset, load_cic_ids2017, load_unsw_nb15,
        )
        loaders = {"edge": load_edge_iiotset, "cic": load_cic_ids2017, "unsw": load_unsw_nb15}
        for dataset, width in (("edge", 42), ("cic", 77), ("unsw", 42)):
            with self.subTest(dataset=dataset):
                groups = [[] for _ in cleaner.SOURCE_FILES[dataset]]
                for i, label in enumerate(cleaner.LABELS[dataset]):
                    groups[i % len(groups)].append(self.row(dataset, i + 1, label))
                self.write(dataset, groups)
                report = self.clean(dataset)
                X, y = loaders[dataset](report["output"])
                classes = len(cleaner.LABELS[dataset])
                self.assertEqual(X.shape, (classes, width))
                self.assertEqual(set(y.tolist()), set(range(classes)))
                self.assertEqual(DATASET_METADATA[dataset]["classes"], classes)
                self.assertEqual(sum(report["class_counts"].values()), classes)

    def test_edge_snippet_taxonomy_is_separate(self):
        from src.shared.data.data_loader import load_edge_iiotset
        snippet = self.root / "snippet.csv"
        snippet.write_text("feature,label\n1,Scanning\n2,MITM_ARP\n3,DoS_TCP\n")
        _, y = load_edge_iiotset(str(snippet))
        self.assertEqual(y.tolist(), [3, 14, 1])
        self.write("edge", [[self.row("edge", label="DDoS_HTTP")], [self.row("edge", 2, "DDoS_ICMP")]])
        report = self.clean("edge")
        _, y = load_edge_iiotset(report["output"])
        self.assertEqual(y.tolist(), [EDGE_FULL_LABELS["DDoS_HTTP"], EDGE_FULL_LABELS["DDoS_ICMP"]])
        with Path(report["output"]).open("a") as handle:
            handle.write(",".join(["3"] * 42 + ["Scanning"]) + "\n")
        with self.assertRaisesRegex(ValueError, "unknown label"):
            load_edge_iiotset(report["output"])

    def test_unsw_keeps_first_headerless_row_and_ignores_reference_splits(self):
        groups = [[self.row("unsw")], [self.row("unsw", 2, "Backdoors")],
                  [self.row("unsw", 3, " Fuzzers ")], [self.row("unsw", 4)]]
        folder = self.write("unsw", groups)
        for name in ("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"):
            (folder / name).write_text("incompatible,reference,schema\nshould,never,be-read\n")
        report = self.clean("unsw")
        rows = self.output_rows(report)
        self.assertEqual([row[-1] for row in rows[1:]], ["Normal", "Backdoor", "Fuzzers", "Normal"])
        self.assertEqual(len(rows), 5)
        self.assertEqual(len(report["reference_only_files"]), 2)

    def test_unsw_numeric_values_match_author_conversion_and_audit_counts(self):
        import numpy as np
        from src.shared.data.data_loader import load_unsw_nb15
        values = ("", "  ", "not-a-number", "0x50", "NaN", "inf", "-Infinity",
                  "3.5e38", "-3.5e38", "1e400", "0", "-0", "12.5", " -2.75 ",
                  "1e-50", "3.4028235e38", str(cleaner.FLOAT32_MAX))
        columns = cleaner.UNSW_SOURCE_COLUMNS
        indexes = [i for i, column in enumerate(columns) if column not in cleaner.UNSW_EXCLUDED_COLUMNS]
        groups, original_values = [[] for _ in range(4)], [[] for _ in range(4)]
        for i, value in enumerate(values):
            row = self.row("unsw", i + 1, "Normal" if i % 2 == 0 else "Generic")
            for index in indexes:
                row[index] = value
            groups[i % 4].append(row)
            original_values[i % 4].append(value)
        self.write("unsw", groups)
        report = self.clean("unsw")
        X, _ = load_unsw_nb15(report["output"])
        # Independent reference: failed float() -> zero, float32, then nan_to_num.
        expected = []
        for group in original_values:
            for value in group:
                try:
                    number = float(value)
                except (ValueError, TypeError):
                    number = 0.0
                expected.append([number] * len(indexes))
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            expected = np.nan_to_num(np.array(expected, dtype=np.float32), nan=0., posinf=0., neginf=0.)
        np.testing.assert_array_equal(X, expected)
        self.assertEqual(X.shape, (len(values), 42))
        self.assertEqual(report["input_rows"], report["kept_rows"])
        self.assertEqual(report["invalid_numeric_rows"], 0)
        self.assertEqual(report["zero_filled_rows"], 10)
        self.assertEqual(report["zero_filled_values"], 10 * 42)
        self.assertEqual(report["zero_filled_fields"], {columns[i]: 10 for i in indexes})
        self.assertEqual(report["zero_filled_reasons"], {
            "missing": 2 * 42, "invalid": 2 * 42, "nonfinite": 4 * 42, "float32_overflow": 2 * 42})
        self.assertEqual(report["zero_filled_class_counts"], {"Normal": 5, "Generic": 5})
        for source in report["sources"]:
            self.assertEqual(source["invalid_fields"], {})
            self.assertEqual(sum(source["zero_filled_fields"].values()), source["zero_filled_values"])
            self.assertEqual(sum(source["zero_filled_reasons"].values()), source["zero_filled_values"])
            self.assertEqual(sum(source["zero_filled_class_counts"].values()), source["zero_filled_rows"])
        self.assertEqual(sum(s["zero_filled_rows"] for s in report["sources"]), 10)
        saved_report = Path(report["output"]).with_name("preprocessing_report.json")
        self.assertEqual(json.loads(saved_report.read_text()), report)

    def test_unsw_deduplicates_originals_without_collapsing_zero_filled_rows(self):
        columns = cleaner.UNSW_SOURCE_COLUMNS
        missing, zero, invalid = (self.row("unsw") for _ in range(3))
        for column in ("ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd"):
            index = columns.index(column)
            missing[index], zero[index], invalid[index] = "", "0", "bad"
        self.write("unsw", [[missing, zero], [missing, invalid],
                            [self.row("unsw", 2)], [self.row("unsw", 3)]])
        report = self.clean("unsw")
        self.assertEqual((report["input_rows"], report["duplicate_rows"], report["invalid_numeric_rows"], report["kept_rows"]), (6, 1, 0, 5))
        self.assertEqual((report["zero_filled_rows"], report["zero_filled_values"]), (2, 6))
        rows = self.output_rows(report)
        self.assertEqual(rows[1], rows[2])
        self.assertEqual(rows[2], rows[3])
        self.assertEqual(report["class_counts"]["Normal"], 5)

    def test_unsw_zero_filling_does_not_relax_schema_or_label_validation(self):
        columns = cleaner.UNSW_SOURCE_COLUMNS
        for failure in ("malformed", "binary", "missing_category", "unknown", "contradiction"):
            with self.subTest(failure=failure):
                groups = [[self.row("unsw", i + 1)] for i in range(4)]
                row = groups[0][0]
                row[columns.index("ct_ftp_cmd")] = ""
                if failure == "malformed":
                    row.pop()
                elif failure == "binary":
                    row[columns.index("Label")] = ""
                elif failure == "missing_category":
                    row[columns.index("Label")] = "1"
                elif failure == "unknown":
                    row[columns.index("attack_cat")] = "Mystery"
                else:
                    row[columns.index("attack_cat")] = "Generic"
                folder = self.write("unsw", groups)
                output = folder.parent / "intrusion_records.csv"
                output.write_text("previous complete CSV")
                with self.assertRaises(ValueError):
                    self.clean("unsw")
                self.assertEqual(output.read_text(), "previous complete CSV")

    def test_cic_dash_aliases_and_duplicate_numeric_column(self):
        labels = ("Web Attack – XSS", "Web Attack - Brute Force", "Web Attack \udc96 Sql Injection",
                  "Web Attack \ufffd XSS", "BENIGN", "FTP-Patator", "SSH-Patator", "DoS GoldenEye")
        groups = [[self.row("cic", i + 1, label)] for i, label in enumerate(labels)]
        # Numerically equal spellings in duplicate measurement columns agree.
        duplicate = [i for i, col in enumerate(cleaner.CIC_SOURCE_COLUMNS) if col == "Fwd Header Length"]
        groups[0][0][duplicate[1]] = "1.0"
        self.write("cic", groups)
        report = self.clean("cic")
        self.assertEqual(report["kept_rows"], 8)
        self.assertEqual(report["collapsed_duplicate_columns"], ["Fwd Header Length"])
        self.assertEqual(report["class_counts"]["Web Attack – XSS"], 2)
        self.assertEqual(report["class_counts"]["Web Attack – Sql Injection"], 1)

    def test_failure_preserves_previous_inputs_and_does_not_publish(self):
        for failure in ("duplicate_column", "unknown_label", "malformed_row", "missing_file", "wrong_header"):
            with self.subTest(failure=failure), TemporaryDirectory() as directory:
                self.root = Path(directory)
                groups = [[self.row("cic", i + 1)] for i in range(8)]
                if failure == "duplicate_column":
                    groups[-1][0][cleaner.CIC_SOURCE_COLUMNS.index("Fwd Header Length")] = "999"
                if failure == "unknown_label":
                    groups[-1][0][-1] = "Mystery attack"
                if failure == "malformed_row":
                    groups[-1][0].pop()
                folder = self.write("cic", groups)
                if failure == "missing_file":
                    (folder / cleaner.CIC_FILES[-1]).unlink()
                if failure == "wrong_header":
                    (folder / cleaner.CIC_FILES[-1]).write_text("wrong,header\n1,2\n")
                raw = self.root / "cic_ids2017/raw/network_flows.csv"
                raw.parent.mkdir()
                raw.write_text("old snippet")
                output = folder.parent / raw.name
                output.write_text("previous complete output")
                report = folder.parent / "preprocessing_report.json"
                report.write_text("previous report")
                with self.assertRaises(ValueError):
                    self.clean("cic")
                self.assertEqual(raw.read_text(), "old snippet")
                self.assertEqual(output.read_text(), "previous complete output")
                self.assertEqual(report.read_text(), "previous report")
                self.assertFalse((raw.parent.parent / "archived").exists())
                self.assertEqual(list(folder.parent.glob(".cleaning-*")), [])

    def test_success_replaces_cleaned_outputs_and_preserves_snippet_without_archives(self):
        folder = self.write("edge")
        raw = self.root / "edge_iiotset/raw/network_traffic_samples.csv"
        raw.parent.mkdir()
        raw.write_text("old snippet")
        output = folder.parent / raw.name
        output.write_text("old cleaned CSV")
        report_file = folder.parent / "preprocessing_report.json"
        report_file.write_text("old report")
        report = self.clean("edge")
        self.assertEqual(raw.read_text(), "old snippet")
        self.assertFalse((raw.parent.parent / "archived").exists())
        self.assertEqual(list(folder.parent.glob(".cleaning-*")), [])
        self.assertNotEqual(output.read_text(), "old cleaned CSV")
        self.assertEqual(json.loads(report_file.read_text()), report)
        self.assertTrue(all((folder / name).exists() for name in cleaner.EDGE_FILES))
        self.assertFalse((raw.parent.parent / "artifacts").exists())

    def test_publication_error_restores_previous_outputs(self):
        folder = self.write("edge")
        output = folder.parent / "network_traffic_samples.csv"
        output.write_text("previous complete CSV")
        previous_report = folder.parent / "preprocessing_report.json"
        previous_report.write_text("previous report")
        real_replace = Path.replace

        def replace(source, destination):
            if source.parent.name.startswith(".cleaning-") and source.name == "preprocessing_report.json":
                raise OSError("simulated publication failure")
            return real_replace(source, destination)

        with patch.object(Path, "replace", replace), self.assertRaisesRegex(OSError, "publication failure"):
            self.clean("edge")
        self.assertEqual(output.read_text(), "previous complete CSV")
        self.assertEqual(previous_report.read_text(), "previous report")

    def test_no_usable_rows_fails_without_publishing(self):
        groups = [[self.row("edge", i + 1)] for i in range(2)]
        for group in groups:
            group[0][cleaner.EDGE_SOURCE_COLUMNS.index("tcp.len")] = "NaN"
        folder = self.write("edge", groups)
        with self.assertRaisesRegex(ValueError, "no usable rows; invalid numeric rows=2"):
            self.clean("edge")
        self.assertFalse((folder.parent / "network_traffic_samples.csv").exists())

    def test_cli_preprocess_only_needs_no_network_and_returns_failure_for_missing_sources(self):
        self.write("unsw")
        with patch.object(downloader, "ensure_kagglehub", side_effect=AssertionError("network dependency")), \
             patch.object(downloader, "ensure_requests", side_effect=AssertionError("network dependency")), \
             patch.object(downloader, "download_unsw", side_effect=AssertionError("network call")), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(downloader.main(["--out", str(self.root), "--only", "unsw", "--preprocess-only"]), 0)
            self.assertEqual(downloader.main(["--out", str(self.root), "--only", "edge", "--preprocess-only"]), 1)

    def test_cli_reports_zero_filled_values_separately_from_excluded_rows(self):
        groups = [[self.row("unsw", i + 1)] for i in range(4)]
        groups[0][0][cleaner.UNSW_SOURCE_COLUMNS.index("ct_ftp_cmd")] = ""
        self.write("unsw", groups)
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(downloader.main(["--out", str(self.root), "--only", "unsw", "--preprocess-only"]), 0)
        self.assertIn("4 retained; 0 identical duplicates removed; 0 invalid numeric rows excluded; 1 values zero-filled across 1 retained rows", output.getvalue())

    def test_smoke_never_cleans(self):
        with patch.object(downloader, "ensure_kagglehub"), patch.object(downloader, "ensure_requests"), \
             patch.object(downloader, "download_edge") as fetch, \
             patch.object(downloader, "preprocess_dataset", side_effect=AssertionError("must not clean")), \
             redirect_stdout(io.StringIO()):
            self.assertEqual(downloader.main(["--out", str(self.root), "--only", "edge", "--smoke"]), 0)
            fetch.assert_called_once()

    def test_http_resume_counts_total_file_size_not_remaining_body_size(self):
        part = self.root / "source.csv.part"
        part.write_bytes(b"abcde")
        response = FakeResponse(206, {"content-length": "3", "content-range": "bytes 5-7/8"}, [b"fgh"])
        get = Mock(return_value=response)
        with patch.dict(sys.modules, {"requests": SimpleNamespace(get=get)}), redirect_stdout(io.StringIO()):
            result = downloader.hf_download("source.csv", self.root)
        self.assertEqual(result.read_bytes(), b"abcdefgh")
        self.assertEqual(get.call_args.kwargs["headers"]["Range"], "bytes=5-")
        self.assertFalse(part.exists())

    def test_http_retry_resumes_after_bytes_written_by_failed_attempt(self):
        first = FakeResponse(200, {"content-length": "8"}, [b"abc", OSError("interrupted")])
        second = FakeResponse(206, {"content-length": "5", "content-range": "bytes 3-7/8"}, [b"defgh"])
        get = Mock(side_effect=[first, second])
        with patch.dict(sys.modules, {"requests": SimpleNamespace(get=get)}), \
             patch.object(downloader.time, "sleep"), redirect_stdout(io.StringIO()):
            result = downloader.hf_download("source.csv", self.root)
        self.assertEqual(result.read_bytes(), b"abcdefgh")
        self.assertNotIn("Range", get.call_args_list[0].kwargs["headers"])
        self.assertEqual(get.call_args_list[1].kwargs["headers"]["Range"], "bytes=3-")

    def test_bad_http_resume_range_preserves_previous_completed_source(self):
        output, part = self.root / "source.csv", self.root / "source.csv.part"
        output.write_bytes(b"previous complete source")
        part.write_bytes(b"abc")
        response = FakeResponse(206, {"content-length": "5", "content-range": "bytes 0-4/8"}, [b"wrong"])
        with patch.dict(sys.modules, {"requests": SimpleNamespace(get=Mock(return_value=response))}), \
             patch.object(downloader.time, "sleep"), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "Failed to download"):
                downloader.hf_download("source.csv", self.root)
        self.assertEqual(output.read_bytes(), b"previous complete source")
        self.assertEqual(part.read_bytes(), b"abc")

    def test_interrupted_kaggle_move_does_not_publish_partial_source(self):
        cached = self.root / "cached.csv"
        cached.write_bytes(b"complete downloaded bytes")
        destination = self.root / "downloads"
        destination.mkdir()
        output = destination / "source.csv"
        output.write_bytes(b"previous complete source")

        def interrupted_move(source, target):
            Path(target).write_bytes(b"partial copy")
            raise OSError("cross-filesystem copy interrupted")

        with patch.dict(sys.modules, {"kagglehub": SimpleNamespace(dataset_download=lambda *a, **k: str(cached))}), \
             patch.object(downloader.shutil, "move", side_effect=interrupted_move), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(OSError, "interrupted"):
                downloader.kaggle_fetch_file("owner/dataset", "source.csv", destination)
        self.assertEqual(output.read_bytes(), b"previous complete source")
        self.assertEqual((destination / "source.csv.kaggle.part").read_bytes(), b"partial copy")

    def test_cleaned_edge_input_builds_existing_tensor_format(self):
        import torch
        from scripts.setup import build_artifacts as builder
        groups = [[], []]
        for i in range(120):
            groups[i % 2].append(self.row("edge", i + 1, "Normal" if i % 2 else "DDoS_HTTP"))
        self.write("edge", groups)
        report = self.clean("edge")
        config = dict(dataset="edge_iiotset", dataset_path=report["output"], random_seed=42,
                      simulate_global_leakage=False, apply_smote=False, test_split=.15, val_split=.15,
                      num_fogs=1, uniform_edges_per_fog=2, custom_fog_topology="[]", dataset_fraction=1.,
                      n_classes_per=2, power_law_a=.4)
        with patch.object(builder, "load_yaml_configs", return_value=config), redirect_stdout(io.StringIO()):
            builder.build_artifacts()
        artifacts = Path(report["output"]).parent.parent / "artifacts"
        for split, size in (("train", 84), ("val", 18), ("test", 18)):
            X, y, classes = torch.load(artifacts / f"edge_iiotset_leakage_False_smote_False_{split}.pt", weights_only=True)
            self.assertEqual(X.shape, (size, 40))
            self.assertEqual(set(y.tolist()), {0, 3})
            self.assertEqual(classes, 15)
            self.assertTrue(torch.isfinite(X).all())

    def test_zero_filled_unsw_input_builds_existing_tensor_format(self):
        import torch
        from scripts.setup import build_artifacts as builder
        groups = [[] for _ in range(4)]
        columns = cleaner.UNSW_SOURCE_COLUMNS
        for i in range(120):
            row = self.row("unsw", i + 1, "Normal" if i % 2 else "Generic")
            for column in ("ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd"):
                row[columns.index(column)] = ""
            row[columns.index("dsport")] = "0x50" if i % 3 else "80"
            groups[i % 4].append(row)
        self.write("unsw", groups)
        report = self.clean("unsw")
        self.assertEqual(report["kept_rows"], 120)
        self.assertEqual(report["zero_filled_rows"], 120)
        config = dict(dataset="unsw_nb15", dataset_path=report["output"], random_seed=42,
                      simulate_global_leakage=False, apply_smote=False, test_split=.15, val_split=.15,
                      num_fogs=1, uniform_edges_per_fog=2, custom_fog_topology="[]", dataset_fraction=1.,
                      n_classes_per=2, power_law_a=.4)
        with patch.object(builder, "load_yaml_configs", return_value=config), redirect_stdout(io.StringIO()):
            builder.build_artifacts()
        artifacts = Path(report["output"]).parent.parent / "artifacts"
        for split, size in (("train", 84), ("val", 18), ("test", 18)):
            X, y, classes = torch.load(artifacts / f"unsw_nb15_leakage_False_smote_False_{split}.pt", weights_only=True)
            self.assertEqual(X.shape, (size, 40))
            self.assertEqual(set(y.tolist()), {0, 1})
            self.assertEqual(classes, 10)
            self.assertTrue(torch.isfinite(X).all())

    def test_builder_replaces_snippet_artifacts_for_full_data_and_regenerates_missing_files(self):
        import torch
        from scripts.setup import build_artifacts as builder
        from src.shared.data.data_loader import get_dataset

        snippet = self.root / "edge_iiotset/raw/network_traffic_samples.csv"
        snippet.parent.mkdir(parents=True)
        with snippet.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([f"feature_{i}" for i in range(40)] + ["label"])
            for i in range(80):
                writer.writerow([str(i + 1)] * 40 + ["Normal" if i % 2 else "DoS_TCP"])
        config = dict(dataset="edge_iiotset", dataset_path=str(snippet), random_seed=42,
                      simulate_global_leakage=False, apply_smote=False, test_split=.15, val_split=.15,
                      num_fogs=1, uniform_edges_per_fog=2, custom_fog_topology="[]", dataset_fraction=1.,
                      n_classes_per=2, power_law_a=.4)

        def build():
            with patch.object(builder, "load_yaml_configs", return_value=config), redirect_stdout(io.StringIO()):
                builder.build_artifacts()

        def consume():
            return get_dataset("edge_iiotset", config["dataset_path"], 15, 42, apply_smote=False)

        build()
        self.assertEqual(consume()[0].shape, (56, 40))
        artifacts = snippet.parent.parent / "artifacts"
        manifest_path = artifacts / "edge_iiotset_leakage_False_smote_False_manifest.json"
        snippet_manifest = json.loads(manifest_path.read_text())
        groups = [[], []]
        for i in range(120):
            groups[i % 2].append(self.row("edge", i + 1, "Normal" if i % 2 else "DDoS_HTTP"))
        self.write("edge", groups)
        report = self.clean("edge")
        # Download/cleaning does not delete artifacts or select the new input.
        self.assertEqual(json.loads(manifest_path.read_text()), snippet_manifest)
        self.assertTrue(snippet.exists())
        config["dataset_path"] = report["output"]
        with self.assertRaisesRegex(ValueError, "provenance changed"):
            consume()
        build()
        full_manifest = json.loads(manifest_path.read_text())
        self.assertNotEqual(full_manifest["base"]["source_sha256"], snippet_manifest["base"]["source_sha256"])
        self.assertEqual(full_manifest["rows_before_smote"], {"train": 84, "val": 18, "test": 18})
        expected_X, expected_y, _ = consume()
        self.assertEqual(set(expected_y.tolist()), {0, 3})
        for path in artifacts.glob("*_edge_*.pt"):
            _, edge_y, _ = torch.load(path, weights_only=True)
            self.assertEqual(set(edge_y.tolist()), {0, 3})
        # An unchanged input reuses base tensors, while partitions are refreshed.
        with patch.object(builder, "load_edge_iiotset", side_effect=AssertionError("base should be cached")):
            build()
        # All deletions here affect only this temporary test's generated files.
        for path in artifacts.iterdir():
            path.unlink()
        with self.assertRaisesRegex(ValueError, "Missing or stale artifact"):
            consume()
        build()
        actual_X, actual_y, _ = consume()
        torch.testing.assert_close(actual_X, expected_X)
        torch.testing.assert_close(actual_y, expected_y)
        self.assertEqual(len(list(artifacts.glob("*_edge_*.pt"))), 2)


class FakeResponse:
    def __init__(self, status, headers, chunks):
        self.status_code, self.headers, self.chunks = status, headers, chunks

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


if __name__ == "__main__":
    unittest.main()
