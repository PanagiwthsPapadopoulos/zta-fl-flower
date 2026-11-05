"""Stream downloaded CSVs into numeric inputs for the existing artifact builder.

No scaling, encoding, resampling or train/validation/test splitting happens here.
UNSW numeric failures become zero, following the original authors' public loader; Edge
and CIC reject affected rows. Exact deduplication compares every original CSV
field before feature selection or numeric replacement.
"""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import sqlite3
import struct
from tempfile import TemporaryDirectory

from src.shared.data.dataset_schema import (
    CIC_IDS2017_LABELS, CIC_SOURCE_COLUMNS, EDGE_EXCLUDED_COLUMNS,
    EDGE_FULL_LABELS, EDGE_SOURCE_COLUMNS, UNSW_EXCLUDED_COLUMNS,
    UNSW_NB15_LABELS, UNSW_SOURCE_COLUMNS,
)

DATASETS = {
    "edge": ("edge_iiotset", "network_traffic_samples.csv", "label"),
    "cic": ("cic_ids2017", "network_flows.csv", "Label"),
    "unsw": ("unsw_nb15", "intrusion_records.csv", "attack_cat"),
}
EDGE_FILES = ("DNN-EdgeIIoT-dataset.csv", "ML-EdgeIIoT-dataset.csv")
CIC_FILES = (
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
)
UNSW_FULL_FILES = tuple(f"UNSW-NB15_{i}.csv" for i in range(1, 5))
SOURCE_FILES = {"edge": EDGE_FILES, "cic": CIC_FILES, "unsw": UNSW_FULL_FILES}
SOURCE_COLUMNS = {"edge": EDGE_SOURCE_COLUMNS, "cic": CIC_SOURCE_COLUMNS, "unsw": UNSW_SOURCE_COLUMNS}
LABELS = {"edge": EDGE_FULL_LABELS, "cic": CIC_IDS2017_LABELS, "unsw": UNSW_NB15_LABELS}
EXCLUDED = {"edge": EDGE_EXCLUDED_COLUMNS, "cic": {"Label"}, "unsw": UNSW_EXCLUDED_COLUMNS}
FLOAT32_MAX = float.fromhex("0x1.fffffep+127")
UNSW_NUMERIC_POLICY_SOURCE = "https://github.com/ssam18/zta-federated-learning/blob/main/src/utils/data_loader.py"
CIC_LABEL_ALIASES = {
    f"Web Attack {dash} {attack}": f"Web Attack – {attack}"
    for attack in ("Brute Force", "XSS", "Sql Injection")
    for dash in ("-", "\udc96", "\ufffd")
}


def download_directory(out_root: Path, dataset: str) -> Path:
    return Path(out_root) / DATASETS[dataset][0] / "full_dataset" / "downloads"


def _label(dataset, row, columns):
    if dataset == "edge":
        label = row[columns.index("Attack_type")].strip()
    elif dataset == "cic":
        label = row[columns.index("Label")].strip()
        # Public copies use a Unicode en dash, a Windows-1252 dash byte, an
        # ASCII dash or the replacement character in these three labels.
        label = CIC_LABEL_ALIASES.get(label, label)
    else:
        label = row[columns.index("attack_cat")].strip()
        binary = row[columns.index("Label")].strip()
        if binary not in {"0", "1"}:
            raise ValueError(f"invalid UNSW binary label {binary!r}")
        if not label and binary == "0":
            label = "Normal"
        if label == "Backdoors":
            label = "Backdoor"
        if label in UNSW_NB15_LABELS and ((label == "Normal") != (binary == "0")):
            raise ValueError("UNSW attack_cat contradicts its binary label")
    if label not in LABELS[dataset]:
        raise ValueError(f"unknown {dataset} label {label!r}; no benign fallback is permitted")
    return label


def _numeric_error(value):
    if not value.strip():
        return "missing"
    try:
        number = float(value)
    except ValueError:
        return "invalid"
    if not math.isfinite(number):
        return "nonfinite"
    if abs(number) > FLOAT32_MAX:
        return "float32_overflow"
    return None


def _unsw_measurement(value):
    """Return CSV value and replacement reason, matching the author's float32 input.

    _safe_float followed by float32 conversion and nan_to_num maps parse
    failures and nonfinite results to zero. Valid values retain their spelling
    unless float32 rounding is needed to pass the strict downstream loader.
    """
    if not value.strip():
        return "0", "missing"
    try:
        number = float(value)
    except (ValueError, TypeError):
        return "0", "invalid"
    if not math.isfinite(number):
        return "0", "nonfinite"
    if abs(number) > FLOAT32_MAX:
        # A value just above float32 max can round back to that finite max.
        # Larger values overflow; struct may return inf or raise by platform.
        try:
            rounded = struct.unpack("f", struct.pack("f", number))[0]
        except OverflowError:
            return "0", "float32_overflow"
        if not math.isfinite(rounded):
            return "0", "float32_overflow"
        return str(rounded), None
    return value.strip(), None


def _equal_measurements(left, right):
    if left.strip() == right.strip():
        return True
    try:
        return Decimal(left.strip()) == Decimal(right.strip())
    except InvalidOperation:
        return False


def _publish(prepared_csv, prepared_report, output, report_path):
    """Replace cleaned outputs; keep rollback copies only until publication succeeds."""
    moved, published = [], []
    try:
        for path in (output, report_path):
            if path.exists():
                destination = prepared_csv.parent / ("previous-" + path.name)
                path.rename(destination)
                moved.append((path, destination))
        for source, destination in ((prepared_csv, output), (prepared_report, report_path)):
            source.replace(destination)
            published.append((source, destination))
    except BaseException:
        for source, destination in reversed(published):
            destination.replace(source)
        for original, backup in reversed(moved):
            backup.replace(original)
        raise


def preprocess_dataset(out_root: Path, dataset: str, log=print):
    """Clean all required sources; return an auditable report after atomic CSV publication."""
    name, basename, label_column = DATASETS[dataset]
    dataset_root = Path(out_root) / name
    destination = dataset_root / "full_dataset"
    downloads = download_directory(out_root, dataset)
    sources = [downloads / filename for filename in SOURCE_FILES[dataset]]
    missing = [str(path) for path in sources if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError("Missing required source files: " + ", ".join(missing))

    columns = SOURCE_COLUMNS[dataset]
    features, feature_indexes = [], []
    for index, column in enumerate(columns):
        if column not in EXCLUDED[dataset] and column not in features:
            features.append(column)
            feature_indexes.append(index)
    repeated = [i for i, column in enumerate(columns) if column in columns[:i]]
    output, report_path = destination / basename, destination / "preprocessing_report.json"
    try:
        # Repository-relative paths also work under the container's /app mount.
        dataset_path = str(output.resolve().relative_to(Path(__file__).resolve().parents[2]))
    except ValueError:
        dataset_path = str(output.resolve())
    report = {
        "dataset": name, "output": str(output), "dataset_path": dataset_path, "feature_columns": features,
        "label_column": label_column, "label_mapping": LABELS[dataset],
        "excluded_source_columns": [c for c in columns if c in EXCLUDED[dataset]],
        "collapsed_duplicate_columns": [columns[i] for i in repeated],
        "deduplication": "exact original CSV field values, before feature selection or label normalization",
        "invalid_numeric_policy": "exclude rows with missing, invalid, nonfinite or float32-overflowing retained measurements",
        "counting_order": "identical duplicates first; numeric exclusions and replacements counted among remaining unique records",
        "invalid_detail_counts": "invalid_fields and invalid_reasons count field occurrences; one excluded row can have several invalid fields",
        "input_rows": 0, "duplicate_rows": 0, "invalid_numeric_rows": 0, "kept_rows": 0,
        "zero_filled_rows": 0, "zero_filled_values": 0,
        "zero_filled_fields": {}, "zero_filled_reasons": {}, "zero_filled_class_counts": {},
        "class_counts": {label: 0 for label in LABELS[dataset]}, "sources": [],
    }
    if dataset == "unsw":
        report["reference_only_files"] = ["UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv"]
        report["invalid_numeric_policy"] = "replace missing, invalid, nonfinite and float32-overflowing retained measurements with zero; retain the row"
        report["numeric_policy_source"] = UNSW_NUMERIC_POLICY_SOURCE
        report["numeric_policy_source_reviewed_on"] = "2026-10-25"
        report["zero_fill_detail_counts"] = "zero_filled_rows counts retained unique records with replacements; zero_filled_values, fields and reasons count replaced cells; class counts count affected rows"
    # csv's default 128-KiB field limit is too small for some payload strings.
    # Rows are still streamed; excluded payloads are never collected in memory.
    previous_field_limit = csv.field_size_limit()
    csv.field_size_limit(2**31 - 1)
    try:
        with TemporaryDirectory(prefix=".cleaning-", dir=destination) as temporary:
            temporary = Path(temporary)
            prepared_csv, prepared_report = temporary / basename, temporary / report_path.name
            connection = sqlite3.connect(temporary / "seen.sqlite")
            try:
                # This disposable database bounds RAM while comparing complete
                # records, without relying on a hash that could collide.
                connection.execute("PRAGMA cache_size = -8192")
                connection.execute("PRAGMA journal_mode = OFF")
                connection.execute("CREATE TABLE seen (record TEXT PRIMARY KEY) WITHOUT ROWID")
                with prepared_csv.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(features + [label_column])
                    for source in sources:
                        counts = {"file": source.name, "input_rows": 0, "duplicate_rows": 0,
                                  "invalid_numeric_rows": 0, "kept_rows": 0,
                                  "zero_filled_rows": 0, "zero_filled_values": 0,
                                  "invalid_fields": {}, "invalid_reasons": {},
                                  "invalid_class_counts": {}, "class_counts": {}}
                        invalid_fields, invalid_reasons = Counter(), Counter()
                        invalid_classes, kept_classes = Counter(), Counter()
                        zero_fields, zero_reasons, zero_classes = Counter(), Counter(), Counter()
                        log(f"  cleaning {source.name}")
                        # Preserve undecodable bytes in excluded text fields and
                        # dedup keys. Only explicit CIC dash variants are repaired.
                        with source.open(newline="", encoding="utf-8-sig", errors="surrogateescape") as source_handle:
                            reader = csv.reader(source_handle, strict=True)
                            if dataset != "unsw":
                                header = tuple(value.strip() for value in next(reader))
                                if header != columns:
                                    raise ValueError(f"{source}: unexpected source header; expected the original {len(columns)}-column schema")
                            for row in reader:
                                counts["input_rows"] += 1
                                location = f"{source}:{reader.line_num}"
                                if len(row) != len(columns):
                                    raise ValueError(f"{location}: expected {len(columns)} fields, received {len(row)}")
                                for index in repeated:
                                    original = columns.index(columns[index])
                                    if not _equal_measurements(row[index], row[original]):
                                        raise ValueError(f"{location}: duplicated {columns[index]!r} columns disagree")
                                try:
                                    label = _label(dataset, row, columns)
                                except ValueError as exc:
                                    raise ValueError(f"{location}: {exc}") from exc
                                key = json.dumps(row, ensure_ascii=True, separators=(",", ":"))
                                inserted = connection.execute("INSERT OR IGNORE INTO seen VALUES (?)", (key,)).rowcount
                                if not inserted:
                                    counts["duplicate_rows"] += 1
                                else:
                                    if dataset == "unsw":
                                        converted = [_unsw_measurement(row[i]) for i in feature_indexes]
                                        values = [value for value, _ in converted]
                                        errors = [(columns[i], reason) for i, (_, reason) in zip(feature_indexes, converted)]
                                    else:
                                        values = [row[i].strip() for i in feature_indexes]
                                        errors = [(columns[i], _numeric_error(row[i])) for i in feature_indexes]
                                    errors = [(column, reason) for column, reason in errors if reason]
                                    if errors and dataset != "unsw":
                                        counts["invalid_numeric_rows"] += 1
                                        invalid_classes[label] += 1
                                        for column, reason in errors:
                                            invalid_fields[column] += 1
                                            invalid_reasons[reason] += 1
                                    else:
                                        if errors:
                                            counts["zero_filled_rows"] += 1
                                            counts["zero_filled_values"] += len(errors)
                                            zero_classes[label] += 1
                                            for column, reason in errors:
                                                zero_fields[column] += 1
                                                zero_reasons[reason] += 1
                                        writer.writerow(values + [label])
                                        counts["kept_rows"] += 1
                                        kept_classes[label] += 1
                                if counts["input_rows"] % 100_000 == 0:
                                    connection.commit()
                                    log(f"    {counts['input_rows']:,} rows read; {counts['kept_rows']:,} retained")
                        connection.commit()
                        if not counts["input_rows"]:
                            raise ValueError(f"{source}: no observations")
                        counts.update(invalid_fields=dict(invalid_fields), invalid_reasons=dict(invalid_reasons),
                                      invalid_class_counts=dict(invalid_classes), class_counts=dict(kept_classes),
                                      zero_filled_fields=dict(zero_fields), zero_filled_reasons=dict(zero_reasons),
                                      zero_filled_class_counts=dict(zero_classes))
                        report["sources"].append(counts)
                        for key in ("input_rows", "duplicate_rows", "invalid_numeric_rows", "kept_rows",
                                    "zero_filled_rows", "zero_filled_values"):
                            report[key] += counts[key]
                        for key in ("zero_filled_fields", "zero_filled_reasons", "zero_filled_class_counts"):
                            for field, count in counts[key].items():
                                report[key][field] = report[key].get(field, 0) + count
                        for label, count in kept_classes.items():
                            report["class_counts"][label] += count
                        log(f"    kept {counts['kept_rows']:,}; identical duplicates {counts['duplicate_rows']:,}; invalid numeric rows {counts['invalid_numeric_rows']:,}")
                        if dataset == "unsw":
                            log(f"    zero-filled {counts['zero_filled_values']:,} values across {counts['zero_filled_rows']:,} retained rows")
            finally:
                connection.close()
            if not report["kept_rows"]:
                raise ValueError(f"{name}: no usable rows; invalid numeric rows={report['invalid_numeric_rows']:,}, duplicates={report['duplicate_rows']:,}")
            report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            prepared_report.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
            _publish(prepared_csv, prepared_report, output, report_path)
    finally:
        csv.field_size_limit(previous_field_limit)
    log(f"  cleaned -> {output} ({report['kept_rows']:,} rows, {len(features)} numeric features)")
    log(f"  counts and exclusions -> {report_path}")
    log(f"  Set dataset_path to {dataset_path}; the existing artifact builder performs train/validation/test splitting and model preprocessing.")
    return report
