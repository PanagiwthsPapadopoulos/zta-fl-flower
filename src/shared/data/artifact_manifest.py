"""Offline artifact provenance. Nodes validate; only the builder creates tensors."""
import ast
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

SCHEMA_VERSION = 1
PIPELINE_VERSION = "whole-dataset-splits-minmax-pca40-validated-smote-v3"
REBUILD = "Run scripts/setup/build_artifacts.py offline with the current configuration before starting nodes."


@lru_cache(maxsize=512)
def _digest(path, size, mtime_ns):
    result = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def file_digest(path):
    path = Path(path).resolve()
    stat = path.stat()
    return _digest(str(path), stat.st_size, stat.st_mtime_ns)


def split_fractions(test_split, val_split):
    test_split, val_split = float(test_split), float(val_split)
    if (not math.isfinite(test_split + val_split) or test_split <= 0 or val_split <= 0
            or test_split + val_split >= 1):
        raise ValueError("test_split and val_split must be positive fractions of the full dataset with sum < 1")
    return test_split + val_split, test_split / (test_split + val_split)


def base_provenance(dataset_name, dataset_path, num_classes, random_seed,
                    simulate_global_leakage, apply_smote, test_split, val_split):
    split_fractions(test_split, val_split)
    return {"schema": SCHEMA_VERSION, "pipeline": PIPELINE_VERSION,
            "loader_sha256": file_digest(Path(__file__).with_name("data_loader.py")),
            "label_schema_sha256": file_digest(Path(__file__).with_name("dataset_schema.py")),
            "source_sha256": file_digest(dataset_path), "dataset": dataset_name,
            "num_classes": int(num_classes), "n_features": 40, "random_seed": int(random_seed),
            "simulate_global_leakage": bool(simulate_global_leakage), "apply_smote": bool(apply_smote),
            "test_split": float(test_split), "val_split": float(val_split)}


def partition_provenance(config):
    custom = config.get("custom_fog_topology", "")
    custom = ast.literal_eval(custom) if isinstance(custom, str) and custom.strip() else custom
    topology = list(custom) if custom else [int(config["uniform_edges_per_fog"])] * int(config["num_fogs"])
    if not topology or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in topology):
        raise ValueError("Topology must contain positive edge counts")
    fraction, power = float(config["dataset_fraction"]), float(config["power_law_a"])
    classes = int(config["n_classes_per"])
    if not 0 < fraction <= 1 or not math.isfinite(power) or power <= 0 or classes <= 0:
        raise ValueError("Invalid partition configuration")
    return {"topology": topology, "dataset_fraction": fraction, "n_classes_per": classes,
            "power_law_a": power}


def validate_artifact(artifact_path, manifest_path, expected_base, expected_partition=None):
    try:
        manifest = json.loads(Path(manifest_path).read_text())
        if manifest["base"] != expected_base:
            raise ValueError("Dataset/preprocessing provenance changed")
        if expected_partition is not None and manifest["partition"] != expected_partition:
            raise ValueError("Edge partition configuration changed")
        if manifest["artifacts"][Path(artifact_path).name]["sha256"] != file_digest(artifact_path):
            raise ValueError("Artifact checksum mismatch")
        return manifest
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"Missing or stale artifact {artifact_path}: {exc}. {REBUILD}") from exc


def write_manifest(path, base, partition, artifact_paths, counts, *, class_counts=None, smote_k_neighbors=None):
    payload = {"base": base, "partition": partition, "rows_before_smote": counts,
               "class_counts": class_counts or {}, "smote_k_neighbors": smote_k_neighbors,
               "artifacts": {Path(p).name: {"sha256": file_digest(p)} for p in artifact_paths}}
    temporary = Path(path).with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def validate_checkpoint_pipeline(metadata):
    """Never evaluate legacy feature-space checkpoints on freshly rebuilt data."""
    if metadata.get("pipeline_version") != PIPELINE_VERSION:
        raise ValueError("Checkpoint preprocessing is missing or incompatible. Rebuild artifacts and restart the experiment; legacy checkpoints cannot be reused.")
