"""Validation and encoding at the model transport boundary."""
import copy
import math
from numbers import Integral

import numpy as np
import torch
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from src.shared.network.compression import compress_weights, decompress_weights


def validate_sample_count(size: int) -> None:
    if isinstance(size, bool) or not isinstance(size, Integral) or size <= 0:
        raise ValueError("num_examples must be a positive integer")


def validate_model(model: torch.nn.Module) -> None:
    for name, value in model.state_dict().items():
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite model tensor: {name}")
        if name.endswith("running_var") and (value < 0).any():
            raise ValueError(f"Negative BatchNorm variance: {name}")


def load_model_arrays(model: torch.nn.Module, arrays: list, bits: int) -> None:
    """Validate the entire payload before changing any model state."""
    if bits not in (8, 16, 32):
        raise ValueError(f"Unsupported quantization width: {bits}")
    expected = model.state_dict()
    count = len(expected) * (2 if bits == 8 else 1)
    if len(arrays) != count:
        raise ValueError(f"Expected {count} payload arrays, received {len(arrays)}")
    for value in arrays:
        if not isinstance(value, np.ndarray) or value.dtype.kind not in "fiu":
            raise ValueError("Model payloads must contain real numeric arrays")
        if not np.isfinite(value).all():
            raise ValueError("Nonfinite payload array")
    if bits == 8:
        for encoded, meta in zip(arrays[::2], arrays[1::2]):
            if encoded.dtype != np.uint8 or meta.shape != (2,) or meta[1] <= 0:
                raise ValueError("Invalid INT8 payload or quantization metadata")
    decoded = decompress_weights(arrays, bits)
    state = {}
    for (name, template), value in zip(expected.items(), decoded):
        if value.shape != tuple(template.shape) or not np.isfinite(value).all():
            raise ValueError(f"Invalid shape or values for {name}")
        if not template.is_floating_point():
            bounds = torch.iinfo(template.dtype)
            if (value < bounds.min).any() or (value >= bounds.max).any() or (value < 0).any():
                raise ValueError(f"Invalid integer buffer: {name}")
        tensor = torch.as_tensor(value.copy()).to(dtype=template.dtype, device=template.device)
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Payload overflows model dtype: {name}")
        if name.endswith("running_var") and (tensor < 0).any():
            raise ValueError(f"Negative BatchNorm variance: {name}")
        state[name] = tensor
    model.load_state_dict(state, strict=True)


def decode_model(reference: torch.nn.Module, parameters, bits: int) -> torch.nn.Module:
    model = copy.deepcopy(reference).cpu()
    try:
        arrays = parameters_to_ndarrays(parameters)
    except Exception as exc:
        raise ValueError("Malformed serialized model payload") from exc
    load_model_arrays(model, arrays, bits)
    return model


def encode_model(model: torch.nn.Module, bits: int):
    validate_model(model)
    arrays = [value.detach().cpu().numpy().copy() for value in model.state_dict().values()]
    encoded = compress_weights(arrays, bits)
    # Includes overflow introduced by FP16 conversion or quantization metadata.
    probe = copy.deepcopy(model).cpu()
    load_model_arrays(probe, encoded, bits)
    return ndarrays_to_parameters(encoded)


def positive_trust(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("Invalid regional trust")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Regional trust must be finite and nonnegative")
    return value


def validate_region_result(num_examples, has_parameters: bool, metrics: dict) -> str:
    """Distinguish a valid empty regional result from a failed or malformed one."""
    status = metrics.get("aggregation_status")
    trust = positive_trust(metrics.get("total_regional_trust"))
    if status == "aggregated":
        validate_sample_count(num_examples)
        if not has_parameters or trust <= 0:
            raise ValueError("Aggregated region requires a model and positive weight")
    elif status in ("all_rejected", "no_results", "no_positive_trust", "error"):
        if (isinstance(num_examples, bool) or not isinstance(num_examples, Integral)
                or num_examples != 0 or has_parameters or trust != 0):
            raise ValueError("Empty regional status requires zero samples, zero weight and no model")
    else:
        raise ValueError("Missing or unknown regional aggregation status")
    return status
