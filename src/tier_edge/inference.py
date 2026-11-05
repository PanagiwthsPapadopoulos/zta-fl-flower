"""CPU inference evaluation, separate from floating-point FL and SHAP models."""
import copy
import io
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.shared.data.data_loader import DATASET_METADATA, get_dataset


def quantize_model(model):
    """Author Linear/LSTM dynamic INT8 mechanism on an isolated CPU copy."""
    if torch.backends.quantized.engine == "none":
        supported = torch.backends.quantized.supported_engines
        engine = next((name for name in ("x86", "fbgemm", "qnnpack") if name in supported), None)
        if engine is None:
            raise RuntimeError("Dynamic INT8 inference requires a PyTorch CPU quantization backend")
        torch.backends.quantized.engine = engine
    return torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(model).cpu().eval(), {nn.Linear, nn.LSTM}, dtype=torch.qint8,
    )


def evaluation_loader(config):
    """A deterministic, clean held-out probe; never the poisoned training data."""
    name = str(config["dataset"]).lower()
    X, y, _ = get_dataset(
        name, config["dataset_path"], DATASET_METADATA[name]["classes"], int(config["random_seed"]),
        simulate_global_leakage=bool(config["simulate_global_leakage"]),
        apply_smote=bool(config["apply_smote"]), split="val",
        test_split=float(config["test_split"]), val_split=float(config["val_split"]),
    )
    limit = int(config["edge_inference_samples"])
    if limit < 1:
        raise ValueError("edge_inference_samples must be positive")
    indices = torch.randperm(len(y), generator=torch.Generator().manual_seed(int(config["random_seed"])))[:limit]
    return DataLoader(TensorDataset(X[indices], y[indices]), batch_size=int(config["batch_size"]), shuffle=False)


def _serialized_size(model):
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


@torch.inference_mode()
def evaluate_cpu_inference(model, loader, int8=True):
    """Compare copies on identical batches; timings exclude conversion and warmup.

    Latency is a local probe, not a hardware-independent benchmark. Serialized
    state bytes include metadata; they are not network traffic or total RAM.
    """
    batches = [(x.cpu(), y.cpu()) for x, y in loader]
    count = sum(len(y) for _, y in batches)
    if not count:
        raise ValueError("Edge inference requires nonempty held-out data")
    fp32 = copy.deepcopy(model).cpu().eval()
    variants = {"fp32": fp32}
    if int8:
        variants["int8"] = quantize_model(fp32)
    metrics = {"edge_inference_samples": count, "edge_inference_int8": int8,
               "edge_inference_cpu_threads": torch.get_num_threads()}
    loss = 0.0
    for label, candidate in variants.items():
        candidate(batches[0][0])  # warmup excluded
        correct, loss_sum, elapsed = 0, 0.0, 0.0
        for x, y in batches:
            start = time.perf_counter()
            logits = candidate(x)
            elapsed += time.perf_counter() - start
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite edge inference predictions")
            correct += int((logits.argmax(-1) == y).sum())
            loss_sum += float(nn.functional.cross_entropy(logits, y, reduction="sum"))
        loss = loss_sum / count
        metrics.update({f"edge_{label}_accuracy": correct / count,
                        f"edge_{label}_loss": loss,
                        f"edge_{label}_state_bytes": _serialized_size(candidate),
                        f"edge_{label}_inference_ms_per_sample": elapsed * 1000 / count})
    metrics["accuracy"] = metrics["edge_int8_accuracy" if int8 else "edge_fp32_accuracy"]
    return loss, count, metrics
