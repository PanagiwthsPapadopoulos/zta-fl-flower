import torch
from src.shared.models.cnn_lstm import CNNLSTMClassifier


def get_model(architecture_name: str, n_features: int, n_classes: int) -> torch.nn.Module:
    """Instantiates and returns the requested neural network architecture."""
    arch = architecture_name.lower()
    if arch == "cnnlstm":
        return CNNLSTMClassifier(n_features=n_features, n_classes=n_classes)
    else:
        raise ValueError(f"❌ Unknown model architecture requested: {arch}")


def get_seeded_model(architecture_name: str, n_features: int, n_classes: int, seed: int):
    """Build the trusted initial CPU model without consuming the caller's RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return get_model(architecture_name, n_features, n_classes)
