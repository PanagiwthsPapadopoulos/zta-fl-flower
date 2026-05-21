import torch

from src.models.cnn_lstm import CNNLSTMClassifier


def get_model(architecture_name: str, n_features: int, n_classes: int) -> torch.nn.Module:
    """
    Instantiates and returns the requested neural network architecture.
    
    Parameters
    ----------
    architecture_name : str
        String identifier specifying the desired model architecture.
    n_features : int
        Number of input dimensions per sample.
    n_classes : int
        Number of distinct output classifications.
        
    Returns
    -------
    torch.nn.Module
        The initialized PyTorch model instance.
    """
    arch = architecture_name.lower()
    
    if arch == "cnnlstm":
        return CNNLSTMClassifier(n_features=n_features, n_classes=n_classes)
    else:
        raise ValueError(f"❌ Unknown model architecture requested: {arch}")