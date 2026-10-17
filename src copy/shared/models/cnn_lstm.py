from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMClassifier(nn.Module):
    """Defines a one-dimensional convolutional neural network paired with a stacked LSTM for multi-class traffic classification."""
    def __init__(
        self,
        n_features: int = 40,
        n_classes: int = 15,
        seq_len: int = 1,
        cnn_filters: int = 64,
        cnn_kernel: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.seq_len = seq_len
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional

        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=cnn_filters,
            kernel_size=cnn_kernel,
            padding=cnn_kernel // 2,
        )
        self.bn1 = nn.BatchNorm1d(cnn_filters)

        self.conv2 = nn.Conv1d(
            in_channels=cnn_filters,
            out_channels=cnn_filters * 2,
            kernel_size=cnn_kernel,
            padding=cnn_kernel // 2,
        )
        self.bn2 = nn.BatchNorm1d(cnn_filters * 2)

        self.conv3 = nn.Conv1d(
            in_channels=cnn_filters * 2,
            out_channels=cnn_filters * 2,
            kernel_size=cnn_kernel,
            padding=cnn_kernel // 2,
        )
        self.bn3 = nn.BatchNorm1d(cnn_filters * 2)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.cnn_dropout = nn.Dropout(p=dropout)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=4)

        self.lstm = nn.LSTM(
            input_size=cnn_filters * 2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_out = lstm_hidden * (2 if bidirectional else 1)

        self.fc_dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(lstm_out, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        if x.size(-1) >= 2:
            x = self.pool(x)
            
        x = F.relu(self.bn3(self.conv3(x)))
        self.cnn_dropout(x)
        x = self.adaptive_pool(x)
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]

        x = self.fc_dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def get_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if x.size(-1) >= 2:
            x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        self.cnn_dropout(x)
        x = self.adaptive_pool(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.fc_dropout(x)
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))