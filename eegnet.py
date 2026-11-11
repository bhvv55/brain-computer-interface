"""EEGNet implementation in PyTorch for SSVEP Classification.

Based on Lawhern et al. (J. Neural Eng. 2018) 'EEGNet: a compact convolutional
neural network for EEG-based brain-computer interfaces'.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from src.models.base import BaseSSVEPClassifier
from src.utils.logging import setup_logger

logger = setup_logger("models_eegnet")


class EEGNetModel(nn.Module):
    """PyTorch EEGNet Architecture (Lawhern et al. 2018)."""

    def __init__(
        self,
        n_classes: int = 40,
        n_channels: int = 9,
        n_times: int = 501,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout_rate: float = 0.25,
    ):
        super().__init__()
        self.F1 = F1
        self.D = D
        self.F2 = F2

        # Block 1: Temporal Conv + Depthwise Spatial Conv
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        # Block 2: Separable Conv (Depthwise + Pointwise)
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )

        # Compute flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            out = self.conv1(dummy)
            out = self.bn1(out)
            out = self.depthwise(out)
            out = self.bn2(out)
            out = self.elu(out)
            out = self.pool1(out)
            out = self.drop1(out)
            out = self.separable(out)
            self.flatten_dim = out.numel()

        self.classifier = nn.Linear(self.flatten_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, 1, n_channels, n_times)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.separable(x)
        x = x.flatten(start_dim=1)
        out = self.classifier(x)
        return out


class EEGNetClassifier(BaseSSVEPClassifier):
    """Wrapper for training and inference with PyTorch EEGNet."""

    def __init__(
        self,
        target_frequencies: Dict[int, float],
        sfreq: float = 250.0,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[str] = None,
    ):
        super().__init__(target_frequencies=target_frequencies, sfreq=sfreq)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[EEGNetModel] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EEGNetClassifier":
        """Trains EEGNet model on provided trials.

        Args:
            X: Array of shape (n_trials, n_channels, n_times)
            y: Array of shape (n_trials,) with target IDs
        """
        n_trials, n_channels, n_times = X.shape

        # Map target IDs (1..40) to 0-indexed labels
        label_to_idx = {tid: i for i, tid in enumerate(self.classes_)}
        y_indices = np.array([label_to_idx[val] for val in y], dtype=np.int64)

        # Standardize per trial across time
        X_norm = (X - np.mean(X, axis=-1, keepdims=True)) / (np.std(X, axis=-1, keepdims=True) + 1e-8)

        # Format as PyTorch tensors: (batch, 1, channels, times)
        X_tensor = torch.tensor(X_norm[:, np.newaxis, :, :], dtype=torch.float32)
        y_tensor = torch.tensor(y_indices, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = EEGNetModel(
            n_classes=self.n_classes_,
            n_channels=n_channels,
            n_times=n_times,
        ).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.model.train()
        logger.info(f"Training EEGNet on {self.device} for {self.epochs} epochs (batch={self.batch_size})...")

        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_y)

        self.is_fitted_ = True
        logger.info("EEGNet training completed.")
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Returns softmax probability distribution as decision scores."""
        if not self.is_fitted_ or self.model is None:
            raise RuntimeError("EEGNet is not fitted yet.")

        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)

        # Normalize
        X_norm = (X - np.mean(X, axis=-1, keepdims=True)) / (np.std(X, axis=-1, keepdims=True) + 1e-8)
        X_tensor = torch.tensor(X_norm[:, np.newaxis, :, :], dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probas = torch.softmax(logits, dim=1).cpu().numpy()

        return probas
