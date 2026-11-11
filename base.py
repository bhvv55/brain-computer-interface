"""Base class for BCI neural intent decoders."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import joblib
import numpy as np

class BaseSSVEPClassifier(ABC):
    """Abstract base class for all SSVEP classification algorithms."""

    def __init__(self, target_frequencies: Dict[int, float], sfreq: float = 250.0):
        """
        Args:
            target_frequencies: Dict mapping target ID (1..40) to stimulus frequency in Hz
            sfreq: Sampling frequency in Hz
        """
        self.target_frequencies = target_frequencies
        self.sfreq = sfreq
        self.classes_ = sorted(list(target_frequencies.keys()))
        self.n_classes_ = len(self.classes_)
        self.is_fitted_ = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseSSVEPClassifier":
        """Fit model to training data (if supervised).

        Args:
            X: np.ndarray of shape (n_trials, n_channels, n_times)
            y: np.ndarray of shape (n_trials,) with target labels
        """
        pass

    @abstractmethod
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Computes raw correlation or decision scores for each class.

        Args:
            X: np.ndarray of shape (n_trials, n_channels, n_times) or (n_channels, n_times)

        Returns:
            scores: np.ndarray of shape (n_trials, n_classes)
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts target classes for trials."""
        scores = self.predict_scores(X)
        pred_indices = np.argmax(scores, axis=1)
        return np.array([self.classes_[i] for i in pred_indices])

    def predict_proba(self, X: np.ndarray, temperature: float = 10.0) -> np.ndarray:
        """Computes calibrated probability distribution across classes using temperature-scaled softmax.

        Args:
            X: np.ndarray of shape (n_trials, n_channels, n_times)
            temperature: Softmax temperature parameter for calibration

        Returns:
            probas: np.ndarray of shape (n_trials, n_classes)
        """
        scores = self.predict_scores(X)
        scaled = scores * temperature
        # Subtract max for numerical stability
        exp_scores = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        probas = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probas

    def predict_with_confidence(
        self,
        X: np.ndarray,
        temperature: float = 10.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns predictions, top confidence scores, and margin over second best.

        Returns:
            predictions: np.ndarray of shape (n_trials,)
            confidences: np.ndarray of shape (n_trials,) with max probability in [0, 1]
            margins: np.ndarray of shape (n_trials,) with difference between 1st and 2nd best scores
        """
        probas = self.predict_proba(X, temperature=temperature)
        predictions = self.predict(X)
        confidences = np.max(probas, axis=1)

        # Margin calculation
        sorted_probas = np.sort(probas, axis=1)
        margins = sorted_probas[:, -1] - sorted_probas[:, -2]

        return predictions, confidences, margins

    def save(self, file_path: Union[str, Path]) -> None:
        """Persists model to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, str(path))

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "BaseSSVEPClassifier":
        """Loads persisted model from disk."""
        return joblib.load(str(file_path))
