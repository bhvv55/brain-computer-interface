"""Support Vector Machine (SVM) for SSVEP Classification.

Extracts canonical correlation or harmonic power spectral features across channels
and trains an SVM with probability calibration.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.models.base import BaseSSVEPClassifier
from src.models.cca import CCAClassifier
from src.utils.logging import setup_logger

logger = setup_logger("models_svm")


class SSVEPSVMClassifier(BaseSSVEPClassifier):
    """Hybrid CCA-Feature + Support Vector Machine Classifier for SSVEP."""

    def __init__(
        self,
        target_frequencies: Dict[int, float],
        sfreq: float = 250.0,
        n_harmonics: int = 3,
        C: float = 5.0,
        kernel: str = "linear",
        gamma: str = "scale",
    ):
        """
        Args:
            target_frequencies: Mapping of target ID to frequency in Hz
            sfreq: Sampling rate in Hz
            n_harmonics: Number of harmonics for CCA feature extraction
            C: Regularization parameter
            kernel: 'linear', 'rbf', etc.
            gamma: Kernel coefficient
        """
        super().__init__(target_frequencies=target_frequencies, sfreq=sfreq)
        self.n_harmonics = n_harmonics
        self.C = C
        self.kernel = kernel
        self.gamma = gamma

        # Base CCA feature extractor
        self.cca_extractor = CCAClassifier(
            target_frequencies=target_frequencies,
            sfreq=sfreq,
            n_harmonics=n_harmonics,
        )

        # Scikit-Learn SVM Pipeline with StandardScaler
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, probability=True, random_state=42)),
        ])

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extracts 40-dimensional CCA correlation vector for each trial."""
        return self.cca_extractor.predict_scores(X)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SSVEPSVMClassifier":
        """Fits SVM on extracted SSVEP features.

        Args:
            X: Array of shape (n_trials, n_channels, n_times)
            y: Array of shape (n_trials,) with target IDs
        """
        self.cca_extractor.fit()
        features = self._extract_features(X)

        logger.info(f"Training SVM on feature matrix of shape {features.shape}...")
        self.pipeline.fit(features, y)
        self.is_fitted_ = True
        logger.info("SVM training complete.")
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Returns decision function scores or probabilities."""
        features = self._extract_features(X)
        if hasattr(self.pipeline.named_steps["svm"], "predict_proba"):
            # Return class probabilities aligned with self.classes_
            probas = self.pipeline.predict_proba(features)
            # Ensure correct column alignment with self.classes_
            svm_classes = list(self.pipeline.named_steps["svm"].classes_)
            reordered = np.zeros((len(X), self.n_classes_))
            for i, c in enumerate(self.classes_):
                if c in svm_classes:
                    reordered[:, i] = probas[:, svm_classes.index(c)]
            return reordered
        else:
            return self.pipeline.decision_function(features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels directly using SVM pipeline."""
        features = self._extract_features(X)
        return self.pipeline.predict(features)
