"""Canonical Correlation Analysis (CCA) for SSVEP Decoding.

Implements multi-channel canonical correlation template matching against
harmonic sine/cosine reference signals using QR-SVD decomposition.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np

from src.models.base import BaseSSVEPClassifier
from src.utils.logging import setup_logger

logger = setup_logger("models_cca")


def generate_cca_references(
    freq: float,
    n_times: int,
    sfreq: float,
    n_harmonics: int = 5,
    phase: Optional[float] = None
) -> np.ndarray:
    """Generates harmonic sine and cosine reference signals for a stimulus frequency.

    Args:
        freq: Stimulus frequency in Hz
        n_times: Number of time sample points
        sfreq: Sampling rate in Hz
        n_harmonics: Number of harmonic components to generate (default: 5)
        phase: Optional initial phase in radians (default: None)

    Returns:
        Y: np.ndarray of shape (2 * n_harmonics, n_times)
    """
    t = np.arange(n_times) / sfreq
    references = []
    base_phase = phase if phase is not None else 0.0

    for h in range(1, n_harmonics + 1):
        harm_freq = h * freq
        harm_phase = h * base_phase
        references.append(np.sin(2.0 * np.pi * harm_freq * t + harm_phase))
        references.append(np.cos(2.0 * np.pi * harm_freq * t + harm_phase))

    return np.array(references, dtype=np.float64)


def compute_canonical_correlation(X: np.ndarray, Y: np.ndarray) -> float:
    """Computes the maximum canonical correlation coefficient between multi-channel EEG X

    and synthetic harmonic references Y using QR and SVD decomposition.

    Args:
        X: EEG data array of shape (n_channels, n_times)
        Y: Reference signals array of shape (2 * n_harmonics, n_times)

    Returns:
        rho: Maximum canonical correlation coefficient in [0, 1]
    """
    # Center signals across time
    X_c = X - np.mean(X, axis=1, keepdims=True)
    Y_c = Y - np.mean(Y, axis=1, keepdims=True)

    # QR decomposition of transposes (time samples x channels)
    q_x, _ = np.linalg.qr(X_c.T)
    q_y, _ = np.linalg.qr(Y_c.T)

    # SVD of projection matrix Q_x^T @ Q_y
    # The singular values represent the canonical correlation coefficients
    s = np.linalg.svd(np.dot(q_x.T, q_y), compute_uv=False)
    return float(s[0]) if len(s) > 0 else 0.0


class CCAClassifier(BaseSSVEPClassifier):
    """Standard Canonical Correlation Analysis classifier for SSVEP."""

    def __init__(
        self,
        target_frequencies: Dict[int, float],
        sfreq: float = 250.0,
        n_harmonics: int = 5,
        target_phases: Optional[Dict[int, float]] = None,
    ):
        """
        Args:
            target_frequencies: Mapping of target ID (1..40) to frequency in Hz
            sfreq: Sampling frequency in Hz
            n_harmonics: Number of harmonics for reference signals (default: 5)
            target_phases: Optional mapping of target ID to stimulus phase in radians
        """
        super().__init__(target_frequencies=target_frequencies, sfreq=sfreq)
        self.n_harmonics = n_harmonics
        self.target_phases = target_phases
        self._cached_references: Dict[int, np.ndarray] = {}

    def _get_or_create_reference(self, target_id: int, n_times: int) -> np.ndarray:
        """Retrieves or creates cached harmonic reference for a given target and duration."""
        cache_key = (target_id, n_times)
        if cache_key not in self._cached_references:
            freq = self.target_frequencies[target_id]
            phase = self.target_phases.get(target_id) if self.target_phases else None
            self._cached_references[cache_key] = generate_cca_references(
                freq=freq,
                n_times=n_times,
                sfreq=self.sfreq,
                n_harmonics=self.n_harmonics,
                phase=phase,
            )
        return self._cached_references[cache_key]

    def fit(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> "CCAClassifier":
        """CCA is an unsupervised template-matching method, so fit marks model as ready."""
        self.is_fitted_ = True
        logger.info(f"Fitted standard CCA classifier for {self.n_classes_} targets (Nh={self.n_harmonics}).")
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Computes CCA correlation coefficients for each class.

        Args:
            X: Array of shape (n_trials, n_channels, n_times) or (n_channels, n_times)

        Returns:
            scores: Array of shape (n_trials, n_classes) with correlation values in [0, 1]
        """
        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)

        n_trials, n_channels, n_times = X.shape
        scores = np.zeros((n_trials, self.n_classes_), dtype=np.float64)

        for c_idx, target_id in enumerate(self.classes_):
            Y_ref = self._get_or_create_reference(target_id, n_times)
            for trial_idx in range(n_trials):
                trial_eeg = X[trial_idx]
                scores[trial_idx, c_idx] = compute_canonical_correlation(trial_eeg, Y_ref)

        return scores
