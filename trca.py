"""Task-Related Component Analysis (TRCA) for SSVEP.

Implements spatial filtering to maximize inter-trial phase consistency across blocks
(Nakanishi et al. IEEE TBME 2018).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import linalg

from src.models.base import BaseSSVEPClassifier
from src.utils.logging import setup_logger

logger = setup_logger("models_trca")


class TRCAClassifier(BaseSSVEPClassifier):
    """Task-Related Component Analysis (TRCA) classifier."""

    def __init__(
        self,
        target_frequencies: Dict[int, float],
        sfreq: float = 250.0,
        ensemble: bool = False,
    ):
        """
        Args:
            target_frequencies: Mapping of target ID (1..40) to frequency in Hz
            sfreq: Sampling rate in Hz
            ensemble: Whether to use ensemble TRCA across all spatial filters
        """
        super().__init__(target_frequencies=target_frequencies, sfreq=sfreq)
        self.ensemble = ensemble
        self.spatial_filters_: Dict[int, np.ndarray] = {}
        self.templates_: Dict[int, np.ndarray] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TRCAClassifier":
        """Fits TRCA spatial filters and class average templates from training data.

        Args:
            X: Array of shape (n_trials, n_channels, n_times)
            y: Array of shape (n_trials,) with target IDs
        """
        n_trials, n_channels, n_times = X.shape
        self.spatial_filters_ = {}
        self.templates_ = {}

        logger.info(f"Fitting TRCA on {n_trials} trials across {self.n_classes_} classes (ensemble={self.ensemble})...")

        for target_id in self.classes_:
            cls_mask = (y == target_id)
            cls_trials = X[cls_mask]

            if len(cls_trials) < 2:
                self.spatial_filters_[target_id] = np.ones((n_channels, 1)) / np.sqrt(n_channels)
                self.templates_[target_id] = cls_trials[0] if len(cls_trials) == 1 else np.zeros((n_channels, n_times))
                continue

            # Class average template
            template = np.mean(cls_trials, axis=0)
            self.templates_[target_id] = template

            # Compute S (inter-trial covariance) and Q (total covariance)
            N_cls = len(cls_trials)
            S = np.zeros((n_channels, n_channels))
            Q = np.zeros((n_channels, n_channels))

            trials_c = np.array([tr - np.mean(tr, axis=1, keepdims=True) for tr in cls_trials])

            for i in range(N_cls):
                for j in range(N_cls):
                    if i != j:
                        S += np.dot(trials_c[i], trials_c[j].T)
                Q += np.dot(trials_c[i], trials_c[i].T)

            # Regularize Q for numerical stability
            Q += 1e-5 * np.eye(n_channels)

            try:
                eigvals, eigvecs = linalg.eigh(S, Q)
                w = eigvecs[:, -1:]
                norm_w = np.linalg.norm(w)
                self.spatial_filters_[target_id] = w / (norm_w if norm_w > 0 else 1.0)
            except Exception as e:
                logger.warning(f"TRCA eigenvalue failure for target {target_id}: {e}. Falling back.")
                self.spatial_filters_[target_id] = np.ones((n_channels, 1)) / np.sqrt(n_channels)

        self.is_fitted_ = True
        logger.info("TRCA model fitting complete.")
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Computes correlation coefficients against learned templates.

        Args:
            X: Array of shape (n_trials, n_channels, n_times) or (n_channels, n_times)

        Returns:
            scores: Array of shape (n_trials, n_classes) with correlation values
        """
        if not self.is_fitted_:
            raise RuntimeError("TRCA model is not fitted yet. Call fit() first.")

        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)

        n_trials, n_channels, n_times = X.shape
        scores = np.zeros((n_trials, self.n_classes_), dtype=np.float64)

        for c_idx, target_id in enumerate(self.classes_):
            w = self.spatial_filters_[target_id]
            template = self.templates_[target_id]
            s_tmpl = np.dot(w.T, template).ravel()
            s_tmpl_std = np.std(s_tmpl)

            for trial_idx in range(n_trials):
                trial = X[trial_idx]
                s_tr = np.dot(w.T, trial).ravel()
                s_tr_std = np.std(s_tr)

                if s_tmpl_std > 1e-9 and s_tr_std > 1e-9:
                    r = np.corrcoef(s_tr, s_tmpl)[0, 1]
                    scores[trial_idx, c_idx] = float(r) if not np.isnan(r) else 0.0
                else:
                    scores[trial_idx, c_idx] = 0.0

        return scores
