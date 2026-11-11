"""Filter Bank Canonical Correlation Analysis (FBCCA) for SSVEP Decoding.

Implements multi-band sub-band decomposition combined with weighted CCA,
substantially improving harmonic SNR and decoding accuracy (Chen et al. 2015).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.signal import butter, sosfiltfilt, cheby1

from src.models.base import BaseSSVEPClassifier
from src.models.cca import generate_cca_references, compute_canonical_correlation
from src.utils.logging import setup_logger

logger = setup_logger("models_fbcca")


class FBCCAClassifier(BaseSSVEPClassifier):
    """Filter Bank Canonical Correlation Analysis (FBCCA) Classifier."""

    def __init__(
        self,
        target_frequencies: Dict[int, float],
        sfreq: float = 250.0,
        n_harmonics: int = 5,
        n_subbands: int = 5,
        target_phases: Optional[Dict[int, float]] = None,
        a_weight: float = 1.25,
        b_weight: float = 0.25,
    ):
        """
        Args:
            target_frequencies: Mapping of target ID (1..40) to frequency in Hz
            sfreq: Sampling rate in Hz
            n_harmonics: Number of harmonics for reference signals (default: 5)
            n_subbands: Number of sub-bands in filter bank (default: 5)
            target_phases: Optional target phases in radians
            a_weight: Power coefficient in weight equation w(m) = m^(-a) + b (default: 1.25)
            b_weight: Additive coefficient in weight equation (default: 0.25)
        """
        super().__init__(target_frequencies=target_frequencies, sfreq=sfreq)
        self.n_harmonics = n_harmonics
        self.n_subbands = n_subbands
        self.target_phases = target_phases
        self.a_weight = a_weight
        self.b_weight = b_weight

        # Precompute sub-band weights w(m) = m^(-a) + b
        m_vals = np.arange(1, self.n_subbands + 1, dtype=float)
        self.subband_weights = (m_vals ** (-self.a_weight)) + self.b_weight

        # Filter SOS coefficients cache for each sub-band
        self._filter_sos_list = self._design_subband_filters()
        self._cached_references: Dict[Tuple[int, int], np.ndarray] = {}

    def _design_subband_filters(self) -> List[np.ndarray]:
        """Designs Chebyshev Type I / Butterworth SOS filters for each sub-band.

        Sub-band m covers: [m * 8.0 - 2.0, 90.0] Hz
        """
        sos_filters = []
        nyq = 0.5 * self.sfreq
        high_cutoff = min(90.0, nyq - 1.0)

        for m in range(1, self.n_subbands + 1):
            low_cutoff = max(6.0, m * 8.0 - 2.0)
            # Ensure cutoffs within (0, Nyquist)
            low_norm = low_cutoff / nyq
            high_norm = high_cutoff / nyq

            # 4th order Chebyshev Type I filter (0.5 dB ripple in passband)
            sos = cheby1(N=4, rp=0.5, Wn=[low_norm, high_norm], btype="bandpass", output="sos")
            sos_filters.append(sos)

        return sos_filters

    def _filter_subbands(self, X: np.ndarray) -> List[np.ndarray]:
        """Applies each sub-band filter to EEG data X (n_channels, n_times).

        Returns:
            subband_signals: List of arrays of shape (n_channels, n_times)
        """
        subbands = []
        for sos in self._filter_sos_list:
            # Zero-phase forward-backward filtering along time axis (axis=-1)
            X_filtered = sosfiltfilt(sos, X, axis=-1)
            subbands.append(X_filtered)
        return subbands

    def _get_or_create_reference(self, target_id: int, n_times: int) -> np.ndarray:
        """Retrieves or generates synthetic reference signals."""
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

    def fit(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> "FBCCAClassifier":
        """Unsupervised template-matching method."""
        self.is_fitted_ = True
        logger.info(
            f"Fitted FBCCA classifier ({self.n_classes_} targets, {self.n_subbands} sub-bands, "
            f"Nh={self.n_harmonics}, a={self.a_weight}, b={self.b_weight})."
        )
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Computes weighted FBCCA scores for each trial and target class.

        Args:
            X: Array of shape (n_trials, n_channels, n_times) or (n_channels, n_times)

        Returns:
            scores: Array of shape (n_trials, n_classes) with integrated scores rho_tilde
        """
        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)

        n_trials, n_channels, n_times = X.shape
        scores = np.zeros((n_trials, self.n_classes_), dtype=np.float64)

        # Precompute references for all classes
        references = [self._get_or_create_reference(tid, n_times) for tid in self.classes_]

        for trial_idx in range(n_trials):
            trial_eeg = X[trial_idx]
            # Decompose trial into sub-bands
            subband_data = self._filter_subbands(trial_eeg)

            for c_idx, target_id in enumerate(self.classes_):
                Y_ref = references[c_idx]
                trial_target_score = 0.0

                for m_idx, X_sub in enumerate(subband_data):
                    r_m = compute_canonical_correlation(X_sub, Y_ref)
                    w_m = self.subband_weights[m_idx]
                    trial_target_score += w_m * (r_m ** 2)

                scores[trial_idx, c_idx] = trial_target_score

        return scores
