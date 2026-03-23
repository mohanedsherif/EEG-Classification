"""
eeg.features.spectral
=====================
Feature extraction from EEG signals.

Features produced
-----------------
1. **Band power** — instantaneous power (Hilbert envelope²) averaged over
   four post-stimulus time windows for each of five canonical frequency bands.
   Shape per trial: 5 bands × 4 windows × n_channels

2. **Phase-Locking Value (PLV)** — theta-band inter-channel synchrony for all
   channel pairs.  Captures functional connectivity which is strongly linked
   to memory consolidation during sleep.
   Shape per trial: n_channels × (n_channels − 1) / 2

3. **Covariance matrices** — regularised spatial covariance per trial, used
   by the Riemannian geometry classifier.
   Shape per trial: n_channels × n_channels

References
----------
Lachaux, J.-P. et al. (1999). Measuring phase synchrony in brain signals.
Human Brain Mapping, 8(4), 194–208.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FS: int = 200  # default sampling frequency (Hz)

FREQ_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1,   4),
    "theta": (4,   8),
    "alpha": (8,  12),
    "beta":  (12, 30),
    "gamma": (30, 45),
}

TIME_WINDOWS: dict[str, tuple[float, float]] = {
    "early": (0.00, 0.33),
    "mid":   (0.33, 0.67),
    "late":  (0.67, 1.00),
    "full":  (0.00, 1.00),
}


# ---------------------------------------------------------------------------
# Signal processing primitives
# ---------------------------------------------------------------------------

def bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: int = FS,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    data    : ndarray, last axis is time
    lowcut  : lower cutoff frequency (Hz)
    highcut : upper cutoff frequency (Hz)
    fs      : sampling frequency (Hz)
    order   : filter order

    Returns
    -------
    ndarray, same shape as input
    """
    nyq  = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data, axis=-1)


def instantaneous_power(
    data: np.ndarray,
    freq_band: tuple[float, float],
    fs: int = FS,
) -> np.ndarray:
    """
    Compute instantaneous power via the Hilbert transform.

    Parameters
    ----------
    data      : ndarray (..., time)
    freq_band : (low, high) in Hz
    fs        : sampling frequency

    Returns
    -------
    power : ndarray, same shape as input
    """
    filtered = bandpass_filter(data, *freq_band, fs=fs)
    return np.abs(hilbert(filtered)) ** 2


# ---------------------------------------------------------------------------
# PLV connectivity
# ---------------------------------------------------------------------------

def compute_plv(
    data: np.ndarray,
    freq_band: tuple[float, float] | None = None,
    fs: int = FS,
) -> np.ndarray:
    """
    Compute theta-band Phase-Locking Value for all channel pairs.

    PLV measures the consistency of the phase difference between two signals
    across trials.  A value of 1 means perfect phase synchrony; 0 means
    completely random phase relationship.

    Parameters
    ----------
    data      : ndarray (n_trials, n_channels, n_timepoints)
    freq_band : frequency band for PLV (default: theta 4–8 Hz)
    fs        : sampling frequency

    Returns
    -------
    plv : ndarray (n_trials, n_pairs)
        where n_pairs = n_channels × (n_channels − 1) / 2
    """
    if freq_band is None:
        freq_band = FREQ_BANDS["theta"]

    filtered  = bandpass_filter(data, *freq_band, fs=fs)
    phase     = np.angle(hilbert(filtered))            # (trials, ch, time)
    n_trials, n_ch, _ = phase.shape

    pairs   = [(i, j) for i in range(n_ch) for j in range(i + 1, n_ch)]
    plv_mat = np.zeros((n_trials, len(pairs)), dtype=np.float32)

    for k, (i, j) in enumerate(pairs):
        diff           = phase[:, i, :] - phase[:, j, :]
        plv_mat[:, k]  = np.abs(np.mean(np.exp(1j * diff), axis=-1))

    return plv_mat


# ---------------------------------------------------------------------------
# Covariance matrices (for Riemannian classifier)
# ---------------------------------------------------------------------------

def compute_covariance_matrices(
    data: np.ndarray,
    regularisation: float = 1e-6,
) -> np.ndarray:
    """
    Compute a regularised spatial covariance matrix for each trial.

    Parameters
    ----------
    data          : ndarray (n_trials, n_channels, n_timepoints)
    regularisation: Tikhonov regularisation added to the diagonal

    Returns
    -------
    covmats : ndarray (n_trials, n_channels, n_channels)
    """
    n_trials, n_ch, _ = data.shape
    covmats = np.zeros((n_trials, n_ch, n_ch), dtype=np.float64)

    for t in range(n_trials):
        C = np.cov(data[t])                              # (n_ch, n_ch)
        C += regularisation * np.eye(n_ch)               # regularise
        covmats[t] = C

    return covmats


# ---------------------------------------------------------------------------
# Full feature vector
# ---------------------------------------------------------------------------

def extract_features(
    data: np.ndarray,
    time_vector: np.ndarray,
    include_plv: bool = True,
) -> np.ndarray:
    """
    Build a fixed-size feature vector per trial.

    Feature layout
    --------------
    For each (band, window) pair:
        mean instantaneous power per channel  →  n_channels values
    Then (if include_plv):
        theta-band PLV for all channel pairs  →  n_ch*(n_ch-1)/2 values

    Total dimensions (example, 16 channels, PLV on):
        5 bands × 4 windows × 16 ch  +  16*15/2 = 320 + 120 = 440

    Parameters
    ----------
    data        : ndarray (n_trials, n_channels, n_timepoints)
    time_vector : ndarray (n_timepoints,) in seconds
    include_plv : whether to append PLV features

    Returns
    -------
    features : float32 ndarray (n_trials, n_features)
    """
    blocks: list[np.ndarray] = []

    for freq_band in FREQ_BANDS.values():
        power = instantaneous_power(data, freq_band)     # (trials, ch, time)

        for t0, t1 in TIME_WINDOWS.values():
            mask = (time_vector >= t0) & (time_vector < t1)
            if mask.sum() == 0:
                continue
            blocks.append(power[:, :, mask].mean(axis=-1).astype(np.float32))

    if include_plv:
        blocks.append(compute_plv(data))

    return np.concatenate(blocks, axis=1)
