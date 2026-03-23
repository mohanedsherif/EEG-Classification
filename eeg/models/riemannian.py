"""
eeg.models.riemannian
=====================
Riemannian geometry classifier for EEG.

Spatial covariance matrices of EEG signals lie on the manifold of Symmetric
Positive Definite (SPD) matrices.  Classifying directly on this manifold
— rather than in flat Euclidean space — consistently outperforms classical
approaches in BCI benchmarks.

The Minimum Distance to Mean (MDM) classifier computes a Riemannian mean for
each class and assigns a new trial to the nearest class mean using the
affine-invariant Riemannian metric.

This module wraps ``pyriemann`` behind an optional-dependency guard.  If
``pyriemann`` is not installed the class raises an informative ImportError
on instantiation.

Install
-------
    pip install pyriemann

References
----------
Barachant, A. et al. (2012). Multiclass brain-computer interface classification
by Riemannian geometry. IEEE Transactions on Biomedical Engineering, 59(4).
https://doi.org/10.1109/TBME.2011.2172210
"""

from __future__ import annotations

import numpy as np

from eeg.features.spectral import compute_covariance_matrices
from eeg.utils.optional_deps import require_pyriemann, _PYRIEMANN_AVAILABLE

if _PYRIEMANN_AVAILABLE:
    from pyriemann.classification import MDM
    from pyriemann.estimation import Covariances


class RiemannianMDM:
    """
    Minimum Distance to Mean classifier on the Riemannian manifold of
    Symmetric Positive Definite (SPD) matrices.

    Input
    -----
    Raw (z-scored) EEG data of shape (n_trials, n_channels, n_timepoints).
    Covariance matrices are computed internally.

    Interface
    ---------
    Same as classical models: ``fit(X, y)`` and ``predict_proba(X)``.

    Requires
    --------
    ``pip install pyriemann``
    """

    def __init__(self, metric: str = "riemann") -> None:
        require_pyriemann("RiemannianMDM")
        self._mdm      = MDM(metric=metric)
        self.classes_: np.ndarray | None = None

    def _get_covmats(self, X: np.ndarray) -> np.ndarray:
        """Compute regularised covariance matrices from raw EEG."""
        return compute_covariance_matrices(X).astype(np.float64)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RiemannianMDM":
        """
        Parameters
        ----------
        X : ndarray (n_trials, n_channels, n_timepoints)
        y : ndarray (n_trials,)
        """
        covmats = self._get_covmats(X)
        self._mdm.fit(covmats, y)
        self.classes_ = self._mdm.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns
        -------
        proba : ndarray (n_trials, n_classes)
        """
        covmats = self._get_covmats(X)
        # MDM returns distances, not probabilities — convert with softmin
        distances = self._mdm.transform(covmats)   # (n_trials, n_classes)
        # Softmin: closer = higher probability
        neg_d  = -distances
        exp_d  = np.exp(neg_d - neg_d.max(axis=1, keepdims=True))
        return exp_d / exp_d.sum(axis=1, keepdims=True)

    def __repr__(self) -> str:
        return "RiemannianMDM(metric='riemann')"
