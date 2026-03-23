"""
eeg.models.classical
====================
Classical ML classifiers for EEG feature vectors.

Both models expose a scikit-learn–compatible interface:
    ``fit(X, y)``  →  self
    ``predict_proba(X)``  →  ndarray (n_samples, 2)

where X is a 2-D feature matrix (n_trials, n_features) produced by
:func:`eeg.features.spectral.extract_features`.
"""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class LDABaseline:
    """
    Starter-pipeline baseline: plain LDA on multi-band power features.
    No regularisation — equivalent to what the course starter does but
    operating on aggregated trial features rather than per-timepoint.
    """

    def __init__(self) -> None:
        self._clf   = LinearDiscriminantAnalysis()
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LDABaseline":
        X = np.nan_to_num(X)
        self._clf.fit(X, y)
        self.classes_ = self._clf.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(np.nan_to_num(X))

    def __repr__(self) -> str:
        return "LDABaseline()"


class ClassicalEnsemble:
    """
    Ensemble of Regularised LDA + RBF-SVM.

    Regularised LDA
    ---------------
    Uses Ledoit-Wolf automatic shrinkage (``shrinkage='auto'``), which is
    critical for EEG data where the number of features often exceeds the
    number of samples.  Outperforms plain LDA in most BCI benchmarks.

    RBF-SVM
    -------
    Preceded by ``StandardScaler`` to zero-mean and unit-variance the
    feature matrix.  ``gamma='scale'`` sets γ = 1 / (n_features × X.var()).

    Ensemble
    --------
    Predicted probabilities from both classifiers are averaged.  This reduces
    variance and typically outperforms either model alone.

    References
    ----------
    Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for large-
    dimensional covariance matrices. Journal of Multivariate Analysis, 88(2).
    """

    def __init__(self, svm_C: float = 1.0, svm_gamma: str = "scale") -> None:
        self.lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
        self.svm = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(
                kernel="rbf",
                C=svm_C,
                gamma=svm_gamma,
                probability=True,
                random_state=42,
            )),
        ])
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassicalEnsemble":
        X = np.nan_to_num(X)
        self.lda.fit(X, y)
        self.svm.fit(X, y)
        self.classes_ = self.lda.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(X)
        return (self.lda.predict_proba(X) + self.svm.predict_proba(X)) / 2

    def __repr__(self) -> str:
        return f"ClassicalEnsemble(svm_C={self.svm['clf'].C})"
