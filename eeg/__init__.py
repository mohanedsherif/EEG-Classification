"""
eeg — EEG Emotional Memory Classification
==========================================
A research-grade pipeline for classifying emotional vs. neutral memory
reactivation from sleep EEG recordings.

Models
------
- ClassicalEnsemble : Regularised LDA + RBF-SVM (multi-band features + PLV)
- EEGNet            : Compact CNN for EEG (Lawhern et al., 2018)  [requires torch]
- RiemannianMDM     : Minimum Distance to Mean on SPD manifold    [requires pyriemann]

Usage
-----
>>> from eeg.data.loader import load_dataset
>>> from eeg.models.classical import ClassicalEnsemble
>>> from eeg.training.loocv import run_loocv
>>> subjects = load_dataset("./training")
>>> result = run_loocv(subjects, lambda: ClassicalEnsemble())
>>> print(f"Mean AUC: {result.mean_auc:.4f}")
"""

__version__ = "1.0.0"
__author__  = "Mohanad Sherif"
