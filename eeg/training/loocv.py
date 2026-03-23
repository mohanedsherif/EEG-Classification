"""
eeg.training.loocv
==================
Leave-One-Participant-Out Cross-Validation (LOOCV) engine.

Design
------
The ``run_loocv`` function accepts a *model factory* — a zero-argument callable
that returns a fresh, untrained model instance.  This pattern keeps the LOOCV
engine completely decoupled from any specific model class and works identically
for classical (scikit-learn style), neural network (EEGNetTrainer), and
Riemannian (RiemannianMDM) models.

Feature vs. raw data
--------------------
Classical models consume a 2-D feature matrix (n_trials, n_features) produced
by ``extract_features``.  EEGNet and Riemannian models consume the raw 3-D
array (n_trials, n_channels, n_timepoints).  The ``use_raw`` flag switches
between the two modes.

Positive class convention
-------------------------
Labels are 0 (neutral) and 1 (emotional).  AUC is computed for class 1.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
from sklearn.metrics import roc_auc_score

from eeg.features.spectral import extract_features
from eeg.utils.metrics import LOOCVResult


def run_loocv(
    subjects_data: list[dict],
    model_factory: Callable,
    *,
    use_raw: bool   = False,
    verbose: bool   = True,
    val_subject: bool = False,
) -> LOOCVResult:
    """
    Leave-One-Participant-Out Cross-Validation.

    For each fold:
      - test  = one held-out participant
      - train = all remaining participants
      - A fresh model is created via ``model_factory()``
      - AUC is computed for the held-out participant

    Parameters
    ----------
    subjects_data  : list of per-subject dicts (from ``load_dataset``).
                     Each dict must have keys ``X`` (n_trials, ch, time),
                     ``y`` (n_trials,), ``time`` (n_timepoints,),
                     ``subject_id`` (str).
    model_factory  : callable → fresh model with ``fit`` and ``predict_proba``.
    use_raw        : if True, pass raw 3-D EEG arrays to the model (for
                     EEGNet / Riemannian).  If False (default), extract
                     multi-band + PLV features first.
    verbose        : print per-fold AUC.
    val_subject    : if True and ``use_raw=True``, carve out one training
                     participant as a validation set for early stopping.

    Returns
    -------
    LOOCVResult
    """
    n = len(subjects_data)

    # Pre-extract features for all subjects (classical models only)
    if not use_raw:
        if verbose:
            print("  Extracting features...", flush=True)
        features = [
            extract_features(s["X"], s["time"])
            for s in subjects_data
        ]

    model_name = model_factory().__class__.__name__
    aucs       = []
    subj_ids   = []
    t_start    = time.time()

    if verbose:
        print(f"\n  Running LOOCV for {model_name} ({n} folds):")

    for i in range(n):
        subj_id = subjects_data[i]["subject_id"]
        subj_ids.append(subj_id)

        if verbose:
            print(f"    Fold {i + 1:2d}/{n}  (test = participant {subj_id})...",
                  end=" ", flush=True)

        # ----------------------------------------------------------------
        # Build train / test splits
        # ----------------------------------------------------------------
        train_idx = [j for j in range(n) if j != i]

        if use_raw:
            X_test  = subjects_data[i]["X"]
            y_test  = subjects_data[i]["y"]
            X_train = np.concatenate([subjects_data[j]["X"] for j in train_idx])
            y_train = np.concatenate([subjects_data[j]["y"] for j in train_idx])

            # Optional: one training participant as validation for early stopping
            X_val, y_val = None, None
            if val_subject and len(train_idx) > 1:
                val_idx    = train_idx[-1]    # last training subject as val
                train_only = train_idx[:-1]
                X_val   = subjects_data[val_idx]["X"]
                y_val   = subjects_data[val_idx]["y"]
                X_train = np.concatenate([subjects_data[j]["X"] for j in train_only])
                y_train = np.concatenate([subjects_data[j]["y"] for j in train_only])
        else:
            X_test  = features[i]
            y_test  = subjects_data[i]["y"]
            X_train = np.concatenate([features[j] for j in train_idx])
            y_train = np.concatenate([subjects_data[j]["y"] for j in train_idx])
            X_val, y_val = None, None

        # ----------------------------------------------------------------
        # Train and evaluate
        # ----------------------------------------------------------------
        model = model_factory()

        if use_raw and X_val is not None:
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        else:
            model.fit(X_train, y_train)

        # AUC for the emotional class (label = 1)
        pos_idx = list(model.classes_).index(1)
        proba   = model.predict_proba(X_test)[:, pos_idx]
        auc     = roc_auc_score(y_test == 1, proba)
        aucs.append(auc)

        if verbose:
            print(f"AUC = {auc:.4f}")

    elapsed = time.time() - t_start

    result = LOOCVResult(
        model_name       = model_name,
        per_subject_auc  = aucs,
        subject_ids      = subj_ids,
        fit_time_seconds = elapsed,
    )

    if verbose:
        print(
            f"\n  {model_name} summary: "
            f"Mean = {result.mean_auc:.4f} ± {result.std_auc:.4f}  "
            f"CI95 = [{result.ci_95[0]:.3f}, {result.ci_95[1]:.3f}]  "
            f"({elapsed:.1f}s)\n"
        )

    return result
