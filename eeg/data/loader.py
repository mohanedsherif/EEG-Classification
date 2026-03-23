"""
eeg.data.loader
===============
Loads MATLAB v7.3 HDF5 files produced by the EEG Emotional Memory competition
and returns normalised numpy arrays ready for feature extraction or neural nets.

Label convention (used everywhere in this package)
---------------------------------------------------
  0 = neutral   (sleep_neu folder)
  1 = emotional (sleep_emo folder)
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import zscore


# ---------------------------------------------------------------------------
# Internal HDF5 helpers
# ---------------------------------------------------------------------------

def _resolve_field(f: h5py.File, data_ref: h5py.Group, field: str):
    """Dereference an HDF5 field that may be a direct dataset or a Reference."""
    node = data_ref[field]
    if not isinstance(node, h5py.Dataset):
        return node

    val = node[()]
    if isinstance(val, h5py.Reference):
        return f[val]
    if hasattr(val, "shape") and val.shape == (1, 1):
        ref = val.item()
        if isinstance(ref, h5py.Reference):
            return f[ref]
        if isinstance(ref, bytes):
            ref = ref.decode("utf-8")
        return f[ref]
    return node


def _load_mat(filepath: str | Path) -> dict:
    """
    Load a single HDF5 .mat file.

    Returns
    -------
    dict with keys:
        ``trial``     – ndarray (n_trials, n_channels, n_timepoints)
        ``trialinfo`` – ndarray (n_trials, n_cols)  or  None
        ``time``      – ndarray (n_timepoints,)  [seconds, t >= 0 only]
    """
    with h5py.File(filepath, "r") as f:
        ref = f["data"]

        time_data  = np.array(_resolve_field(f, ref, "time")).flatten()
        trial_data = np.array(_resolve_field(f, ref, "trial")).T  # → (trials, ch, time)

        try:
            trialinfo = np.array(_resolve_field(f, ref, "trialinfo")).T
        except (KeyError, ValueError, TypeError):
            trialinfo = None

        # keep post-stimulus window only (t >= 0)
        mask = time_data >= 0
        if np.any(~mask):
            time_data  = time_data[mask]
            trial_data = trial_data[:, :, mask]

    return {"trial": trial_data, "trialinfo": trialinfo, "time": time_data}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_subject(train_path: str | Path, subj_file: str) -> dict:
    """
    Load and combine neutral + emotional EEG data for one participant.

    Parameters
    ----------
    train_path : str or Path
        Root directory containing ``sleep_neu/`` and ``sleep_emo/`` sub-folders.
    subj_file : str
        Filename (e.g. ``"S_2_cleaned.mat"``) present in both sub-folders.

    Returns
    -------
    dict with keys:
        ``X``          – float32 ndarray (n_trials, n_channels, n_timepoints),
                         z-scored per participant (no data leakage: statistics
                         are computed only from this subject's own trials)
        ``y``          – int ndarray (n_trials,),  0 = neutral, 1 = emotional
        ``time``       – ndarray (n_timepoints,)  [seconds]
        ``subject_id`` – str
    """
    train_path = Path(train_path)
    neu = _load_mat(train_path / "sleep_neu" / subj_file)
    emo = _load_mat(train_path / "sleep_emo" / subj_file)

    X = np.concatenate([neu["trial"], emo["trial"]], axis=0).astype(np.float32)
    y = np.array(
        [0] * len(neu["trial"]) + [1] * len(emo["trial"]), dtype=np.int64
    )

    # Per-participant z-score (axis=0 → across trials).
    # NOTE: no data leakage — stats are computed from this subject only,
    # and the same transformation is applied to their test trials in LOOCV.
    X = zscore(X, axis=0).astype(np.float32)

    subject_id = subj_file.split("_")[1]  # "S_2_cleaned.mat" → "2"

    return {"X": X, "y": y, "time": neu["time"], "subject_id": subject_id}


def load_dataset(train_path: str | Path) -> list[dict]:
    """
    Load all participants from the training directory.

    Parameters
    ----------
    train_path : str or Path
        Directory containing ``sleep_neu/`` and ``sleep_emo/`` sub-folders.

    Returns
    -------
    List of per-subject dicts as returned by :func:`load_subject`,
    sorted by subject ID.

    Raises
    ------
    FileNotFoundError
        If ``train_path`` or the expected sub-folders do not exist.
    """
    train_path = Path(train_path)
    neu_dir    = train_path / "sleep_neu"

    if not neu_dir.exists():
        raise FileNotFoundError(
            f"Training data not found at '{train_path}'.\n"
            "Expected sub-folders: sleep_neu/ and sleep_emo/\n"
            "Download the data from the Kaggle competition page."
        )

    files = sorted(f for f in os.listdir(neu_dir) if f.endswith(".mat"))
    if not files:
        raise FileNotFoundError(f"No .mat files found in '{neu_dir}'.")

    subjects = []
    for fname in files:
        print(f"  Loading participant {fname.split('_')[1]}...", end=" ", flush=True)
        subj = load_subject(train_path, fname)
        subjects.append(subj)
        print(f"{len(subj['y'])} trials | {subj['X'].shape[1]} ch × {subj['X'].shape[2]} tp")

    return subjects


def load_test_subject(test_path: str | Path, subj_file: str) -> dict:
    """
    Load a single test file (no labels).

    Returns
    -------
    dict with keys ``X``, ``time``, ``subject_id``.
    """
    data = _load_mat(Path(test_path) / subj_file)
    X    = zscore(data["trial"], axis=0).astype(np.float32)
    return {
        "X":          X,
        "time":       data["time"],
        "subject_id": subj_file.split("_")[1],
    }
