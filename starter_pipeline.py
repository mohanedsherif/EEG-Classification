# -*- coding: utf-8 -*-
"""Starter_pipeline

This is part of the Kaggle competition of the detailed AI engineering course (AL Bdaya):
www.aibdaya.com
"""


"""# EEG Emotional Memory Classification - Starter Notebook

Welcome to the EEG Emotional Memory Reactivation Competition!

This notebook demonstrates:
1. Loading and processing training data
2. Extracting theta power features (4-8 Hz)
3. Leave-one-participant-out cross-validation (for local validation)
4. Training final model on all training data
5. Loading and processing test data
6. Making predictions on test set
7. Creating valid submission file

## 1. Setup and Imports
"""

import numpy as np
import h5py
import os
import pandas as pd
import glob
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import zscore
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
print("All packages imported successfully")

"""## 2. Configuration

On Kaggle, data is in `/kaggle/input/dataset-name/`

For local testing, adjust paths accordingly.
"""

# data paths
if os.path.exists('/kaggle/input'):
    # running on Kaggle
    TRAIN_PATH = '/kaggle/input/predicting-emotions-using-brain-waves/training'
    TEST_PATH = '/kaggle/input/predicting-emotions-using-brain-waves/testing'  # will exist on Kaggle when submitting
else:
    # running locally (if you want to download the notebook and work on your machine)
    TRAIN_PATH = './training'
    TEST_PATH = './testing'  # for local testing, pointing to the generated public data

"""## 3. Data Loading and Feature Extraction

we'll define the functions to load HDF5 files and extract power features
"""

def load_hdf5_data(filepath):
    """
    load HDF5 MATLAB v7.3 file

    Parameters:
    -----------
    filepath : str
        path to .mat file

    Returns:
    --------
    data_dict : dict
        dictionary with 'trial', 'trialinfo', 'time'
    """
    def load_field(f, data_ref, field_name):
        """helper to load HDF5 field handling references"""
        field = data_ref[field_name]

        if isinstance(field, h5py.Dataset):
            ref_value = field[()]

            if isinstance(ref_value, h5py.Reference):
                return f[ref_value]
            elif hasattr(ref_value, 'shape') and ref_value.shape == (1, 1):
                ref = ref_value.item()
                if isinstance(ref, h5py.Reference):
                    return f[ref]
                else:
                    if isinstance(ref, bytes):
                        ref = ref.decode('utf-8')
                    return f[ref]
            else:
                return field
        else:
            return field

    with h5py.File(filepath, 'r') as f:
        data_ref = f['data']

        trial_data = load_field(f, data_ref, 'trial')
        try:
            trialinfo_data = load_field(f, data_ref, 'trialinfo')
            trialinfo = np.array(trialinfo_data).T
        except (KeyError, ValueError, TypeError):
            # trialinfo might be missing in public test data
            trialinfo = None

        time_data = np.array(load_field(f, data_ref, 'time')).flatten()
        trial_data = np.array(load_field(f, data_ref, 'trial')).T # (Trials, Channels, Time)

        # enforce 0 to 1s trim (in case input data is not trimmed)
        # we only keep time >= 0
        mask = time_data >= 0

        if np.any(~mask): # if there are negative timepoints
            time_data = time_data[mask]
            trial_data = trial_data[:, :, mask]
            # trialinfo is per trial, not per timepoint, so it stays same

        return {
            'trial': trial_data,
            'trialinfo': trialinfo,
            'time': time_data
        }

print("Data loading function defined")

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    apply Butterworth bandpass filter
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)


def extract_hilbert_power(data, freq_band=(4, 8), fs=200):
    """
    extract instantaneous power using Hilbert transform

    Parameters:
    -----------
    data : ndarray
        EEG data with shape (trials, channels, timepoints)
    freq_band : tuple
        frequency band to analyze (default: 4-8 Hz theta)
    fs : float
        sampling frequency (default: 200 Hz)

    Returns:
    --------
    power : ndarray
        instantaneous power, same shape as input
    """
    n_trials, n_channels, n_timepoints = data.shape

    # bandpass filter
    data_filtered = np.zeros_like(data)
    for trial in range(n_trials):
        for ch in range(n_channels):
            data_filtered[trial, ch, :] = butter_bandpass_filter(
                data[trial, ch, :], freq_band[0], freq_band[1], fs
            )

    # hilbert transform
    analytic_signal = np.zeros(data_filtered.shape, dtype=complex)
    for trial in range(n_trials):
        for ch in range(n_channels):
            analytic_signal[trial, ch, :] = hilbert(data_filtered[trial, ch, :])

    # extract power
    power = np.abs(analytic_signal) ** 2

    return power

print("Feature extraction functions defined")

"""## 4. Load Training Data

load all training participants (14 participants)
"""

print("="*70)
print("LOADING TRAINING DATA")
print("="*70)

# find all training files
neu_path = os.path.join(TRAIN_PATH, 'sleep_neu')
train_files = sorted([f for f in os.listdir(neu_path) if f.endswith('.mat')])

print(f"\nFound {len(train_files)} training participants")

train_data_list = []
train_labels_list = []
train_counts = []
time_vector = None

for subj_file in train_files:
    subj_id = subj_file.split('_')[1]
    print(f"Loading participant {subj_id}...")

    # load neutral and emotional
    neu_data = load_hdf5_data(os.path.join(TRAIN_PATH, 'sleep_neu', subj_file))
    emo_data = load_hdf5_data(os.path.join(TRAIN_PATH, 'sleep_emo', subj_file))

    # set labels
    # use 1 for Neutral, 2 for Emotional
    if neu_data['trialinfo'] is not None:
        neu_data['trialinfo'][:, 0] = 1
    if emo_data['trialinfo'] is not None:
        emo_data['trialinfo'][:, 0] = 2

    # combine
    combined_trials = np.concatenate([neu_data['trial'], emo_data['trial']], axis=0)

    # handle labels concatenation
    if neu_data['trialinfo'] is not None and emo_data['trialinfo'] is not None:
        combined_labels = np.concatenate([
            neu_data['trialinfo'][:, 0],
            emo_data['trialinfo'][:, 0]
        ], axis=0)
    else:
        # should not happen for training data
        raise ValueError(f"Missing trialinfo for training subject {subj_id}")

    # extract power features
    power_features = extract_hilbert_power(combined_trials)

    # z-score per participant
    power_zscore = zscore(power_features, axis=0)

    train_data_list.append(power_zscore)
    train_labels_list.append(combined_labels)
    train_counts.append(len(combined_labels))

    if time_vector is None:
        time_vector = neu_data['time']

    print(f"{len(combined_labels)} trials")

# aggregate all training data
train_data_all = np.concatenate(train_data_list, axis=0)
train_labels_all = np.concatenate(train_labels_list, axis=0)

print(f"\n{'='*70}")
print(f"Total training: {train_data_all.shape[0]} trials from {len(train_counts)} participants")
print(f"Data shape: {train_data_all.shape}")
print(f"Time range: {time_vector[0]:.3f}s to {time_vector[-1]:.3f}s")
print(f"Neutral trials: {np.sum(train_labels_all==1)}")
print(f"Emotional trials: {np.sum(train_labels_all==2)}")
print("="*70)

"""## 5. Leave-One-Participant-Out Cross-Validation

**Purpose:** estimate performance on unseen participants (local validation)

**Note:** this is for validation only. The final submission will use ALL training data.

first we define the classification function:
"""

def classify_timepoint(train_data, train_labels, test_data, test_labels, timepoint_idx):
    """
    classify at a single timepoint using LDA

    Parameters:
    -----------
    train_data : ndarray
        training data (trials, channels, timepoints)
    train_labels : ndarray
        training labels (1=neutral, 2=emotional)
    test_data : ndarray
        test data (trials, channels, timepoints)
    test_labels : ndarray
        test labels
    timepoint_idx : int
        timepoint index to classify

    Returns:
    --------
    auc : float
        area Under ROC Curve
    """
    # extract features at this timepoint
    X_train = train_data[:, :, timepoint_idx]
    X_test = test_data[:, :, timepoint_idx]
 

    # replace NaN with 0
    X_train = np.nan_to_num(X_train, 0)
    X_test = np.nan_to_num(X_test, 0)

    # train classifier
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, train_labels)

    # predict
    try:
        y_pred_proba = clf.predict_proba(X_test)
        auc = roc_auc_score(test_labels, y_pred_proba[:, 1])
    except:
        auc = 0.5  # return chance if can't compute

    return auc

print("Classification function defined")

"""now run the cross-validation:"""

print("="*70)
print("LEAVE-ONE-OUT CROSS-VALIDATION (For Local Validation)")
print("="*70)

# participant boundaries
ranges = np.concatenate([[0], np.cumsum(train_counts)])
n_train = len(train_counts)
n_timepoints = train_data_all.shape[2]

loocv_auc = np.zeros((n_train, n_timepoints))

print(f"\nValidating {n_train} participants at {n_timepoints} timepoints...")
print("This may take a few minutes...\n")

for i in range(n_train):
    print(f"Validating participant {i+1}/{n_train}...", end=' ')

    # split train/test
    test_idx = np.arange(ranges[i], ranges[i+1], dtype=int)
    train_idx = np.concatenate([
        np.arange(0, ranges[i], dtype=int),
        np.arange(ranges[i+1], len(train_labels_all), dtype=int)
    ])

    test_data = train_data_all[test_idx]
    test_labels = train_labels_all[test_idx]
    train_data = train_data_all[train_idx]
    train_labels = train_labels_all[train_idx]

    # classify at each timepoint
    for t in range(n_timepoints):
        loocv_auc[i, t] = classify_timepoint(
            train_data, train_labels,
            test_data, test_labels, t
        )

    print(f"done")

 

"""## 6. Visualize LOOCV Results

this shows the expected performance on unseen participants.
"""

mean_auc_loocv = loocv_auc.mean(axis=0)
sem_auc_loocv = loocv_auc.std(axis=0) / np.sqrt(n_train)

plt.figure(figsize=(8, 5))
plt.plot(time_vector, mean_auc_loocv, 'b-', linewidth=2, label='Mean AUC (LOOCV)')
plt.fill_between(time_vector, mean_auc_loocv - sem_auc_loocv, mean_auc_loocv + sem_auc_loocv,
                 alpha=0.3, color='blue', label='SEM')
plt.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Chance')
plt.axvline(0, color='gray', linestyle='--', alpha=0.5, label='Cue onset')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.title('LOOCV Performance', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim([0.45, 0.6])
plt.show()


"""## Summary
 
this prevents overfitting to brief spikes and rewards sustained effects!

### Next Steps to Improve:

1. **Different frequency bands**: try alpha (8-12 Hz), beta (12-30 Hz) perhaps in combination with theta (4-8 Hz)
2. **Advanced features**: you could try time-frequency representation
3. **Other classifiers**: regularized LDA, KDE, Bayes, Random Forest, neural networks
4. **Feature selection**: identify most informative channels/features
5. **Domain Adaptation**: try to address the covariate shift  
6. **Ensemble methods**: combine multiple models


## Feel free to try ANY new method or model to reach the highest performance you could reach !


## بالتوفيق إن شاء الله

"""