# -*- coding: utf-8 -*-
"""
EEG Emotional Memory Classification - Improved Pipeline

Improvements over the starter:
1. Multi-band power features: delta, theta, alpha, beta, gamma
2. Time-windowed features (early / mid / late / full epoch)
3. Cross-channel Phase-Locking Value (PLV) connectivity (theta band)
4. Regularised LDA with automatic Ledoit-Wolf shrinkage
5. RBF-SVM with StandardScaler
6. Ensemble model (LDA + SVM averaged probabilities)
7. Trial-level LOOCV (more stable than per-timepoint)
8. Final model trained on all data + Kaggle submission generator
"""

import numpy as np
import h5py
import os
import pandas as pd
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import zscore
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

print("All packages imported successfully")

# ======================================================================
# CONFIGURATION
# ======================================================================

if os.path.exists('/kaggle/input'):
    TRAIN_PATH = '/kaggle/input/predicting-emotions-using-brain-waves/training'
    TEST_PATH  = '/kaggle/input/predicting-emotions-using-brain-waves/testing'
else:
    TRAIN_PATH = './training'
    TEST_PATH  = './testing'

FS = 200  # sampling frequency (Hz)

FREQ_BANDS = {
    'delta': (1,  4),
    'theta': (4,  8),
    'alpha': (8,  12),
    'beta':  (12, 30),
    'gamma': (30, 45),
}

# Post-stimulus time windows (seconds)
TIME_WINDOWS = {
    'early': (0.00, 0.33),
    'mid':   (0.33, 0.67),
    'late':  (0.67, 1.00),
    'full':  (0.00, 1.00),
}

INCLUDE_PLV = True   # PLV connectivity features (theta band only)

# ======================================================================
# DATA LOADING
# ======================================================================

def load_hdf5_data(filepath):
    """Load HDF5 MATLAB v7.3 .mat file into a dict."""

    def _resolve(f, ref_or_dataset):
        val = ref_or_dataset[()]
        if isinstance(val, h5py.Reference):
            return f[val]
        if hasattr(val, 'shape') and val.shape == (1, 1):
            ref = val.item()
            if isinstance(ref, h5py.Reference):
                return f[ref]
            if isinstance(ref, bytes):
                ref = ref.decode('utf-8')
            return f[ref]
        return ref_or_dataset

    with h5py.File(filepath, 'r') as f:
        data_ref = f['data']

        # trialinfo (may be absent in public test files)
        try:
            trialinfo = np.array(_resolve(f, data_ref['trialinfo'])).T
        except (KeyError, ValueError, TypeError):
            trialinfo = None

        time_data  = np.array(_resolve(f, data_ref['time'])).flatten()
        trial_data = np.array(_resolve(f, data_ref['trial'])).T  # (trials, ch, time)

        # keep only post-stimulus (t >= 0)
        mask = time_data >= 0
        if np.any(~mask):
            time_data  = time_data[mask]
            trial_data = trial_data[:, :, mask]

    return {'trial': trial_data, 'trialinfo': trialinfo, 'time': time_data}

# ======================================================================
# SIGNAL PROCESSING HELPERS
# ======================================================================

def bandpass(data, lowcut, highcut, fs=FS, order=4):
    """Zero-phase Butterworth bandpass filter. data: (..., time)."""
    nyq   = 0.5 * fs
    b, a  = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)

def instantaneous_power(data, freq_band, fs=FS):
    """Hilbert-envelope power for a frequency band. Returns same shape as data."""
    return np.abs(hilbert(bandpass(data, *freq_band, fs=fs))) ** 2

# ======================================================================
# FEATURE EXTRACTION
# ======================================================================

def compute_plv(data, freq_band=None, fs=FS):
    """
    Theta-band Phase-Locking Value for every channel pair.

    Parameters
    ----------
    data : ndarray (trials, channels, time)

    Returns
    -------
    plv : ndarray (trials, n_pairs)
    """
    if freq_band is None:
        freq_band = FREQ_BANDS['theta']

    phase = np.angle(hilbert(bandpass(data, *freq_band, fs=fs)))
    n_trials, n_ch, _ = phase.shape

    pairs   = [(i, j) for i in range(n_ch) for j in range(i + 1, n_ch)]
    plv_mat = np.zeros((n_trials, len(pairs)), dtype=np.float32)

    for k, (i, j) in enumerate(pairs):
        diff           = phase[:, i, :] - phase[:, j, :]
        plv_mat[:, k]  = np.abs(np.mean(np.exp(1j * diff), axis=-1))

    return plv_mat


def extract_features(data, time_vector, include_plv=INCLUDE_PLV):
    """
    Build a fixed-size feature vector per trial.

    Features
    --------
    - Mean band power per channel, per band, per time window
      → 5 bands × 4 windows × n_channels
    - (optional) Theta-band PLV for all channel pairs

    Parameters
    ----------
    data        : ndarray (trials, channels, time)
    time_vector : ndarray (time,)
    include_plv : bool

    Returns
    -------
    features : ndarray (trials, n_features)
    """
    feat_blocks = []

    for band_name, freq_band in FREQ_BANDS.items():
        power = instantaneous_power(data, freq_band)   # (trials, ch, time)

        for win_name, (t0, t1) in TIME_WINDOWS.items():
            mask = (time_vector >= t0) & (time_vector < t1)
            if mask.sum() == 0:
                continue
            feat_blocks.append(power[:, :, mask].mean(axis=-1))  # (trials, ch)

    if include_plv:
        feat_blocks.append(compute_plv(data))

    return np.concatenate(feat_blocks, axis=1).astype(np.float32)

# ======================================================================
# ENSEMBLE MODEL  (Regularised LDA + RBF-SVM)
# ======================================================================

class EnsembleModel:
    """
    Combines a regularised LDA (Ledoit-Wolf shrinkage) and an RBF-SVM
    by averaging their predicted probabilities.
    """

    def __init__(self):
        self.lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
        self.svm = Pipeline([
            ('scaler', StandardScaler()),
            ('clf',    SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
                           random_state=42))
        ])
        self.classes_ = None

    def fit(self, X, y):
        X = np.nan_to_num(X)
        self.lda.fit(X, y)
        self.svm.fit(X, y)
        self.classes_ = self.lda.classes_
        return self

    def predict_proba(self, X):
        X = np.nan_to_num(X)
        return (self.lda.predict_proba(X) + self.svm.predict_proba(X)) / 2

# ======================================================================
# LOAD TRAINING DATA & EXTRACT FEATURES
# ======================================================================

print("=" * 70)
print("LOADING TRAINING DATA & EXTRACTING FEATURES")
print("=" * 70)

neu_path    = os.path.join(TRAIN_PATH, 'sleep_neu')
train_files = sorted([f for f in os.listdir(neu_path) if f.endswith('.mat')])
print(f"\nFound {len(train_files)} participants\n")

features_per_subj = []
labels_per_subj   = []
time_vector       = None

for subj_file in train_files:
    subj_id = subj_file.split('_')[1]
    print(f"  Participant {subj_id}...", end=' ', flush=True)

    neu = load_hdf5_data(os.path.join(TRAIN_PATH, 'sleep_neu', subj_file))
    emo = load_hdf5_data(os.path.join(TRAIN_PATH, 'sleep_emo', subj_file))

    combined = np.concatenate([neu['trial'], emo['trial']], axis=0)
    labels   = np.array([1] * len(neu['trial']) + [2] * len(emo['trial']))

    # z-score per participant (normalise across trials)
    combined = zscore(combined, axis=0)

    feats = extract_features(combined, neu['time'])
    features_per_subj.append(feats)
    labels_per_subj.append(labels)

    if time_vector is None:
        time_vector = neu['time']

    print(f"{len(labels)} trials | {feats.shape[1]} features")

print(f"\nFeature vector size : {features_per_subj[0].shape[1]}")
print(f"Total participants  : {len(features_per_subj)}")

# ======================================================================
# LEAVE-ONE-PARTICIPANT-OUT CROSS-VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("LEAVE-ONE-PARTICIPANT-OUT CROSS-VALIDATION")
print("=" * 70 + "\n")

n_subj = len(features_per_subj)
aucs   = []

for i in range(n_subj):
    X_test  = features_per_subj[i]
    y_test  = labels_per_subj[i]
    X_train = np.concatenate([features_per_subj[j] for j in range(n_subj) if j != i])
    y_train = np.concatenate([labels_per_subj[j]   for j in range(n_subj) if j != i])

    model = EnsembleModel()
    model.fit(X_train, y_train)

    pos_idx = list(model.classes_).index(2)   # class 2 = emotional
    proba   = model.predict_proba(X_test)[:, pos_idx]
    auc     = roc_auc_score(y_test == 2, proba)
    aucs.append(auc)

    print(f"  Participant {i + 1:2d}:  AUC = {auc:.4f}")

aucs = np.array(aucs)
print(f"\n  Mean AUC : {aucs.mean():.4f} ± {aucs.std():.4f}")
print(f"  Best AUC : {aucs.max():.4f}   Worst AUC : {aucs.min():.4f}")

# ======================================================================
# VISUALISE LOOCV RESULTS
# ======================================================================

colors = ['#2ecc71' if a >= 0.5 else '#e74c3c' for a in aucs]
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(range(1, n_subj + 1), aucs, color=colors, edgecolor='black', alpha=0.85)
ax.axhline(aucs.mean(), color='#3498db', linewidth=2, linestyle='--',
           label=f'Mean AUC = {aucs.mean():.4f}')
ax.axhline(0.5, color='gray', linewidth=1.5, linestyle=':',
           label='Chance (0.50)')
ax.set_xlabel('Participant', fontsize=12)
ax.set_ylabel('AUC', fontsize=12)
ax.set_title('LOOCV AUC per Participant\n(Ensemble: Regularised LDA + RBF-SVM)',
             fontsize=13, fontweight='bold')
ax.set_xticks(range(1, n_subj + 1))
ax.set_ylim([0.3, 1.0])
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('loocv_results.png', dpi=150)
plt.show()
print("Plot saved to loocv_results.png")

# ======================================================================
# TRAIN FINAL MODEL ON ALL DATA
# ======================================================================

print("\n" + "=" * 70)
print("TRAINING FINAL MODEL ON ALL PARTICIPANTS")
print("=" * 70)

X_all = np.concatenate(features_per_subj)
y_all = np.concatenate(labels_per_subj)

final_model = EnsembleModel()
final_model.fit(X_all, y_all)
print(f"Final model trained on {len(y_all)} trials.")

# ======================================================================
# GENERATE KAGGLE SUBMISSION (when test data is available)
# ======================================================================

if os.path.exists(TEST_PATH):
    print("\n" + "=" * 70)
    print("GENERATING TEST PREDICTIONS")
    print("=" * 70)

    test_files  = sorted([f for f in os.listdir(TEST_PATH) if f.endswith('.mat')])
    predictions = []

    for test_file in test_files:
        subj_id   = test_file.split('_')[1]
        print(f"  Processing test participant {subj_id}...", end=' ', flush=True)

        td        = load_hdf5_data(os.path.join(TEST_PATH, test_file))
        test_data = zscore(td['trial'], axis=0)
        feats     = extract_features(test_data, td['time'])

        pos_idx   = list(final_model.classes_).index(2)
        proba     = final_model.predict_proba(feats)[:, pos_idx]

        for trial_idx, p in enumerate(proba):
            predictions.append({
                'id':         f"{test_file.replace('.mat','')}_{trial_idx}",
                'prediction': p
            })

        print(f"{len(proba)} trials")

    submission = pd.DataFrame(predictions)
    submission.to_csv('submission.csv', index=False)
    print(f"\nSubmission saved: {len(submission)} rows → submission.csv")

else:
    print("\nNo ./testing directory found — skipping submission generation.")
    print("On Kaggle, test predictions will be generated automatically.")

print("\nDone!")
