# EEG Emotional Memory Classification

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A research-grade pipeline for classifying **emotional vs. neutral memory reactivation** from sleep EEG recordings, built for the Kaggle competition *Predicting Emotions Using Brain Waves*.

This project implements and compares four progressively more advanced classifiers:

| # | Model | Algorithm |
|---|-------|-----------|
| 1 | LDA Baseline | Theta-band Hilbert power + plain LDA |
| 2 | **Ensemble** | 5-band power + PLV + Regularised LDA / RBF-SVM |
| 3 | **Riemannian MDM** | Covariance matrices on the SPD manifold |
| 4 | **EEGNet** | End-to-end compact CNN (Lawhern et al., 2018) |

---

## The Problem

During sleep, the brain replays memories. EEG signals recorded during this process differ subtly between emotionally charged and neutral memories. The goal is to decode this difference from multichannel EEG with high accuracy — evaluated by ROC-AUC.

---

## Dataset

| Property | Value |
|----------|-------|
| Participants | 14 (training) |
| Channels | 16 |
| Sampling rate | 200 Hz |
| Epoch length | 1 second (post-stimulus, 200 timepoints) |
| Classes | 0 = neutral, 1 = emotional |
| Trials per subject | ~340–1494 (median ≈ 700) |

Data is in HDF5-based MATLAB v7.3 format (`.mat` files).

---

## Methods

### Features (Models 1 & 2)

**Band power** — For each of 5 canonical EEG frequency bands, the instantaneous power (Hilbert envelope²) is averaged over 4 post-stimulus time windows:

| Band | Range | Function |
|------|-------|----------|
| Delta | 1–4 Hz | Deep sleep, memory consolidation |
| Theta | 4–8 Hz | Hippocampal–cortical communication |
| Alpha | 8–12 Hz | Cortical inhibition, attention |
| Beta | 12–30 Hz | Active cognition |
| Gamma | 30–45 Hz | Local neural computation |

Time windows: Early (0–0.33 s), Mid (0.33–0.67 s), Late (0.67–1.0 s), Full epoch.

**Phase-Locking Value (PLV)** — Theta-band inter-channel synchrony for all channel pairs, measuring functional connectivity during memory replay.

Total feature vector: `5 bands × 4 windows × 16 channels + 16×15/2 PLV = 440 features`

---

### EEGNet Architecture

```
Input: (batch, 1, 16 channels, 200 timepoints)
│
├── Block 1 — Temporal + Depthwise Spatial Conv
│   ├── Conv2d(1→8, kernel=(1,64), padding=(0,32))   # temporal: captures ≥3 Hz
│   ├── BatchNorm → DepthwiseConv(8→16, kernel=(16,1)) # spatial: channel mixing
│   ├── BatchNorm → ELU → AvgPool(1,4) → Dropout
│   └── Output: (batch, 16, 1, 50)
│
├── Block 2 — Separable Convolution
│   ├── DepthwiseConv(16→16, kernel=(1,16))           # temporal refinement
│   ├── PointwiseConv(16→16, kernel=(1,1))            # channel mixing
│   ├── BatchNorm → ELU → AvgPool(1,8) → Dropout
│   └── Output: (batch, 16, 1, ~6)
│
└── Classifier: Flatten → Linear(96, 2)

Total parameters: ~4,700
```

The compact design (< 5k parameters) is key for small EEG datasets — reduces overfitting while still learning spatial and temporal patterns directly from raw signals.

---

### Riemannian MDM

EEG covariance matrices lie on the Riemannian manifold of Symmetric Positive Definite (SPD) matrices. The Minimum Distance to Mean (MDM) classifier:

1. Computes per-trial spatial covariance matrices `Σ ∈ ℝ^(C×C)`
2. Estimates a Riemannian mean for each class using the affine-invariant metric
3. Assigns a new trial to the nearest class mean

This approach is **rotation-invariant** and handles the non-Euclidean geometry of covariance matrices, making it state-of-the-art for many BCI tasks.

---

### Evaluation Protocol

**Leave-One-Participant-Out (LOPO) Cross-Validation** — the standard protocol for generalisation testing across subjects in BCI research:

- Held-out test set = all trials from one participant
- Training set = all trials from the remaining 13 participants
- Repeated 14 times (one fold per participant)
- Metric: ROC-AUC (threshold-independent)

Per-participant z-score normalisation is applied independently to each subject's own data — no data leakage between training and test participants.

---

## Results

> Run `python experiments/run_all.py` to populate this table with your results.

| Model | Mean AUC | Std | 95% CI | Best | Worst | Time (s) |
|-------|----------|-----|--------|------|-------|----------|
| LDABaseline | — | — | — | — | — | — |
| ClassicalEnsemble | — | — | — | — | — | — |
| RiemannianMDM | — | — | — | — | — | — |
| EEGNetTrainer | — | — | — | — | — | — |

---

## Installation

```bash
git clone https://github.com/mohanedsherif/EEG-Classification.git
cd EEG-Classification

# Core dependencies
python -m pip install -r requirements.txt

# Optional: deep learning (EEGNet)
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Optional: Riemannian geometry
python -m pip install pyriemann
```

Place competition data under:
```
EEG-Classification/
├── training/
│   ├── sleep_neu/    ← S_2_cleaned.mat … S_17_cleaned.mat
│   └── sleep_emo/    ← S_2_cleaned.mat … S_17_cleaned.mat
└── testing/          ← Kaggle only
```

---

## Running

```bash
# Compare all available models
python experiments/run_all.py

# Quick smoke-test (EEGNet at 30 epochs)
python experiments/run_all.py --fast

# Classical models only (no optional deps needed)
python experiments/run_all.py --no-eegnet --no-riemannian
```

Outputs:
- Console: per-participant AUC + summary table
- `experiments/results/comparison.png` — bar chart
- `experiments/results/per_subject.png` — per-subject line plot
- `experiments/results/summary.json` — full numeric results

---

## Project Structure

```
EEG-Classification/
│
├── eeg/                          # installable package
│   ├── data/
│   │   └── loader.py             # HDF5 loading, per-subject normalisation
│   ├── features/
│   │   └── spectral.py           # band power, PLV, covariance matrices
│   ├── models/
│   │   ├── classical.py          # LDABaseline, ClassicalEnsemble
│   │   ├── eegnet.py             # EEGNet + EEGNetTrainer [requires torch]
│   │   └── riemannian.py         # RiemannianMDM         [requires pyriemann]
│   ├── training/
│   │   └── loocv.py              # unified LOOCV engine
│   └── utils/
│       ├── optional_deps.py      # graceful import guards
│       └── metrics.py            # AUC, CI, results table
│
├── experiments/
│   ├── run_all.py                # model comparison entry point
│   ├── config.yaml               # all hyperparameters
│   └── results/                  # generated outputs (gitignored except .gitkeep)
│
├── tests/
│   └── test_features.py          # unit tests
│
├── starter_pipeline.py           # original course baseline (reference)
├── improved_pipeline.py          # first iteration (reference)
├── pyproject.toml
└── requirements.txt
```

---

## Ideas for Further Work

- **Cross-frequency coupling** — phase-amplitude coupling between theta and gamma
- **Source localisation** — project sensor space to source space (MNE-Python)
- **Domain adaptation** — Euclidean alignment or CORAL to reduce subject-to-subject shift
- **Transformer** — EEG-Conformer or vanilla attention on raw signals
- **Explainability** — SHAP values or channel-contribution maps

---

## References

1. Lawhern, V.J. et al. (2018). EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces. *Journal of Neural Engineering*, 15(5). https://doi.org/10.1088/1741-2552/aace8c

2. Barachant, A. et al. (2012). Multiclass Brain-Computer Interface Classification by Riemannian Geometry. *IEEE Transactions on Biomedical Engineering*, 59(4). https://doi.org/10.1109/TBME.2011.2172210

3. Lachaux, J.-P. et al. (1999). Measuring Phase Synchrony in Brain Signals. *Human Brain Mapping*, 8(4), 194–208.

4. Ledoit, O. & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2).
