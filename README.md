# EEG Emotional Memory Classification

**Kaggle Competition** — Predicting Emotions Using Brain Waves
*(Part of the AL Bdaya Detailed AI Engineering Course — [aibdaya.com](https://www.aibdaya.com))*

---

## Problem

Classify whether an EEG trial during sleep belongs to **neutral** or **emotional** memory reactivation.
Metric: **ROC-AUC** (area under the receiver-operating-characteristic curve).

---

## Dataset

| Split | Participants | Condition |
|-------|-------------|-----------|
| Train | 14 | `sleep_neu` + `sleep_emo` |
| Test  | hidden (Kaggle) | — |

Each `.mat` file contains multichannel EEG recordings sampled at **200 Hz**, trimmed to the 0–1 s post-stimulus window.

---

## Approach

### Starter baseline (`starter_pipeline.py`)
- Theta-band (4–8 Hz) Hilbert power at **each timepoint**
- LDA classifier (no regularisation)
- Leave-one-participant-out (LOPO) cross-validation

### Improved model (`improved_pipeline.py`)

#### Features
| Feature type | Detail |
|---|---|
| Multi-band power | delta (1–4), theta (4–8), alpha (8–12), beta (12–30), gamma (30–45) Hz |
| Time windows | Early (0–0.33 s), Mid (0.33–0.67 s), Late (0.67–1.0 s), Full epoch |
| PLV connectivity | Theta-band Phase-Locking Value for all channel pairs |

Total feature vector: **5 bands × 4 windows × N channels + N(N−1)/2 PLV pairs**

#### Models
| Model | Key settings |
|---|---|
| Regularised LDA | `solver='eigen'`, `shrinkage='auto'` (Ledoit-Wolf) |
| RBF-SVM | `StandardScaler` + `SVC(kernel='rbf', C=1.0, gamma='scale')` |
| **Ensemble** | Average of LDA + SVM predicted probabilities |

#### Evaluation
Trial-level LOPO cross-validation (leave one participant out, predict all their trials at once).

---

## Results

| Model | Mean LOOCV AUC |
|---|---|
| Starter (LDA, theta only) | ~0.54 |
| **Improved (Ensemble, multi-band + PLV)** | **TBD after run** |

---

## How to Run

### 1. Install dependencies
```bash
python -m pip install numpy h5py pandas scipy scikit-learn matplotlib
```

### 2. Prepare data
Place the competition data as:
```
EEG/
├── training/
│   ├── sleep_neu/   ← S_2_cleaned.mat … S_17_cleaned.mat
│   └── sleep_emo/   ← S_2_cleaned.mat … S_17_cleaned.mat
└── testing/         ← (Kaggle only, or local public test set)
```

### 3. Run
```bash
# Baseline (starter)
python starter_pipeline.py

# Improved model
python improved_pipeline.py
```

Outputs:
- Console: per-participant AUC + summary stats
- `loocv_results.png` — bar chart of LOOCV AUC per participant
- `submission.csv` — Kaggle submission file (only when `./testing` exists)

---

## Project Structure

```
EEG/
├── starter_pipeline.py     # Course-provided baseline
├── improved_pipeline.py    # Improved model (this work)
├── requirements.txt
└── README.md
```

---

## Ideas for Further Improvement

- **More classifiers**: Random Forest, XGBoost, MLP neural network
- **Feature selection**: Select most discriminative channels / pairs
- **Domain adaptation**: Address subject-to-subject covariate shift
- **Deep learning**: EEGNet or ShallowConvNet on raw signals
- **Cross-frequency coupling**: phase-amplitude coupling between bands
- **Riemannian geometry**: covariance-matrix classifiers (very strong on EEG)

---

## References

- Delorme, A. & Makeig, S. (2004). EEGLAB: an open source toolbox for analysis of single-trial EEG dynamics. *Journal of Neuroscience Methods*, 134(1), 9–21.
- Lotte, F. et al. (2018). A review of classification algorithms for EEG-based brain-computer interfaces. *Journal of Neural Engineering*, 15(3).
