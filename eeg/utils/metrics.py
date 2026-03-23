"""
eeg.utils.metrics
=================
Evaluation metrics and result formatting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float     = 0.05,
    random_state: int = 42,
) -> tuple[float, float]:
    """
    Bootstrap 95% confidence interval for ROC-AUC.

    Parameters
    ----------
    y_true      : binary ground-truth labels
    y_score     : predicted probabilities for the positive class
    n_bootstrap : number of bootstrap resamples
    alpha       : significance level (default 0.05 → 95% CI)

    Returns
    -------
    (lower, upper) : CI bounds
    """
    rng  = np.random.RandomState(random_state)
    aucs = []
    for _ in range(n_bootstrap):
        idx   = resample(np.arange(len(y_true)), random_state=rng)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))

    aucs = np.array(aucs)
    lo   = np.percentile(aucs, 100 * alpha / 2)
    hi   = np.percentile(aucs, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


@dataclass
class LOOCVResult:
    """Container for leave-one-participant-out cross-validation results."""
    model_name:        str
    per_subject_auc:   list[float]
    subject_ids:       list[str]
    fit_time_seconds:  float = 0.0
    mean_auc:          float = field(init=False)
    std_auc:           float = field(init=False)
    ci_95:             tuple[float, float] = field(init=False)

    def __post_init__(self) -> None:
        a = np.array(self.per_subject_auc)
        self.mean_auc = float(a.mean())
        self.std_auc  = float(a.std())
        # Simple ±1.96 SE interval
        se = a.std() / np.sqrt(len(a))
        self.ci_95 = (float(a.mean() - 1.96 * se), float(a.mean() + 1.96 * se))


def format_results_table(results: list[LOOCVResult]) -> str:
    """
    Format a list of LOOCVResult objects as a markdown table.

    Returns a string ready to paste into README.md.
    """
    header = (
        "| Model | Mean AUC | Std | 95% CI | Best | Worst | Time (s) |\n"
        "|-------|----------|-----|--------|------|-------|----------|\n"
    )
    rows = []
    for r in results:
        ci   = f"[{r.ci_95[0]:.3f}, {r.ci_95[1]:.3f}]"
        best  = max(r.per_subject_auc)
        worst = min(r.per_subject_auc)
        rows.append(
            f"| {r.model_name} "
            f"| **{r.mean_auc:.4f}** "
            f"| {r.std_auc:.4f} "
            f"| {ci} "
            f"| {best:.4f} "
            f"| {worst:.4f} "
            f"| {r.fit_time_seconds:.1f} |"
        )
    return header + "\n".join(rows)
