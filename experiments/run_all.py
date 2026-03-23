"""
experiments/run_all.py
======================
Compare all implemented classifiers using Leave-One-Participant-Out
cross-validation and print a results table.

Usage
-----
    python experiments/run_all.py              # full run
    python experiments/run_all.py --fast       # quick smoke-test (30 epochs)
    python experiments/run_all.py --no-eegnet  # skip EEGNet
    python experiments/run_all.py --no-riemannian

The script degrades gracefully: models whose optional dependencies
(torch, pyriemann) are not installed are skipped with a clear message.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

# Ensure the project root is on sys.path when run from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving plots
import matplotlib.pyplot as plt

from eeg.data.loader import load_dataset
from eeg.models.classical import LDABaseline, ClassicalEnsemble
from eeg.training.loocv import run_loocv
from eeg.utils.metrics import format_results_table
from eeg.utils.optional_deps import check_optional_deps


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EEG classifier comparison via LOOCV"
    )
    p.add_argument("--config",          default="experiments/config.yaml")
    p.add_argument("--fast",            action="store_true",
                   help="Quick run: 30 epochs for EEGNet")
    p.add_argument("--no-eegnet",       action="store_true")
    p.add_argument("--no-riemannian",   action="store_true")
    p.add_argument("--no-classical",    action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_comparison(results, save_path: str = "experiments/results/comparison.png") -> None:
    """Bar chart comparing mean LOOCV AUC across all models."""
    names   = [r.model_name for r in results]
    means   = [r.mean_auc   for r in results]
    errs    = [r.std_auc    for r in results]
    colors  = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"][:len(results)]

    fig, ax = plt.subplots(figsize=(max(6, len(results) * 2), 5))
    bars = ax.bar(names, means, yerr=errs, color=colors, alpha=0.85,
                  edgecolor="black", capsize=5)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.5, label="Chance")
    ax.set_ylabel("Mean LOOCV AUC", fontsize=12)
    ax.set_title("Model Comparison — Leave-One-Participant-Out CV", fontsize=13, fontweight="bold")
    ax.set_ylim([0.4, max(means) + 0.1])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar, m, e in zip(bars, means, errs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + e + 0.005,
                f"{m:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    os.makedirs(Path(save_path).parent, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"  Comparison plot saved → {save_path}")


def plot_per_subject(results, save_path: str = "experiments/results/per_subject.png") -> None:
    """Line plot of per-subject AUC for each model."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
    markers = ["o", "s", "^", "D"]

    x_labels = results[0].subject_ids

    for r, col, mark in zip(results, colors, markers):
        ax.plot(range(len(r.per_subject_auc)), r.per_subject_auc,
                marker=mark, color=col, label=r.model_name, linewidth=1.5, markersize=6)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.5, label="Chance")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels([f"S{s}" for s in x_labels], fontsize=9)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_xlabel("Participant", fontsize=12)
    ax.set_title("Per-Participant LOOCV AUC", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(Path(save_path).parent, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"  Per-subject plot saved → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    deps = check_optional_deps()

    print("=" * 70)
    print("EEG EMOTIONAL MEMORY CLASSIFICATION — MODEL COMPARISON")
    print("=" * 70)
    print(f"\nOptional dependencies:")
    print(f"  torch     : {'✓ available' if deps['torch']     else '✗ not installed'}")
    print(f"  pyriemann : {'✓ available' if deps['pyriemann'] else '✗ not installed'}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print("="*70)
    subjects = load_dataset(cfg["data"]["train_path"])
    print(f"\n  {len(subjects)} participants loaded.")
    print(f"  Data shape: {subjects[0]['X'].shape}  (trials × channels × timepoints)")

    results = []

    # ------------------------------------------------------------------
    # 1. LDA Baseline
    # ------------------------------------------------------------------
    if not args.no_classical:
        print(f"\n{'='*70}")
        print("MODEL 1 — LDA Baseline (starter-pipeline equivalent)")
        print("="*70)
        results.append(run_loocv(subjects, lambda: LDABaseline(), verbose=True))

    # ------------------------------------------------------------------
    # 2. Classical Ensemble (LDA + SVM)
    # ------------------------------------------------------------------
    if not args.no_classical:
        print(f"\n{'='*70}")
        print("MODEL 2 — Ensemble (Regularised LDA + RBF-SVM, multi-band + PLV)")
        print("="*70)
        results.append(run_loocv(subjects, lambda: ClassicalEnsemble(), verbose=True))

    # ------------------------------------------------------------------
    # 3. Riemannian MDM
    # ------------------------------------------------------------------
    if not args.no_riemannian:
        if deps["pyriemann"]:
            from eeg.models.riemannian import RiemannianMDM
            print(f"\n{'='*70}")
            print("MODEL 3 — Riemannian MDM (SPD manifold)")
            print("="*70)
            results.append(
                run_loocv(subjects, lambda: RiemannianMDM(), use_raw=True, verbose=True)
            )
        else:
            print("\n  [skip] Riemannian MDM — install pyriemann: pip install pyriemann")

    # ------------------------------------------------------------------
    # 4. EEGNet
    # ------------------------------------------------------------------
    if not args.no_eegnet:
        if deps["torch"]:
            from eeg.models.eegnet import EEGNetTrainer
            eegnet_cfg = cfg["eegnet"].copy()
            if args.fast:
                eegnet_cfg["n_epochs"] = 30
                print("\n  [fast mode] EEGNet: n_epochs=30")

            print(f"\n{'='*70}")
            print(f"MODEL 4 — EEGNet (end-to-end CNN, {eegnet_cfg['n_epochs']} epochs)")
            print("="*70)
            results.append(
                run_loocv(
                    subjects,
                    lambda: EEGNetTrainer(**eegnet_cfg),
                    use_raw=True,
                    val_subject=cfg["loocv"]["val_subject"],
                    verbose=True,
                )
            )
        else:
            print(
                "\n  [skip] EEGNet — install PyTorch:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
            )

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    if not results:
        print("\nNo models ran. Check your arguments and optional dependencies.")
        return

    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print("="*70)
    print()
    print(format_results_table(results))

    # Save plots
    print()
    plot_comparison(results)
    if len(results) > 1:
        plot_per_subject(results)

    # Save JSON
    summary = [
        {
            "model":            r.model_name,
            "mean_auc":         round(r.mean_auc, 4),
            "std_auc":          round(r.std_auc, 4),
            "ci_95":            [round(r.ci_95[0], 4), round(r.ci_95[1], 4)],
            "per_subject_auc":  [round(a, 4) for a in r.per_subject_auc],
            "subject_ids":      r.subject_ids,
            "fit_time_s":       round(r.fit_time_seconds, 1),
        }
        for r in results
    ]
    out_path = "experiments/results/summary.json"
    os.makedirs("experiments/results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Full results saved → {out_path}")

    print(f"\n{'='*70}")
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    main()
