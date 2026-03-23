"""
eeg.models.eegnet
=================
EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs.

This implementation closely follows the original paper architecture and is
sized for the competition data (200 Hz, variable channels, 1-second epochs).

Architecture overview
---------------------
Input : (batch, 1, n_channels, n_timepoints)

Block 1 — temporal + depthwise spatial convolution
  Conv2d(1 → F1, kernel=(1, K))          # temporal: captures oscillations ≥ fs/K Hz
  BatchNorm → DepthwiseConv(ch-wise)     # spatial: learns channel combinations
  BatchNorm → ELU → AvgPool(1,4) → Dropout

Block 2 — separable convolution
  DepthwiseConv(1,16) → PointwiseConv(F2)
  BatchNorm → ELU → AvgPool(1,8) → Dropout

Classifier
  Flatten → Linear(n_flat → n_classes)

References
----------
Lawhern, V.J. et al. (2018). EEGNet: A Compact Convolutional Neural Network
for EEG-based Brain-Computer Interfaces. Journal of Neural Engineering, 15(5).
https://doi.org/10.1088/1741-2552/aace8c
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from eeg.utils.optional_deps import require_torch, _TORCH_AVAILABLE

if _TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# EEGNet architecture
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class EEGNet(nn.Module):
        """
        EEGNet for EEG classification.

        Parameters
        ----------
        n_channels    : number of EEG channels
        n_timepoints  : number of time samples per trial
        n_classes     : number of output classes (default 2)
        f1            : number of temporal filters in Block 1 (default 8)
        d             : depth multiplier for depthwise conv (default 2)
        f2            : number of pointwise filters in Block 2 (default f1*d)
        kernel_length : temporal conv kernel size (default 64 ≈ fs/3 at 200 Hz,
                        captures oscillations down to ~3 Hz)
        dropout       : dropout probability (default 0.25)
        """

        def __init__(
            self,
            n_channels: int   = 16,
            n_timepoints: int = 200,
            n_classes: int    = 2,
            f1: int           = 8,
            d: int            = 2,
            f2: int           = 16,
            kernel_length: int = 64,
            dropout: float     = 0.25,
        ) -> None:
            super().__init__()

            # ----------------------------------------------------------------
            # Block 1: temporal convolution + depthwise spatial convolution
            # ----------------------------------------------------------------
            self.block1 = nn.Sequential(
                # Temporal filter — "same" padding keeps time dimension intact
                nn.Conv2d(1, f1, kernel_size=(1, kernel_length),
                          padding=(0, kernel_length // 2), bias=False),
                nn.BatchNorm2d(f1),

                # Depthwise spatial filter — one spatial filter per temporal filter
                nn.Conv2d(f1, f1 * d, kernel_size=(n_channels, 1),
                          groups=f1, bias=False),
                nn.BatchNorm2d(f1 * d),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 4)),
                nn.Dropout(dropout),
            )

            # ----------------------------------------------------------------
            # Block 2: separable convolution (depthwise + pointwise)
            # ----------------------------------------------------------------
            self.block2 = nn.Sequential(
                # Depthwise temporal convolution
                nn.Conv2d(f1 * d, f1 * d, kernel_size=(1, 16),
                          padding=(0, 8), groups=f1 * d, bias=False),
                # Pointwise: mix channels
                nn.Conv2d(f1 * d, f2, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(f2),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 8)),
                nn.Dropout(dropout),
            )

            # ----------------------------------------------------------------
            # Classifier — size computed dynamically to avoid hard-coding
            # ----------------------------------------------------------------
            with torch.no_grad():
                dummy  = torch.zeros(1, 1, n_channels, n_timepoints)
                n_flat = self.block2(self.block1(dummy)).numel()

            self.classifier = nn.Linear(n_flat, n_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.block1(x)
            x = self.block2(x)
            x = x.flatten(1)
            return self.classifier(x)


# ---------------------------------------------------------------------------
# Trainer — wraps EEGNet with a fit / predict_proba interface
# ---------------------------------------------------------------------------

class EEGNetTrainer:
    """
    Training wrapper for EEGNet that exposes the same interface as the
    classical models so the LOOCV engine can treat them identically.

    Parameters
    ----------
    n_channels    : EEG channel count
    n_timepoints  : time samples per epoch
    n_classes     : number of classes (default 2)
    n_epochs      : maximum training epochs (default 150)
    lr            : Adam learning rate (default 1e-3)
    weight_decay  : L2 regularisation (default 1e-4)
    batch_size    : mini-batch size (default 64)
    patience      : early-stopping patience in epochs (default 20)
    device        : ``'auto'`` (GPU if available, else CPU), ``'cpu'``, or
                    ``'cuda'``
    **model_kwargs: forwarded to :class:`EEGNet`
    """

    def __init__(
        self,
        n_channels: int    = 16,
        n_timepoints: int  = 200,
        n_classes: int     = 2,
        n_epochs: int      = 150,
        lr: float          = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int    = 64,
        patience: int      = 20,
        device: str        = "auto",
        **model_kwargs,
    ) -> None:
        require_torch("EEGNetTrainer")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._device        = torch.device(device)
        self._n_channels    = n_channels
        self._n_timepoints  = n_timepoints
        self._n_classes     = n_classes
        self._n_epochs      = n_epochs
        self._lr            = lr
        self._weight_decay  = weight_decay
        self._batch_size    = batch_size
        self._patience      = patience
        self._model_kwargs  = model_kwargs
        self.classes_: Optional[np.ndarray] = None
        self._model: Optional["EEGNet"]     = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_tensor(self, X: np.ndarray) -> "torch.Tensor":
        """Convert (n_trials, ch, time) → (n_trials, 1, ch, time) float32."""
        t = torch.from_numpy(np.nan_to_num(X)).float()
        if t.dim() == 3:
            t = t.unsqueeze(1)   # add channel-of-channels dim
        return t.to(self._device)

    def _make_loader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool
    ) -> "DataLoader":
        dataset = TensorDataset(
            self._to_tensor(X),
            torch.from_numpy(y).long().to(self._device),
        )
        return DataLoader(dataset, batch_size=self._batch_size, shuffle=shuffle)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "EEGNetTrainer":
        """
        Train EEGNet.

        Parameters
        ----------
        X_train, y_train : training data and labels
        X_val, y_val     : optional validation data for early stopping
        """
        n_ch = X_train.shape[1]
        n_tp = X_train.shape[2]

        self._model = EEGNet(
            n_channels=n_ch,
            n_timepoints=n_tp,
            n_classes=self._n_classes,
            **self._model_kwargs,
        ).to(self._device)

        self.classes_ = np.unique(y_train)

        optimizer  = torch.optim.Adam(
            self._model.parameters(), lr=self._lr, weight_decay=self._weight_decay
        )
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self._n_epochs
        )
        criterion  = nn.CrossEntropyLoss()
        train_loader = self._make_loader(X_train, y_train, shuffle=True)

        best_val_loss = float("inf")
        no_improve    = 0
        best_state    = None

        for epoch in range(1, self._n_epochs + 1):
            # --- train ---
            self._model.train()
            for Xb, yb in train_loader:
                optimizer.zero_grad()
                loss = criterion(self._model(Xb), yb)
                loss.backward()
                optimizer.step()
            scheduler.step()

            # --- optional early stopping ---
            if X_val is not None and y_val is not None:
                self._model.eval()
                with torch.no_grad():
                    Xv   = self._to_tensor(X_val)
                    yv   = torch.from_numpy(y_val).long().to(self._device)
                    vloss = criterion(self._model(Xv), yv).item()

                if vloss < best_val_loss - 1e-4:
                    best_val_loss = vloss
                    best_state    = {k: v.cpu().clone() for k, v in
                                     self._model.state_dict().items()}
                    no_improve    = 0
                else:
                    no_improve += 1

                if no_improve >= self._patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(
                {k: v.to(self._device) for k, v in best_state.items()}
            )

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n_trials, n_classes) probability array."""
        assert self._model is not None, "Call fit() before predict_proba()."
        self._model.eval()
        with torch.no_grad():
            logits = self._model(self._to_tensor(X))
            proba  = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def __repr__(self) -> str:
        return (
            f"EEGNetTrainer(n_epochs={self._n_epochs}, "
            f"lr={self._lr}, device={self._device})"
        )

else:
    # Stub class when torch is not installed
    class EEGNetTrainer:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            require_torch("EEGNetTrainer")
