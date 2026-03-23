"""
eeg.utils.optional_deps
=======================
Guards for optional dependencies (torch, pyriemann).

Usage
-----
Import the boolean flags to branch at module level:

    from eeg.utils.optional_deps import _TORCH_AVAILABLE

    if _TORCH_AVAILABLE:
        import torch
        class RealModel(nn.Module): ...
    else:
        class RealModel:  # stub
            def __init__(self, *a, **kw): require_torch("RealModel")

Call ``require_torch`` / ``require_pyriemann`` inside methods to raise a
helpful error at use-time rather than import-time.
"""

from __future__ import annotations

try:
    import torch          # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import pyriemann       # noqa: F401
    _PYRIEMANN_AVAILABLE = True
except ImportError:
    _PYRIEMANN_AVAILABLE = False


def require_torch(feature: str = "this feature") -> None:
    """Raise an informative ImportError if PyTorch is not installed."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            f"{feature} requires PyTorch.\n"
            "Install with:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "Or for CUDA:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )


def require_pyriemann(feature: str = "this feature") -> None:
    """Raise an informative ImportError if pyriemann is not installed."""
    if not _PYRIEMANN_AVAILABLE:
        raise ImportError(
            f"{feature} requires pyriemann.\n"
            "Install with:  pip install pyriemann"
        )


def check_optional_deps() -> dict[str, bool]:
    """Return availability of all optional dependencies."""
    return {
        "torch":     _TORCH_AVAILABLE,
        "pyriemann": _PYRIEMANN_AVAILABLE,
    }
