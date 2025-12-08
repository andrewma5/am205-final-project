"""Low-rank approximation algorithms.

This package contains various algorithms for computing low-rank approximations
of matrices, including SVD-based methods, randomized methods, and rank-revealing
factorizations.
"""

from .rsvd import rsvd
from .rsvd_srft import rsvd_srft
from .gaussian_lowrank import gaussian_lowrank
from .svd_lowrank import numpy_svd_lowrank, python_svd_lowrank
from .rrqr_lowrank import rrqr_lowrank

__all__ = [
    "rsvd",
    "rsvd_srft",
    "gaussian_lowrank",
    "numpy_svd_lowrank",
    "python_svd_lowrank",
    "rrqr_lowrank",
]
