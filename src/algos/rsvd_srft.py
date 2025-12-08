"""Randomized SVD with Subsampled Random Fourier Transform (SRFT).

This is a variant of randomized SVD that uses structured random matrices
(SRFT) instead of Gaussian random matrices for improved efficiency.
"""

import numpy as np
import numpy.fft
from scipy.linalg import qr
from numpy.linalg import svd


def rsvd_srft(A, k, p=10, q=0):
    """Randomized SVD with Subsampled Random Fourier Transform.

    Uses SRFT (Subsampled Randomized Fourier Transform) for the random
    projection instead of Gaussian random matrices. This can be faster
    for very large matrices due to the FFT.

    Args:
        A: (m x n) input matrix
        k: desired rank
        p: oversampling parameter (default: 10)
        q: number of power iterations (default: 0)

    Returns:
        U, S, Vt: Rank-k SVD factors
    """
    m, n = A.shape
    # SRFT matrix: Omega = D F R (subsampled)
    # Step 1: random diagonal sign matrix
    D = np.random.choice([1, -1], size=n)
    # Step 2: multiply A by D
    AD = A * D[np.newaxis, :]
    # Step 3: FFT along columns (F^T)
    AF = numpy.fft.fft(AD, axis=1)  # shape m x n
    # Step 4: Subsample k+p columns
    idx = np.random.choice(n, size=k + p, replace=False)
    Omega = AF[:, idx]  # m x (k+p)
    # Power iterations (reshape to real if necessary)
    Y = Omega
    for _ in range(q):
        Y = A @ (A.T @ Y)
    Q, _ = qr(Y, mode="economic")
    B = Q.T @ A
    Ub, S, Vt = svd(B, full_matrices=False)
    U = Q @ Ub
    return U, S, Vt
