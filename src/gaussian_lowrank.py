"""Gaussian (Greedy Cross) Low-Rank Approximation.

Implements a greedy cross approximation algorithm that iteratively selects
pivot rows and columns to build a low-rank approximation using outer products.

Note: This algorithm is based on the cross approximation method for images
and works well for non-negative matrices. When compared to SVD-based methods,
it provides a trade-off between speed (fast greedy selection) and accuracy
(optimal spectral decomposition).
"""

import numpy as np


def gaussian_lowrank(A, rank):
    """
    Greedy cross approximation using maximum element pivoting.

    This algorithm greedily selects pivot elements and builds a rank-k
    approximation using outer products. It's designed for non-negative matrices
    (like images) and provides a fast alternative to SVD-based methods.

    Algorithm:
        1. Initialize approximation A_approx = 0
        2. For k = 1 to rank:
            a. Find location (i, j) of maximum absolute value in residual
            b. Get pivot value p = A[i, j]
            c. Add weighted outer product: A_approx += outer(A[:, j], A[i, :] / p)
            d. Update residual: A := A - outer product
        3. Return A_approx

    Args:
        A: (m x n) numpy array - non-negative input matrix
        rank: int - desired rank for approximation

    Returns:
        A_approx: (m x n) numpy array - rank-k approximation of A

    Raises:
        ValueError: If rank is invalid (< 1 or >= min(m, n))

    Notes:
        - Optimized for non-negative matrices (e.g., images)
        - Faster than SVD but may have higher approximation error
        - Demonstrates trade-off between speed and accuracy
    """
    m, n = A.shape

    # Validate rank
    if rank < 1:
        raise ValueError(f"Rank must be at least 1, got {rank}")
    if rank >= min(m, n):
        raise ValueError(f"Rank must be less than min(m, n) = {min(m, n)}, got {rank}")

    # Initialize
    A_work = A.copy()
    A_approx = np.zeros_like(A)

    # Greedy cross approximation
    for _ in range(rank):
        # Find element with maximum absolute value
        i, j = np.unravel_index(np.argmax(np.abs(A_work)), A_work.shape)
        pivot = A_work[i, j]

        # Check for near-zero pivot
        if np.abs(pivot) < 1e-14:
            break

        # Add rank-1 update: outer(column_j, row_i / pivot)
        rank1 = np.outer(A_work[:, j], A_work[i, :] / pivot)

        A_approx += rank1
        A_work -= rank1

    return A_approx
