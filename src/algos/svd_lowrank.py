"""SVD-based low-rank approximation algorithms.

Implements both NumPy-based and Pure Python SVD for low-rank matrix approximation.
"""

import numpy as np


def numpy_svd_lowrank(A, rank):
    """
    Low-rank approximation using NumPy's SVD.

    This is the standard truncated SVD approach for optimal low-rank approximation
    (Eckart-Young-Mirsky theorem). Fast and accurate baseline.

    Args:
        A: (m x n) numpy array - input matrix
        rank: int - desired rank for approximation

    Returns:
        U: (m x rank) numpy array - left singular vectors
        S: (rank,) numpy array - singular values
        Vt: (rank x n) numpy array - right singular vectors (transposed)

    Raises:
        ValueError: If rank is invalid (< 1 or >= min(m, n))
    """
    m, n = A.shape

    # Validate rank
    if rank < 1:
        raise ValueError(f"Rank must be at least 1, got {rank}")
    if rank >= min(m, n):
        raise ValueError(f"Rank must be less than min(m, n) = {min(m, n)}, got {rank}")

    # Compute full SVD with efficient mode
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Truncate to desired rank
    U = U[:, :rank]
    S = S[:rank]
    Vt = Vt[:rank, :]

    return U, S, Vt


def python_svd_lowrank(A, rank, max_iters=10000, tol=1e-6):
    """
    Low-rank approximation using pure Python SVD implementation.

    Implements SVD from scratch using power iteration with deflation.
    Much slower than NumPy SVD but educational and demonstrates the algorithm.

    Args:
        A: (m x n) numpy array - input matrix
        rank: int - desired rank for approximation
        max_iters: int - maximum iterations for power iteration (default: 1000)
        tol: float - convergence tolerance (default: 1e-10)

    Returns:
        U: (m x rank) numpy array - left singular vectors
        S: (rank,) numpy array - singular values
        Vt: (rank x n) numpy array - right singular vectors (transposed)

    Raises:
        ValueError: If rank is invalid
        RuntimeError: If power iteration fails to converge
    """
    m, n = A.shape

    # Validate rank
    if rank < 1:
        raise ValueError(f"Rank must be at least 1, got {rank}")
    if rank >= min(m, n):
        raise ValueError(f"Rank must be less than min(m, n) = {min(m, n)}, got {rank}")

    # Initialize storage for singular vectors and values
    U = np.zeros((m, rank))
    S = np.zeros(rank)
    Vt = np.zeros((rank, n))

    # Work on a copy of A since we'll deflate it
    A_work = A.copy()

    # Compute singular triplets one at a time using deflation
    for k in range(rank):
        # Find dominant singular triplet using power iteration
        u, sigma, v = _power_iteration(A_work, max_iters, tol)

        # Store the singular triplet
        U[:, k] = u
        S[k] = sigma
        Vt[k, :] = v

        # Deflate: remove the dominant component from A_work
        A_work = _deflate_matrix(A_work, u, sigma, v)

    return U, S, Vt


def _power_iteration(A, max_iters=1000, tol=1e-10):
    """
    Power iteration to find dominant singular vector pair.

    Uses power iteration on A^T A to find the dominant right singular vector,
    then computes the corresponding left singular vector and singular value.

    Args:
        A: (m x n) numpy array
        max_iters: int - maximum iterations
        tol: float - convergence tolerance

    Returns:
        u: (m,) numpy array - dominant left singular vector
        sigma: float - dominant singular value
        v: (n,) numpy array - dominant right singular vector

    Raises:
        RuntimeError: If iteration fails to converge
    """
    m, n = A.shape

    # Initialize random right singular vector
    np.random.seed(None)  # Use different seed each time for robustness
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    # Power iteration: v = A^T A v, normalized
    for iteration in range(max_iters):
        v_old = v.copy()

        # Compute A^T (A v) for numerical stability (not (A^T A) v)
        Av = A @ v
        v = A.T @ Av

        # Normalize
        v_norm = np.linalg.norm(v)
        if v_norm < tol:
            raise RuntimeError(f"Power iteration failed: v_norm = {v_norm} < {tol}")
        v = v / v_norm

        # Check convergence
        diff = np.linalg.norm(v - v_old)
        if diff < tol:
            break
    else:
        # Did not converge within max_iters
        raise RuntimeError(
            f"Power iteration did not converge after {max_iters} iterations"
        )

    # Compute corresponding left singular vector and singular value
    Av = A @ v
    sigma = np.linalg.norm(Av)

    if sigma < tol:
        # Matrix is effectively rank-deficient at this level
        u = np.zeros(m)
        sigma = 0.0
    else:
        u = Av / sigma

    return u, sigma, v


def _deflate_matrix(A, u, sigma, v):
    """
    Remove dominant singular triplet from matrix.

    Computes A_deflated = A - sigma * u * v^T, which removes the
    dominant singular component from the matrix.

    Args:
        A: (m x n) numpy array
        u: (m,) numpy array - left singular vector to remove
        sigma: float - singular value
        v: (n,) numpy array - right singular vector to remove

    Returns:
        A_deflated: (m x n) numpy array - A with dominant component removed
    """
    # A_deflated = A - sigma * outer(u, v)
    return A - sigma * np.outer(u, v)
