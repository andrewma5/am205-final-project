import time
import numpy as np
from scipy.linalg import qr as scipy_qr

def rrqr_lowrank(A, k):
    """Rank-Revealing QR low-rank approximation.

    Uses QR factorization with column pivoting to compute a rank-k
    approximation of matrix A.
    """
    m, n = A.shape
    Q, R, piv = scipy_qr(A, pivoting=True)

    # Compute rank-k approximation in permuted space
    Ak_permuted = Q[:, :k] @ R[:k, :]

    # Unpivot: create inverse permutation and reorder columns
    Ak = np.zeros((m, n), dtype=Ak_permuted.dtype)
    Ak[:, piv] = Ak_permuted

    return Ak
