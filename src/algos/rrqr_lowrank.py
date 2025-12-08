import time
import numpy as np
from scipy.linalg import qr as scipy_qr

def rrqr_lowrank(A, k):
    # The original function calculates timing and error, which will be handled by AlgorithmBenchmark.
    # We only need to return the approximated matrix Ak.
    Q, R, piv = scipy_qr(A, pivoting=True)
    # Form Ak = Q[:, :k] R[:k, :]
    Ak = Q[:, :k] @ R[:k, :]
    return Ak
