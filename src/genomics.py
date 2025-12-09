import numpy as np
import pandas as pd
import time
import requests, gzip, io
from numpy.linalg import qr
from scipy.linalg import svd, qr as scipy_qr
from scipy.linalg import svdvals

# Import clean, fixed implementations from algos
from algos import rsvd_srft as algo_rsvd_srft
from algos import rsvd
from algos import rrqr_lowrank


# Spectral norm error
def spectral_error(A, U, S, Vt, k):
    # Ak = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    Ak = (U[:, :k] * S[:k]) @ Vt[:k, :]
    return np.linalg.norm(A - Ak, ord=2)  # 2-norm = spectral norm


# Best possible error (sigma_{k+1})
def best_error(A, k):
    s = svdvals(A)
    if k < len(s):
        return s[k]  # sigma_{k+1}, since s[0] = sigma_1
    else:
        return 0.0


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------


def frob_error(A, U, S, Vt, k):
    """Compute Frobenius error between A and its rank-k approx."""
    Ak = (U[:, :k] * S[:k]) @ Vt[:k, :]
    return np.linalg.norm(A - Ak, ord="fro")


def optimal_error(A, k):
    """Return optimal achievable error = sum of squares of tail singular values."""
    S = svd(A, full_matrices=False, compute_uv=False)
    return np.sqrt(np.sum(S[k:] ** 2)), S[k]  # Frobenius error, and sigma_{k+1}


# ------------------------------------------------------------
# (1) Direct SVD
# ------------------------------------------------------------


def truncated_svd(A, k):
    t0 = time.time()
    U, S, Vt = svd(A, full_matrices=False)
    t1 = time.time()
    err = frob_error(A, U, S, Vt, k)
    return t1 - t0, err


# ------------------------------------------------------------
# (2) rSVD (Gaussian)
# ------------------------------------------------------------


def rsvd_gaussian(A, k, p=10, q=1):
    t0 = time.time()

    # Use clean implementation from algos
    U, S, Vt = rsvd(A, k, n_oversamples=p, n_subspace_iters=q)

    t1 = time.time()
    err = frob_error(A, U, S, Vt, k)
    return t1 - t0, err


# ------------------------------------------------------------
# (3) rSVD-SRFT (Three q values: 0,1,4)
# ------------------------------------------------------------
def rsvd_srft(A, k, p=10, q=0, return_matrices=True):
    t0 = time.time()

    # Use fixed implementation from algos
    U, S, Vt = algo_rsvd_srft(A, k, p=p, q=q)

    t1 = time.time()

    if return_matrices:
        return t1 - t0, U, S, Vt
    else:
        err = spectral_error(A, U, S, Vt, k)
        return t1 - t0, err


def rrqr(A, k):
    """
    Rank-Revealing QR with column pivoting.
    Returns time and spectral norm error directly.
    """
    t0 = time.time()

    # Use fixed implementation from algos
    Ak = rrqr_lowrank(A, k)

    t1 = time.time()
    err = np.linalg.norm(A - Ak, ord=2)  # spectral norm of residual
    return t1 - t0, err


# ------------------------------------------------------------
# LOAD REAL GENE-EXPRESSION MATRIX (GSE2553)
# ------------------------------------------------------------


def load_gse2553():
    url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE2nnn/GSE2553/matrix/GSE2553_series_matrix.txt.gz"
    r = requests.get(url)
    r.raise_for_status()

    # Parse series_matrix.txt.gz into DataFrame
    with gzip.open(io.BytesIO(r.content), "rt", errors="ignore") as f:
        df = pd.read_csv(f, sep="\t", comment="!", index_col=0)

    # Convert all entries to numeric, drop non-numeric rows
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="any")

    # Convert to numpy array (genes × samples)
    A = df.to_numpy(dtype=np.float32)

    print("Loaded GSE2553 matrix shape:", A.shape)
    return A


# Example usage
A = load_gse2553()

print("HELLO")
print(A)

# ------------------------------------------------------------
# RUN EXPERIMENTS
# ------------------------------------------------------------

ks = [40, 60, 80, 100]  # low-rank approximations
q_values = [0, 1, 4]  # power iteration counts

results = []

# Compute full singular values once for best possible error
full_s = svdvals(A)

for k in ks:
    sigma_k1 = full_s[k] if k < len(full_s) else 0
    for q in q_values:
        t, U, S, Vt = rsvd_srft(A, k, p=10, q=q)
        err = spectral_error(A, U, S, Vt, k)
        results.append(
            {
                "k": k,
                "q": q,
                "time_sec": t,
                "spectral_error": err,
                "best_error": sigma_k1,
            }
        )

# RRQR separately
for k in ks:
    t, err = rrqr(A, k)
    results.append(
        {
            "k": k,
            "q": "RRQR",
            "time_sec": t,
            "spectral_error": err,
            "best_error": full_s[k] if k < len(full_s) else 0,
        }
    )

df_results = pd.DataFrame(results)
print(df_results)


"""
import matplotlib.pyplot as plt
import numpy as np

# Data
k = np.array([40, 60, 80, 100])

srft_q0 = np.array([162.82, 152.48, 137.05, 114.20])
srft_q1 = np.array([109.19, 92.24, 87.47, 81.83])
srft_q4 = np.array([88.78, 80.58, 74.08, 72.93])
rrqr     = np.array([222.84, 173.28, 152.10, 115.24])
best     = np.array([84.78, 72.83, 66.17, 60.20])

# Plot
plt.figure(figsize=(8, 5))

plt.plot(k, srft_q0, marker='o', label='SRFT q=0')
plt.plot(k, srft_q1, marker='o', label='SRFT q=1')
plt.plot(k, srft_q4, marker='o', label='SRFT q=4')
plt.plot(k, rrqr,    marker='o', label='RRQR')
plt.plot(k, best,    marker='o', label='Best Error (σ_{k+1})')

plt.xlabel("k", fontsize=12)
plt.ylabel("Spectral Error", fontsize=12)
plt.title("Low Rank Approximation Error Versus Rank", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.show()
"""
