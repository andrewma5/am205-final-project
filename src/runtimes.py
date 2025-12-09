import numpy as np
import time
import pandas as pd
from numpy.linalg import qr
from scipy.linalg import svd, qr as scipy_qr

# Import clean, fixed implementations from algos
from algos import rsvd_srft as algo_rsvd_srft
from algos import rsvd
from algos import rrqr_lowrank

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------


def frob_error(A, U, S, Vt, k):
    Ak = (U[:, :k] * S[:k]) @ Vt[:k, :]
    return np.linalg.norm(A - Ak, ord="fro")


# ------------------------------------------------------------
# (1) Direct SVD (full SVD, truncate)
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
# (3) rSVD with Randomized Fourier Transform (SRFT / FJLT)
# ------------------------------------------------------------


def rsvd_srft(A, k, p=10, q=0):
    t0 = time.time()

    # Use fixed implementation from algos
    U, S, Vt = algo_rsvd_srft(A, k, p=p, q=q)

    t1 = time.time()
    err = frob_error(A, U, S, Vt, k)
    return t1 - t0, err


# ------------------------------------------------------------
# (4) Rank-Revealing QR (RRQR) – using column pivoting
# ------------------------------------------------------------


def rrqr(A, k):
    t0 = time.time()

    # Use fixed implementation from algos (already returns full matrix)
    Ak = rrqr_lowrank(A, k)

    t1 = time.time()
    err = np.linalg.norm(A - Ak, "fro")
    return t1 - t0, err


# ------------------------------------------------------------
# RUN EXPERIMENT
# ------------------------------------------------------------

sizes = [1024, 2048, 4096]
ks = [10, 20, 40, 80, 160]

all_results = {}  # dictionary of {n: DataFrame}

for n in sizes:
    print(f"\n=== MATRIX SIZE {n} × {n} ===")
    np.random.seed(0)
    A = np.random.randn(n, n)

    rows = []  # rows of dataframe

    for k in ks:
        print(f"  k = {k}")

        methods = {
            "svd": truncated_svd,
            "rsvd": rsvd_gaussian,
            "rsvd_srft": rsvd_srft,
            "rrqr": rrqr,
        }

        for method_name, method_func in methods.items():
            t, err = method_func(A, k)
            rows.append({"k": k, "method": method_name, "time_sec": t, "error": err})

    df = pd.DataFrame(rows)
    all_results[n] = df

print("\nDONE. Access results using all_results[n], e.g. all_results[1024]")

# Per-size nicely formatted tables
for n, df in all_results.items():
    print(f"\n========================================")
    print(f"DETAILED RESULTS FOR MATRIX SIZE {n} × {n}")
    print("========================================")
    print(df.sort_values(["k", "method"]).to_string(index=False))

    # Pivot table: k × method for time
    time_pivot = df.pivot(index="k", columns="method", values="time_sec")
    err_pivot = df.pivot(index="k", columns="method", values="error")

    print("\n-- Time (seconds) --")
    print(time_pivot.to_string())

    print("\n-- Frobenius Error ||A - Ak||_F --")
    print(err_pivot.to_string())


# Save or print results
print("\nDONE. 'results' dictionary contains all timing and error data.")
