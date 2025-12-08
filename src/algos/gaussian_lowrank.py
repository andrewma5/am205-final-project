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
        1. Initialize column matrix C and row matrix R
        2. For k = 1 to rank:
            a. Find location (i, j) of maximum absolute value in residual
            b. Get pivot value p = A[i, j]
            c. Store column A[:, j] in C and row A[i, :]/p in R
            d. Update residual: A := A - outer(C[:, k], R[k, :])
        3. Return (C, S, R) where approximation = C @ diag(S) @ R

    Args:
        A: (m x n) numpy array - non-negative input matrix
        rank: int - desired rank for approximation

    Returns:
        Tuple of (U, S, Vt):
            U: (m x rank) numpy array - selected columns
            S: (rank,) numpy array - scaling factors (all ones)
            Vt: (rank x n) numpy array - selected scaled rows

    Raises:
        ValueError: If rank is invalid (< 1 or >= min(m, n))

    Notes:
        - Optimized for non-negative matrices (e.g., images)
        - Faster than SVD but may have higher approximation error
        - Demonstrates trade-off between speed and accuracy
        - Returns factors in SVD-like format for memory efficiency
    """
    m, n = A.shape

    # Validate rank
    if rank < 1:
        raise ValueError(f"Rank must be at least 1, got {rank}")
    if rank >= min(m, n):
        raise ValueError(f"Rank must be less than min(m, n) = {min(m, n)}, got {rank}")

    # Initialize
    A_work = A.copy()
    C = np.zeros((m, rank))  # Column matrix
    R = np.zeros((rank, n))  # Row matrix

    # Greedy cross approximation
    actual_rank = 0
    for k in range(rank):
        # Find element with maximum absolute value
        i, j = np.unravel_index(np.argmax(np.abs(A_work)), A_work.shape)
        pivot = A_work[i, j]

        # Check for near-zero pivot
        if np.abs(pivot) < 1e-14:
            break

        # Store column and scaled row
        C[:, k] = A_work[:, j]
        R[k, :] = A_work[i, :] / pivot

        # Update residual
        A_work -= np.outer(C[:, k], R[k, :])
        actual_rank += 1

    # Trim to actual rank if early stopping occurred
    if actual_rank < rank:
        C = C[:, :actual_rank]
        R = R[:actual_rank, :]

    # Return in SVD-like format (U, S, Vt)
    # S is all ones since scaling is already incorporated in R
    S = np.ones(actual_rank)

    return C, S, R


def gaussian_lowrank_lecture(A, rank):
    """
    Greedy cross approximation using maximum element pivoting (lecture version).

    This is the lecture implementation that differs from the main implementation
    in how it handles pivoting and normalization.

    Algorithm:
        1. Initialize approximation A_approx = 0
        2. For k = 1 to rank:
            a. Find overall maximum value in residual
            b. Find location (i, j) of maximum value in residual
            c. Add weighted outer product: A_approx += outer(A[:, j], A[i, :] / maxerr)
            d. Update residual: A := A - outer product
        3. Return A_approx

    Key differences from gaussian_lowrank:
        - Uses np.max(A_work) for normalization instead of the pivot element
        - Uses np.argmax(A_work) without absolute value
        - Simpler algorithm without early stopping check

    Args:
        A: (m x n) numpy array - input matrix
        rank: int - desired rank for approximation

    Returns:
        A_approx: (m x n) numpy array - rank-k approximation of A
    """
    A_work = A.copy()
    A_approx = np.zeros_like(A)

    for k in range(rank):
        maxerr = np.max(A_work)
        i, j = np.unravel_index(np.argmax(A_work), A_work.shape)
        Ak = np.outer(A_work[:, j], A_work[i, :] / maxerr)
        A_approx += Ak
        A_work -= Ak

    return A_approx


"""
Gaussian low rank from lecture

## this will load an image and convert to grayscale and flt.pt.
A = io.imread('lincoln.png')[:,:,0] # just take first channel since its already gray    
#A = color.rgb2gray(A), if you have a color image, uncomment this line              
A = np.array(A, dtype=float)

## initialize zeros and rank
Aapprox = np.zeros_like(A)
rankA = np.linalg.matrix_rank(A)   
sq = np.sqrt(A.size)                

## being the iterative approximation of A
for k in range(rankA + 1):
    maxerr = np.max(A)              
    if k % 600 == 0: ## 600 is the chosen_num here. this should print out a plot less than 6 times. 
        ## I chose this "600" value so you won't have to wait too long to get the idea. 
        err = norm(A, 'fro') / sq
        plt.imshow(Aapprox, cmap='gray')   ## displays the current rank approximation as an image
        plt.title(f'rank = {k},  rmserr = {err:.4f}')
        plt.draw()
        plt.pause(0.1) ## be patient as this generates several plots one after the other
        print("Be patient, I am working...")
    
    i, j = np.unravel_index(np.argmax(A, axis=None), A.shape)  
    Ak = np.outer(A[:, j], A[i, :] / maxerr)  
    Aapprox = Aapprox + Ak                    
    A = A - Ak                                
    
plt.show();
"""


if __name__ == "__main__":
    def test_function(func_name, func, matrix, matrix_type):
        """Test a low-rank approximation function across multiple ranks."""
        ranks = list(range(10, 260, 10))
        errors = []
        violations = []

        print(f"\n{'='*60}")
        print(f"=== Testing {func_name} with {matrix_type} ===")
        print(f"{'='*60}")
        print(f"Matrix size: {matrix.shape[0]}x{matrix.shape[1]}")
        print(f"Ranks tested: {ranks[0]}, {ranks[1]}, {ranks[2]}, ..., {ranks[-1]}")
        print()

        for rank in ranks:
            A_approx = func(matrix, rank)
            error = np.linalg.norm(matrix - A_approx, 'fro')
            errors.append(error)

            # Check for monotonic decrease
            if len(errors) > 1 and errors[-1] >= errors[-2]:
                violations.append((ranks[len(errors)-2], ranks[len(errors)-1],
                                 errors[-2], errors[-1]))

        is_monotonic = len(violations) == 0
        return ranks, errors, is_monotonic, violations

    def print_results(func_name, matrix_type, ranks, errors, is_monotonic, violations):
        """Print formatted test results."""
        print(f"Rank | Frobenius Error")
        print(f"-----|----------------")
        for rank, error in zip(ranks, errors):
            print(f"{rank:4d} | {error:.6f}")

        print()
        print(f"Monotonically decreasing: {is_monotonic}")

        if not is_monotonic:
            print(f"\nViolations found (error increased):")
            for r1, r2, e1, e2 in violations:
                print(f"  Rank {r1} -> {r2}: {e1:.6f} -> {e2:.6f} (increase: {e2-e1:.6f})")
        print()

    # Generate test matrices
    print("\n" + "="*60)
    print("GAUSSIAN LOW RANK APPROXIMATION TESTING")
    print("="*60)

    np.random.seed(42)
    matrix_random = np.random.rand(1000, 1000)

    np.random.seed(43)
    matrix_image = np.random.uniform(0, 255, (1000, 1000))

    # Generate low-rank matrices
    np.random.seed(44)
    # Create a true low-rank matrix: rank 50 matrix
    U = np.random.rand(1000, 50)
    V = np.random.rand(50, 1000)
    matrix_lowrank_pure = U @ V

    np.random.seed(45)
    # Create a low-rank matrix with noise
    U_noisy = np.random.rand(1000, 50)
    V_noisy = np.random.rand(50, 1000)
    noise = np.random.randn(1000, 1000) * 0.1  # Small noise
    matrix_lowrank_noisy = (U_noisy @ V_noisy) + noise

    # Track overall results
    all_monotonic = []

    # Test 1: gaussian_lowrank with random matrix (0-1)
    ranks, errors, is_monotonic, violations = test_function(
        "gaussian_lowrank", gaussian_lowrank, matrix_random, "Random Matrix (0-1)")
    print_results("gaussian_lowrank", "Random Matrix (0-1)",
                  ranks, errors, is_monotonic, violations)
    all_monotonic.append(is_monotonic)

    # Test 2: gaussian_lowrank with image-like matrix (0-255)
    ranks, errors, is_monotonic, violations = test_function(
        "gaussian_lowrank", gaussian_lowrank, matrix_image, "Image-like Matrix (0-255)")
    print_results("gaussian_lowrank", "Image-like Matrix (0-255)",
                  ranks, errors, is_monotonic, violations)
    all_monotonic.append(is_monotonic)

    # Test 3: gaussian_lowrank_lecture with random matrix (0-1)
    ranks, errors, is_monotonic, violations = test_function(
        "gaussian_lowrank_lecture", gaussian_lowrank_lecture, matrix_random,
        "Random Matrix (0-1)")
    print_results("gaussian_lowrank_lecture", "Random Matrix (0-1)",
                  ranks, errors, is_monotonic, violations)
    all_monotonic.append(is_monotonic)

    # Test 4: gaussian_lowrank_lecture with image-like matrix (0-255)
    ranks, errors, is_monotonic, violations = test_function(
        "gaussian_lowrank_lecture", gaussian_lowrank_lecture, matrix_image,
        "Image-like Matrix (0-255)")
    print_results("gaussian_lowrank_lecture", "Image-like Matrix (0-255)",
                  ranks, errors, is_monotonic, violations)
    all_monotonic.append(is_monotonic)

    # Test 5: gaussian_lowrank with pure low-rank matrix
    ranks, errors, is_monotonic, violations = test_function(
        "gaussian_lowrank", gaussian_lowrank, matrix_lowrank_pure,
        "Pure Low-Rank Matrix (rank 50)")
    print_results("gaussian_lowrank", "Pure Low-Rank Matrix (rank 50)",
                  ranks, errors, is_monotonic, violations)
    all_monotonic.append(is_monotonic)

    # Test 6: gaussian_lowrank with noisy low-rank matrix
    ranks, errors, is_monotonic, violations = test_function(
        "gaussian_lowrank", gaussian_lowrank, matrix_lowrank_noisy,
        "Low-Rank Matrix + Noise (rank 50)")
    print_results("gaussian_lowrank", "Low-Rank Matrix + Noise (rank 50)",
                  ranks, errors, is_monotonic, violations)
    all_monotonic.append(is_monotonic)

    # Test 7: gaussian_lowrank_lecture with pure low-rank matrix
    ranks, errors, is_monotonic, violations = test_function(
        "gaussian_lowrank_lecture", gaussian_lowrank_lecture, matrix_lowrank_pure,
        "Pure Low-Rank Matrix (rank 50)")
    print_results("gaussian_lowrank_lecture", "Pure Low-Rank Matrix (rank 50)",
                  ranks, errors, is_monotonic, violations)
    all_monotonic.append(is_monotonic)

    # Test 8: gaussian_lowrank_lecture with noisy low-rank matrix
    ranks, errors, is_monotonic, violations = test_function(
        "gaussian_lowrank_lecture", gaussian_lowrank_lecture, matrix_lowrank_noisy,
        "Low-Rank Matrix + Noise (rank 50)")
    print_results("gaussian_lowrank_lecture", "Low-Rank Matrix + Noise (rank 50)",
                  ranks, errors, is_monotonic, violations)
    all_monotonic.append(is_monotonic)

    # Final summary
    print("="*60)
    print("=== SUMMARY ===")
    print("="*60)
    print(f"All tests monotonically decreasing: {all(all_monotonic)}")
    if not all(all_monotonic):
        print(f"Tests with violations: {8 - sum(all_monotonic)} out of 8")
    else:
        print(f"All 8 tests passed monotonicity check!")
    print("="*60)
