import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from algos import rsvd_srft, rrqr_lowrank

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Algorithms to run. To test other algorithms from src/algos,
# simply comment/uncomment entries in this list and extend
# `_apply_algorithm` below.
ACTIVE_ALGOS = [
    "rsvd_srft",
    "rrqr",
    # "rsvd",
    # "numpy_svd_lowrank",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def str2bool(v: str) -> bool:
    """Parse a string into a boolean value."""
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in {"yes", "y", "true", "t", "1"}:
        return True
    if v in {"no", "n", "false", "f", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v!r}")


def load_image(image_path: str, grayscale: bool) -> np.ndarray:
    """Load an image as a float64 numpy array in [0, 1].

    If `grayscale` is True, returns shape (H, W).
    Otherwise returns shape (H, W, 3).
    """
    img = Image.open(image_path)
    if grayscale:
        img = img.convert("L")
    else:
        img = img.convert("RGB")

    arr = np.asarray(img, dtype=np.float64) / 255.0
    return arr


def image_to_matrix(img_array: np.ndarray) -> np.ndarray:
    """Convert a 2D/3D image array to a 2D matrix for norm computations."""
    if img_array.ndim == 2:
        return img_array
    elif img_array.ndim == 3:
        h, w, c = img_array.shape
        return img_array.reshape(h, w * c)
    else:
        raise ValueError(f"Unsupported image array shape: {img_array.shape}")


def _apply_algorithm(A: np.ndarray, rank: int, algo_name: str) -> np.ndarray:
    """Apply a low-rank algorithm from src/algos to a 2D matrix.

    Returns the approximated matrix Ak with the same shape as A.
    """
    if algo_name == "rsvd_srft":
        # rsvd_srft returns U, S, Vt (already truncated to rank k)
        U, S, Vt = rsvd_srft(A, k=rank, q=2)
        # Reconstruct rank-k approximation
        Ak = U @ np.diag(S) @ Vt
        return Ak.real

    elif algo_name == "rrqr":
        # rrqr_lowrank already returns Ak
        Ak = rrqr_lowrank(A, rank)
        return Ak.real

    # Template for extending with additional algorithms:
    # elif algo_name == "rsvd":
    #     from algos import rsvd
    #     U, S, Vt = rsvd(A, rank)
    #     U_k = U[:, :rank]
    #     S_k = S[:rank]
    #     Vt_k = Vt[:rank, :]
    #     Ak = (U_k * S_k) @ Vt_k
    #     return Ak.real
    #
    elif algo_name == "numpy_svd_lowrank":
        from algos import numpy_svd_lowrank

        U, S, Vt = numpy_svd_lowrank(A, rank)
        Ak = U @ np.diag(S) @ Vt
        return Ak.real

    else:
        raise ValueError(f"Unknown algorithm name: {algo_name}")


def approximate_image(img_array: np.ndarray, rank: int, algo_name: str) -> np.ndarray:
    """Apply a low-rank algorithm channel-wise to an image.

    Args:
        img_array: (H, W) or (H, W, 3) float array in [0, 1]
        rank: target rank (per channel)
        algo_name: name in ACTIVE_ALGOS

    Returns:
        Approximated image array with same shape as img_array.
    """
    if img_array.ndim == 2:
        A = img_array
        Ak = _apply_algorithm(A, rank, algo_name)
        return np.clip(Ak, 0.0, 1.0)

    elif img_array.ndim == 3:
        h, w, c = img_array.shape
        Ak_channels = []
        for ch in range(c):
            A_ch = img_array[:, :, ch]
            Ak_ch = _apply_algorithm(A_ch, rank, algo_name)
            Ak_channels.append(np.clip(Ak_ch, 0.0, 1.0))

        Ak = np.stack(Ak_channels, axis=2)
        return Ak

    else:
        raise ValueError(f"Unsupported image array shape: {img_array.shape}")


def compute_errors(A_orig: np.ndarray, A_approx: np.ndarray) -> Tuple[float, float]:
    """Compute relative Frobenius and spectral (2-norm) errors.

    The errors are computed as:
        ||A - Ak|| / ||A||
    where A and Ak are in matrix form (H x (W*C)).
    """
    A_mat = image_to_matrix(A_orig)
    Ak_mat = image_to_matrix(A_approx)

    diff = A_mat - Ak_mat

    frob_orig = np.linalg.norm(A_mat, "fro")
    spec_orig = np.linalg.norm(A_mat, 2)

    frob_err = np.linalg.norm(diff, "fro") / frob_orig if frob_orig > 0 else 0.0
    spec_err = np.linalg.norm(diff, 2) / spec_orig if spec_orig > 0 else 0.0

    return float(frob_err), float(spec_err)


def compute_memory_usage(img_array: np.ndarray, rank: int) -> Tuple[float, float]:
    """Compute memory usage for rank-k approximation vs full image.

    Args:
        img_array: Original image array
        rank: Rank k for the approximation

    Returns:
        Tuple of (lowrank_memory_mb, full_memory_mb)
    """
    # Convert image to matrix form to get dimensions
    A_mat = image_to_matrix(img_array)
    m, n = A_mat.shape

    # Memory for rank-k factorization: U (m x k) + S (k) + Vt (k x n)
    lowrank_elements = m * rank + rank + rank * n
    lowrank_bytes = lowrank_elements * 8  # float64 = 8 bytes
    lowrank_mb = lowrank_bytes / (1024**2)

    # Memory for full image
    full_elements = m * n
    full_bytes = full_elements * 8
    full_mb = full_bytes / (1024**2)

    return float(lowrank_mb), float(full_mb)


def compute_ranks(max_rank: int, num_runs: int) -> List[int]:
    """Compute a list of ranks from 1 to max_rank, split into ~num_runs points."""
    if num_runs <= 1:
        return [max_rank]

    if num_runs > max_rank:
        # No point in having more runs than ranks.
        num_runs = max_rank

    raw = np.linspace(1, max_rank, num_runs)
    ranks = [int(round(r)) for r in raw]
    ranks[0] = 1
    ranks[-1] = max_rank

    # Ensure uniqueness and sorted order
    ranks = sorted(set(ranks))
    return ranks


def prepare_output_dir(image_path: str, img_array: np.ndarray) -> Path:
    """Prepare an output directory under <project_root>/results/."""
    h, w = img_array.shape[:2]
    stem = Path(image_path).stem

    # project_root is one level above this file (src/)
    project_root = Path(__file__).resolve().parent.parent
    results_root = project_root / "results"
    out_dir = results_root / f"image_{stem}_{h}x{w}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def save_approximation_image(
    img_array: np.ndarray,
    output_dir: Path,
    algo_name: str,
    rank: int,
    grayscale: bool,
) -> None:
    """Save a single approximation as an image file."""
    algo_dir = output_dir / algo_name
    algo_dir.mkdir(parents=True, exist_ok=True)

    # Convert back to uint8 [0, 255] for saving
    arr = np.clip(img_array * 255.0, 0, 255).astype(np.uint8)

    if grayscale:
        img = Image.fromarray(arr, mode="L")
    else:
        img = Image.fromarray(arr)

    filename = f"{algo_name}_rank_{rank:03d}.png"
    img.save(algo_dir / filename)


def plot_rsvd_approximations(
    image_original: np.ndarray,
    ranks: List[int],
    approx_images: List[np.ndarray],
    grayscale: bool,
    output_dir: Path,
) -> None:
    """Plot original + rsvd_srft approximations in a single figure."""
    num_images = 1 + len(approx_images)  # original + approximations
    ncols = min(4, num_images)
    nrows = int(np.ceil(num_images / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    # Original image
    axes[0].imshow(
        image_original,
        cmap="gray" if grayscale else None,
    )
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Approximations
    for idx, (rank, img_rank) in enumerate(zip(ranks, approx_images), start=1):
        ax = axes[idx]
        ax.imshow(
            img_rank,
            cmap="gray" if grayscale else None,
        )
        ax.set_title(f"RSVD-SRFT rank={rank}")
        ax.axis("off")

    # Hide any unused axes
    for ax in axes[num_images:]:
        ax.axis("off")

    fig.suptitle("Randomized SVD (SRFT) Low-Rank Approximations", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "rsvd_srft_approximations.png", dpi=150)
    plt.close(fig)


def plot_error_curves(
    error_data: Dict[str, Dict[str, List[float]]],
    output_dir: Path,
) -> None:
    """Plot Frobenius and spectral error vs rank for all algorithms."""
    # Frobenius
    plt.figure(figsize=(6, 4))
    for algo, metrics in error_data.items():
        ranks = metrics["ranks"]
        frob = metrics["frobenius"]
        plt.plot(ranks, frob, marker="o", label=algo)
    plt.xlabel("Rank k")
    plt.ylabel("Relative Frobenius error")
    plt.title("Frobenius error vs rank")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "error_frobenius_vs_rank.png", dpi=150)
    plt.close()

    # Spectral
    plt.figure(figsize=(6, 4))
    for algo, metrics in error_data.items():
        ranks = metrics["ranks"]
        spec = metrics["spectral"]
        plt.plot(ranks, spec, marker="o", label=algo)
    plt.xlabel("Rank k")
    plt.ylabel("Relative spectral (2-norm) error")
    plt.title("Spectral error vs rank")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "error_spectral_vs_rank.png", dpi=150)
    plt.close()


def plot_memory_curves(
    memory_data: Dict[str, List[float]],
    full_memory_mb: float,
    output_dir: Path,
) -> None:
    """Plot memory usage vs rank for low-rank approximations.

    Args:
        memory_data: Dict mapping algorithm names to lists of memory values (MB)
        full_memory_mb: Memory of full image in MB
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(6, 4))

    # Plot each algorithm's memory usage
    for algo, data in memory_data.items():
        if algo != "ranks":
            ranks = memory_data["ranks"]
            memory = data
            plt.plot(ranks, memory, marker="o", label=f"{algo} (low-rank)")

    # Plot full image memory as horizontal line
    ranks = memory_data["ranks"]
    plt.axhline(
        y=full_memory_mb, color="red", linestyle="--", linewidth=2, label="Full image"
    )

    plt.xlabel("Rank k")
    plt.ylabel("Memory (MB)")
    plt.title("Memory usage vs rank")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "memory_vs_rank.png", dpi=150)
    plt.close()


def save_errors_to_csv(
    error_data: Dict[str, Dict[str, List[float]]],
    output_dir: Path,
) -> None:
    """Save error metrics to CSV."""
    csv_path = output_dir / "errors_vs_rank.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "rank", "frobenius_error", "spectral_error"])
        for algo, metrics in error_data.items():
            for r, frob, spec in zip(
                metrics["ranks"], metrics["frobenius"], metrics["spectral"]
            ):
                writer.writerow([algo, r, frob, spec])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute low-rank image approximations using algorithms from src/algos.\n"
            "Currently uses RSVD-SRFT and RRQR; extend ACTIVE_ALGOS to test others."
        )
    )
    parser.add_argument("image_path", type=str, help="Path to input image file.")
    parser.add_argument(
        "--grayscale",
        type=str2bool,
        default=False,
        help="Whether to convert the image to grayscale (default: False).",
    )
    parser.add_argument(
        "--max-rank",
        type=int,
        required=True,
        help="Maximum target rank k.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        required=True,
        help="Number of ranks to sample between 1 and max_rank.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = args.image_path
    grayscale = bool(args.grayscale)
    max_rank = int(args.max_rank)
    num_runs = int(args.num_runs)

    if max_rank < 1:
        raise ValueError("max_rank must be >= 1")
    if num_runs < 1:
        raise ValueError("num_runs must be >= 1")

    # Load image
    img_array = load_image(image_path, grayscale=grayscale)
    h, w = img_array.shape[:2]

    # Ensure rank is feasible
    max_possible_rank = min(h, w)
    if max_rank > max_possible_rank:
        print(
            f"[warning] Requested max_rank={max_rank} exceeds "
            f"min(image_height, image_width)={max_possible_rank}. "
            f"Using max_rank={max_possible_rank} instead."
        )
        max_rank = max_possible_rank

    # Prepare ranks and output directory
    ranks = compute_ranks(max_rank, num_runs)
    out_dir = prepare_output_dir(image_path, img_array)

    print(f"Image shape: {img_array.shape}")
    print(f"Using ranks: {ranks}")
    print(f"Algorithms: {ACTIVE_ALGOS}")
    print(f"Saving results to: {out_dir}")

    # Data structure for errors
    error_data: Dict[str, Dict[str, List[float]]] = {
        algo: {"ranks": [], "frobenius": [], "spectral": []} for algo in ACTIVE_ALGOS
    }

    # Data structure for memory tracking
    memory_data: Dict[str, List[float]] = {algo: [] for algo in ACTIVE_ALGOS}
    memory_data["ranks"] = []
    full_memory_mb = None  # Will be set in the first iteration

    # To visualize RSVD-SRFT approximations
    rsvd_srft_approxs: List[np.ndarray] = []

    # Main loop over ranks
    for k in ranks:
        print(f"\n=== Rank k={k} ===")

        # Compute memory usage for this rank (same for all algos)
        lowrank_mem_mb, full_mem_mb = compute_memory_usage(img_array, k)
        if full_memory_mb is None:
            full_memory_mb = full_mem_mb
        memory_data["ranks"].append(k)

        for algo in ACTIVE_ALGOS:
            print(f"  -> Running {algo}...")
            Ak_img = approximate_image(img_array, k, algo)
            frob_err, spec_err = compute_errors(img_array, Ak_img)

            error_data[algo]["ranks"].append(k)
            error_data[algo]["frobenius"].append(frob_err)
            error_data[algo]["spectral"].append(spec_err)
            memory_data[algo].append(lowrank_mem_mb)

            print(
                f"     Frobenius error: {frob_err:.6f}, "
                f"Spectral error: {spec_err:.6f}, "
                f"Memory: {lowrank_mem_mb:.2f} MB"
            )

            save_approximation_image(
                img_array=Ak_img,
                output_dir=out_dir,
                algo_name=algo,
                rank=k,
                grayscale=grayscale,
            )

            if algo == "rsvd_srft":
                rsvd_srft_approxs.append(Ak_img)

    # Plots and CSV
    plot_rsvd_approximations(
        image_original=img_array,
        ranks=ranks,
        approx_images=rsvd_srft_approxs,
        grayscale=grayscale,
        output_dir=out_dir,
    )
    plot_error_curves(error_data, out_dir)
    if full_memory_mb is not None:
        plot_memory_curves(memory_data, full_memory_mb, out_dir)
    save_errors_to_csv(error_data, out_dir)

    print("\nDone.")
    print(f"- RSVD-SRFT approximation grid: {out_dir / 'rsvd_srft_approximations.png'}")
    print(f"- Frobenius error plot: {out_dir / 'error_frobenius_vs_rank.png'}")
    print(f"- Spectral error plot: {out_dir / 'error_spectral_vs_rank.png'}")
    print(f"- Memory usage plot: {out_dir / 'memory_vs_rank.png'}")
    print(f"- Error CSV: {out_dir / 'errors_vs_rank.csv'}")
    print(f"- Full image memory: {full_memory_mb:.2f} MB")


if __name__ == "__main__":
    main()
