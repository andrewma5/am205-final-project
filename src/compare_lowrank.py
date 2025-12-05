"""Low-Rank Approximation Algorithm Comparison Framework.

Benchmarks and compares multiple low-rank approximation algorithms across
various ranks, measuring execution time, approximation error, and memory usage.
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from .rsvd import rsvd
from .gaussian_lowrank import gaussian_lowrank
from .svd_lowrank import numpy_svd_lowrank, python_svd_lowrank


@dataclass
class BenchmarkResult:
    """Store results for a single algorithm at a single rank."""

    method_name: str
    rank: int
    time_sec: float
    error_frobenius: float
    memory_theoretical_bytes: int
    success: bool = True
    error_message: str = ""


class MatrixGenerator:
    """Generate test matrices with controlled properties."""

    @staticmethod
    def random_matrix(m: int, n: int, seed: int = 42) -> np.ndarray:
        """
        Generate non-negative random matrix for testing.

        Creates a matrix with all non-negative entries by taking the absolute value
        of samples from a standard normal distribution. This is ideal for testing
        the Gaussian (greedy cross approximation) algorithm, which works best on
        non-negative matrices like images.

        Args:
            m: number of rows
            n: number of columns
            seed: random seed for reproducibility

        Returns:
            A: (m x n) non-negative matrix with entries from |N(0,1)|
        """
        np.random.seed(seed)
        return np.abs(np.random.randn(m, n))


class AlgorithmBenchmark:
    """Benchmark a single algorithm."""

    def __init__(self, name: str, func: Callable, returns_factors: bool = True):
        """
        Args:
            name: Display name for algorithm
            func: Function implementing algorithm
            returns_factors: True if func returns (U, S, Vt), False if returns full matrix
        """
        self.name = name
        self.func = func
        self.returns_factors = returns_factors

    def run(self, A: np.ndarray, rank: int) -> BenchmarkResult:
        """
        Run benchmark for given matrix and rank.

        Args:
            A: input matrix
            rank: desired rank

        Returns:
            BenchmarkResult with all metrics
        """
        m, n = A.shape

        try:
            # Run algorithm once and measure time
            start_time = time.perf_counter()
            result = self.func(A.copy(), rank)
            end_time = time.perf_counter()
            time_sec = end_time - start_time

            # Compute approximation error using result from single run
            error = self._compute_error(A, result)

            # Compute theoretical memory
            memory_theoretical = self._compute_theoretical_memory(m, n, rank)

            return BenchmarkResult(
                method_name=self.name,
                rank=rank,
                time_sec=time_sec,
                error_frobenius=error,
                memory_theoretical_bytes=memory_theoretical,
                success=True,
            )

        except Exception as e:
            # Algorithm failed - return error result
            return BenchmarkResult(
                method_name=self.name,
                rank=rank,
                time_sec=0.0,
                error_frobenius=np.inf,
                memory_theoretical_bytes=0,
                success=False,
                error_message=str(e),
            )

    def _compute_error(self, A_original: np.ndarray, result: any) -> float:
        """Compute Frobenius norm error."""
        if self.returns_factors:
            # Result is (U, S, Vt) - reconstruct matrix
            U, S, Vt = result
            A_approx = U @ np.diag(S) @ Vt
        else:
            # Result is full approximation matrix
            A_approx = result

        # Compute Frobenius norm of difference
        error = np.linalg.norm(A_original - A_approx, ord="fro")
        return error

    def _compute_theoretical_memory(self, m: int, n: int, rank: int) -> int:
        """Compute theoretical memory for storing factors."""
        if self.returns_factors:
            # Storage for U (m x rank) + S (rank) + Vt (rank x n)
            return (m * rank + rank + rank * n) * 8
        else:
            # Storage for full matrix (m x n)
            return m * n * 8


class ComparisonRunner:
    """Run comparison across multiple algorithms and ranks."""

    def __init__(
        self,
        matrix_size: int,
        max_rank: int,
        num_ranks: Optional[int] = None,
        seed: int = 42,
        skip_python_svd: bool = False,
    ):
        """
        Args:
            matrix_size: N for N×N matrix
            max_rank: maximum rank to test
            num_ranks: number of ranks to test (evenly distributed from 1 to max_rank)
                       if None, tests every rank from 1 to max_rank
            seed: random seed for reproducibility
            skip_python_svd: if True, skip pure Python SVD
        """
        self.matrix_size = matrix_size
        self.max_rank = max_rank
        self.num_ranks = num_ranks
        self.seed = seed
        self.skip_python_svd = skip_python_svd
        self.A: Optional[np.ndarray] = None
        self.results: List[BenchmarkResult] = []

        # Define algorithms to benchmark
        self.algorithms = [
            AlgorithmBenchmark("rSVD", rsvd, returns_factors=True),
            AlgorithmBenchmark("Gaussian", gaussian_lowrank, returns_factors=False),
            AlgorithmBenchmark("NumPy SVD", numpy_svd_lowrank, returns_factors=True),
        ]

        if not skip_python_svd:
            self.algorithms.append(
                AlgorithmBenchmark(
                    "Python SVD", python_svd_lowrank, returns_factors=True
                )
            )

    def setup(self):
        """Generate test matrix and validate inputs."""
        self._validate_inputs()

        print(
            f"Generating {self.matrix_size}×{self.matrix_size} non-negative test matrix (seed={self.seed})..."
        )
        self.A = MatrixGenerator.random_matrix(
            self.matrix_size, self.matrix_size, self.seed
        )

    def run_all(self) -> List[BenchmarkResult]:
        """Run all algorithms for all ranks."""
        if self.A is None:
            raise RuntimeError("Must call setup() before run_all()")

        # Generate ranks to test
        if self.num_ranks is None:
            # Test every rank from 1 to max_rank
            ranks_to_test = list(range(1, self.max_rank + 1))
        else:
            # Evenly distribute num_ranks samples from 1 to max_rank
            ranks_to_test = np.linspace(1, self.max_rank, self.num_ranks, dtype=int)
            ranks_to_test = sorted(set(ranks_to_test))  # Remove duplicates and sort

        print(
            f"Running comparisons for {len(ranks_to_test)} ranks up to {self.max_rank}..."
        )
        print(f"Testing ranks: {ranks_to_test}\n")

        for idx, rank in enumerate(ranks_to_test, 1):
            print(f"Rank {rank} ({idx}/{len(ranks_to_test)}):")

            for algo in self.algorithms:
                result = algo.run(self.A, rank)
                self.results.append(result)

                if result.success:
                    print(
                        f"  {algo.name:12s}: {result.time_sec:.4f}s, "
                        f"error={result.error_frobenius:.4f}, "
                        f"mem={result.memory_theoretical_bytes // 1024}KB"
                    )
                else:
                    print(f"  {algo.name:12s}: FAILED - {result.error_message}")

            print()

        return self.results

    def _validate_inputs(self):
        """Validate matrix_size, max_rank, and num_ranks."""
        if self.matrix_size < 2:
            raise ValueError(f"Matrix size must be at least 2, got {self.matrix_size}")

        if self.max_rank < 1:
            raise ValueError(f"Max rank must be at least 1, got {self.max_rank}")

        if self.max_rank >= self.matrix_size:
            raise ValueError(
                f"Max rank ({self.max_rank}) must be less than "
                f"matrix size ({self.matrix_size})"
            )

        if self.num_ranks is not None:
            if self.num_ranks < 1:
                raise ValueError(
                    f"Number of ranks must be at least 1, got {self.num_ranks}"
                )
            if self.num_ranks > self.max_rank:
                raise ValueError(
                    f"Number of ranks ({self.num_ranks}) cannot exceed "
                    f"max rank ({self.max_rank})"
                )

        # Warn about Python SVD performance
        if not self.skip_python_svd and self.max_rank > 50:
            print(
                f"WARNING: Pure Python SVD with max_rank={self.max_rank} will be very slow.\n"
                f"Consider using --no-python-svd flag.\n"
            )


class ResultsVisualizer:
    """Create visualization plots from benchmark results."""

    def __init__(self, results: List[BenchmarkResult]):
        self.results = results
        self.methods = sorted(list(set(r.method_name for r in results if r.success)))

        # Define colors and markers for each method
        self.colors = {
            "rSVD": "#1f77b4",
            "Gaussian": "#ff7f0e",
            "NumPy SVD": "#2ca02c",
            "Python SVD": "#d62728",
        }
        self.markers = {
            "rSVD": "o",
            "Gaussian": "s",
            "NumPy SVD": "^",
            "Python SVD": "D",
        }

    def plot_all(self, save_dir: str = "."):
        """Generate all three plots and save to files."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Get matrix size from first result
        matrix_size = int(np.sqrt(self.results[0].memory_theoretical_bytes // 8))
        if self.results[0].method_name == "Gaussian":
            # Gaussian stores full matrix, infer size differently
            for r in self.results:
                if r.method_name != "Gaussian":
                    # Use a factored method to infer size
                    m_times_k = (r.memory_theoretical_bytes // 8 - r.rank) // 2
                    matrix_size = m_times_k // r.rank
                    break

        # Infer from actual results
        ranks = sorted(list(set(r.rank for r in self.results if r.success)))
        if ranks:
            # Use first rank to infer matrix size
            for r in self.results:
                if r.rank == ranks[0] and r.method_name == "NumPy SVD":
                    # m*k + k + k*n = theoretical_bytes / 8
                    # For square matrix: 2*m*k + k = theoretical_bytes / 8
                    # m = (theoretical_bytes / 8 - k) / (2*k)
                    matrix_size = (r.memory_theoretical_bytes // 8 - r.rank) // (
                        2 * r.rank
                    )
                    break

        # Create all plots
        self.plot_time_vs_rank(save_path / f"time_vs_rank_n{matrix_size}.png")
        self.plot_error_vs_rank(save_path / f"error_vs_rank_n{matrix_size}.png")
        self.plot_theoretical_memory_vs_rank(
            save_path / f"memory_vs_rank_n{matrix_size}.png"
        )

        print(f"\nPlots saved to:")
        print(f"  - {save_path / f'time_vs_rank_n{matrix_size}.png'}")
        print(f"  - {save_path / f'error_vs_rank_n{matrix_size}.png'}")
        print(f"  - {save_path / f'memory_vs_rank_n{matrix_size}.png'}")

    def plot_time_vs_rank(self, save_path: Path):
        """Plot execution time vs rank for all methods."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for method in self.methods:
            data = [
                (r.rank, r.time_sec)
                for r in self.results
                if r.method_name == method and r.success
            ]
            if data:
                ranks, times = zip(*data)
                ax.plot(
                    ranks,
                    times,
                    marker=self.markers.get(method, "o"),
                    color=self.colors.get(method, "gray"),
                    linewidth=2,
                    markersize=6,
                    label=method,
                )

        self._format_plot(
            ax,
            title="Execution Time vs Rank",
            xlabel="Rank",
            ylabel="Time (seconds)",
            use_log_scale=True,
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_error_vs_rank(self, save_path: Path):
        """Plot approximation error vs rank for all methods."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for method in self.methods:
            data = [
                (r.rank, r.error_frobenius)
                for r in self.results
                if r.method_name == method and r.success
            ]
            if data:
                ranks, errors = zip(*data)
                ax.plot(
                    ranks,
                    errors,
                    marker=self.markers.get(method, "o"),
                    color=self.colors.get(method, "gray"),
                    linewidth=2,
                    markersize=6,
                    label=method,
                )

        self._format_plot(
            ax,
            title="Approximation Error vs Rank",
            xlabel="Rank",
            ylabel="Frobenius Norm Error",
            use_log_scale=True,
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_theoretical_memory_vs_rank(self, save_path: Path):
        """Plot theoretical memory vs rank for all methods."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for method in self.methods:
            data = [
                (r.rank, r.memory_theoretical_bytes / (1024 * 1024))  # Convert to MB
                for r in self.results
                if r.method_name == method and r.success
            ]
            if data:
                ranks, memory = zip(*data)
                ax.plot(
                    ranks,
                    memory,
                    marker=self.markers.get(method, "o"),
                    color=self.colors.get(method, "gray"),
                    linewidth=2,
                    markersize=6,
                    label=method,
                )

        self._format_plot(
            ax,
            title="Memory vs Rank",
            xlabel="Rank",
            ylabel="Memory (MB)",
            use_log_scale=False,
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _format_plot(
        self, ax, title: str, xlabel: str, ylabel: str, use_log_scale: bool = False
    ):
        """Helper to format plot with consistent style."""
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=10, loc="best")

        if use_log_scale:
            ax.set_yscale("log")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare low-rank approximation algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage: 500×500 matrix, test ranks 1-50 (outputs to results/)
  python -m src.compare_lowrank --matrix-size 500 --max-rank 50

  # Test only 10 evenly-spaced ranks from 1 to 100
  python -m src.compare_lowrank -n 500 -r 100 --num-ranks 10

  # With custom output directory and seed
  python -m src.compare_lowrank -n 1000 -r 100 -o my_results/ --seed 123

  # Skip slow Python SVD
  python -m src.compare_lowrank -n 2000 -r 200 --no-python-svd
        """,
    )

    parser.add_argument(
        "--matrix-size",
        "-n",
        type=int,
        required=True,
        help="Size N of N×N test matrix",
    )
    parser.add_argument(
        "--max-rank",
        "-r",
        type=int,
        required=True,
        help="Maximum rank to test",
    )
    parser.add_argument(
        "--num-ranks",
        "-k",
        type=int,
        default=None,
        help="Number of ranks to test (evenly distributed from 1 to max-rank). "
        "If not specified, tests every rank from 1 to max-rank.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="results",
        help="Directory to save plots (default: results)",
    )
    parser.add_argument(
        "--no-python-svd",
        action="store_true",
        help="Skip pure Python SVD (very slow for large ranks)",
    )

    args = parser.parse_args()

    # Run comparison
    runner = ComparisonRunner(
        matrix_size=args.matrix_size,
        max_rank=args.max_rank,
        num_ranks=args.num_ranks,
        seed=args.seed,
        skip_python_svd=args.no_python_svd,
    )

    runner.setup()
    results = runner.run_all()

    # Generate plots
    print("Generating plots...")
    visualizer = ResultsVisualizer(results)
    visualizer.plot_all(save_dir=args.output_dir)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    successful_results = [r for r in results if r.success]
    if successful_results:
        # Group by method
        for method in visualizer.methods:
            method_results = [r for r in successful_results if r.method_name == method]
            if method_results:
                avg_time = np.mean([r.time_sec for r in method_results])
                avg_error = np.mean([r.error_frobenius for r in method_results])
                avg_mem_kb = np.mean(
                    [r.memory_theoretical_bytes / 1024 for r in method_results]
                )
                print(f"\n{method}:")
                print(f"  Average time:   {avg_time:.4f}s")
                print(f"  Average error:  {avg_error:.4f}")
                print(f"  Average memory: {avg_mem_kb:.1f} KB")

    # Report failures
    failed_results = [r for r in results if not r.success]
    if failed_results:
        print(f"\n{len(failed_results)} algorithm runs failed:")
        for r in failed_results:
            print(f"  {r.method_name} at rank {r.rank}: {r.error_message}")

    print("\nComparison complete!")


if __name__ == "__main__":
    main()
