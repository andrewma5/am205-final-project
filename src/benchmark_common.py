"""Shared benchmarking infrastructure for low-rank approximation comparisons.

This module provides common classes and utilities for benchmarking multiple
low-rank approximation algorithms across different matrix types and ranks.
"""

import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import matplotlib.pyplot as plt

from .algos import (
    rsvd,
    rsvd_srft,
    gaussian_lowrank,
    numpy_svd_lowrank,
    python_svd_lowrank,
    rrqr_lowrank,
)


@dataclass
class BenchmarkResult:
    """Store results for a single algorithm at a single rank."""

    method_name: str
    rank: int
    time_sec: float
    error_spectral: float
    error_frobenius: float
    memory_bytes: int
    success: bool = True
    error_message: str = ""


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
        try:
            # Start memory tracking
            tracemalloc.start()

            # Run algorithm once and measure time
            start_time = time.perf_counter()
            result = self.func(A.copy(), rank)
            end_time = time.perf_counter()
            time_sec = end_time - start_time

            # Get peak memory usage
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_bytes = peak

            # Compute both approximation errors using result from single run
            error_spectral, error_frobenius = self._compute_error(A, result)

            return BenchmarkResult(
                method_name=self.name,
                rank=rank,
                time_sec=time_sec,
                error_spectral=error_spectral,
                error_frobenius=error_frobenius,
                memory_bytes=memory_bytes,
                success=True,
            )

        except Exception as e:
            # Stop memory tracking if it was started
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            # Algorithm failed - return error result
            return BenchmarkResult(
                method_name=self.name,
                rank=rank,
                time_sec=0.0,
                error_spectral=np.inf,
                error_frobenius=np.inf,
                memory_bytes=0,
                success=False,
                error_message=str(e),
            )

    def _compute_error(self, A_original: np.ndarray, result: any) -> tuple[float, float]:
        """Compute both spectral and Frobenius norm errors.

        Returns:
            tuple: (error_spectral, error_frobenius)
        """
        if self.returns_factors:
            # Result is (U, S, Vt) - reconstruct matrix
            U, S, Vt = result
            A_approx = U @ np.diag(S) @ Vt
        else:
            # Result is full approximation matrix
            A_approx = result

        # Compute both norms of the difference
        diff = A_original - A_approx
        error_spectral = np.linalg.norm(diff, ord=2)
        error_frobenius = np.linalg.norm(diff, ord="fro")
        return error_spectral, error_frobenius


class ComparisonRunner:
    """Run comparison across multiple algorithms and ranks."""

    def __init__(
        self,
        matrix: Optional[np.ndarray] = None,
        matrix_generator: Optional[Callable] = None,
        matrix_size: Optional[int] = None,
        max_rank: int = None,
        num_ranks: Optional[int] = None,
        seed: int = 42,
        skip_python_svd: bool = False,
        matrix_description: str = "test matrix",
    ):
        """
        Initialize comparison runner.

        Accepts matrix input in three modes:
        1. Pre-generated matrix (for images)
        2. Generator function (for random/noisy matrices)
        3. Matrix size (backward compatibility, will use random generation)

        Args:
            matrix: Pre-generated matrix (optional)
            matrix_generator: Function that returns matrix (optional)
            matrix_size: N for N×N matrix (optional, for backward compatibility)
            max_rank: maximum rank to test
            num_ranks: number of ranks to test (evenly distributed from 1 to max_rank)
                       if None, tests every rank from 1 to max_rank
            seed: random seed for reproducibility
            skip_python_svd: if True, skip pure Python SVD
            matrix_description: description for logging
        """
        # Validate input mode
        input_count = sum(
            [matrix is not None, matrix_generator is not None, matrix_size is not None]
        )
        if input_count != 1:
            raise ValueError(
                "Must provide exactly one of: matrix, matrix_generator, or matrix_size"
            )

        self.matrix = matrix
        self.matrix_generator = matrix_generator
        self.matrix_size = matrix_size
        self.max_rank = max_rank
        self.num_ranks = num_ranks
        self.seed = seed
        self.skip_python_svd = skip_python_svd
        self.matrix_description = matrix_description
        self.A: Optional[np.ndarray] = None
        self.results: List[BenchmarkResult] = []

        # Define algorithms to benchmark
        self.algorithms = [
            AlgorithmBenchmark("rSVD", rsvd, returns_factors=True),
            AlgorithmBenchmark("rSVD-SRFT", rsvd_srft, returns_factors=True),
            AlgorithmBenchmark("RRQR", rrqr_lowrank, returns_factors=False),
            # AlgorithmBenchmark("Gaussian", gaussian_lowrank, returns_factors=False),
            AlgorithmBenchmark("NumPy SVD", numpy_svd_lowrank, returns_factors=True),
        ]

        # if not skip_python_svd:
        #     self.algorithms.append(
        #         AlgorithmBenchmark(
        #             "Python SVD", python_svd_lowrank, returns_factors=True
        #         )
        #     )

    def setup(self):
        """Generate/validate test matrix and validate inputs."""
        # Generate or use provided matrix
        if self.matrix is not None:
            # Pre-generated matrix
            self.A = self.matrix
            print(f"Using provided {self.matrix_description}...")
        elif self.matrix_generator is not None:
            # Generator function
            print(f"Generating {self.matrix_description} (seed={self.seed})...")
            self.A = self.matrix_generator()
        else:
            # Backward compatibility: use matrix_size
            from .matrix_generators import MatrixGenerator

            print(
                f"Generating {self.matrix_size}×{self.matrix_size} {self.matrix_description} (seed={self.seed})..."
            )
            self.A = MatrixGenerator.random_matrix(
                self.matrix_size, self.matrix_size, self.seed
            )

        # Validate inputs
        self._validate_inputs()

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
                        f"spectral={result.error_spectral:.4f}, "
                        f"frobenius={result.error_frobenius:.4f}, "
                        f"mem={result.memory_bytes // 1024}KB"
                    )
                else:
                    print(f"  {algo.name:12s}: FAILED - {result.error_message}")

            print()

        return self.results

    def _validate_inputs(self):
        """Validate matrix dimensions, max_rank, and num_ranks."""
        if self.A is None:
            raise RuntimeError("Matrix not initialized")

        m, n = self.A.shape

        if m < 2 or n < 2:
            raise ValueError(f"Matrix dimensions must be at least 2×2, got {m}×{n}")

        if self.max_rank < 1:
            raise ValueError(f"Max rank must be at least 1, got {self.max_rank}")

        if self.max_rank >= min(m, n):
            raise ValueError(
                f"Max rank ({self.max_rank}) must be less than "
                f"min matrix dimension ({min(m, n)})"
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

    def __init__(self, results: List[BenchmarkResult], matrix_type_label: str = ""):
        """
        Initialize visualizer.

        Args:
            results: List of benchmark results
            matrix_type_label: Label for plot filenames (e.g., "lowrank_r25_noise0.1_n500")
        """
        self.results = results
        self.matrix_type_label = matrix_type_label
        self.methods = sorted(list(set(r.method_name for r in results if r.success)))

        # Define colors and markers for each method
        self.colors = {
            "rSVD": "#1f77b4",
            "rSVD-SRFT": "#ff7f0e",  # New color for rSVD-SRFT
            "RRQR": "#2ca02c",  # New color for RRQR
            "Gaussian": "#9467bd",  # Original Gaussian color, but it's commented out
            "NumPy SVD": "#8c564b",  # Original NumPy SVD color, but it's commented out
            "Python SVD": "#d62728",
        }
        self.markers = {
            "rSVD": "o",
            "rSVD-SRFT": "s",  # New marker for rSVD-SRFT
            "RRQR": "^",  # New marker for RRQR
            "Gaussian": "D",  # Original Gaussian marker, but it's commented out
            "NumPy SVD": "P",  # Original NumPy SVD marker, but it's commented out
            "Python SVD": "X",
        }

    def plot_all(self, save_dir: str = "."):
        """Generate all three plots and save to files."""
        save_path = Path(save_dir)

        # Use matrix_type_label if provided, otherwise infer from results
        if self.matrix_type_label:
            label = self.matrix_type_label
        else:
            # Infer matrix size from results
            ranks = sorted(list(set(r.rank for r in self.results if r.success)))
            if ranks:
                for r in self.results:
                    if r.rank == ranks[0] and r.method_name == "NumPy SVD":
                        matrix_size = (r.memory_bytes // 8 - r.rank) // (
                            2 * r.rank
                        )
                        label = f"n{matrix_size}"
                        break
            else:
                label = "unknown"

        # Create experiment subfolder
        experiment_dir = save_path / label
        experiment_dir.mkdir(parents=True, exist_ok=True)

        # Create all plots inside experiment folder
        self.plot_time_vs_rank(experiment_dir / "time_vs_rank.png")
        self.plot_error_spectral_vs_rank(experiment_dir / "error_spectral_vs_rank.png")
        self.plot_error_frobenius_vs_rank(experiment_dir / "error_frobenius_vs_rank.png")
        self.plot_memory_vs_rank(experiment_dir / "memory_vs_rank.png")

        print(f"\nPlots saved to {experiment_dir}:")
        print(f"  - time_vs_rank.png")
        print(f"  - error_spectral_vs_rank.png")
        print(f"  - error_frobenius_vs_rank.png")
        print(f"  - memory_vs_rank.png")

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

    def plot_error_spectral_vs_rank(self, save_path: Path):
        """Plot spectral norm approximation error vs rank for all methods."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for method in self.methods:
            data = [
                (r.rank, r.error_spectral)
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
            title="Approximation Error (Spectral Norm) vs Rank",
            xlabel="Rank",
            ylabel="Spectral Norm Error",
            use_log_scale=True,
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_error_frobenius_vs_rank(self, save_path: Path):
        """Plot Frobenius norm approximation error vs rank for all methods."""
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
            title="Approximation Error (Frobenius Norm) vs Rank",
            xlabel="Rank",
            ylabel="Frobenius Norm Error",
            use_log_scale=True,
        )
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_memory_vs_rank(self, save_path: Path):
        """Plot actual memory usage vs rank for all methods."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for method in self.methods:
            data = [
                (r.rank, r.memory_bytes / (1024 * 1024))  # Convert to MB
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
            title="Peak Memory Usage vs Rank",
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
