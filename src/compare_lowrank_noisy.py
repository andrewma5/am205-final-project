"""Compare low-rank approximation algorithms on low-rank matrices with noise.

Tests how algorithms perform when the true underlying structure is low-rank
but corrupted by additive Gaussian noise. This is common in denoising applications.
"""

import argparse
import numpy as np

from .benchmark_common import ComparisonRunner, ResultsVisualizer
from .matrix_generators import MatrixGenerator


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare low-rank approximation algorithms on noisy low-rank matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 500x500 matrix with true rank 25, light noise
  python -m src.compare_lowrank_noisy -n 500 --true-rank 25 --noise 0.1 -r 50

  # 1000x1000 matrix, rank 50, heavy noise, test 10 ranks
  python -m src.compare_lowrank_noisy -n 1000 --true-rank 50 --noise 1.0 -r 100 -k 10

  # Pure low-rank (no noise) - should see error drop at true rank
  python -m src.compare_lowrank_noisy -n 800 --true-rank 30 --noise 0.0 -r 60
        """,
    )

    parser.add_argument(
        "--matrix-size", "-n",
        type=int,
        required=True,
        help="Size N of N×N test matrix",
    )
    parser.add_argument(
        "--true-rank",
        type=int,
        required=True,
        help="True rank of underlying low-rank structure",
    )
    parser.add_argument(
        "--noise-level", "--noise",
        type=float,
        required=True,
        help="Noise level (standard deviation of additive Gaussian noise). "
             "0.0 = no noise, 0.1 = light, 1.0 = heavy",
    )
    parser.add_argument(
        "--max-rank", "-r",
        type=int,
        required=True,
        help="Maximum rank to test (should be > true-rank to see noise floor)",
    )
    parser.add_argument(
        "--num-ranks", "-k",
        type=int,
        default=None,
        help="Number of ranks to test (evenly distributed). "
             "If not specified, tests every rank from 1 to max-rank.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir", "-o",
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

    # Validate true_rank vs matrix_size
    if args.true_rank >= args.matrix_size:
        parser.error(
            f"true-rank ({args.true_rank}) must be less than "
            f"matrix-size ({args.matrix_size})"
        )

    if args.max_rank <= args.true_rank:
        print(
            f"WARNING: max-rank ({args.max_rank}) should be > true-rank ({args.true_rank}) "
            f"to observe noise floor in error plots.\n"
        )

    # Generate matrix
    print(f"Generating {args.matrix_size}×{args.matrix_size} low-rank matrix:")
    print(f"  True rank: {args.true_rank}")
    print(f"  Noise level: {args.noise_level}")
    print(f"  Seed: {args.seed}")

    A = MatrixGenerator.lowrank_with_noise(
        m=args.matrix_size,
        n=args.matrix_size,
        true_rank=args.true_rank,
        noise_level=args.noise_level,
        seed=args.seed,
    )

    # Print matrix statistics
    info = MatrixGenerator.get_matrix_info(A)
    print(f"\nMatrix statistics:")
    print(f"  Shape: {info['shape']}")
    print(f"  Range: [{info['min']:.4f}, {info['max']:.4f}]")
    print(f"  Mean: {info['mean']:.4f}, Std: {info['std']:.4f}")
    print(f"  Numerical rank: {info['estimated_rank']}")

    # Run comparison
    runner = ComparisonRunner(
        matrix=A,
        max_rank=args.max_rank,
        num_ranks=args.num_ranks,
        seed=args.seed,
        skip_python_svd=args.no_python_svd,
        matrix_description=f"low-rank (rank {args.true_rank}) + noise ({args.noise_level})",
    )

    runner.setup()
    results = runner.run_all()

    # Generate plots
    print("Generating plots...")
    matrix_label = f"lowrank_r{args.true_rank}_noise{args.noise_level}_n{args.matrix_size}"
    visualizer = ResultsVisualizer(results, matrix_type_label=matrix_label)
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
                avg_error_spectral = np.mean([r.error_spectral for r in method_results])
                avg_error_frobenius = np.mean([r.error_frobenius for r in method_results])
                avg_mem_kb = np.mean(
                    [r.memory_bytes / 1024 for r in method_results]
                )
                print(f"\n{method}:")
                print(f"  Average time:            {avg_time:.4f}s")
                print(f"  Average spectral error:  {avg_error_spectral:.4f}")
                print(f"  Average Frobenius error: {avg_error_frobenius:.4f}")
                print(f"  Average memory:          {avg_mem_kb:.1f} KB")

    # Report failures
    failed_results = [r for r in results if not r.success]
    if failed_results:
        print(f"\n{len(failed_results)} algorithm runs failed:")
        for r in failed_results:
            print(f"  {r.method_name} at rank {r.rank}: {r.error_message}")

    print("\nComparison complete!")


if __name__ == "__main__":
    main()
