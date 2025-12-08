"""Low-Rank Approximation Algorithm Comparison Framework.

Benchmarks and compares multiple low-rank approximation algorithms across
various ranks, measuring execution time, approximation error, and memory usage.
"""

import argparse
import numpy as np

from .benchmark_common import ComparisonRunner, ResultsVisualizer


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
        matrix_description="random test matrix",
    )

    runner.setup()
    results = runner.run_all()

    # Generate plots
    print("Generating plots...")
    visualizer = ResultsVisualizer(results, matrix_type_label=f"n{args.matrix_size}")
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
