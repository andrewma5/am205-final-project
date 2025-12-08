"""Compare low-rank approximation algorithms on image matrices.

Tests how algorithms compress real image data. The Gaussian algorithm
is optimized for non-negative matrices like images, while SVD-based
methods provide optimal approximations for any matrix.
"""

import argparse
from pathlib import Path
import numpy as np

from .benchmark_common import ComparisonRunner, ResultsVisualizer
from .matrix_generators import MatrixGenerator


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare low-rank approximation algorithms on image matrices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on image at original size
  python -m src.compare_lowrank_image --image photo.jpg -r 100

  # Test with resize to 512x512, sample 20 ranks
  python -m src.compare_lowrank_image --image large.png --resize 512 512 -r 200 -k 20

  # Keep color (use first channel), custom output directory
  python -m src.compare_lowrank_image --image color.jpg --no-grayscale -r 50 -o img_results/
        """,
    )

    parser.add_argument(
        "--image-path", "--image",
        type=str,
        required=True,
        help="Path to input image file (PNG, JPG, etc.)",
    )
    parser.add_argument(
        "--max-rank", "-r",
        type=int,
        required=True,
        help="Maximum rank to test",
    )
    parser.add_argument(
        "--num-ranks", "-k",
        type=int,
        default=None,
        help="Number of ranks to test (evenly distributed). "
             "If not specified, tests every rank from 1 to max-rank.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Resize image to (width, height) before testing. "
             "If not specified, uses original image dimensions.",
    )
    parser.add_argument(
        "--no-grayscale",
        action="store_true",
        help="Don't convert to grayscale (use first channel of color image)",
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

    # Load image
    image_path = Path(args.image_path)
    if not image_path.exists():
        parser.error(f"Image file not found: {args.image_path}")

    print(f"Loading image: {args.image_path}")

    resize_target = tuple(args.resize) if args.resize else None
    convert_gray = not args.no_grayscale

    A = MatrixGenerator.from_image(
        filepath=args.image_path,
        convert_grayscale=convert_gray,
        target_size=resize_target,
    )

    # Print image/matrix information
    info = MatrixGenerator.get_matrix_info(A)
    print(f"\nImage matrix properties:")
    print(f"  Dimensions: {info['shape'][0]} × {info['shape'][1]}")
    print(f"  Pixel range: [{info['min']:.1f}, {info['max']:.1f}]")
    print(f"  Mean: {info['mean']:.2f}, Std: {info['std']:.2f}")
    print(f"  Numerical rank: {info['estimated_rank']}")
    print(f"  Grayscale: {convert_gray}")

    # Validate max_rank against image dimensions
    min_dim = min(A.shape)
    if args.max_rank >= min_dim:
        parser.error(
            f"max-rank ({args.max_rank}) must be less than "
            f"min image dimension ({min_dim})"
        )

    # Suggest resize if image is very large
    if max(A.shape) > 2000 and resize_target is None:
        print(
            f"\nWARNING: Large image ({A.shape[0]}×{A.shape[1]}). "
            f"Consider using --resize for faster computation.\n"
        )

    # Run comparison
    img_name = image_path.stem
    runner = ComparisonRunner(
        matrix=A,
        max_rank=args.max_rank,
        num_ranks=args.num_ranks,
        seed=args.seed,
        skip_python_svd=args.no_python_svd,
        matrix_description=f"image '{img_name}' ({A.shape[0]}×{A.shape[1]})",
    )

    runner.setup()
    results = runner.run_all()

    # Generate plots
    print("Generating plots...")
    matrix_label = f"image_{img_name}_{A.shape[0]}x{A.shape[1]}"
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
