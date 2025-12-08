"""Matrix generators for testing low-rank approximation algorithms.

This module provides various matrix generation methods for benchmarking,
including random matrices, low-rank matrices with noise, and image loading.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class MatrixGenerator:
    """Generate test matrices with controlled properties."""

    @staticmethod
    def random_matrix(m: int, n: int, seed: int = 42) -> np.ndarray:
        """
        Generate random matrix for testing.

        Creates a matrix with entries from standard normal distribution.

        Args:
            m: number of rows
            n: number of columns
            seed: random seed for reproducibility

        Returns:
            A: (m x n) matrix with entries from N(0,1)
        """
        np.random.seed(seed)
        return np.random.randn(m, n)

    @staticmethod
    def lowrank_with_noise(
        m: int,
        n: int,
        true_rank: int,
        noise_level: float,
        seed: int = 42
    ) -> np.ndarray:
        """
        Generate low-rank matrix with additive Gaussian noise.

        Creates matrix: A = U @ V + noise
        where U is (m x true_rank), V is (true_rank x n)

        Algorithm:
            1. Generate U ~ N(0, 1/sqrt(true_rank)) of size (m x true_rank)
            2. Generate V ~ N(0, 1/sqrt(true_rank)) of size (true_rank x n)
            3. Compute low-rank component: L = U @ V
            4. Generate noise ~ N(0, noise_level^2)
            5. Return A = L + noise

        The scaling ensures:
            - Expected Frobenius norm of L ≈ sqrt(m * n)
            - Noise level parameter directly controls noise magnitude

        Args:
            m: number of rows
            n: number of columns
            true_rank: true rank of underlying signal
            noise_level: standard deviation of additive noise
                        (0.0 = no noise, 0.1 = light noise, 1.0 = heavy noise)
            seed: random seed for reproducibility

        Returns:
            A: (m x n) matrix = low-rank + noise

        Raises:
            ValueError: if true_rank >= min(m, n)
            ValueError: if noise_level < 0

        Examples:
            # Pure low-rank matrix (no noise)
            A = MatrixGenerator.lowrank_with_noise(1000, 1000, 50, 0.0)

            # Low-rank with light noise (SNR ≈ 10)
            A = MatrixGenerator.lowrank_with_noise(1000, 1000, 50, 0.1)

            # Low-rank with heavy noise (SNR ≈ 1)
            A = MatrixGenerator.lowrank_with_noise(1000, 1000, 50, 1.0)
        """
        # Validation
        if true_rank < 1:
            raise ValueError(f"true_rank must be at least 1, got {true_rank}")
        if true_rank >= min(m, n):
            raise ValueError(
                f"true_rank ({true_rank}) must be less than min(m, n) = {min(m, n)}"
            )
        if noise_level < 0:
            raise ValueError(f"noise_level must be non-negative, got {noise_level}")

        np.random.seed(seed)

        # Generate low-rank factors with appropriate scaling
        # Scale by 1/sqrt(true_rank) so that E[||UV||_F^2] ≈ mn
        scale = 1.0 / np.sqrt(true_rank)
        U = np.random.randn(m, true_rank) * scale
        V = np.random.randn(true_rank, n) * scale

        # Compute low-rank component
        L = U @ V

        # Add Gaussian noise
        if noise_level > 0:
            noise = np.random.randn(m, n) * noise_level
            A = L + noise
        else:
            A = L

        return A

    @staticmethod
    def from_image(
        filepath: str,
        convert_grayscale: bool = True,
        target_size: Optional[tuple] = None,
    ) -> np.ndarray:
        """
        Load image file and convert to matrix.

        Args:
            filepath: path to image file (supports PNG, JPG, etc.)
            convert_grayscale: if True, convert to grayscale
                             if False, use first channel if multi-channel
            target_size: optional (width, height) tuple for resizing
                        if None, uses original image dimensions

        Returns:
            A: (height x width) matrix with pixel values
               - Grayscale: values in [0, 255] (not normalized)
               - Color: values from first channel [0, 255]

        Raises:
            ImportError: if PIL is not installed
            FileNotFoundError: if image file doesn't exist
            ValueError: if image cannot be loaded

        Examples:
            # Load grayscale image at original size
            A = MatrixGenerator.from_image("photo.jpg")

            # Load and resize to 512x512
            A = MatrixGenerator.from_image("photo.jpg", target_size=(512, 512))

            # Load color image (first channel only)
            A = MatrixGenerator.from_image("photo.png", convert_grayscale=False)
        """
        if not PIL_AVAILABLE:
            raise ImportError(
                "PIL (Pillow) is required for image loading. "
                "Install it with: pip install pillow"
            )

        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Image file not found: {filepath}")

        try:
            # Load image using PIL
            img = Image.open(filepath)

            # Resize if requested
            if target_size is not None:
                img = img.resize(target_size, Image.Resampling.LANCZOS)

            # Convert to grayscale if requested
            if convert_grayscale:
                img = img.convert('L')  # L = grayscale

            # Convert to numpy array
            A = np.array(img, dtype=float)

            # If image is color and we didn't convert to grayscale, take first channel
            if len(A.shape) == 3:
                A = A[:, :, 0]

            return A

        except Exception as e:
            raise ValueError(f"Failed to load image {filepath}: {str(e)}")

    @staticmethod
    def get_matrix_info(A: np.ndarray) -> Dict:
        """
        Get information about a matrix for logging.

        Args:
            A: input matrix

        Returns:
            Dictionary with matrix properties:
                - shape: tuple of matrix dimensions
                - dtype: numpy data type
                - min: minimum value
                - max: maximum value
                - mean: mean value
                - std: standard deviation
                - estimated_rank: numerical rank (using default tolerance)
        """
        return {
            'shape': A.shape,
            'dtype': A.dtype,
            'min': np.min(A),
            'max': np.max(A),
            'mean': np.mean(A),
            'std': np.std(A),
            'estimated_rank': np.linalg.matrix_rank(A),
        }
