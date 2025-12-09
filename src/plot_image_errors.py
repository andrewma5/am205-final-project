import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_errors(csv_path: Path) -> pd.DataFrame:
    """Load an errors_vs_rank.csv file into a DataFrame."""
    return pd.read_csv(csv_path)


def main():
    # Base directory is the directory of this script
    base_dir = Path(__file__).resolve().parent

    # Paths to the CSV files (relative, as in your comment)
    trefethen_path = (
        base_dir / ".." / "results" / "image_trefethen_900x600" / "errors_vs_rank.csv"
    )
    boston_path = (
        base_dir / ".." / "results" / "image_boston_799x1200" / "errors_vs_rank.csv"
    )

    # Load the data
    data_by_image = {
        "Trefethen 900x600": load_errors(trefethen_path),
        "Boston 799x1200": load_errors(boston_path),
    }

    # Prepare figure with two subplots: Frobenius and Spectral error
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    metrics = [
        ("frobenius_error", "Frobenius error"),
        ("spectral_error", "Spectral error"),
    ]

    # Styling: different markers for images, line styles for algorithms
    style_map = {
        ("Trefethen 900x600", "rsvd_srft"): {"marker": "o", "linestyle": "-"},
        ("Trefethen 900x600", "rrqr"): {"marker": "o", "linestyle": "--"},
        ("Boston 799x1200", "rsvd_srft"): {"marker": "s", "linestyle": "-"},
        ("Boston 799x1200", "rrqr"): {"marker": "s", "linestyle": "--"},
    }

    for ax, (col_name, y_label) in zip(axes, metrics):
        for image_label, df in data_by_image.items():
            for algo in df["algorithm"].unique():
                sub = df[df["algorithm"] == algo].sort_values("rank")
                style = style_map.get((image_label, algo), {})
                ax.plot(
                    sub["rank"],
                    sub[col_name],
                    label=f"{image_label} - {algo}",
                    **style,
                )

        ax.set_xlabel("Rank")
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle=":", alpha=0.5)

    axes[0].legend(title="Image - Algorithm", fontsize=8)
    fig.suptitle("Approximation Error vs Rank for Two Images", fontsize=12)
    fig.tight_layout()

    # Save figure to ../results/error_plots.png
    save_path = base_dir / ".." / "results" / "error_plots.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    main()
