# Evaluation script for NAIP generation model

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_fidelity_metrics(real_dir, generated_dir, cuda, batch_size, kid_subset_size):
    """Compute FID, KID, and Precision/Recall using torch-fidelity."""
    import torch_fidelity

    logger.info("Computing fidelity metrics (FID, KID, Precision/Recall)...")
    logger.info(f"  Generated: {generated_dir}")
    logger.info(f"  Reference: {real_dir}")

    metrics = torch_fidelity.calculate_metrics(
        input1=str(generated_dir),
        input2=str(real_dir),
        cuda=cuda,
        batch_size=batch_size,
        fid=True,
        kid=True,
        kid_subset_size=kid_subset_size,
        prc=True,
    )

    return metrics


def compute_histograms_streaming(image_dir):
    """Compute per-band histograms by streaming images one at a time."""
    image_dir = Path(image_dir)
    image_paths = sorted(image_dir.glob("*.png"))

    if not image_paths:
        raise ValueError(f"No PNG images found in {image_dir}")

    histograms = [np.zeros(256, dtype=np.float64) for _ in range(3)]

    for path in tqdm(image_paths, desc=f"Computing histograms for {image_dir.name}"):
        img = np.array(Image.open(path).convert("RGB"))
        for band in range(3):
            hist, _ = np.histogram(img[:, :, band], bins=256, range=(0, 255))
            histograms[band] += hist

    # Normalize to probability distributions
    for band in range(3):
        total = histograms[band].sum()
        if total > 0:
            histograms[band] /= total

    return histograms


def compute_band_wasserstein(hists_real, hists_gen):
    """Compute Wasserstein-1 distance per band between histogram distributions."""
    from scipy.stats import wasserstein_distance

    bin_centers = np.arange(256)
    distances = []

    for band in range(3):
        dist = wasserstein_distance(
            bin_centers, bin_centers,
            u_weights=hists_real[band],
            v_weights=hists_gen[band],
        )
        distances.append(dist)

    return distances


def plot_histograms(hists_real, hists_gen, wasserstein_dists, output_dir):
    """Plot histogram comparisons for each band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    band_names = ["Red", "Green", "Blue"]
    band_colors = ["red", "green", "blue"]
    bins = np.arange(256)

    # Combined 1x3 figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (ax, name, color) in enumerate(zip(axes, band_names, band_colors)):
        ax.plot(bins, hists_real[i], color=color, linewidth=1.0, label="Real")
        ax.plot(bins, hists_gen[i], color="black", linewidth=1.0, linestyle="--", label="Generated")
        ax.set_title(f"{name} (W₁ = {wasserstein_dists[i]:.2f})")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Density")
        ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "histogram_comparison.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved {output_dir / 'histogram_comparison.png'}")

    # Individual band figures
    for i, (name, color) in enumerate(zip(band_names, band_colors)):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(bins, hists_real[i], color=color, linewidth=1.0, label="Real")
        ax.plot(bins, hists_gen[i], color="black", linewidth=1.0, linestyle="--", label="Generated")
        ax.set_title(f"{name} Channel (W₁ = {wasserstein_dists[i]:.2f})")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Density")
        ax.legend()
        plt.tight_layout()
        fig.savefig(output_dir / f"histogram_{name[0]}.png", dpi=150)
        plt.close(fig)
    logger.info("Saved individual band histograms")


def print_results(fidelity_metrics, wasserstein_dists):
    """Print formatted results table."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    if fidelity_metrics is not None:
        print("\nFidelity Metrics (Inception v3 features):")
        print(f"  FID                         {fidelity_metrics['frechet_inception_distance']:.2f}")
        print(f"  KID (mean +/- std)          {fidelity_metrics['kernel_inception_distance_mean']:.4f} +/- {fidelity_metrics['kernel_inception_distance_std']:.4f}")
        print(f"  Precision                   {fidelity_metrics['precision']:.2f}")
        print(f"  Recall                      {fidelity_metrics['recall']:.2f}")

    if wasserstein_dists is not None:
        band_names = ["Red", "Green", "Blue"]
        mean_w = np.mean(wasserstein_dists)
        print("\nPer-Band Histogram Analysis (Wasserstein-1 Distance):")
        for name, dist in zip(band_names, wasserstein_dists):
            print(f"  {name} channel                 {dist:.2f}")
        print(f"  Mean across bands           {mean_w:.2f}")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated NAIP images against real test images"
    )

    # Required arguments
    parser.add_argument(
        "--real_dir",
        type=str,
        required=True,
        help="Path to real test images",
    )
    parser.add_argument(
        "--generated_dir",
        type=str,
        required=True,
        help="Path to generated images",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Where to save plots and metrics.json",
    )

    # Computation arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for feature extraction",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for Inception feature extraction",
    )
    parser.add_argument(
        "--kid_subset_size",
        type=int,
        default=1000,
        help="Subset size for KID estimation",
    )
    parser.add_argument(
        "--skip_fidelity",
        action="store_true",
        help="Skip FID/KID/Precision-Recall (histogram-only mode)",
    )
    parser.add_argument(
        "--skip_histograms",
        action="store_true",
        help="Skip histogram analysis",
    )

    args = parser.parse_args()

    # Validate directories
    real_dir = Path(args.real_dir)
    generated_dir = Path(args.generated_dir)

    if not real_dir.is_dir():
        raise FileNotFoundError(f"Real image directory not found: {real_dir}")
    if not generated_dir.is_dir():
        raise FileNotFoundError(f"Generated image directory not found: {generated_dir}")

    real_pngs = list(real_dir.glob("*.png"))
    gen_pngs = list(generated_dir.glob("*.png"))

    if not real_pngs:
        raise ValueError(f"No PNG images found in {real_dir}")
    if not gen_pngs:
        raise ValueError(f"No PNG images found in {generated_dir}")

    logger.info(f"Real images: {len(real_pngs)} in {real_dir}")
    logger.info(f"Generated images: {len(gen_pngs)} in {generated_dir}")

    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fidelity metrics
    fidelity_metrics = None
    if not args.skip_fidelity:
        cuda = args.device == "cuda" and torch.cuda.is_available()
        fidelity_metrics = compute_fidelity_metrics(
            real_dir, generated_dir, cuda, args.batch_size, args.kid_subset_size
        )
        logger.info("Fidelity metrics computed successfully")

    # Histogram analysis
    wasserstein_dists = None
    if not args.skip_histograms:
        logger.info("Computing pixel intensity histograms...")
        hists_real = compute_histograms_streaming(real_dir)
        hists_gen = compute_histograms_streaming(generated_dir)
        wasserstein_dists = compute_band_wasserstein(hists_real, hists_gen)
        plot_histograms(hists_real, hists_gen, wasserstein_dists, output_dir)
        logger.info("Histogram analysis completed")

    # Print results
    print_results(fidelity_metrics, wasserstein_dists)

    # Save metrics.json
    results = {}
    if fidelity_metrics is not None:
        results["fidelity"] = {
            "fid": fidelity_metrics["frechet_inception_distance"],
            "kid_mean": fidelity_metrics["kernel_inception_distance_mean"],
            "kid_std": fidelity_metrics["kernel_inception_distance_std"],
            "precision": fidelity_metrics["precision"],
            "recall": fidelity_metrics["recall"],
        }
    if wasserstein_dists is not None:
        results["histograms"] = {
            "wasserstein_red": wasserstein_dists[0],
            "wasserstein_green": wasserstein_dists[1],
            "wasserstein_blue": wasserstein_dists[2],
            "wasserstein_mean": float(np.mean(wasserstein_dists)),
        }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
