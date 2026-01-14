# Sample generation script for FID evaluation

import argparse
import logging
import os
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from flow_matching.solver import ODESolver

from model import UNet, FlowMatchingModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_samples(args):
    """Generate samples from a trained model."""
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    if torch.cuda.is_available():
        logger.info(f"Device Name: {torch.cuda.get_device_name(0)}")

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    epoch = checkpoint.get("epoch", "unknown")
    logger.info(f"Loaded model from epoch {epoch}")

    # Load architecture from checkpoint (with defaults for older checkpoints)
    model_channels = checkpoint.get("model_channels", 64)
    channel_mult = checkpoint.get("channel_mult", (1, 2, 4, 8))
    num_res_blocks = checkpoint.get("num_res_blocks", 2)
    image_size = checkpoint.get("image_size", 512)

    logger.info(f"Model config: channels={model_channels}, mult={channel_mult}, res_blocks={num_res_blocks}, size={image_size}")

    # Create model
    model = UNet(
        in_channels=3,
        model_channels=model_channels,
        out_channels=3,
        channel_mult=channel_mult,
        num_res_blocks=num_res_blocks,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create flow matching wrapper and solver
    flow_model = FlowMatchingModel(model).to(device)
    solver = ODESolver(velocity_model=flow_model)

    # Generate samples in batches
    num_generated = 0

    logger.info(f"Generating {args.num_samples} samples...")

    with torch.no_grad():
        pbar = tqdm(total=args.num_samples, desc="Generating samples")

        while num_generated < args.num_samples:
            # Determine batch size for this iteration
            remaining = args.num_samples - num_generated
            current_batch_size = min(args.batch_size, remaining)

            # Sample from noise
            x_0 = torch.randn(
                current_batch_size, 3, image_size, image_size
            ).to(device)

            # Generate samples using ODE solver
            x_1 = solver.sample(
                x_init=x_0,
                step_size=None,
                method=args.ode_method,
                atol=args.atol,
                rtol=args.rtol,
            )

            # Convert samples back to [0, 1] range
            samples = (x_1 * 0.5 + 0.5).clamp(0, 1)

            # Save individual images
            for i in range(current_batch_size):
                sample_path = output_dir / f"sample_{num_generated:06d}.png"
                save_image(samples[i], sample_path)
                num_generated += 1

            pbar.update(current_batch_size)

        pbar.close()

    logger.info(f"Generated {num_generated} samples in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate samples from trained flow matching model"
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        required=True,
        help="Number of samples to generate",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_samples",
        help="Output directory for generated samples",
    )

    # Sampling arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for sampling",
    )
    parser.add_argument(
        "--ode_method",
        type=str,
        default="dopri5",
        help="ODE solver method",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for ODE solver",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for ODE solver",
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Set seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        logger.info(f"Set random seed to {args.seed}")

    generate_samples(args)


if __name__ == "__main__":
    main()