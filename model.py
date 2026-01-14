# model A

import argparse
import logging
import os
import math
from pathlib import Path
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from flow_matching.path import CondOTProbPath
from flow_matching.solver import ODESolver
from flow_matching.utils import ModelWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NAIPDataset(Dataset):
    """Dataset for NAIP aerial imagery."""

    def __init__(self, data_path, split="train", transform=None):
        self.data_dir = data_path / split
        self.transform = transform
        self.image_paths = list(self.data_dir.glob("*.png"))

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {self.data_dir}")

        logger.info(f"Found {len(self.image_paths)} images in {self.data_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Use PIL to load images
        from PIL import Image

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Return the image and a dummy label (0) as we're doing unconditional generation
        return image, torch.tensor(0)


# Modified UNet architecture based on the examples
class ResBlock(nn.Module):
    """Residual block with time embedding."""

    def __init__(self, in_channels, out_channels, time_channels, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_channels, out_channels))

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        # Add time embedding
        h = h + self.time_mlp(time_emb)[:, :, None, None]

        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention block."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Attention
        scale = 1.0 / math.sqrt(math.sqrt(C))
        w = torch.einsum("bci,bcj->bij", q * scale, k * scale)
        w = torch.softmax(w, dim=-1)
        h = torch.einsum("bij,bcj->bci", w, v)
        h = h.reshape(B, C, H, W)

        return x + self.proj(h)


class Upsample(nn.Module):
    """Upsampling layer with optional convolution."""

    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """Downsampling layer with optional convolution."""

    def __init__(self, channels, use_conv=True):
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        else:
            self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        if self.use_conv:
            return self.conv(x)
        else:
            return self.pool(x)


class UNet(nn.Module):
    """U-Net architecture for flow matching."""

    def __init__(
        self,
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=2,
        attention_resolutions=(8, 16),
        channel_mult=(1, 2, 4, 8),
        dropout=0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions

        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Input blocks
        self.input_blocks = nn.ModuleList(
            [nn.Conv2d(in_channels, model_channels, 3, padding=1)]
        )

        input_block_chans = [model_channels]
        ch = model_channels

        # Downsampling
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, mult * model_channels, time_embed_dim, dropout)]
                ch = mult * model_channels

                if mult * model_channels in attention_resolutions:
                    layers.append(AttentionBlock(ch))

                self.input_blocks.append(nn.ModuleList(layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                self.input_blocks.append(nn.ModuleList([Downsample(ch, True)]))
                input_block_chans.append(ch)

        # Middle blocks
        self.middle_block = nn.ModuleList(
            [
                ResBlock(ch, ch, time_embed_dim, dropout),
                AttentionBlock(ch),
                ResBlock(ch, ch, time_embed_dim, dropout),
            ]
        )

        # Upsampling
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        model_channels * mult,
                        time_embed_dim,
                        dropout,
                    )
                ]
                ch = model_channels * mult

                if ch in attention_resolutions:
                    layers.append(AttentionBlock(ch))

                if level > 0 and i == num_res_blocks:
                    layers.append(Upsample(ch, True))

                self.output_blocks.append(nn.ModuleList(layers))

        # Final layers
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps):
        # Time embedding
        emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(emb)

        # Input blocks
        h = x
        hs = []
        for module in self.input_blocks:
            if isinstance(module, nn.ModuleList):
                for layer in module:
                    if isinstance(layer, ResBlock):
                        h = layer(h, emb)
                    else:
                        h = layer(h)
            else:
                h = module(h)
            hs.append(h)

        # Middle blocks
        for layer in self.middle_block:
            if isinstance(layer, ResBlock):
                h = layer(h, emb)
            else:
                h = layer(h)

        # Output blocks
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in module:
                if isinstance(layer, ResBlock):
                    h = layer(h, emb)
                else:
                    h = layer(h)

        return self.out(h)


def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device) / half
    )
    args = timesteps[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class FlowMatchingModel(ModelWrapper):
    """Wrapper for UNet model to handle flow matching interface."""

    def __init__(self, model):
        super().__init__(model)

    def forward(self, x, t, **extras):
        # Ensure t has the right shape
        if t.ndim == 0:
            t = t.view(1)
        if len(t) == 1 and len(x) > 1:
            t = t.repeat(len(x))

        # Scale from [-1,1] to [0,1] for visualization purposes when sampling
        return self.model(x, t)


def train(args):
    """Main training function."""
    # Set up output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    # Set device
    device = torch.device(args.device)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

    # Set up data loading
    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),  # Scales to [0, 1]
            transforms.Normalize(0.5, 0.5),  # Scales to [-1, 1]
        ]
    )

    data_path = Path("data")
    dataset_path = data_path / args.dataset_name
    dataset = NAIPDataset(dataset_path, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        prefetch_factor=2,
    )

    # Create model
    model = UNet(
        in_channels=3,
        model_channels=args.model_channels,
        out_channels=3,
        channel_mult=args.channel_mult,
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
    ).to(device)

    flow_model = FlowMatchingModel(model).to(device)

    # Set up optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    if args.resume_from:
        if os.path.exists(args.resume_from):
            logger.info(f"Loading checkpoint from {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            logger.info(f"Resuming from epoch {start_epoch}")
        else:
            logger.warning(f"Checkpoint not found at {args.resume_from}, starting from scratch")

    # Set up flow matching path
    path = CondOTProbPath()

    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0.0

        start_time = time.time()
        losses = []
        with tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}") as pbar:
            for batch_idx, (x, _) in enumerate(pbar):
                x = x.to(device)

                # Zero gradients
                optimizer.zero_grad()

                # Sample noise
                noise = torch.randn_like(x)

                # Sample time points
                t = torch.rand(x.shape[0], device=device)

                # Sample path
                path_sample = path.sample(t=t, x_0=noise, x_1=x)
                x_t = path_sample.x_t
                u_t = path_sample.dx_t

                # Forward pass and compute loss
                pred_u_t = flow_model(x_t, t)
                loss = F.mse_loss(pred_u_t, u_t)

                # Backward and optimize
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                losses.append(loss.item())

                # # Log and sample periodically
                # if batch_idx % args.log_interval == 0:
                #     logger.info(
                #         f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}"
                #     )
                epoch_loss += loss.item()
                avg_loss = epoch_loss / (batch_idx + 1)
                pbar.set_postfix(loss=avg_loss)

        # # End of epoch
        # epoch_loss /= len(dataloader)
        # epoch_time = time.time() - start_time
        #
        # logger.info(
        #     f"Epoch {epoch} completed in {epoch_time:.2f}s, Avg Loss: {epoch_loss:.6f}"
        # )

        # Generate samples after each epoch
        if epoch % args.sample_interval == 0 or epoch == args.num_epochs - 1:
            model.eval()

            # Create solver
            solver = ODESolver(velocity_model=flow_model)

            # Draw samples
            with torch.no_grad():
                x_0 = torch.randn(
                    args.num_samples, 3, args.image_size, args.image_size
                ).to(device)

                # Set up sampling options
                time_grid = torch.tensor([0.0, 1.0], device=device)
                ode_options = {
                    "atol": 1e-5,
                    "rtol": 1e-5,
                }

                # Sample from the model
                x_1 = solver.sample(
                    x_init=x_0,
                    step_size=None,
                    method="dopri5",
                    atol=1e-5,
                    rtol=1e-5,
                )

                # Convert samples back to [0, 1] range for visualization
                samples = (x_1 * 0.5 + 0.5).clamp(0, 1)

                # Save samples
                save_path = samples_dir / f"epoch_{epoch+1}.png"
                save_image(samples, save_path, nrow=int(math.sqrt(args.num_samples)))

        # Save model checkpoint
        if epoch % args.save_interval == 0 or epoch == args.num_epochs - 1:
            checkpoint_path = output_dir / f"checkpoint_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                    "model_channels": args.model_channels,
                    "channel_mult": args.channel_mult,
                    "num_res_blocks": args.num_res_blocks,
                    "image_size": args.image_size,
                },
                checkpoint_path,
            )

    logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Flow Matching for NAIP Image Generation"
    )

    # Data arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="coconino_chunk",
        help="Name of the dataset subdirectory",
    )
    parser.add_argument(
        "--image_size", type=int, default=512, help="Size to resize images to"
    )

    # Model arguments
    parser.add_argument(
        "--model_channels", type=int, default=64, help="Base channels for UNet"
    )
    parser.add_argument(
        "--channel_mult",
        type=str,
        default="1,2,4,8",
        help="Channel multipliers (comma-separated)",
    )
    parser.add_argument(
        "--num_res_blocks", type=int, default=2, help="Number of ResBlocks per level"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Sampling arguments
    parser.add_argument(
        "--num_samples", type=int, default=1, help="Number of samples to generate"
    )
    parser.add_argument(
        "--ode_method", type=str, default="dopri5", help="ODE solver method"
    )

    # Logging arguments
    parser.add_argument(
        "--output_dir", type=str, default="./output", help="Output directory"
    )
    parser.add_argument(
        "--log_interval", type=int, default=10, help="Logging interval (batches)"
    )
    parser.add_argument(
        "--sample_interval", type=int, default=1, help="Sampling interval (epochs)"
    )
    parser.add_argument(
        "--save_interval", type=int, default=10, help="Model saving interval (epochs)"
    )

    args = parser.parse_args()

    # Process channel_mult
    args.channel_mult = tuple(map(int, args.channel_mult.split(",")))

    train(args)


if __name__ == "__main__":
    main()
