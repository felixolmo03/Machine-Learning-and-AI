#!/usr/bin/env python3
"""
Quick training script using preset configurations.

Usage:
    python3 train_with_preset.py --config small --epochs 5
    python3 train_with_preset.py --config base --epochs 10
"""

import argparse
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.storyteller.model import get_small_config, get_base_config, get_moe_config

def main():
    parser = argparse.ArgumentParser(description="Train with preset configs")
    parser.add_argument("--config", type=str, default="small",
                       choices=["small", "base", "moe"],
                       help="Which preset config to use")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--download_data", action="store_true")
    parser.add_argument("--use_scheduler", action="store_true", default=True)
    parser.add_argument("--label_smoothing", type=float, default=0.1)

    args = parser.parse_args()

    # Get the preset config
    if args.config == "small":
        config = get_small_config()
        print("Using SMALL config (~125M params)")
    elif args.config == "base":
        config = get_base_config()
        print("Using BASE config (~350M params)")
    else:
        config = get_moe_config()
        print("Using MOE config (~500M total, ~100M active)")

    # Print estimated parameters
    total_params = config.num_parameters() / 1e6
    active_params = config.active_parameters() / 1e6
    print(f"Total parameters: {total_params:.1f}M")
    print(f"Active parameters: {active_params:.1f}M")

    # Build command
    cmd_parts = [
        "python3 train_custom.py",
        f"--epochs {args.epochs}",
        f"--batch_size {args.batch_size}",
        f"--hidden_size {config.hidden_size}",
        f"--num_layers {config.num_layers}",
        f"--num_heads {config.num_attention_heads}",
        f"--label_smoothing {args.label_smoothing}",
        "--save_every_n_steps 2000",
    ]

    if args.download_data:
        cmd_parts.append("--download_data")

    if args.use_scheduler:
        cmd_parts.append("--use_scheduler")

    if config.use_moe:
        cmd_parts.append("--use_moe")

    cmd = " ".join(cmd_parts)

    print("\nRunning command:")
    print(cmd)
    print("\n" + "="*60 + "\n")

    import os
    os.system(cmd)

if __name__ == "__main__":
    main()
