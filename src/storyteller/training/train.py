"""
Main training script for the Storyteller model.

Usage:
    storyteller-train --config configs/base_model.yaml
    storyteller-train --config configs/moe_model.yaml --resume checkpoints/moe_model/checkpoint_step_10000.pt
"""

import argparse

import torch
import yaml
from transformers import PreTrainedTokenizerFast

from storyteller.model import StorytellerModel, ModelConfig
from storyteller.data.dataset import StoryDataset, StoryDatasetPreloaded, get_dataloader
from storyteller.training.trainer import Trainer
from storyteller.utils.device_utils import smart_select_device


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_model_from_config(config_dict: dict) -> StorytellerModel:
    """Create model from configuration dictionary."""
    # Extract model config and remove fields that aren't ModelConfig parameters
    model_cfg = config_dict["model"].copy()

    # Remove metadata fields that aren't part of ModelConfig
    model_cfg.pop("config_name", None)

    model_config = ModelConfig(**model_cfg)
    model = StorytellerModel(model_config)
    return model


def create_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    Create optimizer with weight decay.

    Uses AdamW with separate weight decay for different parameter groups.
    """
    # Separate parameters that should and shouldn't have weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No weight decay for biases and layer norms
        if "bias" in name or "ln" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {
            "params": decay_params,
            "weight_decay": config["weight_decay"],
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=config["learning_rate"],
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
    num_training_steps: int,
):
    """Create learning rate scheduler."""
    warmup_steps = config.get("warmup_steps", 2000)
    scheduler_type = config.get("lr_scheduler", "cosine")

    if scheduler_type == "cosine":
        from torch.optim.lr_scheduler import LambdaLR
        import math

        # Create a lambda function that implements warmup + cosine decay
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine decay
                progress = float(current_step - warmup_steps) / float(
                    max(1, num_training_steps - warmup_steps)
                )
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                # Scale to end at 0.1 of initial LR
                return 0.1 + (1.0 - 0.1) * cosine_decay

        scheduler = LambdaLR(optimizer, lr_lambda)

    elif scheduler_type == "linear":
        from torch.optim.lr_scheduler import LinearLR

        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=num_training_steps,
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


def main():
    parser = argparse.ArgumentParser(description="Train Storyteller model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to trained tokenizer (default: uses config or 'data/tokenizers/storyteller-tokenizer')",
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config_dict = load_config(args.config)
    train_config = config_dict["training"]

    # Set device
    device_config = train_config.get("device", "smart")

    if device_config == "smart":
        device = smart_select_device()
    else:
        device = torch.device(device_config)
        print(f"Using device: {device}")

    # Determine tokenizer path (priority: CLI arg > config > default)
    if args.tokenizer_path is not None:
        tokenizer_path = args.tokenizer_path
    else:
        tokenizer_path = train_config.get(
            "tokenizer_path", "data/tokenizers/storyteller-tokenizer"
        )

    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    print(f"  Vocabulary size: {len(tokenizer):,}")

    # Update vocab size in config
    config_dict["model"]["vocab_size"] = len(tokenizer)

    # Create datasets
    print("\nCreating datasets...")

    # Determine which dataset class to use based on config
    use_cached = train_config.get("use_cached_dataset", False)

    if use_cached:
        cache_dir = train_config.get("cache_dir", "data/cache")
        print(f"  Using cached dataset (cache_dir: {cache_dir})")

        train_dataset = StoryDatasetPreloaded(
            data_path=train_config["train_data_path"],
            tokenizer=tokenizer,
            max_seq_length=config_dict["model"]["max_seq_length"],
            cache_dir=cache_dir,
        )

        val_dataset = StoryDatasetPreloaded(
            data_path=train_config["val_data_path"],
            tokenizer=tokenizer,
            max_seq_length=config_dict["model"]["max_seq_length"],
            cache_dir=cache_dir,
        )
    else:
        print("  Using standard dataset (no caching)")

        train_dataset = StoryDataset(
            data_path=train_config["train_data_path"],
            tokenizer=tokenizer,
            max_seq_length=config_dict["model"]["max_seq_length"],
        )

        val_dataset = StoryDataset(
            data_path=train_config["val_data_path"],
            tokenizer=tokenizer,
            max_seq_length=config_dict["model"]["max_seq_length"],
        )

    print(f"  Train dataset: {len(train_dataset):,} examples")
    print(f"  Val dataset: {len(val_dataset):,} examples")

    # Configure pin_memory based on device
    # MPS doesn't support pin_memory, so disable it automatically
    pin_memory = train_config.get("pin_memory", True)
    if device.type == "mps" and pin_memory:
        print("  Note: Disabling pin_memory (not supported on MPS)")
        pin_memory = False

    # Create dataloaders
    train_dataloader = get_dataloader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        num_workers=train_config.get("num_workers", 0),
        pin_memory=pin_memory,
    )

    val_dataloader = get_dataloader(
        val_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        num_workers=train_config.get("num_workers", 0),
        pin_memory=pin_memory,
    )

    # Create model
    print("\nCreating model...")
    model = create_model_from_config(config_dict)

    # Enable gradient checkpointing if specified
    if config_dict["model"].get("gradient_checkpointing", False):
        print("  Enabling gradient checkpointing...")
        # This would need to be implemented in the model
        # model.gradient_checkpointing_enable()

    # Calculate training steps
    num_epochs = train_config["num_epochs"]
    gradient_accumulation_steps = train_config.get("gradient_accumulation_steps", 1)
    num_training_steps = (
        len(train_dataloader) // gradient_accumulation_steps * num_epochs
    )
    print("\nTraining configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {train_config['batch_size']}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(
        f"  Effective batch size: {train_config['batch_size'] * gradient_accumulation_steps}"
    )
    print(f"  Total training steps: {num_training_steps:,}")

    # Create optimizer
    print("\nCreating optimizer...")
    optimizer = create_optimizer(model, train_config)

    # Create scheduler
    print("Creating learning rate scheduler...")
    scheduler = create_scheduler(optimizer, train_config, num_training_steps)

    # Get evaluation config
    eval_config = train_config.get("evaluation", {})

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=train_config.get("use_amp", True),
        amp_dtype=train_config.get("amp_dtype", "bfloat16"),
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=train_config.get("max_grad_norm", 1.0),
        save_dir=train_config.get("save_dir", "checkpoints"),
        save_every_n_steps=train_config.get("save_every_n_steps", 5000),
        eval_every_n_steps=train_config.get("eval_every_n_steps", 1000),
        log_every_n_steps=train_config.get("log_every_n_steps", 100),
        keep_last_n_checkpoints=train_config.get("keep_last_n_checkpoints", 3),
        use_mlflow=train_config.get("use_mlflow", False),
        mlflow_experiment_name=train_config.get("mlflow_experiment_name"),
        mlflow_run_name=train_config.get("mlflow_run_name"),
        mlflow_tracking_uri=train_config.get("mlflow_tracking_uri"),
        mlflow_log_system_metrics=train_config.get("mlflow_log_system_metrics", True),
        tokenizer=tokenizer,
        num_eval_samples=eval_config.get("num_eval_samples", 50),
        eval_max_length=eval_config.get("eval_max_length", 512),
        eval_temperature=eval_config.get("eval_temperature", 1.0),
        eval_top_k=eval_config.get("eval_top_k", 50),
        eval_top_p=eval_config.get("eval_top_p", 0.95),
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60 + "\n")

    trainer.train(num_epochs=num_epochs)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
