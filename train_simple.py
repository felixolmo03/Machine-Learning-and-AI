#!/usr/bin/env python3
"""
Simple, Working Training Script for Storyteller

This script uses the proven components from src/storyteller/ to ensure
reliable training and good generation quality.

Usage:
    python train_simple.py --preset tinystories --epochs 3
    python train_simple.py --preset gpu --epochs 5 --use_scheduler
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

# Import proven Storyteller components
from src.storyteller.model import ModelConfig, StorytellerModel
from src.storyteller.data.dataset import StoryDataset
from src.storyteller.training.trainer import Trainer
from src.storyteller.utils.device_utils import smart_select_device


def download_and_prepare_data(data_dir: Path):
    """Download TinyStories dataset if needed."""
    from datasets import load_dataset
    
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = processed_dir / "train.txt"
    val_file = processed_dir / "val.txt"
    
    if train_file.exists() and val_file.exists():
        print(f"✓ Data already exists at {processed_dir}")
        return train_file, val_file
    
    print("\n" + "="*60)
    print("DOWNLOADING TINYSTORIES DATASET")
    print("="*60 + "\n")
    
    print("Downloading TinyStories from HuggingFace...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    print(f"✓ Downloaded {len(dataset):,} stories")
    
    # Extract texts
    all_texts = [story["text"] for story in dataset]
    
    # Split 95/5 train/val
    split_idx = int(len(all_texts) * 0.95)
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]
    
    print(f"\nSaving to disk...")
    print(f"  Train: {len(train_texts):,} stories")
    print(f"  Val: {len(val_texts):,} stories")
    
    with open(train_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(train_texts))
    
    with open(val_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(val_texts))
    
    print(f"✓ Data saved to {processed_dir}\n")
    return train_file, val_file


def get_or_train_tokenizer(data_file: Path, tokenizer_dir: Path, vocab_size: int = 32000):
    """Get existing tokenizer or train a new BPE tokenizer."""
    
    # Check if tokenizer exists
    if tokenizer_dir.exists() and (tokenizer_dir / "tokenizer.json").exists():
        print(f"✓ Loading tokenizer from {tokenizer_dir}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))
        print(f"  Vocabulary size: {len(tokenizer):,}")
        return tokenizer
    
    print(f"\n" + "="*60)
    print(f"TRAINING BPE TOKENIZER (vocab_size={vocab_size})")
    print("="*60 + "\n")
    
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    
    # Create tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # Train
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        show_progress=True,
    )
    
    print(f"Training on {data_file}...")
    tokenizer.train([str(data_file)], trainer)
    
    # Save
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    
    # Convert to HuggingFace format
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_dir / "tokenizer.json"),
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    tokenizer.save_pretrained(str(tokenizer_dir))
    
    print(f"✓ Tokenizer saved to {tokenizer_dir}")
    print(f"  Vocabulary size: {len(tokenizer):,}\n")
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Simple Storyteller Training")
    
    # Presets
    parser.add_argument("--preset", type=str, default="tinystories",
                       choices=["tinystories", "small", "medium"],
                       help="Model preset configuration")
    
    # Training
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--use_scheduler", action="store_true", help="Use LR scheduler with warmup")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    
    # Data
    parser.add_argument("--download_data", action="store_true", help="Download TinyStories dataset")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--max_seq_length", type=int, default=None, help="Max sequence length")
    
    # Paths
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda:0, cpu, etc)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Apply presets
    if args.preset == "tinystories":
        # Optimized for TinyStories - small, fast training
        num_layers = 6
        hidden_size = 384
        num_heads = 6
        batch_size = args.batch_size or 32
        learning_rate = args.learning_rate or 5e-4
        max_seq_length = args.max_seq_length or 256
        vocab_size = 16000  # Smaller vocab for simple stories
    elif args.preset == "small":
        # Small GPT-2 style model
        num_layers = 8
        hidden_size = 512
        num_heads = 8
        batch_size = args.batch_size or 24
        learning_rate = args.learning_rate or 3e-4
        max_seq_length = args.max_seq_length or 512
        vocab_size = 32000
    else:  # medium
        # Medium model
        num_layers = 12
        hidden_size = 768
        num_heads = 12
        batch_size = args.batch_size or 16
        learning_rate = args.learning_rate or 3e-4
        max_seq_length = args.max_seq_length or 512
        vocab_size = 32000
    
    print("\n" + "="*60)
    print("STORYTELLER TRAINING - SIMPLE & RELIABLE")
    print("="*60 + "\n")
    
    print(f"Preset: {args.preset.upper()}")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden Size: {hidden_size}")
    print(f"  Attention Heads: {num_heads}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate:.2e}")
    print(f"  Max Seq Length: {max_seq_length}")
    print(f"  Epochs: {args.epochs}")
    print()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Select device
    if args.device:
        device = torch.device(args.device)
    else:
        device = smart_select_device(verbose=True)
    
    print(f"Device: {device}\n")
    
    # Prepare data
    if args.download_data:
        train_file, val_file = download_and_prepare_data(data_dir)
    else:
        train_file = data_dir / "processed" / "train.txt"
        val_file = data_dir / "processed" / "val.txt"
        
        if not train_file.exists():
            print(f"❌ Training data not found at {train_file}")
            print("   Run with --download_data to download TinyStories")
            sys.exit(1)
    
    # Get tokenizer
    tokenizer_dir = data_dir / "tokenizers" / f"storyteller-{vocab_size}"
    tokenizer = get_or_train_tokenizer(train_file, tokenizer_dir, vocab_size)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = StoryDataset(
        data_path=str(train_file),
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    val_dataset = StoryDataset(
        data_path=str(val_file),
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    
    print(f"✓ Train batches: {len(train_loader):,}")
    print(f"✓ Val batches: {len(val_loader):,}\n")
    
    # Build model
    print("Building model...")
    config = ModelConfig(
        vocab_size=len(tokenizer),
        max_seq_length=max_seq_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 4,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        use_flash_attention=False,  # Disable for compatibility
        positional_encoding="learned",  # Use learned positions for simplicity
    )
    
    model = StorytellerModel(config)
    print()
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    
    # Setup scheduler
    scheduler = None
    if args.use_scheduler:
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        
        steps_per_epoch = len(train_loader)
        
        # Simple cosine schedule with warmup
        def get_scheduler(optimizer, warmup_steps, total_steps):
            from torch.optim.lr_scheduler import LambdaLR
            import math
            
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
            return LambdaLR(optimizer, lr_lambda)
        
        total_steps = steps_per_epoch * args.epochs
        scheduler = get_scheduler(optimizer, args.warmup_steps, total_steps)
        print(f"✓ Scheduler configured (warmup: {args.warmup_steps} steps)\n")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=str(device),
        use_amp=True,
        amp_dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        save_dir=str(save_dir),
        save_every_n_steps=2000,
        eval_every_n_steps=1000,
        log_every_n_steps=100,
        keep_last_n_checkpoints=3,
        use_mlflow=False,
        tokenizer=tokenizer,
        num_eval_samples=5,  # Generate 5 samples during eval
        eval_max_length=200,
        eval_temperature=0.8,
        eval_top_k=50,
        eval_top_p=0.9,
    )
    
    # Resume if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")
    
    trainer.train(num_epochs=args.epochs)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nCheckpoints saved to: {save_dir}")
    print(f"Best model: {save_dir / 'best_model.pt'}")
    
    # Test generation
    print("\n" + "="*60)
    print("TESTING GENERATION")
    print("="*60 + "\n")
    
    model.eval()
    prompts = [
        "Once upon a time",
        "There was a little",
        "One day a",
    ]
    
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Prompt: '{prompt}'")
        print(f"Generated: {text}\n")


if __name__ == "__main__":
    main()
