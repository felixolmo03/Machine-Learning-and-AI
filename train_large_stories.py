#!/usr/bin/env python3
"""
Large Model Training for TinyStories Generation

This creates a larger, more capable model specifically for generating
high-quality short stories like those in TinyStories dataset.

Model: ~120M parameters (similar to GPT-2 small)
Goal: Generate coherent, creative short stories

Usage:
    python train_large_stories.py --epochs 15
"""

import argparse
import sys
import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import PreTrainedTokenizerFast

from src.storyteller.model import ModelConfig, StorytellerModel
from src.storyteller.data.dataset import StoryDataset
from src.storyteller.training.trainer import Trainer
from src.storyteller.utils.device_utils import smart_select_device


def download_tinystories(data_dir: Path):
    """Download TinyStories if needed."""
    from datasets import load_dataset
    
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = processed_dir / "train.txt"
    val_file = processed_dir / "val.txt"
    
    if train_file.exists() and val_file.exists():
        return train_file, val_file
    
    print("\n" + "="*60)
    print("DOWNLOADING TINYSTORIES")
    print("="*60 + "\n")
    
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    print(f"✓ Downloaded {len(dataset):,} stories\n")
    
    all_texts = [story["text"] for story in dataset]
    split_idx = int(len(all_texts) * 0.95)
    
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]
    
    print(f"Saving {len(train_texts):,} train stories...")
    with open(train_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(train_texts))
    
    print(f"Saving {len(val_texts):,} val stories...")
    with open(val_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(val_texts))
    
    print(f"✓ Data saved to {processed_dir}\n")
    return train_file, val_file


def get_or_train_tokenizer(data_file: Path, tokenizer_dir: Path, vocab_size: int = 32000):
    """Get or train tokenizer."""
    if tokenizer_dir.exists() and (tokenizer_dir / "tokenizer.json").exists():
        print(f"✓ Loading tokenizer from {tokenizer_dir}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))
        print(f"  Vocab size: {len(tokenizer):,}\n")
        return tokenizer
    
    print(f"\n" + "="*60)
    print(f"TRAINING TOKENIZER (vocab={vocab_size:,})")
    print("="*60 + "\n")
    
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        show_progress=True,
    )
    
    tokenizer.train([str(data_file)], trainer)
    
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_dir / "tokenizer.json"),
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    tokenizer.save_pretrained(str(tokenizer_dir))
    
    print(f"✓ Tokenizer saved to {tokenizer_dir}\n")
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train large model for TinyStories")
    
    # Training
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("--max_examples", type=int, default=100000, help="Max training examples")
    
    # Model size
    parser.add_argument("--model_size", type=str, default="large", 
                       choices=["medium", "large", "xlarge"],
                       help="Model size: medium(~50M), large(~120M), xlarge(~200M)")
    
    # Data
    parser.add_argument("--download_data", action="store_true", help="Download TinyStories")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Tokenizer vocab size")
    
    # Other
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--save_dir", type=str, default="checkpoints_large", help="Save directory")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("LARGE MODEL TRAINING FOR TINYSTORIES")
    print("="*60 + "\n")
    
    # Model configurations
    if args.model_size == "medium":
        # ~50M params - Good balance
        num_layers = 8
        hidden_size = 512
        num_heads = 8
        print("📊 Model: MEDIUM (~50M parameters)")
    elif args.model_size == "large":
        # ~120M params - GPT-2 small size, RECOMMENDED
        num_layers = 12
        hidden_size = 768
        num_heads = 12
        print("📊 Model: LARGE (~120M parameters) - RECOMMENDED")
    else:  # xlarge
        # ~200M params - Maximum quality
        num_layers = 16
        hidden_size = 1024
        num_heads = 16
        print("📊 Model: XLARGE (~200M parameters)")
    
    print(f"   Layers: {num_layers}")
    print(f"   Hidden: {hidden_size}")
    print(f"   Heads: {num_heads}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Learning Rate: {args.learning_rate:.2e}\n")
    
    # Setup paths
    data_dir = Path("data")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    device = smart_select_device() if not args.device else torch.device(args.device)
    print(f"Device: {device}\n")
    
    # Get data
    if args.download_data:
        train_file, val_file = download_tinystories(data_dir)
    else:
        train_file = data_dir / "processed" / "train.txt"
        val_file = data_dir / "processed" / "val.txt"
        
        if not train_file.exists():
            print("❌ Data not found. Run with --download_data")
            sys.exit(1)
    
    # Get tokenizer
    tokenizer_dir = data_dir / "tokenizers" / f"storyteller-{args.vocab_size}"
    tokenizer = get_or_train_tokenizer(train_file, tokenizer_dir, args.vocab_size)
    
    # Create datasets
    print("="*60)
    print("LOADING DATA")
    print("="*60 + "\n")
    
    full_train = StoryDataset(
        data_path=str(train_file),
        tokenizer=tokenizer,
        max_seq_length=512,  # Longer sequences for better stories
    )
    
    full_val = StoryDataset(
        data_path=str(val_file),
        tokenizer=tokenizer,
        max_seq_length=512,
    )
    
    # Limit dataset size
    max_train = min(len(full_train), args.max_examples)
    max_val = min(len(full_val), args.max_examples // 10)
    
    train_dataset = Subset(full_train, range(max_train))
    val_dataset = Subset(full_val, range(max_val))
    
    print(f"✓ Train examples: {len(train_dataset):,}")
    print(f"✓ Val examples: {len(val_dataset):,}\n")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    
    print(f"📊 Training Configuration:")
    print(f"   Steps/epoch: {steps_per_epoch:,}")
    print(f"   Total steps: {total_steps:,}")
    print(f"   Est. time/epoch: {steps_per_epoch / 5.5 / 60:.1f} minutes")
    print(f"   Est. total time: {total_steps / 5.5 / 3600:.1f} hours\n")
    
    # Build model
    print("="*60)
    print("BUILDING MODEL")
    print("="*60 + "\n")
    
    config = ModelConfig(
        vocab_size=len(tokenizer),
        max_seq_length=512,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden_size * 4,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        use_flash_attention=False,
        positional_encoding="learned",
    )
    
    model = StorytellerModel(config)
    print()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    
    # Scheduler with warmup
    from torch.optim.lr_scheduler import LambdaLR
    import math
    
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        progress = (step - args.warmup_steps) / (total_steps - args.warmup_steps)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    print(f"✓ Scheduler: Warmup {args.warmup_steps} steps, then cosine decay\n")
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=str(device),
        use_amp=True,
        amp_dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        save_dir=str(save_dir),
        save_every_n_steps=min(steps_per_epoch, 2000),
        eval_every_n_steps=steps_per_epoch,  # Eval once per epoch
        log_every_n_steps=100,
        tokenizer=tokenizer,
        num_eval_samples=5,  # Generate 5 stories during eval
        eval_max_length=300,  # Longer stories
        eval_temperature=0.8,
        eval_top_k=50,
        eval_top_p=0.9,
    )
    
    print("="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")
    
    print("🎯 Goal: Generate coherent TinyStories-style short stories")
    print("📈 Expected progress:")
    print("   Epoch 0-2: Learning basic structure")
    print("   Epoch 3-5: Simple coherent stories")
    print("   Epoch 6-10: Good quality stories")
    print("   Epoch 11-15: High quality, creative stories\n")
    
    trainer.train(num_epochs=args.epochs)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60 + "\n")
    
    print(f"✓ Checkpoints saved to: {save_dir}")
    print(f"✓ Best model: {save_dir / 'best_model.pt'}")
    print(f"\nTest generation:")
    print(f"  python diagnose_training.py --checkpoint {save_dir}/best_model.pt")


if __name__ == "__main__":
    main()
