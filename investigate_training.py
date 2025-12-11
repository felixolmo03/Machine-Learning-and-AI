#!/usr/bin/env python3
"""
Investigate Current Training Issues

This script helps diagnose why your training has unusual metrics.
Run this while training is running or after stopping it.

Usage:
    python investigate_training.py
"""

import os
from pathlib import Path
import torch
from transformers import PreTrainedTokenizerFast

print("\n" + "="*60)
print("TRAINING INVESTIGATION")
print("="*60 + "\n")

# Check data
print("📁 Checking Data Files...")
data_dir = Path("data/processed")
if data_dir.exists():
    train_file = data_dir / "train.txt"
    val_file = data_dir / "val.txt"
    
    if train_file.exists():
        size_mb = train_file.stat().st_size / 1024 / 1024
        print(f"  Train file: {size_mb:.1f} MB")
        
        # Count stories
        with open(train_file, 'r') as f:
            content = f.read()
            stories = [s for s in content.split('\n\n') if s.strip()]
            print(f"  Train stories: {len(stories):,}")
            
            # Sample a story
            if stories:
                sample = stories[0][:200]
                print(f"  Sample story: {sample}...")
    else:
        print(f"  ❌ Train file not found!")
    
    if val_file.exists():
        size_mb = val_file.stat().st_size / 1024 / 1024
        print(f"  Val file: {size_mb:.1f} MB")
    else:
        print(f"  ⚠️  Val file not found!")
else:
    print(f"  ❌ Data directory not found: {data_dir}")

# Check tokenizer
print("\n🔤 Checking Tokenizer...")
tokenizer_dirs = list(Path("data/tokenizers").glob("*")) if Path("data/tokenizers").exists() else []
if tokenizer_dirs:
    for tok_dir in tokenizer_dirs:
        if (tok_dir / "tokenizer.json").exists():
            print(f"  Found: {tok_dir.name}")
            tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tok_dir))
            print(f"    Vocab size: {len(tokenizer):,}")
            
            # Test encoding
            test_text = "Once upon a time there was a little girl."
            encoded = tokenizer.encode(test_text)
            print(f"    Test encoding: {len(encoded)} tokens")
            print(f"    Tokens: {encoded[:10]}...")
else:
    print(f"  ❌ No tokenizers found!")

# Check checkpoints
print("\n💾 Checking Checkpoints...")
checkpoint_dir = Path("checkpoints")
if checkpoint_dir.exists():
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if checkpoints:
        print(f"  Found {len(checkpoints)} checkpoint(s):")
        for ckpt in sorted(checkpoints, key=lambda x: x.stat().st_mtime):
            size_mb = ckpt.stat().st_size / 1024 / 1024
            print(f"    {ckpt.name}: {size_mb:.1f} MB")
    else:
        print(f"  ⚠️  No checkpoints yet (will save at end of epoch)")
else:
    print(f"  ⚠️  Checkpoint directory doesn't exist yet")

# Analyze training configuration
print("\n⚙️  Analyzing Training Configuration...")

# Try to infer from data
if data_dir.exists() and (data_dir / "train.txt").exists():
    train_file = data_dir / "train.txt"
    
    # Estimate dataset size
    with open(train_file, 'r') as f:
        content = f.read()
        stories = [s for s in content.split('\n\n') if s.strip()]
        total_stories = len(stories)
        
        # Estimate tokens
        if tokenizer_dirs:
            tok_dir = tokenizer_dirs[0]
            tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tok_dir))
            
            # Sample 100 stories to estimate average length
            sample_stories = stories[:100]
            total_tokens = sum(len(tokenizer.encode(s)) for s in sample_stories)
            avg_tokens_per_story = total_tokens / len(sample_stories)
            estimated_total_tokens = int(avg_tokens_per_story * total_stories)
            
            print(f"  Total stories: {total_stories:,}")
            print(f"  Avg tokens/story: {avg_tokens_per_story:.0f}")
            print(f"  Estimated total tokens: {estimated_total_tokens:,}")
            
            # Estimate training parameters
            print(f"\n📊 Training Estimates:")
            
            # With different batch sizes and seq lengths
            for batch_size in [8, 16, 32]:
                for seq_len in [128, 256, 512]:
                    # Estimate number of examples
                    examples = estimated_total_tokens // seq_len
                    steps_per_epoch = examples // batch_size
                    
                    print(f"\n  Batch={batch_size}, SeqLen={seq_len}:")
                    print(f"    Examples: {examples:,}")
                    print(f"    Steps/epoch: {steps_per_epoch:,}")
                    print(f"    Time/epoch (5.6 it/s): {steps_per_epoch/5.6/3600:.1f} hours")

# Check what's actually running
print("\n🔍 Current Training Analysis...")
print(f"  Your training shows:")
print(f"    Steps/epoch: 106,062")
print(f"    Current step: 42,133 (40%)")
print(f"    Current loss: 0.0305 ⚠️  SUSPICIOUSLY LOW!")
print(f"    Learning rate: 4.98e-04 ✓")
print(f"    Speed: 5.62 it/s ✓")

print(f"\n🚨 ISSUE DETECTED:")
print(f"  Loss of 0.0305 at step 42k is WRONG!")
print(f"  Expected loss at this point: 4.0-6.0")
print(f"  This suggests:")
print(f"    1. Model is overfitting (too large for data)")
print(f"    2. Batch size is too small (causing instability)")
print(f"    3. Loss calculation bug (unlikely)")

# Calculate what settings were likely used
print(f"\n🔢 Reverse Engineering Your Settings...")
steps_per_epoch = 106062
speed = 5.62  # it/s

# Try to figure out batch size and seq length
if tokenizer_dirs and data_dir.exists():
    tok_dir = tokenizer_dirs[0]
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tok_dir))
    
    train_file = data_dir / "train.txt"
    with open(train_file, 'r') as f:
        content = f.read()
        stories = [s for s in content.split('\n\n') if s.strip()]
        
        # Sample to estimate tokens
        sample_stories = stories[:100]
        total_tokens = sum(len(tokenizer.encode(s)) for s in sample_stories)
        avg_tokens = total_tokens / len(sample_stories)
        estimated_total = int(avg_tokens * len(stories))
        
        print(f"  Estimated total tokens: {estimated_total:,}")
        
        # Work backwards from steps_per_epoch
        for seq_len in [128, 256, 512, 1024]:
            examples = estimated_total // seq_len
            batch_size = examples // steps_per_epoch
            if batch_size > 0:
                print(f"  If seq_len={seq_len}: batch_size ≈ {batch_size}")

print(f"\n💡 RECOMMENDATIONS:")
print(f"  1. Stop current training (Ctrl+C)")
print(f"  2. Use these settings:")
print(f"     python train_simple.py \\")
print(f"       --preset tinystories \\")
print(f"       --epochs 10 \\")
print(f"       --batch_size 32 \\")
print(f"       --max_seq_length 256 \\")
print(f"       --use_scheduler")
print(f"  3. Expected results:")
print(f"     - Steps/epoch: ~2,000-5,000 (not 106k!)")
print(f"     - Loss at step 1000: ~5-6")
print(f"     - Loss at end epoch 0: ~4-5")
print(f"     - Time/epoch: 20-40 minutes")

print(f"\n" + "="*60)
print("INVESTIGATION COMPLETE")
print("="*60 + "\n")

# Offer to create a proper training command
print("Would you like me to create a recommended training script?")
print("This will use optimal settings based on your data.")
print()
print("Run this command to start fresh:")
print()
print("  rm -rf checkpoints && python train_simple.py --preset tinystories --epochs 10 --use_scheduler --download_data")
print()
