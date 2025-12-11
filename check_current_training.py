#!/usr/bin/env python3
"""
Check Current Training Status

Quick script to see what's happening with your training RIGHT NOW.

Usage:
    python check_current_training.py
"""

import psutil
import os
from pathlib import Path

print("\n" + "="*60)
print("CURRENT TRAINING STATUS")
print("="*60 + "\n")

# Check if training is running
print("🔍 Checking for running training process...")
training_found = False
for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        cmdline = proc.info['cmdline']
        if cmdline and 'train_simple.py' in ' '.join(cmdline):
            training_found = True
            print(f"  ✓ Found training process (PID: {proc.info['pid']})")
            print(f"  Command: {' '.join(cmdline)}")
            
            # Get process info
            try:
                cpu_percent = proc.cpu_percent(interval=1)
                mem_info = proc.memory_info()
                mem_mb = mem_info.rss / 1024 / 1024
                
                print(f"  CPU usage: {cpu_percent:.1f}%")
                print(f"  Memory: {mem_mb:.0f} MB")
            except:
                pass
            
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

if not training_found:
    print("  ⚠️  No training process found")
    print("  Either training hasn't started or has finished")

# Check GPU usage
print("\n🎮 Checking GPU Status...")
try:
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"    Memory: {mem_allocated:.1f} / {mem_total:.1f} GB allocated")
            print(f"    Reserved: {mem_reserved:.1f} GB")
    else:
        print("  No CUDA GPUs available")
except ImportError:
    print("  PyTorch not available to check GPU")

# Check checkpoints
print("\n💾 Checking Checkpoints...")
checkpoint_dir = Path("checkpoints")
if checkpoint_dir.exists():
    checkpoints = sorted(checkpoint_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime)
    if checkpoints:
        print(f"  Found {len(checkpoints)} checkpoint(s):")
        for ckpt in checkpoints[-3:]:  # Show last 3
            size_mb = ckpt.stat().st_size / 1024 / 1024
            import time
            mod_time = time.ctime(ckpt.stat().st_mtime)
            print(f"    {ckpt.name}: {size_mb:.1f} MB (modified: {mod_time})")
    else:
        print(f"  No checkpoints yet")
        print(f"  Checkpoint will save at end of epoch (~3 hours remaining)")
else:
    print(f"  Checkpoint directory doesn't exist yet")

# Analyze the loss issue
print("\n🚨 LOSS ANALYSIS:")
print(f"  Your reported loss: 0.0305")
print(f"  Expected loss at 40% of epoch 0: 4.5-5.5")
print(f"  Difference: {abs(0.0305 - 5.0):.2f} (HUGE!)")
print()
print(f"  Possible causes:")
print(f"    1. ❌ Loss is scaled incorrectly (divided by something)")
print(f"    2. ❌ Model is overfitting (memorizing data)")
print(f"    3. ❌ Batch size is 1 (causing weird behavior)")
print(f"    4. ❌ Dataset is corrupted or too small")

# Recommendations
print("\n💡 IMMEDIATE ACTIONS:")
print()
print(f"  Option 1: STOP and restart with correct settings")
print(f"    1. Press Ctrl+C to stop training")
print(f"    2. Run: rm -rf checkpoints")
print(f"    3. Run: python train_simple.py --preset tinystories --epochs 10 --use_scheduler --download_data")
print()
print(f"  Option 2: WAIT for epoch to finish (3 hours)")
print(f"    1. Let it complete epoch 0")
print(f"    2. Check checkpoint: python diagnose_training.py --checkpoint checkpoints/checkpoint_step_106062.pt")
print(f"    3. If generation is garbage, restart with Option 1")
print()
print(f"  Option 3: INVESTIGATE current settings")
print(f"    Run: python investigate_training.py")
print()

# Show what GOOD training looks like
print("📊 What NORMAL training should look like:")
print()
print("  Epoch 0:  10% | loss=6.234, lr=5.0e-04")
print("  Epoch 0:  20% | loss=5.456, lr=5.0e-04")
print("  Epoch 0:  40% | loss=4.789, lr=5.0e-04  ← You should be here")
print("  Epoch 0:  60% | loss=4.234, lr=5.0e-04")
print("  Epoch 0: 100% | loss=3.890, lr=5.0e-04")
print()
print("  Your loss (0.0305) is 100x too small!")
print()

print("="*60)
print("STATUS CHECK COMPLETE")
print("="*60 + "\n")

# Final recommendation
print("🎯 MY RECOMMENDATION:")
print()
print("  Stop training now and restart with proper settings.")
print("  The loss of 0.0305 indicates something is fundamentally wrong.")
print()
print("  Quick restart command:")
print("  $ rm -rf checkpoints && python train_simple.py --preset tinystories --epochs 10 --use_scheduler --download_data")
print()
