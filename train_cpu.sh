#!/bin/bash
# Quick start script for CPU training on Mac
# This will train a small 40M parameter model optimized for CPU

echo "Starting CPU training..."
echo "This will train a ~40M parameter model optimized for your Mac"
echo ""
echo "Expected timeline:"
echo "  - Training speed: ~10-20 seconds per step"
echo "  - Checkpoints saved every 1,000 steps"
echo "  - Coherent text expected around 20,000-30,000 steps"
echo "  - Estimated time: 3-7 days of continuous training"
echo ""
echo "You can stop training anytime with Ctrl+C"
echo "Resume later with: python3 train_custom.py --preset cpu --use_scheduler --resume checkpoints/checkpoint_step_XXXXX.pt"
echo ""
echo "Starting in 5 seconds..."
sleep 5

python3 train_custom.py \
  --preset cpu \
  --use_scheduler \
  --monitor_generation \
  --generation_interval 1 \
  2>&1 | tee training.log
