# CPU Training Guide for Mac/Laptop

This guide will help you train a storytelling language model on your Mac's CPU.

## Quick Start (Easiest)

```bash
./train_cpu.sh
```

That's it! This will start training with optimal CPU settings.

## What to Expect

### Model Size
- **~40M parameters** (6 layers, 512 hidden size)
- Small enough to train on CPU, large enough for coherent stories
- Comparable to small GPT-2 models

### Training Speed
- **~10-20 seconds per step** on modern Mac (M1/M2/Intel i7+)
- **~1,000 steps in 3-5 hours**
- **~20,000 steps in 3-7 days**

### Quality Timeline
- **1,000-5,000 steps**: Random gibberish, learning basic tokenization
- **5,000-15,000 steps**: Starting to form words and short phrases
- **15,000-30,000 steps**: **Coherent sentences and simple stories** ✨
- **50,000+ steps**: Well-formed stories with good grammar

### Checkpoints
- Saved every **1,000 steps** in `checkpoints/` folder
- Each checkpoint is ~160MB
- You can test any checkpoint with: `python3 test_model.py checkpoints/checkpoint_step_XXXXX.pt`

## Manual Training Commands

### Basic CPU Training
```bash
python3 train_custom.py --preset cpu --use_scheduler
```

### With Generation Monitoring (see sample stories every epoch)
```bash
python3 train_custom.py --preset cpu --use_scheduler --monitor_generation
```

### Resume from Checkpoint
```bash
python3 train_custom.py --preset cpu --use_scheduler --resume checkpoints/checkpoint_step_10000.pt
```

### Custom Settings (advanced)
```bash
python3 train_custom.py \
  --num_layers 8 \
  --hidden_size 512 \
  --batch_size 4 \
  --gradient_accumulation_steps 2 \
  --use_scheduler
```

## Monitoring Training

### View Training Log
```bash
tail -f training.log
```

### Watch Progress (updates every 5 seconds)
```bash
watch -n 5 'tail -20 training.log'
```

### Test Current Checkpoint
```bash
# Find latest checkpoint
ls -t checkpoints/*.pt | head -1

# Test it
python3 test_model.py checkpoints/checkpoint_step_20000.pt --num_stories 5
```

## Optimizations for CPU Training

The `--preset cpu` configuration includes:
- **Small model**: 6 layers, 512 hidden size (~40M params)
- **Efficient batch size**: 4 with 2x gradient accumulation (effective=8)
- **Shorter sequences**: 384 tokens instead of 512 (20% faster)
- **Higher learning rate**: 5e-4 (helps smaller models learn faster)
- **Frequent checkpoints**: Every 1,000 steps
- **CPU-optimized dataloading**: No multiprocessing overhead

## Tips for Best Results

1. **Let it run continuously**: CPU training is slow but steady. Best to let it run for several days.

2. **Don't interrupt during saves**: Wait for "Checkpoint saved" message to complete.

3. **Monitor disk space**: Each checkpoint is ~160MB. Training to 30K steps = ~30 checkpoints = ~4.8GB.

4. **Test early checkpoints**: Even at 5K steps you can see interesting patterns emerging.

5. **Use generation monitoring**: Add `--monitor_generation` to see sample stories every epoch.

## GPU Training (When Available)

If you get access to a GPU later:

### Medium Model (110M params)
```bash
CUDA_VISIBLE_DEVICES=1 python3 train_custom.py --preset gpu --use_scheduler
```

### Large Model (300M params)
```bash
CUDA_VISIBLE_DEVICES=1 python3 train_custom.py --preset gpu_large --use_scheduler
```

## Troubleshooting

### "Out of Memory" Error
Reduce batch size:
```bash
python3 train_custom.py --preset cpu --batch_size 2 --use_scheduler
```

### Training Too Slow
- Close other applications
- Use shorter sequences: `--max_seq_length 256`
- Reduce model size: `--num_layers 4 --hidden_size 384`

### Checkpoint Size Too Large
Clean up old checkpoints:
```bash
cd checkpoints && ls -t checkpoint_step_*.pt | tail -n +5 | xargs rm -f
```
This keeps only the 4 most recent checkpoints.

## Expected Timeline

**Realistic expectations for CPU training:**

| Steps | Time (approx) | Quality |
|-------|---------------|---------|
| 1,000 | 3-5 hours | Random tokens |
| 5,000 | 1-2 days | Learning words |
| 10,000 | 2-3 days | Short phrases |
| 20,000 | 4-6 days | **Coherent sentences** |
| 30,000 | 6-8 days | **Simple stories** |
| 50,000 | 10-14 days | Good quality stories |

**Remember**: This is training from scratch with 40M parameters on CPU. Even getting coherent text in a week is impressive!

## Questions?

- Check the training log: `tail -100 training.log`
- Monitor progress: `watch -n 5 'tail -20 training.log'`
- Test a checkpoint: `python3 test_model.py checkpoints/checkpoint_step_XXXXX.pt`
