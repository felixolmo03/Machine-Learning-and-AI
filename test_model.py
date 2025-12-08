#!/usr/bin/env python3
"""
Test a trained Storyteller model checkpoint.

Usage:
    python3 test_model.py checkpoints/checkpoint_step_5500.pt
    python3 test_model.py checkpoints/best_model.pt --prompt "Once upon a time"
"""

import argparse
import torch
from pathlib import Path

from src.storyteller.model import ModelConfig, StorytellerModel
from train_custom import TikTokenWrapper


def load_model_from_checkpoint(checkpoint_path: Path):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Recreate model config
    config = ModelConfig(**checkpoint["config"])
    model = StorytellerModel(config)
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    epoch = checkpoint.get('epoch', 0)
    global_step = checkpoint.get('global_step', 0)
    
    print(f"✓ Model loaded (epoch {epoch}, step {global_step})")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Show scheduler info if available
    if 'scheduler_state_dict' in checkpoint:
        print(f"  Scheduler: ✓ (with warmup)")
    
    return model, config


def generate_story(model, tokenizer, prompt: str, max_length: int = 200, temperature: float = 0.8):
    """Generate a story from a prompt."""
    print(f"\nPrompt: {prompt}")
    print("-" * 60)
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            do_sample=True,
        )
    
    # Decode
    generated_text = tokenizer.decode(output_ids[0])
    print(generated_text)
    print("-" * 60)
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Test Storyteller model")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Story prompt")
    parser.add_argument("--max_length", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--num_stories", type=int, default=3, help="Number of stories to generate")
    
    args = parser.parse_args()
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    model, config = load_model_from_checkpoint(checkpoint_path)
    
    # Load tokenizer (assuming TikToken was used)
    print("\nLoading tokenizer...")
    tokenizer = TikTokenWrapper(encoding_name="cl100k_base")
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer):,})")
    
    # Generate stories
    print(f"\n{'='*60}")
    print(f"GENERATING {args.num_stories} STORIES")
    print(f"{'='*60}")
    
    prompts = [
        args.prompt,
        "In a land far away",
        "There once was a",
        "Long ago",
        "One day",
    ]
    
    for i in range(args.num_stories):
        prompt = prompts[i % len(prompts)]
        print(f"\n\nStory {i+1}:")
        generate_story(model, tokenizer, prompt, args.max_length, args.temperature)
    
    print(f"\n{'='*60}")
    print("TESTING COMPLETE!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
