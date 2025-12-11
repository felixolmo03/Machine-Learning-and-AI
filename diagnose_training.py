#!/usr/bin/env python3
"""
Diagnostic Script for Storyteller Training Issues

This script helps identify why the model generates garbage output.
Run this to check for common issues.

Usage:
    python diagnose_training.py --checkpoint checkpoints/checkpoint_step_1000.pt
"""

import argparse
from pathlib import Path
import torch
from transformers import PreTrainedTokenizerFast

from src.storyteller.model import StorytellerModel, ModelConfig


def check_checkpoint(checkpoint_path: Path):
    """Analyze a checkpoint for common issues."""
    print("\n" + "="*60)
    print("CHECKPOINT DIAGNOSTICS")
    print("="*60 + "\n")
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Check what's in the checkpoint
    print("\n📦 Checkpoint Contents:")
    for key in checkpoint.keys():
        print(f"  - {key}")
    
    # Check config
    if "config" in checkpoint:
        config_dict = checkpoint["config"]
        print("\n⚙️  Model Configuration:")
        print(f"  Vocab Size: {config_dict.get('vocab_size', 'N/A'):,}")
        print(f"  Hidden Size: {config_dict.get('hidden_size', 'N/A')}")
        print(f"  Num Layers: {config_dict.get('num_layers', 'N/A')}")
        print(f"  Num Heads: {config_dict.get('num_attention_heads', 'N/A')}")
        print(f"  Max Seq Length: {config_dict.get('max_seq_length', 'N/A')}")
    
    # Check training state
    if "epoch" in checkpoint:
        print(f"\n📊 Training State:")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Global Step: {checkpoint.get('global_step', 'N/A')}")
        print(f"  Best Val Loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
    
    # Check model weights
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"\n🔍 Model Weights Analysis:")
        
        # Check embedding weights
        if "token_embeddings.weight" in state_dict:
            emb_weight = state_dict["token_embeddings.weight"]
            print(f"  Token Embeddings:")
            print(f"    Shape: {emb_weight.shape}")
            print(f"    Mean: {emb_weight.mean().item():.6f}")
            print(f"    Std: {emb_weight.std().item():.6f}")
            print(f"    Min: {emb_weight.min().item():.6f}")
            print(f"    Max: {emb_weight.max().item():.6f}")
            
            # Check for NaN or Inf
            if torch.isnan(emb_weight).any():
                print(f"    ❌ WARNING: NaN values detected!")
            if torch.isinf(emb_weight).any():
                print(f"    ❌ WARNING: Inf values detected!")
        
        # Check LM head weights
        if "lm_head.weight" in state_dict:
            lm_weight = state_dict["lm_head.weight"]
            print(f"  LM Head:")
            print(f"    Shape: {lm_weight.shape}")
            print(f"    Mean: {lm_weight.mean().item():.6f}")
            print(f"    Std: {lm_weight.std().item():.6f}")
            
            # Check if weights are tied
            if "token_embeddings.weight" in state_dict:
                emb_weight = state_dict["token_embeddings.weight"]
                if torch.equal(emb_weight, lm_weight):
                    print(f"    ✓ Weights are tied with embeddings")
                else:
                    print(f"    ⚠️  Weights are NOT tied (this is unusual)")
        
        # Check a transformer block
        block_keys = [k for k in state_dict.keys() if k.startswith("blocks.0.")]
        if block_keys:
            print(f"  First Transformer Block:")
            for key in sorted(block_keys)[:5]:  # Show first 5 weights
                weight = state_dict[key]
                print(f"    {key}: mean={weight.mean().item():.6f}, std={weight.std().item():.6f}")
    
    # Check optimizer state
    if "optimizer_state_dict" in checkpoint:
        opt_state = checkpoint["optimizer_state_dict"]
        print(f"\n🎯 Optimizer State:")
        if "param_groups" in opt_state:
            for i, group in enumerate(opt_state["param_groups"]):
                print(f"  Group {i}:")
                print(f"    LR: {group.get('lr', 'N/A'):.2e}")
                print(f"    Weight Decay: {group.get('weight_decay', 'N/A')}")
                print(f"    Betas: {group.get('betas', 'N/A')}")


def test_generation(checkpoint_path: Path, tokenizer_path: Path, device: str = "cuda"):
    """Test generation with the checkpoint."""
    print("\n" + "="*60)
    print("GENERATION TEST")
    print("="*60 + "\n")
    
    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer):,})")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Create model
    config_dict = checkpoint["config"]
    config = ModelConfig(**config_dict)
    model = StorytellerModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Move to device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"✓ Model loaded on {device}")
    
    # Test prompts
    prompts = [
        "Once upon a time",
        "There was a little",
        "One day a",
        "The big dog",
    ]
    
    print("\n" + "-"*60)
    print("GENERATION SAMPLES")
    print("-"*60 + "\n")
    
    for prompt in prompts:
        print(f"Prompt: '{prompt}'")
        
        # Encode
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=100,
                temperature=0.8,
                top_k=50,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Generated: {text}\n")
        
        # Analyze tokens
        tokens = output_ids[0].tolist()
        print(f"Token IDs (first 20): {tokens[:20]}")
        print(f"Unique tokens: {len(set(tokens))}")
        print(f"Total tokens: {len(tokens)}")
        
        # Check for repetition
        if len(tokens) > 10:
            # Check if same token repeats
            repeats = sum(1 for i in range(len(tokens)-1) if tokens[i] == tokens[i+1])
            print(f"Adjacent repeats: {repeats} ({repeats/len(tokens)*100:.1f}%)")
        
        print("-"*60 + "\n")


def check_tokenizer(tokenizer_path: Path):
    """Check tokenizer for issues."""
    print("\n" + "="*60)
    print("TOKENIZER DIAGNOSTICS")
    print("="*60 + "\n")
    
    if not tokenizer_path.exists():
        print(f"❌ Tokenizer not found: {tokenizer_path}")
        return
    
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
    
    print(f"\n📝 Tokenizer Info:")
    print(f"  Vocab Size: {len(tokenizer):,}")
    print(f"  PAD Token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    print(f"  EOS Token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    print(f"  BOS Token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
    print(f"  UNK Token: '{tokenizer.unk_token}' (ID: {tokenizer.unk_token_id})")
    
    # Test encoding/decoding
    print(f"\n🧪 Encoding/Decoding Test:")
    test_text = "Once upon a time there was a little girl."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    print(f"  Original: '{test_text}'")
    print(f"  Encoded: {encoded}")
    print(f"  Decoded: '{decoded}'")
    
    if test_text.lower() == decoded.lower():
        print(f"  ✓ Round-trip successful")
    else:
        print(f"  ⚠️  Round-trip mismatch!")
    
    # Check special tokens
    print(f"\n🔍 Special Token Encoding:")
    for token_name in ["pad_token", "eos_token", "bos_token", "unk_token"]:
        token = getattr(tokenizer, token_name)
        if token:
            encoded = tokenizer.encode(token, add_special_tokens=False)
            print(f"  {token_name}: '{token}' -> {encoded}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose Storyteller training issues")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizers/storyteller-tokenizer",
                       help="Path to tokenizer")
    parser.add_argument("--device", type=str, default="cuda", help="Device for generation test")
    parser.add_argument("--skip_generation", action="store_true", help="Skip generation test")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)
    
    # Run diagnostics
    check_checkpoint(checkpoint_path)
    check_tokenizer(tokenizer_path)
    
    if not args.skip_generation:
        test_generation(checkpoint_path, tokenizer_path, args.device)
    
    print("\n" + "="*60)
    print("DIAGNOSTICS COMPLETE")
    print("="*60 + "\n")
    
    print("Common Issues to Check:")
    print("  1. ❌ NaN/Inf in weights -> Learning rate too high or gradient explosion")
    print("  2. ❌ All weights near zero -> Model not learning, check loss")
    print("  3. ❌ Repetitive output -> Label smoothing too high or vocab mismatch")
    print("  4. ❌ Nonsense output -> Not trained enough or tokenizer mismatch")
    print("  5. ❌ Same token repeated -> Temperature too low or top_k too small")
    print()


if __name__ == "__main__":
    main()
