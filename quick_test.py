#!/usr/bin/env python3
"""
Quick Test Script - Verify Everything Works

This script does a minimal training run to verify your setup works.
Should complete in 5-10 minutes on GPU.

Usage:
    python quick_test.py
"""

import torch
from pathlib import Path
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from src.storyteller.model import ModelConfig, StorytellerModel
from src.storyteller.data.dataset import StoryDataset
from torch.utils.data import DataLoader

print("\n" + "="*60)
print("STORYTELLER QUICK TEST")
print("="*60 + "\n")

# Create tiny test dataset
print("Creating test dataset...")
test_dir = Path("test_data")
test_dir.mkdir(exist_ok=True)

test_stories = [
    "Once upon a time there was a little girl named Lucy.",
    "She had a big red ball that she loved to play with.",
    "One day Lucy went to the park with her mom.",
    "At the park she saw a friendly dog.",
    "The dog wanted to play with Lucy's ball.",
    "Lucy threw the ball and the dog ran to get it.",
    "They played together all afternoon.",
    "When it was time to go home Lucy was very happy.",
    "She said goodbye to the dog and went home with her mom.",
    "That night Lucy dreamed about her new friend.",
] * 100  # Repeat to have enough data

train_file = test_dir / "train.txt"
with open(train_file, "w") as f:
    f.write("\n\n".join(test_stories))

print(f"✓ Created {len(test_stories)} test stories")

# Create tiny tokenizer
print("\nCreating test tokenizer...")
tokenizer_dir = test_dir / "tokenizer"
tokenizer_dir.mkdir(exist_ok=True)

tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=1000,  # Very small vocab
    special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
)

tokenizer.train([str(train_file)], trainer)
tokenizer.save(str(tokenizer_dir / "tokenizer.json"))

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=str(tokenizer_dir / "tokenizer.json"),
    unk_token="<unk>",
    pad_token="<pad>",
    bos_token="<s>",
    eos_token="</s>",
)
tokenizer.save_pretrained(str(tokenizer_dir))

print(f"✓ Tokenizer created (vocab size: {len(tokenizer)})")

# Create tiny model
print("\nCreating tiny model...")
config = ModelConfig(
    vocab_size=len(tokenizer),
    max_seq_length=64,  # Very short
    hidden_size=128,    # Very small
    num_layers=2,       # Very shallow
    num_attention_heads=4,
    intermediate_size=512,
    use_flash_attention=False,
    positional_encoding="learned",
)

model = StorytellerModel(config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"✓ Model created on {device}")

# Create dataset
print("\nCreating dataset...")
dataset = StoryDataset(
    data_path=str(train_file),
    tokenizer=tokenizer,
    max_seq_length=64,
)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    drop_last=True,
)

print(f"✓ Dataset created ({len(dataset)} examples, {len(dataloader)} batches)")

# Train for a few steps
print("\nTraining for 10 steps...")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
model.train()

losses = []
for step, batch in enumerate(dataloader):
    if step >= 10:
        break
    
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        return_dict=True,
    )
    
    loss = outputs["loss"]
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    
    losses.append(loss.item())
    print(f"  Step {step+1}/10: Loss = {loss.item():.4f}")

print(f"\n✓ Training complete!")
print(f"  Initial loss: {losses[0]:.4f}")
print(f"  Final loss: {losses[-1]:.4f}")
print(f"  Loss decreased: {losses[0] - losses[-1]:.4f}")

if losses[-1] < losses[0]:
    print(f"  ✅ Loss is decreasing - model is learning!")
else:
    print(f"  ⚠️  Loss is not decreasing - something might be wrong")

# Test generation
print("\nTesting generation...")
model.eval()

prompt = "Once upon a time"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=30,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
    )

text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"\nPrompt: '{prompt}'")
print(f"Generated: {text}")

print("\n" + "="*60)
print("TEST COMPLETE!")
print("="*60 + "\n")

print("Results:")
if losses[-1] < losses[0]:
    print("  ✅ Model can train (loss decreases)")
else:
    print("  ❌ Model not training properly (loss not decreasing)")

if len(output_ids[0]) > len(input_ids[0]):
    print("  ✅ Model can generate (produces new tokens)")
else:
    print("  ❌ Model not generating (no new tokens)")

print("\nNext steps:")
print("  1. If tests passed, run: python train_simple.py --preset tinystories --epochs 3 --download_data")
print("  2. If tests failed, check your PyTorch installation")
print()
