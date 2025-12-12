"""
Complete TinyStories Training Script
Run this entire file to train a model and generate stories
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast, GPT2Tokenizer
from datasets import load_dataset
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# MODEL DEFINITION
# ============================================================================

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    max_seq_length: int = 256
    hidden_size: int = 384
    num_layers: int = 6
    num_attention_heads: int = 6
    intermediate_size: int = 1536
    dropout: float = 0.1


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.scale = math.sqrt(self.head_dim)
        
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.max_seq_length, config.max_seq_length), diagonal=1).bool()
        )
    
    def forward(self, x):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        scores = scores.masked_fill(self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout),
        )
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.mlp(self.ln2(x))
        return x


class StoryModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_emb = nn.Embedding(config.max_seq_length, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # Weight tying
        
        self.apply(self._init_weights)
        print(f"Model parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        
        tok_emb = self.token_emb(input_ids)
        pos_emb = self.pos_emb(torch.arange(T, device=input_ids.device))
        x = self.dropout(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), 
                                   labels[:, 1:].reshape(-1))
        
        return {"logits": logits, "loss": loss}
    
    @torch.no_grad()
    def generate(self, input_ids, max_length=200, temperature=0.8, top_k=50):
        self.eval()
        for _ in range(max_length - input_ids.size(1)):
            input_ids_cond = input_ids[:, -self.config.max_seq_length:]
            logits = self.forward(input_ids_cond)["logits"][:, -1, :]
            logits = logits / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


# ============================================================================
# DATASET
# ============================================================================

class StoryDataset(Dataset):
    def __init__(self, stories, tokenizer, max_length=256):
        self.examples = []
        
        # Get EOS token ID (use a fallback if not set)
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            eos_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        
        print(f"Using EOS token ID: {eos_id}")
        print("Tokenizing stories...")
        
        all_tokens = []
        for story in tqdm(stories):
            tokens = tokenizer.encode(story, add_special_tokens=False)
            all_tokens.extend(tokens)
            all_tokens.append(eos_id)
        
        # Create non-overlapping chunks
        all_tokens = torch.tensor(all_tokens, dtype=torch.long)
        for i in range(0, len(all_tokens) - max_length, max_length):
            self.examples.append(all_tokens[i:i + max_length])
        
        print(f"Created {len(self.examples):,} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return {"input_ids": self.examples[idx], "labels": self.examples[idx]}


# ============================================================================
# TRAINING
# ============================================================================

def train():
    # Config
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_STORIES = 200000
    BATCH_SIZE = 32
    EPOCHS = 8
    LR = 3e-4
    MAX_LENGTH = 256
    
    print(f"Device: {DEVICE}")
    print(f"Training on {NUM_STORIES:,} stories for {EPOCHS} epochs\n")
    
    # Load data
    print("Downloading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    stories = [s["text"] for s in dataset.select(range(NUM_STORIES))]
    print(f"Loaded {len(stories):,} stories\n")
    
    # Tokenizer - use GPT2Tokenizer and set eos token properly
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    
    # Dataset and DataLoader
    train_dataset = StoryDataset(stories, tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Model
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_length=MAX_LENGTH,
    )
    model = StoryModel(config).to(DEVICE)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    total_steps = len(train_loader) * EPOCHS
    
    # Training loop
    print(f"\nStarting training...")
    print(f"Steps per epoch: {len(train_loader)}")
    print(f"Total steps: {total_steps}\n")
    
    model.train()
    step = 0
    
    for epoch in range(EPOCHS):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            output = model(input_ids, labels=labels)
            loss = output["loss"]
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Generate sample every 500 steps
            if step % 500 == 0:
                model.eval()
                prompt = "Once upon a time"
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
                output_ids = model.generate(input_ids, max_length=100)
                print(f"\n[Step {step}] Sample: {tokenizer.decode(output_ids[0])[:200]}...\n")
                model.train()
    
    # Save model
    print("\nSaving model...")
    torch.save({"model": model.state_dict(), "config": config}, "story_model.pt")
    print("✓ Model saved to story_model.pt")
    
    return model, tokenizer


# ============================================================================
# GENERATION
# ============================================================================

def generate_story(model, tokenizer, prompt="Once upon a time", max_length=300):
    device = next(model.parameters()).device
    model.eval()
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(input_ids, max_length=max_length, temperature=0.8, top_k=50)
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    # Train the model
    model, tokenizer = train()
    
    # Generate stories
    print("\n" + "="*60)
    print("GENERATING STORIES")
    print("="*60 + "\n")
    
    prompts = [
        "Once upon a time there was a little girl",
        "Tom was a young boy who loved",
        "The magical forest was full of",
    ]
    
    for prompt in prompts:
        print(f"Prompt: {prompt}")
        print(f"Story: {generate_story(model, tokenizer, prompt)}")
        print("-" * 40 + "\n")