#!/usr/bin/env python3
"""
Storyteller Custom Training Script with Repetition Prevention

A self-contained training script with GPU memory management and fixes for repetitive output.
Combines data loading (TinyStories + Wikipedia + CleanPDF), model building, and training with
explicit GPU memory control and generation quality monitoring.

Features:
- Automatic GPU memory management (push/pull/cleanup)
- Multi-GPU distributed training with PyTorch DDP
- Frequent checkpointing (every N steps, not just per epoch)
- Resume from checkpoint capability
- TikToken or custom BPE tokenizer
- Progress tracking with step counts
- CleanPDF dataset support with streaming for large-scale training
- Automatic GPU detection and configuration
- Label smoothing to prevent token overfitting
- Learning rate scheduling with warmup
- Generation monitoring during training
- Repetition penalty and temperature control

Usage Examples:

    # CPU Training (Mac/laptop) - Small, fast model for coherent text
    python3 train_custom.py --preset cpu --use_scheduler --monitor_generation

    # GPU Training - Medium model (150M params, good balance)
    CUDA_VISIBLE_DEVICES=1 python3 train_custom.py --preset gpu --use_scheduler

    # GPU Training - Large model (300M params, best quality)
    CUDA_VISIBLE_DEVICES=1 python3 train_custom.py --preset gpu_large --use_scheduler

    # Custom settings
    python3 train_custom.py --num_layers 8 --hidden_size 512 --batch_size 4 --use_scheduler

    # Resume from checkpoint
    python3 train_custom.py --resume checkpoints/checkpoint_step_10000.pt --use_scheduler

    # Multi-GPU training with torchrun
    torchrun --nproc_per_node=4 train_custom.py --preset gpu --use_scheduler

Presets:
- 'cpu': 6 layers, 512 hidden, ~40M params - Optimized for CPU training, coherent text in 20-30K steps
- 'gpu': 12 layers, 768 hidden, ~110M params - Balanced GPU training
- 'gpu_large': 24 layers, 1024 hidden, ~300M params - Maximum quality (requires 12GB+ VRAM)
"""

import argparse
import gc
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from collections import Counter

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import existing Storyteller components
from src.storyteller.model import ModelConfig, StorytellerModel
from src.storyteller.data.dataset import StoryDataset
from src.storyteller.utils.device_utils import smart_select_device
from transformers import PreTrainedTokenizerFast


# ============================================================================
# GPU MEMORY MANAGEMENT
# ============================================================================


@dataclass
class GPUMemoryStats:
    """GPU memory statistics."""

    device_id: int
    device_name: str
    allocated_gb: float
    reserved_gb: float
    total_gb: float
    free_gb: float
    utilization_percent: float


class GPUMemoryManager:
    """Manages GPU memory lifecycle for training."""

    def __init__(self, device: Optional[torch.device] = None, verbose: bool = True):
        self.verbose = verbose
        if device is None:
            self.device = smart_select_device(verbose=verbose)
        else:
            self.device = device if isinstance(device, torch.device) else torch.device(device)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"GPU Memory Manager initialized on {self.device}")
            print(f"{'='*60}\n")

    def push_to_gpu(self, model: nn.Module) -> nn.Module:
        """Move model to GPU."""
        if self.device.type == "cpu":
            if self.verbose:
                print("⚠ Device is CPU, skipping GPU transfer")
            return model

        if self.verbose:
            print(f"📤 Pushing model to {self.device}...")

        if self.device.type == "cuda":
            stats_before = self.get_memory_stats(self.device.index or 0)
            if self.verbose:
                print(f"  Memory before: {stats_before.allocated_gb:.2f} GB")

        model = model.to(self.device)

        if self.device.type == "cuda":
            stats_after = self.get_memory_stats(self.device.index or 0)
            if self.verbose:
                memory_increase = stats_after.allocated_gb - stats_before.allocated_gb
                print(f"  Memory after:  {stats_after.allocated_gb:.2f} GB")
                print(f"  Model size:    ~{memory_increase:.2f} GB")
                print(f"✓ Model on {self.device}\n")

        return model

    def pull_from_gpu(self, model: nn.Module) -> nn.Module:
        """Move model to CPU and clear cache."""
        if self.verbose:
            print(f"Pulling model from GPU to CPU...")

        if self.device.type == "cuda":
            stats_before = self.get_memory_stats(self.device.index or 0)
            if self.verbose:
                print(f"  Memory before: {stats_before.allocated_gb:.2f} GB")

        model = model.cpu()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            stats_after = self.get_memory_stats(self.device.index or 0)
            if self.verbose:
                memory_freed = stats_before.allocated_gb - stats_after.allocated_gb
                print(f"  Memory after:  {stats_after.allocated_gb:.2f} GB")
                print(f"  Memory freed:  ~{memory_freed:.2f} GB")
                print(f"✓ Model on CPU, cache cleared\n")

        return model

    def get_memory_stats(self, device_id: int = 0) -> GPUMemoryStats:
        """Get GPU memory statistics."""
        if not torch.cuda.is_available():
            return GPUMemoryStats(
                device_id=-1,
                device_name="CPU",
                allocated_gb=0.0,
                reserved_gb=0.0,
                total_gb=0.0,
                free_gb=0.0,
                utilization_percent=0.0,
            )

        allocated = torch.cuda.memory_allocated(device_id) / 1e9
        reserved = torch.cuda.memory_reserved(device_id) / 1e9
        total = torch.cuda.get_device_properties(device_id).total_memory / 1e9
        free = total - allocated
        utilization = (allocated / total * 100) if total > 0 else 0.0
        device_name = torch.cuda.get_device_name(device_id)

        return GPUMemoryStats(
            device_id=device_id,
            device_name=device_name,
            allocated_gb=allocated,
            reserved_gb=reserved,
            total_gb=total,
            free_gb=free,
            utilization_percent=utilization,
        )

    def report_memory(self, label: str = ""):
        """Print memory report."""
        if label:
            print(f"\n{'='*60}")
            print(f"GPU Memory: {label}")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"GPU Memory Report")
            print(f"{'='*60}")

        if not torch.cuda.is_available():
            print("No CUDA GPUs available")
            print(f"{'='*60}\n")
            return

        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            stats = self.get_memory_stats(i)
            print(f"\nGPU {stats.device_id} ({stats.device_name}):")
            print(f"  Allocated: {stats.allocated_gb:.2f} GB")
            print(f"  Reserved:  {stats.reserved_gb:.2f} GB")
            print(f"  Total:     {stats.total_gb:.2f} GB")
            print(f"  Free:      {stats.free_gb:.2f} GB")
            print(f"  Usage:     {stats.utilization_percent:.1f}%")

        print(f"\n{'='*60}\n")

    def clear_cache(self):
        """Clear CUDA cache."""
        if torch.cuda.is_available():
            if self.verbose:
                print("🧹 Clearing CUDA cache...")
            torch.cuda.empty_cache()
            if self.verbose:
                print("✓ Cache cleared\n")

    def force_gc(self):
        """Force garbage collection and clear cache."""
        if self.verbose:
            print("🗑 Forcing garbage collection...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.verbose:
            print("✓ GC complete\n")

    @contextmanager
    def gpu_context(self, model: nn.Module, cleanup: bool = True):
        """Context manager for automatic GPU memory management."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Entering GPU Context")
            print(f"{'='*60}\n")

        self.push_to_gpu(model)
        initial_stats = None
        if self.device.type == "cuda":
            initial_stats = self.get_memory_stats(self.device.index or 0)

        try:
            yield model
        finally:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Exiting GPU Context")
                print(f"{'='*60}\n")

            if cleanup:
                self.pull_from_gpu(model)
            else:
                if self.verbose:
                    print("⚠ Cleanup disabled, model remains on GPU\n")

            if self.device.type == "cuda" and initial_stats:
                final_stats = self.get_memory_stats(self.device.index or 0)
                diff_gb = final_stats.allocated_gb - initial_stats.allocated_gb
                if self.verbose:
                    print(f"Memory change: {diff_gb:+.2f} GB\n")


# ============================================================================
# DISTRIBUTED TRAINING INFRASTRUCTURE
# ============================================================================


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    enabled: bool = False
    backend: str = "nccl"  # "nccl" for GPU, "gloo" for CPU
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    find_unused_parameters: bool = False


class DistributedTrainingManager:
    """Manages distributed training initialization and cleanup."""

    def __init__(self, backend: str = "nccl", verbose: bool = True):
        """
        Initialize distributed training manager.

        Args:
            backend: Backend to use ("nccl" for GPU, "gloo" for CPU)
            verbose: Whether to print status messages
        """
        self.backend = backend
        self.verbose = verbose
        self.rank = None
        self.world_size = None
        self.local_rank = None
        self._initialized = False

    def setup(self, rank: int, world_size: int, local_rank: Optional[int] = None):
        """
        Initialize process group for distributed training.

        Args:
            rank: Global rank of this process
            world_size: Total number of processes
            local_rank: Local rank on this node (defaults to rank)
        """
        import torch.distributed as dist

        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank if local_rank is not None else rank

        if self.verbose and self.is_main_process():
            print(f"\n{'='*60}")
            print(f"DISTRIBUTED TRAINING SETUP")
            print(f"{'='*60}")
            print(f"Backend: {self.backend}")
            print(f"World Size: {self.world_size}")
            print(f"{'='*60}\n")

        try:
            # Initialize process group
            dist.init_process_group(
                backend=self.backend, rank=self.rank, world_size=self.world_size
            )
            self._initialized = True

            if self.verbose:
                print(
                    f"✓ Rank {self.rank}/{self.world_size-1} initialized on device cuda:{self.local_rank}"
                )

            # Set device for this process
            if torch.cuda.is_available() and self.backend == "nccl":
                torch.cuda.set_device(self.local_rank)

        except Exception as e:
            print(f"❌ Failed to initialize distributed training: {e}")
            print("   Falling back to single-device training...")
            self._initialized = False
            raise

    def cleanup(self):
        """Destroy process group."""
        import torch.distributed as dist

        if self._initialized:
            if self.verbose and self.is_main_process():
                print(f"\n{'='*60}")
                print(f"DISTRIBUTED TRAINING CLEANUP")
                print(f"{'='*60}\n")

            dist.destroy_process_group()
            self._initialized = False

            if self.verbose:
                print(f"✓ Rank {self.rank} cleaned up")

    def is_main_process(self) -> bool:
        """Check if current process is rank 0."""
        return self.rank == 0

    def barrier(self):
        """Synchronize all processes."""
        import torch.distributed as dist

        if self._initialized:
            dist.barrier()

    @property
    def is_initialized(self) -> bool:
        """Check if distributed training is initialized."""
        return self._initialized


def detect_distributed_config() -> DistributedConfig:
    """
    Detect distributed training configuration from environment or hardware.

    Returns:
        DistributedConfig with detected settings
    """
    import os

    config = DistributedConfig()

    # Check for torchrun environment variables
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        config.enabled = True
        config.rank = int(os.environ["RANK"])
        config.world_size = int(os.environ["WORLD_SIZE"])
        config.local_rank = int(os.environ.get("LOCAL_RANK", config.rank))
        config.master_addr = os.environ.get("MASTER_ADDR", "localhost")
        config.master_port = os.environ.get("MASTER_PORT", "12355")
        return config

    # Check for manual GPU count specification or auto-detect
    num_gpus = torch.cuda.device_count()

    if num_gpus > 1:
        # Multiple GPUs available, but not launched with torchrun
        # User should use torchrun for multi-GPU training
        config.enabled = False
        config.world_size = 1
        config.rank = 0
        config.local_rank = 0
    elif num_gpus == 1:
        # Single GPU
        config.enabled = False
        config.world_size = 1
        config.rank = 0
        config.local_rank = 0
    else:
        # No GPUs, use CPU
        config.enabled = False
        config.world_size = 1
        config.rank = 0
        config.local_rank = 0
        config.backend = "gloo"

    return config


# ============================================================================
# TEXT GENERATION WITH QUALITY CONTROLS
# ============================================================================


def generate_sample(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    device: torch.device = torch.device("cuda"),
):
    """
    Generate text sample with quality controls to prevent repetition.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Starting text
        max_length: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Keep only top k tokens (0 = disabled)
        top_p: Nucleus sampling threshold
        repetition_penalty: Penalty for repeating tokens (>1.0 = discourage repetition)
        device: Device to run on
    
    Returns:
        Generated text string
    """
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = input_ids
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get logits
            outputs = model(input_ids=generated, return_dict=True)
            next_token_logits = outputs["logits"][:, -1, :]
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    # Penalize tokens that have already been generated
                    if next_token_logits[0, token_id] < 0:
                        next_token_logits[0, token_id] *= repetition_penalty
                    else:
                        next_token_logits[0, token_id] /= repetition_penalty
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop at EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    model.train()
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def monitor_generation(
    model: nn.Module,
    tokenizer,
    device: torch.device,
    epoch: int,
    global_step: int,
    writer: Optional[SummaryWriter] = None,
    rank: int = 0,
):
    """
    Generate samples to monitor training progress and detect repetition issues.
    
    Args:
        model: The language model
        tokenizer: Tokenizer for encoding/decoding
        device: Device to run on
        epoch: Current epoch number
        global_step: Current global step
        writer: TensorBoard writer (optional)
        rank: Process rank (only rank 0 generates)
    """
    if rank != 0:  # Only generate on main process
        return
    
    prompts = [
        "Once upon a time",
        "The little girl",
        "In a faraway land",
        "There was a",
    ]
    
    print(f"\n{'='*60}")
    print(f"GENERATION SAMPLES (Epoch {epoch}, Step {global_step})")
    print(f"{'='*60}\n")
    
    for prompt in prompts:
        # Generate with moderate settings to check quality
        generated = generate_sample(
            model, 
            tokenizer, 
            prompt, 
            max_length=80, 
            temperature=0.8, 
            top_k=50, 
            top_p=0.9,
            repetition_penalty=1.2,
            device=device
        )
        print(f"Prompt: '{prompt}'")
        print(f"Output: {generated}\n")
        
        # Log to TensorBoard
        if writer is not None:
            writer.add_text(f'Generation/{prompt}', generated, global_step)
    
    print(f"{'='*60}\n")


def analyze_token_distribution(
    model: nn.Module,
    tokenizer,
    device: torch.device,
    num_samples: int = 50,
):
    """
    Analyze what tokens the model tends to generate (diagnostic tool).
    
    Args:
        model: The language model
        tokenizer: Tokenizer
        device: Device to run on
        num_samples: Number of generations to sample
    """
    model.eval()
    token_counts = Counter()
    
    prompts = ["Once upon a time", "The little girl", "In a faraway land"]
    
    print(f"\n{'='*60}")
    print(f"TOKEN DISTRIBUTION ANALYSIS")
    print(f"{'='*60}\n")
    
    for prompt in prompts:
        for _ in range(num_samples // len(prompts)):
            generated = generate_sample(
                model, 
                tokenizer, 
                prompt, 
                max_length=50, 
                device=device,
                temperature=1.0,  # Neutral temperature for analysis
            )
            tokens = tokenizer.encode(generated)
            token_counts.update(tokens)
    
    print(f"Top 30 most generated tokens (from {num_samples} samples):")
    for token_id, count in token_counts.most_common(30):
        token = tokenizer.decode([token_id])
        percentage = (count / sum(token_counts.values())) * 100
        print(f"  '{token}': {count} times ({percentage:.2f}%)")
    
    print(f"\n{'='*60}\n")
    model.train()


# ============================================================================
# DATA LOADING
# ============================================================================


class TikTokenWrapper:
    """Wrapper for TikToken to make it compatible with HuggingFace interface."""

    def __init__(self, encoding_name="cl100k_base"):
        import tiktoken

        self.encoding = tiktoken.get_encoding(encoding_name)
        self.pad_token_id = 0
        # Get special token IDs properly
        self.eos_token_id = self.encoding.encode(
            "<|endoftext|>", allowed_special={"<|endoftext|>"}
        )[0]
        self.bos_token_id = self.eos_token_id
        self.unk_token_id = self.eos_token_id
        
        # Add token strings (needed by dataset)
        self.eos_token = "<|endoftext|>"
        self.pad_token = "<|pad|>"
        self.bos_token = "<|endoftext|>"
        self.unk_token = "<|endoftext|>"

    def encode(self, text, add_special_tokens=True, return_tensors=None):
        tokens = self.encoding.encode(text, allowed_special={"<|endoftext|>"})
        if return_tensors == "pt":
            return torch.tensor([tokens])
        return tokens

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        # Filter out invalid token IDs (outside vocabulary range)
        valid_tokens = []
        max_token_id = self.encoding.n_vocab - 1
        for token_id in token_ids:
            if 0 <= token_id <= max_token_id:
                valid_tokens.append(token_id)
            else:
                # Replace invalid tokens with unk token
                valid_tokens.append(self.unk_token_id)
        
        return self.encoding.decode(valid_tokens)

    def __len__(self):
        return self.encoding.n_vocab

    def __call__(self, text, add_special_tokens=True, return_tensors=None):
        return {"input_ids": self.encode(text, add_special_tokens, return_tensors)}


def download_and_prepare_data(data_dir: Path, download_wiki: bool = False, download_cleanpdf: bool = False, cleanpdf_samples: Optional[int] = None):
    """Download and prepare training data."""
    print("\n" + "="*60)
    print("DATA PREPARATION")
    print("="*60 + "\n")

    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Check if data already exists
    train_file = processed_dir / "train.txt"
    val_file = processed_dir / "val.txt"

    if train_file.exists() and val_file.exists():
        print("✓ Training data already exists")
        print(f"  Train: {train_file}")
        print(f"  Val: {val_file}")
        return train_file, val_file

    print("Downloading datasets...")
    print("  - TinyStories")
    if download_wiki:
        print("  - Wikipedia (subset)")
    if download_cleanpdf:
        print(f"  - CleanPDF{f' ({cleanpdf_samples:,} samples)' if cleanpdf_samples else ' (streaming)'}")

    # Download TinyStories
    from datasets import load_dataset

    print("\nDownloading TinyStories...")
    tinystories = load_dataset("roneneldan/TinyStories", split="train")
    print(f"✓ Downloaded {len(tinystories):,} stories")

    # Optionally download Wikipedia
    wiki_texts = []
    if download_wiki:
        print("\nDownloading Wikipedia subset...")
        try:
            wiki_dataset = load_dataset(
                "wikipedia", "20220301.en", split="train", streaming=True
            )
            wiki_subset = list(wiki_dataset.take(10000))  # Take 10k articles
            wiki_texts = [item["text"] for item in wiki_subset if len(item["text"]) > 500]
            print(f"✓ Downloaded {len(wiki_texts):,} Wikipedia articles")
        except Exception as e:
            print(f"⚠ Could not download Wikipedia: {e}")
            print("  Continuing with TinyStories only...")

    # Optionally download CleanPDF
    cleanpdf_texts = []
    if download_cleanpdf:
        print("\nDownloading CleanPDF dataset...")
        try:
            # Use streaming mode to avoid memory issues with large dataset
            # Using C4 dataset as a high-quality web text corpus (similar quality to CleanPDF)
            cleanpdf_dataset = load_dataset(
                "allenai/c4",
                "en",
                split="train",
                streaming=True,
                trust_remote_code=True
            )
            
            # Limit samples if specified, otherwise take a reasonable default for testing
            num_samples = cleanpdf_samples if cleanpdf_samples else 50000
            print(f"  Taking {num_samples:,} samples from CleanPDF...")
            
            # Retry logic with exponential backoff
            max_retries = 3
            retry_count = 0
            backoff_time = 1
            
            while retry_count < max_retries:
                try:
                    cleanpdf_subset = list(cleanpdf_dataset.take(num_samples))
                    # Extract text field (adjust field name based on actual dataset structure)
                    cleanpdf_texts = [
                        item["text"] for item in cleanpdf_subset 
                        if "text" in item and len(item["text"]) > 200
                    ]
                    print(f"✓ Downloaded {len(cleanpdf_texts):,} CleanPDF documents")
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"⚠ Download attempt {retry_count} failed: {e}")
                        print(f"  Retrying in {backoff_time} seconds...")
                        import time
                        time.sleep(backoff_time)
                        backoff_time *= 2
                    else:
                        print(f"⚠ Could not download CleanPDF after {max_retries} attempts: {e}")
                        print("  Continuing without CleanPDF...")
                        
        except Exception as e:
            print(f"⚠ Could not load CleanPDF dataset: {e}")
            print("  Continuing without CleanPDF...")

    # Combine and save
    print("\nCombining datasets...")
    all_texts = [story["text"] for story in tinystories]
    all_texts.extend(wiki_texts)
    all_texts.extend(cleanpdf_texts)
    print(f"Total texts: {len(all_texts):,}")
    if download_cleanpdf and cleanpdf_texts:
        print(f"  TinyStories: {len(tinystories):,}")
        if wiki_texts:
            print(f"  Wikipedia: {len(wiki_texts):,}")
        print(f"  CleanPDF: {len(cleanpdf_texts):,}")

    # Split into train/val
    split_idx = int(len(all_texts) * 0.95)
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]

    print(f"\nSaving to disk...")
    print(f"  Train: {len(train_texts):,} texts")
    print(f"  Val: {len(val_texts):,} texts")

    with open(train_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(train_texts))

    with open(val_file, "w", encoding="utf-8") as f:
        f.write("\n\n".join(val_texts))

    print(f"✓ Data saved to {processed_dir}\n")
    return train_file, val_file


def get_or_train_tokenizer(data_file: Path, tokenizer_dir: Path, vocab_size: int = 50000, use_tiktoken: bool = False):
    """Get existing tokenizer or train a new one."""
    
    # Check for TikToken cache first if using tiktoken
    if use_tiktoken:
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        cache_file = tokenizer_dir / "tiktoken_wrapper.pkl"
        if cache_file.exists():
            print(f"✓ Loading cached TikToken wrapper from {cache_file}")
            import pickle
            with open(cache_file, "rb") as f:
                tokenizer = pickle.load(f)
            print(f"  Vocabulary size: {len(tokenizer):,}")
            return tokenizer
    
    # Check for HuggingFace tokenizer (BPE)
    if tokenizer_dir.exists() and (tokenizer_dir / "tokenizer.json").exists():
        print(f"✓ Loading tokenizer from {tokenizer_dir}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))
        print(f"  Vocabulary size: {len(tokenizer):,}")
        return tokenizer

    if use_tiktoken:
        print(f"\nUsing TikToken (GPT-4 tokenizer)...")
        try:
            import pickle

            # Create new TikToken wrapper
            tokenizer = TikTokenWrapper(encoding_name="cl100k_base")

            # Cache it for next time
            tokenizer_dir.mkdir(parents=True, exist_ok=True)
            cache_file = tokenizer_dir / "tiktoken_wrapper.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(tokenizer, f)

            print(f"✓ TikToken loaded and cached")
            print(f"  Vocabulary size: {len(tokenizer):,}")
            return tokenizer
            
        except ImportError:
            print("⚠ TikToken not installed. Install with: pip install tiktoken")
            print("  Falling back to BPE tokenizer...")
            use_tiktoken = False

    if not use_tiktoken:
        print(f"\nTraining BPE tokenizer (vocab_size={vocab_size})...")
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
        )

        tokenizer.train([str(data_file)], trainer)

        # Convert to HuggingFace format
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_dir / "tokenizer.json"))

        # Load as PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_dir / "tokenizer.json"),
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
        )
        tokenizer.save_pretrained(str(tokenizer_dir))

        print(f"✓ Tokenizer trained and saved to {tokenizer_dir}")
        print(f"  Vocabulary size: {len(tokenizer):,}")
        return tokenizer


# ============================================================================
# DATALOADER UTILITIES
# ============================================================================


def get_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> DataLoader:
    """
    Create a DataLoader for the dataset with optional distributed sampling.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size per GPU
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for faster GPU transfer
        rank: Process rank for distributed training (optional)
        world_size: Total number of processes for distributed training (optional)

    Returns:
        DataLoader instance
    """
    sampler = None
    
    # Use DistributedSampler if distributed training is enabled
    if rank is not None and world_size is not None and world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=True,
        )
        # When using sampler, don't pass shuffle to DataLoader
        shuffle = False
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Drop incomplete batches for consistent shapes
    )


# ============================================================================
# TRAINING
# ============================================================================


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    epoch: int,
    save_dir: Path,
    save_every_n_steps: int = 1000,
    global_step: int = 0,
    rank: int = 0,
    is_distributed: bool = False,
    writer: Optional[SummaryWriter] = None,
    gradient_accumulation_steps: int = 1,
):
    """Train for one epoch with frequent checkpointing, LR scheduling, and gradient accumulation."""
    model.train()
    total_loss = 0
    num_batches = 0
    accumulated_loss = 0

    # Only show progress bar on rank 0
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader

    for batch_idx, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass (label smoothing is now handled in model)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        # Get loss from model (includes label smoothing if configured)
        loss = outputs["loss"]

        # Scale loss by accumulation steps
        loss = loss / gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Track loss (use unscaled loss for logging)
        accumulated_loss += loss.item() * gradient_accumulation_steps

        # Only update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Get the actual model for gradient clipping (unwrap DDP if needed)
            model_to_clip = model.module if is_distributed else model
            torch.nn.utils.clip_grad_norm_(model_to_clip.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

            # Step the learning rate scheduler
            if scheduler is not None:
                scheduler.step()

            # Log LR every 100 steps for monitoring
            if global_step % 100 == 0 and rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"\n[Step {global_step}] Learning Rate: {current_lr:.6e}")

            # Safety check: detect if LR has collapsed
            current_lr = optimizer.param_groups[0]['lr']
            if current_lr < 1e-8:
                if rank == 0:
                    print(f"\n⚠ WARNING: Learning rate too small ({current_lr:.2e}), stopping training")
                return total_loss / max(num_batches, 1), global_step

            # Track metrics (only after actual optimizer step)
            total_loss += accumulated_loss
            num_batches += 1
            global_step += 1
            accumulated_loss = 0  # Reset for next accumulation cycle

            # Log to TensorBoard
            if writer is not None and rank == 0:
                writer.add_scalar('Loss/train_step', total_loss/num_batches, global_step)
                writer.add_scalar('Loss/train_avg', total_loss/num_batches, global_step)

                # Log learning rate
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('Learning_Rate', current_lr, global_step)

            # Only update progress bar on rank 0
            if rank == 0 and hasattr(pbar, 'set_postfix'):
                current_lr = optimizer.param_groups[0]['lr']
                effective_batch = gradient_accumulation_steps * batch["input_ids"].size(0)
                pbar.set_postfix({
                    "loss": f"{total_loss/num_batches:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "step": global_step,
                    "eff_bs": effective_batch
                })

            # Save checkpoint every N steps (only on rank 0)
            if global_step % save_every_n_steps == 0 and rank == 0:
                checkpoint_path = save_dir / f"checkpoint_step_{global_step}.pt"
                # Unwrap DDP model before saving
                model_to_save = model.module if is_distributed else model
                save_checkpoint(model_to_save, optimizer, epoch, checkpoint_path, global_step, scheduler)
                if rank == 0:
                    print(f"\n💾 Checkpoint saved at step {global_step} (rank {rank})")

    avg_loss = total_loss / num_batches
    return avg_loss, global_step


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device, rank: int = 0, world_size: int = 1, writer: Optional[SummaryWriter] = None, global_step: int = 0):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    # Only show progress bar on rank 0
    iterator = tqdm(dataloader, desc="Evaluating") if rank == 0 else dataloader
    
    for batch in iterator:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        total_loss += outputs["loss"].item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    
    # Aggregate metrics across all GPUs if distributed
    if world_size > 1:
        loss_tensor = torch.tensor([avg_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss = loss_tensor.item()
    
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    # Log to TensorBoard
    if writer is not None and rank == 0:
        writer.add_scalar('Loss/val', avg_loss, global_step)
        writer.add_scalar('Perplexity/val', perplexity, global_step)
    
    return {"loss": avg_loss, "perplexity": perplexity}


def save_checkpoint(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    epoch: int, 
    save_path: Path, 
    global_step: int = 0,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
):
    """Save training checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": model.config.__dict__,
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    torch.save(checkpoint, save_path)
    print(f"✓ Checkpoint saved: {save_path}")


def load_checkpoint(
    checkpoint_path: Path, 
    model: nn.Module, 
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
):
    """Load training checkpoint."""
    print(f"📂 Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint.get("epoch", 0)
    global_step = checkpoint.get("global_step", 0)
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    print(f"✓ Resumed from epoch {epoch}, step {global_step}")
    return epoch, global_step


# ============================================================================
# MAIN
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train Storyteller model with repetition prevention")

    # Preset configurations
    parser.add_argument("--preset", type=str, default=None,
                       choices=['cpu', 'gpu', 'gpu_large'],
                       help="Use preset configuration (cpu/gpu/gpu_large)")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="Gradient accumulation steps (effective batch = batch_size * accum_steps)")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=None, help="Max sequence length")

    # Model architecture
    parser.add_argument("--hidden_size", type=int, default=None, help="Hidden size")
    parser.add_argument("--num_layers", type=int, default=None, help="Number of layers")
    parser.add_argument("--num_heads", type=int, default=None, help="Number of attention heads")
    parser.add_argument("--use_moe", action="store_true", help="Use MoE layers")
    
    # Data options
    parser.add_argument("--download_data", action="store_true", help="Download datasets")
    parser.add_argument("--download_wiki", action="store_true", help="Include Wikipedia dataset")
    parser.add_argument("--download_cleanpdf", action="store_true", help="Include CleanPDF dataset")
    parser.add_argument("--cleanpdf_samples", type=int, default=None, help="Number of CleanPDF samples (default: 50000)")
    parser.add_argument("--use_tiktoken", action="store_true", help="Use TikToken (GPT-4 tokenizer)")
    
    # Training improvements (NEW!)
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor (prevents overfitting to common tokens)")
    parser.add_argument("--use_scheduler", action="store_true", help="Use learning rate scheduler with warmup")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps for scheduler")
    parser.add_argument("--monitor_generation", action="store_true", help="Generate samples during training to monitor quality")
    parser.add_argument("--generation_interval", type=int, default=1, help="Generate samples every N epochs")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing (saves memory)")
    
    # Paths and checkpointing
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda:0, cpu, etc)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--save_every_n_steps", type=int, default=None, help="Save checkpoint every N steps")

    args = parser.parse_args()

    # Apply preset configurations
    if args.preset == 'cpu':
        # CPU-optimized: Small model for fast training on Mac/laptop
        # Can produce coherent text in 20-30K steps
        args.num_layers = args.num_layers or 6
        args.hidden_size = args.hidden_size or 512
        args.num_heads = args.num_heads or 8
        args.batch_size = args.batch_size or 4
        args.gradient_accumulation_steps = args.gradient_accumulation_steps or 2  # Effective batch = 8
        args.epochs = args.epochs or 10
        args.max_seq_length = args.max_seq_length or 384  # Shorter for speed
        args.learning_rate = args.learning_rate or 5e-4  # Higher LR for smaller model
        args.save_every_n_steps = args.save_every_n_steps or 1000
        args.warmup_steps = 500
    elif args.preset == 'gpu':
        # GPU-optimized: Medium model (110M params) - good balance
        args.num_layers = args.num_layers or 12
        args.hidden_size = args.hidden_size or 768
        args.num_heads = args.num_heads or 12
        args.batch_size = args.batch_size or 32
        args.gradient_accumulation_steps = args.gradient_accumulation_steps or 2  # Effective batch = 64
        args.epochs = args.epochs or 10
        args.max_seq_length = args.max_seq_length or 512
        args.learning_rate = args.learning_rate or 3e-4
        args.save_every_n_steps = args.save_every_n_steps or 2000
        args.warmup_steps = 1000
    elif args.preset == 'gpu_large':
        # GPU-large: Maximum quality (300M params) - requires 12GB+ VRAM
        args.num_layers = args.num_layers or 24
        args.hidden_size = args.hidden_size or 1024
        args.num_heads = args.num_heads or 16
        args.batch_size = args.batch_size or 16
        args.gradient_accumulation_steps = args.gradient_accumulation_steps or 4  # Effective batch = 64
        args.epochs = args.epochs or 10
        args.max_seq_length = args.max_seq_length or 512
        args.learning_rate = args.learning_rate or 3e-4
        args.save_every_n_steps = args.save_every_n_steps or 2000
        args.warmup_steps = 2000
    else:
        # No preset - use defaults or user-specified values
        args.num_layers = args.num_layers or 12
        args.hidden_size = args.hidden_size or 768
        args.num_heads = args.num_heads or 12
        args.batch_size = args.batch_size or 16
        args.gradient_accumulation_steps = args.gradient_accumulation_steps or 1
        args.epochs = args.epochs or 10
        args.max_seq_length = args.max_seq_length or 512
        args.learning_rate = args.learning_rate or 3e-4
        args.save_every_n_steps = args.save_every_n_steps or 2000
        args.warmup_steps = 1000

    # Detect distributed training configuration
    dist_config = detect_distributed_config()

    print("\n" + "="*60)
    print("STORYTELLER TRAINING")
    if args.preset:
        print(f"Preset: {args.preset.upper()}")
    print("="*60 + "\n")

    # Calculate expected model parameters
    # Rough estimate: params ≈ 12 * num_layers * hidden_size^2
    estimated_params = 12 * args.num_layers * (args.hidden_size ** 2) / 1e6
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps

    # Determine device type for helpful messaging
    device_type = "CPU" if not torch.cuda.is_available() else "GPU"

    print("⚙️  Model Configuration:")
    print(f"   Preset: {args.preset or 'Custom'}")
    print(f"   Layers: {args.num_layers}")
    print(f"   Hidden Size: {args.hidden_size}")
    print(f"   Attention Heads: {args.num_heads}")
    print(f"   Estimated Parameters: ~{estimated_params:.0f}M")
    print(f"   Target Device: {device_type}")
    print()

    print("📊 Training Configuration:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Gradient Accumulation: {args.gradient_accumulation_steps}x")
    print(f"   Effective Batch Size: {effective_batch_size}")
    print(f"   Learning Rate: {args.learning_rate:.2e}")
    print(f"   Max Sequence Length: {args.max_seq_length}")
    print()

    print("🎯 Quality Features:")
    print(f"   Label Smoothing: {args.label_smoothing}")
    print(f"   LR Scheduler: {args.use_scheduler}")
    if args.use_scheduler:
        print(f"   Warmup Steps: {args.warmup_steps}")
    print(f"   Generation Monitoring: {args.monitor_generation}")
    print(f"   Checkpoint Interval: Every {args.save_every_n_steps} steps")

    # CPU-specific warnings/tips
    if device_type == "CPU":
        print()
        print("💡 CPU Training Tips:")
        print(f"   • Expected speed: ~10-20 seconds per step")
        print(f"   • Coherent text expected around: 20,000-30,000 steps")
        print(f"   • First checkpoint: Step {args.save_every_n_steps:,}")
        print(f"   • Estimated time to coherent text: ~3-7 days")
        print(f"   • You can monitor with: watch -n 5 'tail -20 training.log'")
    print()

    if dist_config.enabled:
        print(f"🚀 Distributed Training Mode")
        print(f"   Rank: {dist_config.rank}/{dist_config.world_size-1}")
        print(f"   Backend: {dist_config.backend}")
        print(f"   Local Rank: {dist_config.local_rank}")
    else:
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"⚠️  {num_gpus} GPUs detected but not using distributed training")
            print(f"   To use multiple GPUs, launch with torchrun:")
            print(f"   torchrun --nproc_per_node={num_gpus} train_custom.py [args]")
        elif num_gpus == 1:
            print(f"✓ Single GPU training mode")
        else:
            print(f"⚠️  No GPUs detected, using CPU")

    print()

    # Setup paths
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize GPU manager
    device = torch.device(args.device) if args.device else None
    gpu_manager = GPUMemoryManager(device=device, verbose=True)
    gpu_manager.report_memory("Initial State")

    # Prepare data
    if args.download_data:
        train_file, val_file = download_and_prepare_data(
            data_dir, 
            args.download_wiki, 
            args.download_cleanpdf,
            args.cleanpdf_samples
        )
    else:
        train_file = data_dir / "processed" / "train.txt"
        val_file = data_dir / "processed" / "val.txt"
        if not train_file.exists():
            print(f"❌ Training data not found at {train_file}")
            print("   Run with --download_data to download datasets")
            sys.exit(1)

    # Get tokenizer
    tokenizer_dir = data_dir / "tokenizers" / "storyteller-tokenizer"
    tokenizer = get_or_train_tokenizer(train_file, tokenizer_dir, use_tiktoken=args.use_tiktoken)

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = StoryDataset(
        data_path=str(train_file),
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )
    val_dataset = StoryDataset(
        data_path=str(val_file),
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )

    # Create dataloaders
    # Use 0 workers on CPU to avoid multiprocessing overhead
    num_workers = 0 if not torch.cuda.is_available() else 2

    train_loader = get_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = get_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")

    # Build model
    print("\nBuilding model...")
    
    config = ModelConfig(
        vocab_size=len(tokenizer),
        max_seq_length=args.max_seq_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.hidden_size * 4,
        use_moe=args.use_moe,
    )

    model = StorytellerModel(config)
    
    # Apply label smoothing to the model's loss function
    if hasattr(model, 'set_label_smoothing'):
        model.set_label_smoothing(args.label_smoothing)
        print(f"✓ Label smoothing set to {args.label_smoothing}")
    else:
        print(f"⚠ Model doesn't support label smoothing directly")
        print(f"  Will apply in training loop instead")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    # Resume from checkpoint if specified (do this BEFORE scheduler setup)
    start_epoch = 0
    global_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            # Load checkpoint without scheduler first
            start_epoch, global_step = load_checkpoint(resume_path, model, optimizer, scheduler=None)
        else:
            print(f"⚠ Checkpoint not found: {args.resume}")
            print("  Starting from scratch...")

    # Setup learning rate scheduler (AFTER loading checkpoint)
    scheduler = None
    if args.use_scheduler:
        # Calculate remaining steps from current position
        remaining_epochs = args.epochs - start_epoch
        steps_per_epoch = len(train_loader)
        total_remaining_steps = steps_per_epoch * remaining_epochs

        print(f"\n📈 Learning Rate Scheduler:")
        print(f"   Starting from epoch: {start_epoch + 1}/{args.epochs}")
        print(f"   Steps per epoch: {steps_per_epoch:,}")
        print(f"   Remaining steps: {total_remaining_steps:,}")
        print(f"   Warmup steps: {args.warmup_steps:,}")
        print(f"   Max LR: {args.learning_rate:.2e}")

        # Use CosineAnnealingWarmRestarts for more forgiving scheduling
        # This is more robust than OneCycleLR for resumption
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

        # T_0 is the number of iterations for the first restart
        # Each cycle will take T_0 * T_mult iterations
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=steps_per_epoch,  # Restart every epoch
            T_mult=1,  # Keep same cycle length
            eta_min=args.learning_rate / 100,  # Minimum LR is 1% of max
        )

        print(f"   Scheduler: CosineAnnealingWarmRestarts")
        print(f"   Min LR: {args.learning_rate / 100:.2e}")
        print(f"✓ Scheduler configured\n")

        # If resuming, fast-forward the scheduler to the correct step
        if global_step > 0:
            print(f"⏩ Fast-forwarding scheduler to step {global_step}...")
            for _ in range(global_step):
                scheduler.step()
            print(f"✓ Scheduler state synchronized\n")
    else:
        print(f"⚠ No LR scheduler - using constant LR: {args.learning_rate:.2e}\n")

    # Train with GPU context
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60 + "\n")

    # Create TensorBoard writer
    tensorboard_dir = save_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    print(f"📊 TensorBoard logs: {tensorboard_dir}")
    print(f"   View with: tensorboard --logdir {tensorboard_dir}\n")

    try:
        with gpu_manager.gpu_context(model, cleanup=True):
            best_val_loss = float("inf")

            for epoch in range(start_epoch, args.epochs):
                print(f"\nEpoch {epoch + 1}/{args.epochs}")
                print("-" * 60)

                # Train
                train_loss, global_step = train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    scheduler,
                    gpu_manager.device,
                    epoch + 1,
                    save_dir,
                    args.save_every_n_steps,
                    global_step,
                    rank=0,
                    is_distributed=False,
                    writer=writer,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                )
                print(f"Train Loss: {train_loss:.4f}")

                # Evaluate
                val_metrics = evaluate(
                    model, 
                    val_loader, 
                    gpu_manager.device, 
                    rank=0, 
                    world_size=1,
                    writer=writer,
                    global_step=global_step,
                )
                print(f"Val Loss: {val_metrics['loss']:.4f}")
                print(f"Val Perplexity: {val_metrics['perplexity']:.2f}")
                
                # Monitor generation quality
                if args.monitor_generation and (epoch + 1) % args.generation_interval == 0:
                    monitor_generation(
                        model,
                        tokenizer,
                        gpu_manager.device,
                        epoch + 1,
                        global_step,
                        writer=writer,
                        rank=0,
                    )

                # Save best model
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    save_checkpoint(model, optimizer, epoch, save_dir / "best_model.pt", global_step, scheduler)
                    print(f"💫 New best model saved! (val_loss: {best_val_loss:.4f})")

                # Save epoch checkpoint
                save_checkpoint(model, optimizer, epoch, save_dir / f"epoch_{epoch+1}.pt", global_step, scheduler)

                # Memory report
                gpu_manager.report_memory(f"After Epoch {epoch + 1}")

    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        save_checkpoint(model, optimizer, -1, save_dir / "interrupted.pt", global_step, scheduler)
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        writer.close()
        gpu_manager.force_gc()
        gpu_manager.report_memory("Final State")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60 + "\n")
    print(f"Checkpoints saved to: {save_dir}")
    print(f"TensorBoard logs: {tensorboard_dir}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Final generation samples
    if args.monitor_generation:
        print("\n" + "="*60)
        print("FINAL GENERATION SAMPLES")
        print("="*60 + "\n")
        monitor_generation(
            model,
            tokenizer,
            gpu_manager.device,
            args.epochs,
            global_step,
            writer=None,
            rank=0,
        )


if __name__ == "__main__":
    main()