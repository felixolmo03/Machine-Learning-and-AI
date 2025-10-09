# The Storyteller 📖

A story-generating language model built from scratch using Mixture of Experts (MoE) architecture, designed to run on consumer hardware (32GB VRAM).

## Overview

This project is a comprehensive curriculum for learning modern LLM concepts by building a small language model capable of generating creative stories. The model uses a sparse Mixture of Experts architecture to achieve high capacity while maintaining computational efficiency.

## Architecture

- **Model Size**: ~400-500M parameters (only ~100M active per forward pass)
- **Architecture**: Decoder-only transformer with MoE layers
- **MoE Configuration**: 8 experts per layer, Top-2 routing
- **Context Length**: 2048 tokens
- **Vocabulary**: ~50k tokens

## Project Structure

```
storyteller/
├── src/
│   └── storyteller/         # Main package
│       ├── data/            # Data preparation and preprocessing
│       ├── model/           # Core model architecture
│       ├── training/        # Training scripts and optimization
│       ├── inference/       # Generation and inference
│       ├── evaluation/      # Metrics and evaluation
│       └── utils/           # Utility functions (device selection, etc.)
├── configs/                 # Configuration files
├── docs/                    # Documentation
├── notebooks/               # Interactive tutorials
├── tests/                   # Unit tests
├── pyproject.toml           # Package configuration
└── README.md                # This file
```

## Setup

### Prerequisites

- Python 3.9+
- **Hardware** (one of the following):
  - CUDA-capable GPU with 32GB+ VRAM (recommended for training)
  - Apple Silicon Mac with 32GB+ unified memory (MPS supported)
  - CPU-only (works but significantly slower)
- 50GB+ disk space for datasets

### Installation

**Option 1: Using pip install (recommended)**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Optional: Install development dependencies
pip install -e ".[dev]"

# Optional: Install DeepSpeed for advanced optimization
pip install -e ".[deepspeed]"
```

**Option 2: Using requirements.txt**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install development dependencies
pip install -r requirements-dev.txt

# Optional: Install DeepSpeed
pip install -r requirements-deepspeed.txt

# Install package in editable mode
pip install -e .
```

This will install the `storyteller` package and all CLI tools:
- `storyteller-download` - Download story datasets
- `storyteller-preprocess` - Preprocess and clean data
- `storyteller-tokenizer` - Train custom tokenizer
- `storyteller-train` - Train the model
- `storyteller-generate` - Generate stories

## Quick Start

### 1. Prepare Data

```bash
# Download story datasets
storyteller-download --output_dir data/raw

# Preprocess and clean data
storyteller-preprocess --input_dir data/raw --output_dir data/processed

# Train custom tokenizer
storyteller-tokenizer --input_file data/processed/train.txt --output_dir data/tokenizers
```

### 2. Train Model

```bash
# Train base model (350M params)
storyteller-train --config configs/base_model.yaml

# Train MoE model (500M total, 100M active)
storyteller-train --config configs/moe_model.yaml

# Resume from checkpoint
storyteller-train --config configs/moe_model.yaml --resume checkpoints/checkpoint_step_10000.pt
```

**Device Selection:**
The training script supports smart device selection. In your config file, set:
- `device: "smart"` - Automatically selects best available (MPS > available CUDA GPU > CPU)
- `device: "mps"` - Force Apple Silicon
- `device: "cuda"` or `"cuda:0"` - Force specific CUDA GPU
- `device: "cpu"` - Force CPU

The smart mode checks GPU memory usage and selects an available GPU in multi-GPU systems.

### 3. Generate Stories

```bash
# Interactive generation
storyteller-generate --checkpoint checkpoints/best_model.pt --interactive

# Generate from prompt
storyteller-generate --checkpoint checkpoints/best_model.pt --prompt "Once upon a time"

# Batch generation
storyteller-generate --checkpoint checkpoints/best_model.pt --prompts_file prompts.txt --output stories.txt
```

## Curriculum Modules

The project follows a 10-week curriculum:

1. **Foundations** (Week 1-2): Tokenization, transformer basics, data preparation
2. **Base Model** (Week 3-4): Simple decoder-only transformer, training loops
3. **Mixture of Experts** (Week 5-6): MoE architecture, routing, load balancing
4. **Advanced Training** (Week 7-8): Memory optimization, scaling
5. **Inference & Deployment** (Week 9-10): Sampling strategies, quantization

## Experiment Tracking with MLflow

The project uses MLflow for tracking experiments, metrics, and models.

### Start MLflow UI

```bash
mlflow ui
```

Then open http://localhost:5000 to view your experiments, compare runs, and analyze metrics.

### Logged Metrics
- Training/validation loss and perplexity
- Learning rate schedule
- MoE expert utilization and balance
- Model checkpoints and artifacts

See [MLflow.md](MLflow.md) for detailed usage guide.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black .
isort .
```

### Type Checking

```bash
mypy model/ training/ inference/
```

## License

MIT

## Acknowledgments

- Design inspired by modern LLM architectures (GPT, Mixtral, Switch Transformer)
- Datasets: TinyStories, WritingPrompts, Project Gutenberg, BookCorpus
