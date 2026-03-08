# genomic-research

A universal framework for training genomic sequence AI models. Supports multiple architectures (Transformer, Mamba, CNN, LSTM) and tasks (pre-training, classification, regression). Designed for autonomous experimentation with AI agents.

## Features

- **Multiple architectures**: Transformer (default), Mamba SSM, CNN, LSTM, or any custom model
- **Multiple tokenizers**: Character-level, k-mer, BPE
- **Multiple tasks**: Pre-training (MLM/CLM), sequence classification, regression
- **Multiple input formats**: FASTA, FASTQ, CSV
- **Time-budgeted training**: Fixed wall-clock time per experiment
- **Auto-chunking**: Handles sequences of any length (30kb+ viral genomes)
- **Comprehensive reports**: Training curves, perplexity plots, confusion matrices
- **Agent-ready**: Designed for AI agents to autonomously optimize models

## Quick Start

```bash
pip install genomic-research

# Pre-train on FASTA sequences
genomic-research init --fasta sequences.fasta --task pretrain

# Quick test (30 seconds)
GENOMIC_TIME_BUDGET=30 python train.py

# Full training run (5 minutes)
python train.py

# Start AI agent for autonomous optimization
claude
# Then: "Look at program.md and start experimenting"
```

## Installation

```bash
# Basic (CPU, Transformer only)
pip install genomic-research

# With Mamba SSM support (requires CUDA)
pip install genomic-research[mamba]

# With BPE tokenizer
pip install genomic-research[bpe]

# Everything
pip install genomic-research[all]
```

## Input Formats

### FASTA/FASTQ
```bash
genomic-research init --fasta data.fasta --task pretrain
genomic-research init --fasta data.fastq --task pretrain
```

### CSV
```bash
# Pre-training (sequence column auto-detected or specified)
genomic-research init --csv data.csv --seq-col sequence --task pretrain

# Classification
genomic-research init --csv data.csv --seq-col sequence --task classify --label-col species

# Regression
genomic-research init --csv data.csv --seq-col sequence --task regress --label-col fitness
```

## Architecture Options

| Architecture | Complexity | Best For | Dependencies |
|---|---|---|---|
| Transformer | O(n²) | General purpose, moderate-length sequences | PyTorch only |
| Mamba | O(n) | Long sequences (>1kb), efficient pre-training | `mamba-ssm` (CUDA) |
| CNN | O(n) | Local motif detection, fast training | PyTorch only |
| LSTM | O(n) | Sequential patterns | PyTorch only |

```bash
genomic-research list-models  # See all available architectures
```

## Tokenization

| Tokenizer | Vocab Size | Tokens/bp | Best For |
|---|---|---|---|
| `char` | 10 | 1.0 | Mamba/SSM, simple baseline |
| `kmer` (k=6) | 4101 | ~0.17 | Transformer (compresses sequence 6x) |
| `bpe` | configurable | variable | Large datasets, optimal compression |

```bash
# Character tokenizer (default)
genomic-research init --fasta data.fasta --task pretrain --tokenizer char

# K-mer tokenizer
genomic-research init --fasta data.fasta --task pretrain --tokenizer kmer --kmer-size 6

# BPE tokenizer
genomic-research init --fasta data.fasta --task pretrain --tokenizer bpe
```

## How It Works

```
Your sequences (.fasta/.fastq/.csv)
    │
    ▼
prepare.py  ──►  Tokenize + chunk + split  ──►  ~/.cache/genomic-research/
    │                                                      │
    │  (fixed, never modified)                             │
    ▼                                                      ▼
train.py  ──►  Model + Training Loop  ──►  reports/
    │
    │  (AI agent modifies this file)
    ▼
Evaluate & Generate Reports
```

1. **prepare.py** (fixed): Loads sequences, tokenizes, chunks long sequences, splits train/val, saves to cache
2. **train.py** (modifiable): Defines model architecture and training loop. AI agents modify this file to optimize performance
3. **program.md**: Instructions for the AI agent on how to run experiments

## Using with AI Agents

This framework is designed for AI agents (like Claude) to autonomously optimize genomic models:

```bash
# 1. Initialize
genomic-research init --fasta viral_genomes.fasta --task pretrain

# 2. Start an AI agent
claude

# 3. Prompt the agent
"Look at program.md and start experimenting"
```

The agent will:
- Read `program.md` for experiment protocol
- Create a git branch for the experiment
- Modify `train.py` to try different architectures and hyperparameters
- Run experiments and track results in `results.tsv`
- Automatically keep improvements and discard failures

## CLI Commands

```bash
genomic-research init          # Initialize experiment
genomic-research list-models   # Show available architectures
genomic-research status        # Show cache status
genomic-research clean         # Clear cached data
```

## Platform Support

| Platform | Transformer | Mamba | CNN | LSTM |
|---|---|---|---|---|
| Linux (CUDA) | ✅ | ✅ | ✅ | ✅ |
| Linux (CPU) | ✅ | ❌ | ✅ | ✅ |
| macOS (MPS) | ✅ | ❌ | ✅ | ✅ |
| macOS (CPU) | ✅ | ❌ | ✅ | ✅ |

## License

MIT
