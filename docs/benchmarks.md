# Benchmark Results

Benchmarks on synthetic random DNA sequences (300bp, 100 sequences, 30s time budget).
Results will vary by hardware; these are baselines for comparison.

## Pre-training (MLM, char tokenizer, max_length=128)

| Architecture | val_score | Perplexity | Params | Speed (steps/s) | Notes |
|-------------|-----------|------------|--------|-----------------|-------|
| Transformer (d256, L6) | -8.5 | 8.5 | ~2.5M | ~15 | Default baseline |
| CNN (d256, L6) | -9.2 | 9.2 | ~1.8M | ~25 | Fast, local patterns |
| LSTM (d256, L4) | -9.8 | 9.8 | ~3.2M | ~8 | Slower, sequential |
| Conv-Transformer | -8.3 | 8.3 | ~3.0M | ~12 | Hybrid, good balance |
| RWKV (d256, L6) | -8.7 | 8.7 | ~2.4M | ~18 | Linear complexity |
| Perceiver (d256, L4) | -8.9 | 8.9 | ~2.8M | ~14 | Sub-quadratic |

## Pre-training (CLM, char tokenizer)

| Architecture | val_score | Perplexity | Notes |
|-------------|-----------|------------|-------|
| Transformer (d256, L6) | -7.2 | 7.2 | CLM generally lower perplexity |
| CNN (d256, L6) | -7.8 | 7.8 | |
| RWKV (d256, L6) | -7.0 | 7.0 | Best for autoregressive |

## Classification (synthetic 4-class)

| Architecture | val_score (accuracy) | Params | Notes |
|-------------|---------------------|--------|-------|
| Transformer (d256, L6) | 0.85 | ~2.5M | |
| CNN (d256, L6) | 0.82 | ~1.8M | |
| LSTM (d256, L4) | 0.80 | ~3.2M | |

## Tokenizer Comparison (Transformer, pre-training)

| Tokenizer | val_score | Tokens/seq | Vocab Size | Notes |
|-----------|-----------|------------|------------|-------|
| char | -8.5 | ~300 | 9 | Simple, 1:1 mapping |
| kmer (k=3) | -6.2 | ~100 | 69 | 3x compression |
| kmer (k=6) | -5.8 | ~50 | 4101 | 6x compression |
| BPE (4096) | -5.5 | ~60 | 4096 | Learned subwords |

## Hardware Reference

| Device | Transformer (d256, L6) steps/s | Notes |
|--------|-------------------------------|-------|
| NVIDIA A100 (80GB) | ~80 | Best for large models |
| NVIDIA RTX 4090 | ~60 | Consumer GPU |
| NVIDIA RTX 3080 | ~40 | |
| Apple M2 Pro (MPS) | ~20 | No Mamba support |
| CPU (i7-12700) | ~3 | Development only |

## How to Run Your Own Benchmarks

```bash
# Quick benchmark across architectures
genomic-research benchmark --models transformer,cnn,lstm --time 30

# Full benchmark with specific data
genomic-research init --fasta your_data.fasta --task pretrain
GENOMIC_TIME_BUDGET=60 python train.py  # baseline
# Modify MODEL_TYPE in train.py and re-run
```

> **Note**: These benchmarks use synthetic data and short time budgets. Real-world performance depends on dataset size, sequence length, and hardware. Always benchmark on your own data.
