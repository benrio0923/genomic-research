# Architecture Guide

## Data Flow

```
FASTA/FASTQ/CSV/GenBank
        │
        ▼
   load_sequences()         ─── prepare.py
        │
        ▼
   Tokenize (char/kmer/bpe/codon/protein)
        │
        ▼
   Chunk + Pad (max_length)
        │
        ▼
   Train/Val Split (80/20)
        │
        ▼
   Save to ~/.cache/genomic-research/
        │
        ▼
   train.py loads cached data
        │
        ▼
   ┌─────────────────────┐
   │   Embedding Layer    │   token_ids → (B, L, D)
   │   + Positional Enc   │
   └─────────┬───────────┘
             │
             ▼
   ┌─────────────────────┐
   │   Encoder Layers     │   N_LAYERS × architecture block
   │   (arch-dependent)   │
   └─────────┬───────────┘
             │
             ▼
   ┌─────────────────────┐
   │   Task Head          │   MLM: Linear(D, vocab)
   │                      │   CLM: Linear(D, vocab)
   │                      │   Classify: [CLS] → Linear(D, n_classes)
   │                      │   Regress: [CLS] → Linear(D, 1)
   └─────────────────────┘
```

## Architectures

### Transformer (default)
```
Input → Embedding → [TransformerEncoderLayer × N] → LayerNorm → Head

TransformerEncoderLayer:
  ┌──────────────┐
  │ Multi-Head    │
  │ Self-Attention│──► Add & Norm ──► FFN ──► Add & Norm
  │ (N_HEADS)    │
  └──────────────┘

Complexity: O(L² × D)
Best for: General purpose, L < 2048
```

### CNN
```
Input → Embedding → Conv1D blocks × N → Global Pool → Head

Conv1D Block:
  ┌────────────┐
  │ Conv1D     │
  │ BatchNorm  │──► ReLU ──► (optional MaxPool)
  │ (kernel=7) │
  └────────────┘

Complexity: O(L × D × K)
Best for: Local patterns (motifs, binding sites)
```

### LSTM/GRU
```
Input → Embedding → BiLSTM × N_LAYERS → LayerNorm → Head

BiLSTM:
  Forward:  ──►──►──►──►──►
  Backward: ◄──◄──◄──◄──◄──
  Output = concat(forward, backward)

Complexity: O(L × D²)
Best for: Sequential dependencies
```

### Conv-Transformer (Hybrid)
```
Input → Embedding → [Conv1D × 2] → [Transformer × N] → Head

CNN captures local motifs, Transformer adds global context.
Complexity: O(L² × D) but with better local features
```

### RWKV
```
Input → Embedding → [RWKV Block × N] → LayerNorm → Head

RWKV Block:
  Time-mixing (linear attention with exponential decay)
  Channel-mixing (FFN with gating)

Complexity: O(L × D)
Best for: Long sequences without CUDA requirement
```

### Hyena
```
Input → Embedding → [Hyena Block × N] → LayerNorm → Head

Hyena Block:
  Long convolution via FFT (sub-quadratic)
  Inspired by HyenaDNA for genomics

Complexity: O(L × log(L) × D)
Best for: Very long genomic sequences
```

### Perceiver
```
Input → Embedding → Cross-Attention(latents, input) → [Self-Attention × N] → Head

Latents: M fixed-size vectors (M << L)
Cross-attention maps L tokens to M latents

Complexity: O(L × M × D + M² × D)
Best for: Very long sequences with latent bottleneck
```

### Mamba (requires CUDA)
```
Input → Embedding → [Mamba Block × N] → LayerNorm → Head

Mamba Block:
  Selective State Space Model (S6)
  Input-dependent state transitions
  Hardware-efficient scan algorithm

Complexity: O(L × D)
Best for: Long genomic sequences with CUDA GPU
```

## Architecture Selection Guide

| Sequence Length | CPU/MPS | CUDA (no Mamba) | CUDA + Mamba |
|----------------|---------|-----------------|--------------|
| < 512 | Transformer | Transformer | Transformer |
| 512 - 2048 | Conv-Transformer | Conv-Transformer | Mamba |
| 2048 - 10000 | RWKV / CNN | RWKV / Hyena | Mamba |
| > 10000 | CNN | Hyena / Perceiver | Mamba |

| Task Type | Recommended |
|-----------|-------------|
| Pre-training (MLM) | Transformer, Mamba |
| Pre-training (CLM) | RWKV, Mamba |
| Classification | Transformer, CNN |
| Per-position prediction | U-Net, Transformer |
| Motif detection | CNN, Conv-Transformer |
