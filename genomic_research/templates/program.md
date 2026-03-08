# genomic-research

Autonomous genomic foundation model training. You modify `train.py` to optimize a genomic language model.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar9`). The branch `genomic-research/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b genomic-research/<tag>` from current HEAD.
3. **Read the in-scope files**:
   - `README.md` — repository context.
   - `prepare.py` — fixed data loading, tokenization, evaluation. **Do not modify.**
   - `train.py` — the file you modify. Model, optimizer, training loop, hyperparameters.
4. **Verify data exists**: Check `~/.cache/genomic-research/` has the data files and `task_config.json`. If not, tell the human to run `python prepare.py --fasta <file>`.
5. **Read task config**: `cat ~/.cache/genomic-research/task_config.json` to understand:
   - `task_type`: pretrain, classify, or regress
   - `vocab_size`: tokenizer vocabulary size
   - `max_length`: sequence length in tokens
   - `tokenizer_type`: char, kmer, or bpe
   - `n_train`, `n_val`: dataset sizes
6. **Initialize results.tsv** with header row and run the baseline first.
7. **Confirm and go**.

## Experimentation

Each experiment runs for a **fixed time budget** (default 300s). Launch: `python train.py`.

**What you CAN do:**
- Modify `train.py` — everything is fair game:
  - Model architecture (Transformer, Mamba, CNN, LSTM, hybrid, etc.)
  - Embedding strategies (token embeddings, positional encodings, rotary PE)
  - Attention mechanisms (standard, flash, linear, local)
  - Optimizer choice (AdamW, SGD, Adam, LAMB, etc.)
  - Learning rate schedule (cosine, step, exponential, warmup, etc.)
  - Regularization (dropout, weight decay, layer norm, etc.)
  - Objective (MLM, CLM) — change `OBJECTIVE` variable
  - Mask ratio for MLM
  - Batch size, gradient accumulation, gradient clipping
  - Any technique that improves `val_score`

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only (fixed evaluation + data loading).
- Install new packages beyond what's in `pyproject.toml` (torch, numpy, biopython, scikit-learn, matplotlib). Exception: `mamba-ssm` is available if the user installed `genomic-research[mamba]`.
- Modify evaluation functions.
- Leak validation data into training.

## Goal

**Maximize `val_score`** (higher is better in all cases):

| Task | val_score | Meaning |
|------|-----------|---------|
| pretrain | -perplexity | Lower perplexity = higher score |
| classify | accuracy | Higher accuracy = higher score |
| regress | -MSE | Lower MSE = higher score |

**Simplicity criterion**: All else equal, simpler is better.

## Available architectures

### Transformer (default)
```python
MODEL_TYPE = "transformer"
D_MODEL = 256; N_LAYERS = 6; N_HEADS = 8; D_FF = 1024
```
- Pure PyTorch, no extra dependencies.
- Good baseline, well-understood.
- O(n²) attention — may be slow on very long sequences.

### Mamba (requires CUDA + `pip install genomic-research[mamba]`)
```python
MODEL_TYPE = "mamba"
# Replace GenomicTransformer with:
from mamba_ssm import Mamba
# D_MODEL = 256; D_STATE = 16; D_CONV = 4; EXPAND = 2; N_LAYERS = 6
```
- Linear complexity O(n), excellent for long genomic sequences.
- **Must check**: `torch.cuda.is_available()` — Mamba requires CUDA.
- If no CUDA, fall back to Transformer or CNN.

### CNN (1D Convolutions)
```python
MODEL_TYPE = "cnn"
# Stack of Conv1d → BatchNorm → ReLU → MaxPool
# Good at capturing local motifs (promoters, binding sites)
```

### LSTM/GRU
```python
MODEL_TYPE = "lstm"
# Bidirectional LSTM with packed sequences
# Moderate for genomics, but slower than alternatives
```

### Conv-Transformer (hybrid)
```python
MODEL_TYPE = "conv_transformer"
# CNN layers for local patterns + Transformer for global context
```

### Perceiver (cross-attention latents)
```python
MODEL_TYPE = "perceiver"
# Sub-quadratic O(n*m), good for very long sequences
```

### RWKV (linear attention)
```python
MODEL_TYPE = "rwkv"
# O(n) linear complexity, efficient inference
```

### Hyena (long convolution via FFT)
```python
MODEL_TYPE = "hyena"
# HyenaDNA-inspired, sub-quadratic, good for long sequences
```

### Reformer (LSH attention)
```python
MODEL_TYPE = "reformer"
# O(n log n) approximate attention
```

### U-Net (encoder-decoder with skip connections)
```python
MODEL_TYPE = "unet"
# Excellent for per-position prediction tasks
```

### Multi-Scale CNN / Deep Sets
```python
MODEL_TYPE = "multiscale_cnn"  # Inception-style multi-kernel CNN
MODEL_TYPE = "deep_sets"       # Permutation-invariant for metagenomics
```

## Pre-training specific guidance

### Tokenizer selection
- **char** (default): Simple, 1 token per nucleotide. Best for Mamba/SSM models that handle long sequences naturally.
- **kmer**: Maps k consecutive nucleotides to one token. Compresses sequence length by k. Good for Transformers with max_length constraints.
- **bpe**: Learned subword vocabulary. Best compression ratio. Good for large datasets.

### When to try different settings
- If perplexity plateaus → try larger model (increase D_MODEL, N_LAYERS)
- If training is slow → try smaller batch size or gradient accumulation
- If char tokenizer + Transformer is slow → switch to kmer (reduces sequence length ~6x)
- If you have CUDA → try Mamba (much better for long sequences)
- If MLM performance stalls → try CLM (set `OBJECTIVE = "clm"`)
- If model is overfitting → increase dropout, decrease model size

### Recommended experiment progression
1. Baseline Transformer + MLM (default settings)
2. Tune learning rate (1e-3, 5e-4, 1e-4, 5e-5)
3. Adjust model size (d_model: 128/256/512, layers: 4/6/8/12)
4. Try CLM objective
5. Architecture experiments (CNN, Mamba if CUDA, hybrid)
6. Advanced: different positional encodings (rotary, ALiBi)
7. Advanced: mixed precision training (fp16 or bfloat16)

### Advanced experiment strategies
- **Curriculum learning**: `USE_CURRICULUM = True` — start with short sequences, gradually increase.
- **Progressive resizing**: `USE_PROGRESSIVE_RESIZE = True` — start at `PROGRESSIVE_START_LEN` tokens, grow to full length.
- **Knowledge distillation**: Train a large teacher first, then distill with `USE_DISTILLATION = True`.
- **Multi-objective**: `AUX_LOSSES = ["contrastive", "denoise"]` with `AUX_LOSS_WEIGHT = 0.1`.
- **Data augmentation**: Enable SNP noise, indel noise, reverse complement, local shuffle.
- **Regularization**: R-Drop (`USE_RDROP`), SAM optimizer (`USE_SAM`), mixup/cutmix for classification.
- **Architecture search**: Use `genomic-research hypersearch --trials 20` for Optuna-based search.

### Common failure modes and recovery
| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Loss = NaN | Learning rate too high | Reduce LR by 10x, add gradient clipping |
| Loss doesn't decrease | LR too low or model too small | Increase LR, increase d_model |
| val_score oscillates | Batch too small or LR too high | Increase batch size, reduce LR |
| OOM (out of memory) | Model too large | Reduce d_model/n_layers, enable `USE_GRAD_CHECKPOINT`, use AMP |
| Mamba import fails | No CUDA | Fall back to Transformer/CNN/RWKV |
| Overfitting (train >> val) | Model too complex for data | Increase dropout, reduce layers, add data augmentation |

### Genomics-specific tips
- For viral genomes (10-30kb): char tokenizer + Mamba/Hyena works well
- For bacterial genomes (>1Mb): kmer tokenizer + Transformer, use streaming loader
- Codon tokenizer (`"codon"`) is best for coding regions when reading frame is known
- GC content bias: check with `GC_BIAS_ANALYSIS` in reports
- If sequences are very similar: use `phylogenetic_split()` to prevent data leakage

## Output format

```
---
task_type:        pretrain
objective:        mlm
val_score:        -15.234567
val_perplexity:   15.2346
val_loss:         2.724567
val_token_acc:    0.3245
training_seconds: 300.1
total_seconds:    305.2
peak_vram_mb:     1234.5
num_steps:        5000
num_epochs:       3
num_params:       12,345,678
model_type:       transformer
device:           cuda
```

### Generated reports
| File | Description |
|------|-------------|
| `metrics.json` | All metrics + run info |
| `training_curve.png` | Training loss & learning rate |
| `val_score_curve.png` | Validation score at checkpoints |
| `perplexity_curve.png` | (Pretrain) Perplexity over time |
| `token_accuracy_curve.png` | (Pretrain MLM) Masked token accuracy |
| `confusion_matrix.png` | (Classification) Confusion matrix |
| `per_class_metrics.png` | (Classification) Per-class metrics |
| `predicted_vs_actual.png` | (Regression) Scatter plot |
| `residuals.png` | (Regression) Residual analysis |

Extract key metric: `grep "^val_score:" run.log`

## Logging results

Log every experiment to `results.tsv` (tab-separated):

```
commit	val_score	status	description
a1b2c3d	-15.2346	keep	baseline transformer MLM d256 l6
b2c3d4e	-12.8901	keep	increased to d512 l8
c3d4e5f	-13.5000	discard	CLM objective, worse than MLM
d4e5f6g	0.000000	crash	mamba import failed (no CUDA)
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state: current branch/commit
2. Think of an experimental idea. Consider:
   - Architecture (Transformer depth/width, Mamba, CNN, hybrid)
   - Hyperparameters (LR, batch size, warmup, weight decay)
   - Objective (MLM vs CLM, mask ratio)
   - Training tricks (gradient clipping, mixed precision, gradient accumulation)
   - Positional encoding (sinusoidal, rotary, ALiBi, learned)
   - Regularization (dropout, layer drop, weight tying)
3. Modify `train.py` with the idea
4. git commit
5. Run: `python train.py > run.log 2>&1`
6. Read results: `grep "^val_score:" run.log`
7. If grep is empty → crash. Run `tail -n 50 run.log` for traceback.
8. Record in results.tsv
9. If val_score improved → keep (advance branch)
10. If equal or worse → discard: `git checkout -- train.py` then move on

**Quick testing**: `GENOMIC_TIME_BUDGET=30 python train.py` for a 30-second sanity check.

**Timeout**: Kill if a run exceeds 8 minutes.

**Crashes**: Fix simple bugs and re-run. If fundamentally broken, log "crash" and move on.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. You are autonomous. If out of ideas, think harder — try combining near-misses, radical architectures, read task config for data shape clues. The loop runs until the human interrupts you.
