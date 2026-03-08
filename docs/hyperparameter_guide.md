# Hyperparameter Tuning Guide

## Quick Start Recommendations

| Parameter | Conservative | Moderate | Aggressive |
|-----------|-------------|----------|------------|
| D_MODEL | 128 | 256 | 512 |
| N_LAYERS | 4 | 6 | 12 |
| N_HEADS | 4 | 8 | 16 |
| D_FF | 512 | 1024 | 2048 |
| LEARNING_RATE | 5e-5 | 1e-4 | 5e-4 |
| BATCH_SIZE | 16 | 32 | 64 |
| DROPOUT | 0.2 | 0.1 | 0.05 |
| WEIGHT_DECAY | 0.01 | 1e-4 | 0 |

## Parameter-by-Parameter Guide

### Learning Rate
- **Range**: 1e-5 to 1e-3
- **Default**: 1e-4
- **Too high**: Loss becomes NaN or oscillates wildly
- **Too low**: Training is extremely slow, may not converge
- **Tip**: Use cosine schedule with warmup (`LR_SCHEDULE = "cosine"`, `WARMUP_RATIO = 0.05`)

### Model Size (D_MODEL)
- **Range**: 64 to 1024
- **Rule of thumb**: Larger datasets benefit from larger models
- **< 1000 sequences**: Use 128
- **1000 - 10000 sequences**: Use 256
- **> 10000 sequences**: Try 512

### Number of Layers (N_LAYERS)
- **Range**: 2 to 24
- **More layers = more capacity** but diminishing returns
- **Recommended**: Start with 6, try 4 if overfitting, 8-12 if underfitting

### Batch Size
- **Range**: 8 to 256
- **Larger = more stable gradients** but needs more memory
- **If OOM**: Reduce batch size and use `GRAD_ACCUMULATION_STEPS` to maintain effective batch size
- **Example**: `BATCH_SIZE = 8, GRAD_ACCUMULATION_STEPS = 4` ≈ effective batch 32

### Dropout
- **Range**: 0.0 to 0.5
- **Overfitting** (train >> val): Increase dropout
- **Underfitting** (both high): Decrease dropout
- **Small datasets**: 0.2-0.3
- **Large datasets**: 0.05-0.1

### Mask Ratio (MLM only)
- **Range**: 0.10 to 0.30
- **Default**: 0.15 (following BERT)
- **Higher mask ratio**: Harder task, may learn better representations
- **Lower mask ratio**: Easier task, faster convergence

### Gradient Clipping
- **Default**: 1.0
- **If loss is NaN**: Try 0.5 or 0.1
- **If training is stable**: Can increase to 5.0 or disable (0)

## Tuning Strategy

### Phase 1: Learning Rate Search (most impactful)
```python
# Try these values, keep everything else default
LEARNING_RATE = 1e-3   # Often too high
LEARNING_RATE = 5e-4   # Good for small models
LEARNING_RATE = 1e-4   # Default, usually good
LEARNING_RATE = 5e-5   # Conservative
LEARNING_RATE = 1e-5   # Usually too low
```

### Phase 2: Model Size
```python
# Small
D_MODEL = 128; N_LAYERS = 4; D_FF = 512

# Medium (default)
D_MODEL = 256; N_LAYERS = 6; D_FF = 1024

# Large
D_MODEL = 512; N_LAYERS = 8; D_FF = 2048
```

### Phase 3: Regularization
```python
DROPOUT = 0.2          # If overfitting
WEIGHT_DECAY = 0.01    # If overfitting
USE_SNP_NOISE = True   # Data augmentation
USE_RC_DOUBLE = True   # Double dataset with reverse complement
```

### Phase 4: Advanced
```python
OBJECTIVE = "clm"              # Try CLM instead of MLM
LR_SCHEDULE = "cosine"         # vs "step" vs "exponential"
USE_AMP = True                 # Mixed precision
AMP_DTYPE = "bfloat16"         # Better stability than float16
```

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Changing too many things at once | Change ONE parameter per experiment |
| Not running long enough | Use `GENOMIC_TIME_BUDGET=300` minimum |
| Ignoring training curves | Check `training_curve.png` for diagnosis |
| Model too large for data | If < 1000 seqs, keep D_MODEL ≤ 128 |
| Forgetting gradient accumulation | `BATCH_SIZE * GRAD_ACCUMULATION_STEPS` = effective batch |

## Automated Tuning

Use Optuna for automated search:
```bash
genomic-research hypersearch --trials 20 --time-budget 60
```
This searches learning rate, model size, and dropout automatically.
