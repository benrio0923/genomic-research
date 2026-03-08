# Troubleshooting Guide

## Common Errors

### `FileNotFoundError: task_config.json not found`
**Cause**: Data hasn't been prepared yet.
**Fix**: Run `genomic-research init --fasta your_data.fasta --task pretrain` first.

### `RuntimeError: CUDA out of memory`
**Fix** (try in order):
1. Reduce `BATCH_SIZE` (e.g., 32 → 16 → 8)
2. Enable gradient checkpointing: `USE_GRAD_CHECKPOINT = True`
3. Reduce model size: decrease `D_MODEL` or `N_LAYERS`
4. Enable mixed precision: `USE_AMP = True`
5. Use gradient accumulation: `GRAD_ACCUMULATION_STEPS = 4` with smaller batch

### `Loss = NaN after a few steps`
**Cause**: Learning rate too high or numerical instability.
**Fix**:
1. Reduce `LEARNING_RATE` by 10x (e.g., 1e-4 → 1e-5)
2. Add gradient clipping: `GRAD_CLIP = 1.0`
3. Try `AMP_DTYPE = "bfloat16"` instead of float16 (better numerical stability)

### `ImportError: mamba_ssm`
**Cause**: Mamba requires CUDA and separate installation.
**Fix**: `pip install genomic-research[mamba]` (requires NVIDIA GPU + CUDA toolkit)
**Workaround**: Use `MODEL_TYPE = "transformer"` or `"rwkv"` instead.

### `ValueError: sequences are different lengths`
**Cause**: Sequences aren't chunked/padded to uniform length.
**Fix**: Set `--max-length` during init, or increase the value if sequences are very long.

### `ModuleNotFoundError: No module named 'tokenizers'`
**Cause**: BPE tokenizer requires the `tokenizers` package.
**Fix**: `pip install genomic-research[bpe]` or use `--tokenizer char` instead.

### `torch.compile failed`
**Cause**: Requires PyTorch 2.0+ and may not support all operations.
**Fix**: Set `USE_COMPILE = False` in train.py. This is a performance optimization, not required.

### `Permission denied` when writing to cache
**Cause**: No write access to `~/.cache/genomic-research/`.
**Fix**: `chmod -R u+w ~/.cache/genomic-research/` or set `GENOMIC_CACHE_DIR` environment variable.

### Model overfitting (train loss << val loss)
**Fix**:
1. Increase `DROPOUT` (e.g., 0.1 → 0.2 → 0.3)
2. Reduce model size (fewer layers or smaller d_model)
3. Enable data augmentation: `USE_RC_DOUBLE = True`, `USE_SNP_NOISE = True`
4. Add weight decay: `WEIGHT_DECAY = 0.01`

### Training is very slow
**Fix**:
1. Enable mixed precision: `USE_AMP = True`
2. Use `torch.compile`: `USE_COMPILE = True` (PyTorch 2.0+)
3. Switch tokenizer: kmer compresses sequences ~6x vs char
4. Reduce `max_length` if sequences have lots of padding

## FAQ

**Q: Which tokenizer should I use?**
- **char**: Simplest, 1 token per nucleotide. Best for Mamba/SSM models.
- **kmer** (k=6): Compresses sequences ~6x. Best for Transformers with long sequences.
- **bpe**: Learned subwords, best compression. Requires `tokenizers` package.

**Q: Which architecture is best?**
- Start with **Transformer** (default, no extra dependencies)
- For long sequences (>10kb): try **RWKV** or **Hyena** (linear complexity)
- If you have CUDA: try **Mamba** (best for genomics)
- For local patterns (motifs): try **CNN** or **Conv-Transformer**

**Q: How do I use multiple GPUs?**
- DDP: Set `USE_DDP = True` and launch with `torchrun --nproc_per_node=N python train.py`
- FSDP: Set `USE_FSDP = True` for memory-efficient sharding

**Q: How long should I train?**
- Default is 300 seconds (`GENOMIC_TIME_BUDGET=300`)
- For quick tests: `GENOMIC_TIME_BUDGET=30 python train.py`
- For serious training: `GENOMIC_TIME_BUDGET=3600` (1 hour) or more

**Q: Can I resume training from a checkpoint?**
- Yes: `RESUME_FROM = "checkpoints/best_model.pt"` in train.py

**Q: How do I add my own architecture?**
- Edit `train.py` — define your model class and update `build_model()` to return it
- The agent will do this automatically during the experiment loop
