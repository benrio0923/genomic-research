# OpenSpec: genomic-research Continuous Optimization Plan

> 120+ tasks for autonomous, incremental improvement of the genomic-research framework.
> Designed for long-running AI agent sessions (8+ hours).
> Each task is self-contained, testable, and commits independently.

## Ground Rules

1. **Working directory**: `/Users/benrio0923/Downloads/genomic-research`
2. **Never modify `prepare.py` structure** — only add new functions or fix bugs
3. **Always test after each change** — `GENOMIC_TIME_BUDGET=15 python train.py` or `pytest tests/`
4. **Commit after each completed task** — English commit messages, explain WHY
5. **Log progress** to this file by marking tasks `[x]` when done
6. **If a task fails 3 times**, skip it and move to the next
7. **Priority order**: Phase 1 → 2 → 3 → ... (within each phase, tasks are independent)
8. **Templates are in**: `genomic_research/templates/` — this is the source of truth
9. **Test dir**: Use `/tmp/genomic-test/` for quick validation runs
10. **Push to GitHub** after every 5-10 completed tasks

---

## Phase 1: Core Robustness & Bug Fixes (T1–T12)

Priority: CRITICAL — fix existing issues and harden the codebase.

### T1: Fix MPS autocast warning suppression ✅
- [x] Route non-CUDA autocast to CPU context (`amp_device`) to avoid MPS warnings
- [x] Verify: no more `UserWarning: In MPS autocast` messages
- [x] File: `genomic_research/templates/train.py`

### T2: Fix classification with long sequences ✅
- [x] Classification now chunks long sequences — each chunk inherits the label
- [x] Verify: `--task classify` with coronaviridae CSV: 334 → 37,937 training samples
- [x] File: `genomic_research/templates/prepare.py`

### T3: Add input validation to prepare.py ✅
- [x] Validate file exists, CSV columns, task_type, label distribution warnings
- [x] File: `genomic_research/templates/prepare.py`

### T4: Add input validation to train.py ✅
- [x] Improved load_config() with FileNotFoundError and required key validation
- [x] File: `genomic_research/templates/prepare.py`

### T5: Fix gradient accumulation with DDP ✅
- [x] Use `model.no_sync()` on non-boundary accumulation steps
- [x] Only allreduce on accumulation boundary steps
- [x] File: `genomic_research/templates/train.py`

### T6: Fix CLM causal mask generation ✅
- [x] Fixed mask type mismatch (float causal mask vs bool padding mask)
- [x] Verified CLM training converges (perplexity 3.84)
- [x] File: `genomic_research/templates/train.py`

### T7: Handle empty validation set gracefully ✅
- [x] Return safe defaults for empty val set, adjust batch_size for small sets
- [x] File: `genomic_research/templates/prepare.py`

### T8: Fix checkpoint loading device mismatch ✅
- [x] All `torch.load()` calls already use `map_location=device`
- [x] Verified in prepare.py, train.py, and inference.py
- [x] File: `genomic_research/templates/train.py`, `inference.py`

### T9: Add reproducibility (seed everything) ✅
- [x] Added `SEED = _cfg("seed", 42)`, seeds random/numpy/torch/cuda
- [x] File: `genomic_research/templates/train.py`

### T10: Fix BPE tokenizer integration ✅
- [x] BPE tokenizer has complete save/load/train flow with ImportError handling
- [x] Special tokens match prepare.py constants (PAD=0, MASK=1, CLS=2, SEP=3, UNK=4)
- [x] File: `genomic_research/templates/prepare.py`

### T11: Add signal handling for graceful shutdown ✅
- [x] SIGINT/SIGTERM handler sets `_interrupted` flag, training loop exits cleanly
- [x] File: `genomic_research/templates/train.py`

### T12: Fix results.tsv concurrent write safety ✅
- [x] Use `fcntl.flock(LOCK_EX)` for file locking, with Windows fallback
- [x] File: `genomic_research/templates/train.py`

---

## Phase 2: Testing & Quality Assurance (T13–T30)

Priority: HIGH — comprehensive test coverage.

### T13: Test classification end-to-end ✅
- [x] Add `TestEndToEnd.test_classify_e2e` using synthetic CSV with 3 classes
- [x] Verify: prepare → train → reports with confusion matrix
- [x] File: `tests/test_smoke.py`

### T14: Test regression end-to-end ✅
- [x] Add `TestEndToEnd.test_regress_e2e` using synthetic CSV with float labels
- [x] Verify: prepare → train → reports with predicted_vs_actual
- [x] File: `tests/test_smoke.py`

### T15: Test CLM objective ✅
- [x] Add test for causal language modeling objective
- [x] Verify loss decreases and val_perplexity is reasonable
- [x] File: `tests/test_smoke.py`

### T16: Test k-mer tokenizer with various k values ✅
- [x] Test k=3, k=4, k=6, k=8
- [x] Verify vocab sizes are correct (5 + 4^k)
- [x] Verify roundtrip encode/decode for each
- [x] File: `tests/test_smoke.py`

### T17: Test inference.py ✅
- [x] Add `TestInference` class
- [x] Test: load model from checkpoint, run prediction
- [x] Test: extract embeddings, verify shape
- [x] File: `tests/test_smoke.py`

### T18: Test CLI commands ✅
- [x] Add `TestCLI` class
- [x] Test: `genomic-research list-models` outputs all architectures
- [x] Test: `genomic-research status` when no cache exists
- [x] Test: `genomic-research clean` removes cache
- [x] File: `tests/test_smoke.py`

### T19: Test config override system ✅
- [x] Create test JSON config with custom hyperparameters
- [x] Verify `_cfg()` correctly returns overridden values
- [x] Verify non-overridden values use defaults
- [x] File: `tests/test_smoke.py`

### T20: Test data augmentation functions ✅
- [x] Test `reverse_complement_tokens` with known inputs
- [x] Test `span_mask_tokens` produces correct mask ratios
- [x] Verify augmentation doesn't corrupt PAD tokens
- [x] File: `tests/test_smoke.py`

### T21: Test checkpoint save/load roundtrip ✅
- [x] Save model → load model → verify outputs match
- [x] Test with each architecture (transformer, cnn, lstm)
- [x] File: `tests/test_smoke.py`

### T22: Test FASTQ quality filtering ✅
- [x] Generate synthetic FASTQ with quality scores
- [x] Verify low-quality sequences are filtered
- [x] File: `tests/test_smoke.py`

### T23: Test long sequence chunking ✅
- [x] Create sequence longer than max_length
- [x] Verify correct number of chunks with overlap
- [x] Verify chunk boundaries are correct
- [x] File: `tests/test_smoke.py`

### T24: Test stratified split for classification ✅
- [x] Create imbalanced dataset (class A: 100, class B: 10, class C: 5)
- [x] Verify each class appears in both train and val sets
- [x] File: `tests/test_smoke.py`

### T25: Test class weights computation ✅
- [x] Verify weights are inversely proportional to class frequency
- [x] Verify weights sum to n_classes (approximately)
- [x] File: `tests/test_smoke.py`

### T26: Test report generation ✅
- [x] Verify all expected files are created for pretrain task
- [x] Verify all expected files are created for classify task
- [x] Verify metrics.json contains all expected fields
- [x] File: `tests/test_smoke.py`

### T27: Add property-based tests for tokenizers ✅
- [x] Use hypothesis library to generate random sequences
- [x] Property: `decode(encode(seq)) == seq` for all valid sequences
- [x] Property: `len(encode(seq)) > 0` for non-empty sequences
- [x] File: `tests/test_properties.py` (new)

### T28: Test multi-file input ✅
- [x] Create multiple FASTA files
- [x] Test glob pattern input (if supported) or concatenation
- [x] File: `tests/test_smoke.py`

### T29: Test edge cases ✅
- [x] Single sequence dataset
- [x] Very short sequence (< max_length)
- [x] Sequence with all N's
- [x] Empty FASTA file
- [x] CSV with missing values
- [x] File: `tests/test_edge_cases.py` (new)

### T30: Add benchmark tests ✅
- [x] Test that training doesn't regress in speed (tokens/sec baseline)
- [x] Test memory usage doesn't exceed expected bounds
- [x] File: `tests/test_benchmark.py` (new)

---

## Phase 3: Advanced Model Architectures (T31–T50)

Priority: HIGH — expand model capabilities.

### T31: Implement Flash Attention support ✅
- [x] Use `torch.nn.functional.scaled_dot_product_attention` (PyTorch 2.0+ SDPA)
- [x] Falls back to manual attention for ALiBi (needs attn_bias)
- [x] File: `genomic_research/templates/train.py`

### T32: Implement Convolutional Transformer (hybrid) ✅
- [x] Conv layers for local features → Transformer for global context
- [x] First N layers = Conv1d, remaining = Transformer
- [x] Add `MODEL_TYPE = "conv_transformer"` option
- [x] File: `genomic_research/templates/train.py`

### T33: Implement Perceiver-style architecture ✅
- [x] Use learned latent array (much smaller than sequence length)
- [x] Cross-attention: latents attend to input
- [x] Self-attention: latents attend to each other
- [x] Good for very long sequences without O(n²)
- [x] File: `genomic_research/templates/train.py`

### T34: Implement MoE (Mixture of Experts) layer ✅
- [x] Top-k routing (k=2 default)
- [x] Expert load balancing loss
- [x] Replace feed-forward layers with MoE
- [x] Add `USE_MOE`, `N_EXPERTS`, `MOE_TOP_K` hyperparameters
- [x] File: `genomic_research/templates/train.py`

### T35: Implement GRU variant for LSTM architecture ✅
- [x] `RNN_TYPE` hyperparameter + `model_type="gru"` shortcut
- [x] GenomicLSTM now uses `nn.GRU` or `nn.LSTM` based on config
- [x] File: `genomic_research/templates/train.py`

### T36: Implement dilated CNN ✅
- [x] `CNN_DILATION` hyperparameter for exponential dilation (1, 2, 4, 8...)
- [x] Dilated padding preserves sequence length
- [x] File: `genomic_research/templates/train.py`

### T37: Implement Reformer-style LSH attention ✅
- [x] Locality-Sensitive Hashing for approximate attention
- [x] O(n log n) complexity
- [x] Add as attention variant option
- [x] File: `genomic_research/templates/train.py`

### T38: Implement relative position bias (T5-style) ✅
- [x] Learnable relative position buckets
- [x] Add `POS_ENCODING = "relative"` option
- [x] File: `genomic_research/templates/train.py`

### T39: Implement sliding window attention ✅
- [x] Fixed window size for local attention
- [x] Optional global tokens (first/last positions)
- [x] Add `ATTENTION_WINDOW = 128` hyperparameter
- [x] Combine with Longformer-style global attention
- [x] File: `genomic_research/templates/train.py`

### T40: Implement RWKV-style architecture ✅
- [x] Linear attention with time-decay
- [x] O(n) complexity for inference
- [x] Pure PyTorch implementation
- [x] File: `genomic_research/templates/train.py`

### T41: Implement Hyena architecture ✅
- [x] Long convolution + gating mechanism
- [x] Sub-quadratic complexity
- [x] Excellent for genomic sequences (demonstrated in HyenaDNA paper)
- [x] File: `genomic_research/templates/train.py`

### T42: Add multi-head classification ✅
- [x] Support multiple classification outputs simultaneously
- [x] E.g., species + host + geographic origin
- [x] Add `MULTI_LABEL = True/False` hyperparameter
- [x] File: `genomic_research/templates/train.py`

### T43: Implement multi-scale CNN ✅
- [x] Parallel conv branches with different kernel sizes (3, 7, 15, 31)
- [x] Concatenate features from all scales
- [x] Inception-style architecture for genomics
- [x] File: `genomic_research/templates/train.py`

### T44: Implement U-Net style architecture for sequence-level tasks ✅
- [x] Encoder-decoder with skip connections
- [x] Good for per-position prediction tasks (e.g., secondary structure)
- [x] File: `genomic_research/templates/train.py`

### T45: Implement Deep Sets for set-level features ✅
- [x] Permutation invariant architecture
- [x] Useful when sequence order doesn't matter (e.g., metagenomics)
- [x] File: `genomic_research/templates/train.py`

### T46: Add weight initialization strategies ✅
- [x] `_init_weights()` with trunc_normal for Linear/Embedding, kaiming for Conv1d, orthogonal for LSTM
- [x] Applied via `model.apply()` in `build_model()`
- [x] File: `genomic_research/templates/train.py`

### T47: Implement stochastic depth (layer drop) ✅
- [x] `DropPath` module with linearly increasing drop rate per layer
- [x] `STOCHASTIC_DEPTH` hyperparameter (0.0 = disabled)
- [x] Applied to both attention and FF residual connections
- [x] File: `genomic_research/templates/train.py`

### T48: Implement Pre-LayerNorm vs Post-LayerNorm toggle ✅
- [x] Current: Pre-norm (default for stability)
- [x] Add `NORM_POSITION = "pre" | "post"` option
- [x] File: `genomic_research/templates/train.py`

### T49: Implement model EMA (Exponential Moving Average) ✅
- [x] `ModelEMA` class with configurable decay, updated after each optimizer step
- [x] Final eval compares EMA vs best checkpoint, uses better one
- [x] `USE_EMA`, `EMA_DECAY` hyperparameters
- [x] File: `genomic_research/templates/train.py`

### T50: Implement DeepNorm for stable deep training ✅
- [x] Scale residual connection by α, scale initialization by β
- [x] Allows training 1000+ layer Transformers
- [x] Add `USE_DEEPNORM = True/False` option
- [x] File: `genomic_research/templates/train.py`

---

## Phase 4: Training Optimization (T51–T68)

Priority: MEDIUM-HIGH — better training efficiency and convergence.

### T51: Implement additional LR schedules ✅
- [x] Added `linear`, `one_cycle`, `exponential` schedules
- [x] File: `genomic_research/templates/train.py`

### T52: Implement OneCycleLR schedule ✅
- [x] Merged into T51 as `one_cycle` schedule
- [x] File: `genomic_research/templates/train.py`

### T53: Implement AdamW with decoupled weight decay ✅
- [x] Current AdamW may not correctly decouple WD
- [x] Verify implementation matches original paper
- [x] File: `genomic_research/templates/train.py`

### T54: Add LAMB optimizer option ✅
- [x] `OPTIMIZER` hyperparameter: adamw, sgd, lamb
- [x] LAMB class with trust ratio scaling
- [x] File: `genomic_research/templates/train.py`

### T55: Implement label smoothing ✅
- [x] `LABEL_SMOOTHING` hyperparameter applied to all CrossEntropyLoss instances
- [x] File: `genomic_research/templates/train.py`

### T56: Implement focal loss for classification ✅
- [x] `FocalLoss` class with configurable gamma and label smoothing
- [x] `LOSS_FN`, `FOCAL_GAMMA` hyperparameters
- [x] File: `genomic_research/templates/train.py`

### T57: Implement curriculum learning ✅
- [x] Start with shorter sequences, gradually increase length
- [x] Sort by sequence length and sample from easy → hard
- [x] Add `USE_CURRICULUM = True/False` hyperparameter
- [x] File: `genomic_research/templates/train.py`

### T58: Implement R-Drop regularization ✅
- [x] Train with dropout twice, minimize KL divergence between outputs
- [x] Add `USE_RDROP = True/False`, `RDROP_ALPHA = 1.0`
- [x] File: `genomic_research/templates/train.py`

### T59: Add layer-wise learning rate decay ✅
- [x] `LR_LAYER_DECAY` hyperparameter (1.0 = uniform, <1.0 = lower for earlier layers)
- [x] Auto-detects layer depth from parameter names
- [x] File: `genomic_research/templates/train.py`

### T60: Implement progressive resizing ✅
- [x] Start training with shorter max_length, increase over time
- [x] Good for pre-training: 128 → 256 → 512
- [x] Add `PROGRESSIVE_RESIZE = True/False`
- [x] File: `genomic_research/templates/train.py`

### T61: Add gradient noise injection ✅
- [x] Add Gaussian noise to gradients during early training
- [x] Helps escape sharp minima
- [x] Add `GRAD_NOISE = 0.01` hyperparameter
- [x] File: `genomic_research/templates/train.py`

### T62: Implement SWA (Stochastic Weight Averaging) ✅
- [x] `USE_SWA`, `SWA_LR` hyperparameters, collects in final 25%
- [x] Uses `torch.optim.swa_utils.AveragedModel`, updates BN stats
- [x] File: `genomic_research/templates/train.py`

### T63: Add dynamic batch sizing ✅
- [x] Start with small batch, increase during training
- [x] Simulates LR warmup via batch size warmup
- [x] Add `DYNAMIC_BATCH = True/False`
- [x] File: `genomic_research/templates/train.py`

### T64: Implement SAM optimizer (Sharpness-Aware Minimization) ✅
- [x] Two forward passes per step (overhead, but better generalization)
- [x] Add `USE_SAM = True/False`, `SAM_RHO = 0.05`
- [x] File: `genomic_research/templates/train.py`

### T65: Implement multi-objective training ✅
- [x] Support MLM + contrastive loss simultaneously
- [x] Weighted sum of losses with configurable weights
- [x] Add `AUX_LOSSES = []` list configuration
- [x] File: `genomic_research/templates/train.py`

### T66: Implement early stopping ✅
- [x] `EARLY_STOP_PATIENCE` hyperparameter (0 = disabled)
- [x] Stops training after N evals without improvement, still respects time budget
- [x] File: `genomic_research/templates/train.py`

### T67: Add training resume from checkpoint ✅
- [x] `RESUME_FROM` config option loads model + optimizer state
- [x] Checkpoint now saves optimizer_state_dict and step
- [x] File: `genomic_research/templates/train.py`

### T68: Implement knowledge distillation ✅
- [x] Large model (teacher) → small model (student)
- [x] Soft label distillation loss
- [x] Add `DISTILL_FROM = "path/to/teacher.pt"` config
- [x] File: `genomic_research/templates/train.py`

---

## Phase 5: Data Pipeline Enhancement (T69–T82)

Priority: MEDIUM — expand data format and processing capabilities.

### T69: Support multi-file FASTA input ✅
- [x] `load_sequences()` accepts directory path or glob pattern
- [x] Recursively loads all .fasta/.fastq/.csv files from directory
- [x] File: `genomic_research/templates/prepare.py`

### T70: Support GenBank format input ✅
- [x] Parse .gb/.gbk files using BioPython
- [x] Extract sequence + annotations (gene boundaries, CDS, etc.)
- [x] File: `genomic_research/templates/prepare.py`

### T71: Support GFF/GTF annotation files ✅
- [x] Load gene annotations alongside sequences
- [x] Use for per-position classification tasks
- [x] File: `genomic_research/templates/prepare.py`

### T72: Add data statistics report ✅
- [x] GC content, N content, nucleotide counts, length stats
- [x] Saved as `data_report.json` in cache directory
- [x] File: `genomic_research/templates/prepare.py`

### T73: Implement streaming data loader for large datasets ✅
- [x] Use `IterableDataset` for datasets that don't fit in memory
- [x] Stream from FASTA file directly
- [x] Add `USE_STREAMING = True/False` when n_sequences > threshold
- [x] File: `genomic_research/templates/prepare.py`

### T74: Add sequence deduplication ✅
- [x] MD5-based exact duplicate removal before tokenization
- [x] Reports count of removed duplicates
- [x] File: `genomic_research/templates/prepare.py`

### T75: Add reverse complement data doubling ✅
- [x] Option to add reverse complement of every sequence as additional training data
- [x] Doubles dataset size for free
- [x] Add `--rc-double` flag
- [x] File: `genomic_research/templates/prepare.py`

### T76: Implement dynamic chunking strategies ✅
- [x] Current: fixed overlap (50%)
- [x] Add options: no overlap, random offset, sliding window
- [x] Add `--chunk-strategy fixed|random|slide` argument
- [x] File: `genomic_research/templates/prepare.py`

### T77: Add data sampling for large datasets ✅
- [x] `--sample-n` and `--sample-frac` CLI args
- [x] Random subsampling with fixed seed for reproducibility
- [x] File: `genomic_research/templates/prepare.py`, `cli.py`

### T78: Support paired-end reads ✅
- [x] Load R1/R2 FASTQ pairs
- [x] Merge or concatenate with separator token
- [x] File: `genomic_research/templates/prepare.py`

### T79: Implement k-fold cross-validation ✅
- [x] Instead of single 80/20 split, support k-fold
- [x] Add `--n-folds 5` argument
- [x] Save each fold separately
- [x] File: `genomic_research/templates/prepare.py`

### T80: Add sequence weighting by uniqueness ✅
- [x] Cluster sequences, weight inversely by cluster size
- [x] Prevents over-representation of common sequences
- [x] File: `genomic_research/templates/prepare.py`

### T81: Support protein sequences ✅
- [x] Detect amino acid sequences (not just DNA)
- [x] Expand alphabet: 20 amino acids + special tokens
- [x] Add `--seq-type dna|protein|auto` argument
- [x] File: `genomic_research/templates/prepare.py`

### T82: Add VCF variant format support ✅
- [x] Load variant call files
- [x] Generate sequences with variants applied to reference
- [x] Useful for population genomics tasks
- [x] File: `genomic_research/templates/prepare.py`

---

## Phase 6: Data Augmentation (T83–T92)

Priority: MEDIUM — improve model generalization.

### T83: Implement SNP noise injection ✅
- [x] Randomly substitute bases with configurable error rate (e.g., 1%)
- [x] Simulates sequencing errors
- [x] Add `USE_SNP_NOISE = True/False`, `SNP_RATE = 0.01`
- [x] File: `genomic_research/templates/train.py`

### T84: Implement insertion/deletion noise ✅
- [x] Randomly insert or delete 1-3 bases
- [x] Simulates indel errors
- [x] Add `USE_INDEL_NOISE = True/False`, `INDEL_RATE = 0.005`
- [x] File: `genomic_research/templates/train.py`

### T85: Implement sequence shuffling augmentation ✅
- [x] Shuffle within k-bp windows (preserves local composition)
- [x] Add `USE_LOCAL_SHUFFLE = True/False`, `SHUFFLE_WINDOW = 10`
- [x] File: `genomic_research/templates/train.py`

### T86: Implement random subsequence cropping ✅
- [x] Randomly crop a contiguous subsequence instead of using full chunk
- [x] Good for pre-training with variable context lengths
- [x] Add `USE_RANDOM_CROP = True/False`, `MIN_CROP_RATIO = 0.5`
- [x] File: `genomic_research/templates/train.py`

### T87: Implement mixup augmentation for sequences ✅
- [x] Interpolate between two sequences' embeddings
- [x] Mixup labels for classification
- [x] Add `USE_MIXUP = True/False`, `MIXUP_ALPHA = 0.2`
- [x] File: `genomic_research/templates/train.py`

### T88: Implement CutMix for sequences ✅
- [x] Replace random spans with spans from another sequence
- [x] Mix labels proportionally
- [x] Add `USE_CUTMIX = True/False`, `CUTMIX_ALPHA = 1.0`
- [x] File: `genomic_research/templates/train.py`

### T89: Implement token dropout augmentation ✅
- [x] Randomly drop tokens (replace with PAD) during training
- [x] Different from masking — no prediction target
- [x] Add `TOKEN_DROPOUT = 0.0` (0 = disabled)
- [x] File: `genomic_research/templates/train.py`

### T90: Implement whole-word masking for k-mer tokenizer ✅
- [x] When using k-mer tokenizer, mask whole k-mers not individual tokens
- [x] More biologically meaningful masking
- [x] File: `genomic_research/templates/train.py`

### T91: Implement denoising autoencoder objective ✅
- [x] Input = corrupted sequence, output = original sequence
- [x] Corruption: random deletion + shuffling + insertion
- [x] Add `OBJECTIVE = "denoise"` option
- [x] File: `genomic_research/templates/train.py`

### T92: Implement contrastive learning augmentation ✅
- [x] Create positive pairs via augmentation (RC, crop, noise)
- [x] Add contrastive loss (InfoNCE/SimCLR-style)
- [x] Add `USE_CONTRASTIVE = True/False`, `CONTRASTIVE_TEMP = 0.07`
- [x] File: `genomic_research/templates/train.py`

---

## Phase 7: Evaluation & Analysis (T93–T108)

Priority: MEDIUM — deeper understanding of model behavior.

### T93: Add attention weight visualization ✅
- [x] Extract attention weights from Transformer layers
- [x] Plot attention heatmap for sample sequences
- [x] Save as `reports/attention_map.png`
- [x] File: `genomic_research/templates/train.py`, `prepare.py`

### T94: Add embedding PCA visualization ✅
- [x] Alternative to t-SNE: faster, linear
- [x] Save as `reports/embedding_pca.png`
- [x] Show explained variance ratio
- [x] File: `genomic_research/templates/prepare.py`

### T95: Add UMAP embedding visualization ✅
- [x] Better structure preservation than t-SNE
- [x] Optional: `pip install umap-learn`
- [x] Save as `reports/embedding_umap.png`
- [x] File: `genomic_research/templates/prepare.py`

### T96: Add per-position prediction accuracy ✅
- [x] For MLM: which positions are easiest/hardest to predict?
- [x] Plot accuracy by position in sequence
- [x] Save as `reports/position_accuracy.png`
- [x] File: `genomic_research/templates/prepare.py`

### T97: Add nucleotide-level prediction analysis ✅
- [x] For each base (A/T/C/G/N), what's the prediction accuracy?
- [x] Confusion matrix at the nucleotide level
- [x] Save as `reports/nucleotide_confusion.png`
- [x] File: `genomic_research/templates/prepare.py`

### T98: Add learning curve analysis ✅
- [x] Train with 10%, 25%, 50%, 75%, 100% of data
- [x] Plot val_score vs data size
- [x] Helps determine if more data would help
- [x] File: `genomic_research/templates/train.py` or new script

### T99: Add model complexity analysis ✅
- [x] FLOPs per forward pass
- [x] Memory footprint (model + activations)
- [x] Throughput (sequences/second)
- [x] Save as `reports/complexity.json`
- [x] File: `genomic_research/templates/train.py`

### T100: Add GC content bias analysis ✅
- [x] Does the model perform differently on AT-rich vs GC-rich sequences?
- [x] Bin sequences by GC content, compute val_score per bin
- [x] Save as `reports/gc_bias.png`
- [x] File: `genomic_research/templates/prepare.py`

### T101: Add sequence length bias analysis ✅
- [x] Does the model perform differently on short vs long sequences?
- [x] Bin sequences by length, compute val_score per bin
- [x] Save as `reports/length_bias.png`
- [x] File: `genomic_research/templates/prepare.py`

### T102: Add ROC curve visualization ✅
- [x] Per-class ROC curves for classification
- [x] Save as `reports/roc_curves.png`
- [x] File: `genomic_research/templates/prepare.py`

### T103: Add precision-recall curve visualization ✅
- [x] Per-class PR curves for classification
- [x] Save as `reports/pr_curves.png`
- [x] File: `genomic_research/templates/prepare.py`

### T104: Add calibration plot for classification ✅
- [x] Expected vs observed probability
- [x] Brier score
- [x] Save as `reports/calibration.png`
- [x] File: `genomic_research/templates/prepare.py`

### T105: Add gradient norm tracking ✅
- [x] Track gradient norms per layer during training
- [x] Detect vanishing/exploding gradients
- [x] Save as `reports/gradient_norms.png`
- [x] File: `genomic_research/templates/train.py`

### T106: Add weight distribution analysis ✅
- [x] Histogram of weight values per layer
- [x] Detect dead neurons
- [x] Save as `reports/weight_distribution.png`
- [x] File: `genomic_research/templates/train.py`

### T107: Add motif discovery from attention ✅
- [x] Extract high-attention regions from sequences
- [x] Align and find consensus motifs
- [x] Compare with known biological motifs
- [x] File: `genomic_research/templates/train.py`

### T108: Add statistical significance testing ✅
- [x] When comparing two models, compute p-value (paired t-test on val scores)
- [x] Bootstrap confidence intervals for metrics
- [x] File: `genomic_research/templates/prepare.py`

---

## Phase 8: CLI & User Experience (T109–T120)

Priority: MEDIUM — better developer experience.

### T109: Add `genomic-research evaluate` command ✅
- [x] Load trained model and run evaluation on new data
- [x] Support: `--checkpoint model.pt --fasta new_data.fasta`
- [x] File: `genomic_research/cli.py`

### T110: Add `genomic-research predict` command ✅
- [x] Wrapper around inference.py
- [x] `genomic-research predict --checkpoint model.pt --fasta query.fasta --output predictions.csv`
- [x] File: `genomic_research/cli.py`

### T111: Add `genomic-research embed` command ✅
- [x] Extract embeddings and save as numpy/HDF5
- [x] `genomic-research embed --checkpoint model.pt --fasta data.fasta --output embeddings.npy`
- [x] File: `genomic_research/cli.py`

### T112: Add `genomic-research compare` command ✅
- [x] Compare two results.tsv files or two metrics.json files
- [x] Show delta for each metric, highlight improvements
- [x] File: `genomic_research/cli.py`

### T113: Add `genomic-research export` command ✅
- [x] Export model to ONNX, TorchScript, or SafeTensors
- [x] `genomic-research export --checkpoint model.pt --format onnx --output model.onnx`
- [x] File: `genomic_research/cli.py`

### T114: Add `genomic-research info` command ✅
- [x] Show model architecture, parameters, hyperparameters from checkpoint
- [x] File: `genomic_research/cli.py`

### T115: Add `genomic-research search` command ✅
- [x] Search NCBI for sequences by organism, gene, etc.
- [x] Download and prepare data in one step
- [x] Uses BioPython Entrez API
- [x] File: `genomic_research/cli.py`

### T116: Add progress bar to training ✅
- [x] Show real-time progress with tqdm or custom bar
- [x] Display: step, loss, lr, ETA, val_score
- [x] File: `genomic_research/templates/train.py`

### T117: Add colored terminal output ✅
- [x] Green for improvements, red for degradation
- [x] Bold for important metrics
- [x] Use ANSI escape codes (no external dependency)
- [x] File: `genomic_research/templates/train.py`

### T118: Add `--dry-run` flag to train.py ✅
- [x] Print model architecture, parameter count, estimated time
- [x] Don't actually train
- [x] File: `genomic_research/templates/train.py`

### T119: Add `--profile` flag to train.py ✅
- [x] Run PyTorch profiler for first 10 steps
- [x] Save Chrome trace to `reports/profile.json`
- [x] File: `genomic_research/templates/train.py`

### T120: Add interactive experiment dashboard ✅
- [x] HTML report with all plots, metrics, and experiment history
- [x] Auto-generated from results.tsv and reports/
- [x] Single self-contained HTML file
- [x] File: `genomic_research/templates/prepare.py` or new script

---

## Phase 9: Biological Domain Features (T121–T135)

Priority: MEDIUM — genomics-specific intelligence.

### T121: Add codon-aware tokenization ✅
- [x] Tokenize in reading frames (codons = 3 bases)
- [x] Handles coding regions more naturally
- [x] Add `--tokenizer codon` option
- [x] File: `genomic_research/templates/prepare.py`

### T122: Add ORF detection and annotation ✅
- [x] Find open reading frames in sequences
- [x] Mark coding vs non-coding regions
- [x] Can use as auxiliary task for multi-task learning
- [x] File: `genomic_research/templates/prepare.py`

### T123: Add GC content as auxiliary feature ✅
- [x] Compute local GC content in sliding windows
- [x] Feed as additional input channel to model
- [x] File: `genomic_research/templates/train.py`

### T124: Add k-mer frequency spectrum features ✅
- [x] Compute k-mer frequencies (k=2,3,4) per sequence
- [x] Use as additional features for classification
- [x] File: `genomic_research/templates/prepare.py`

### T125: Add phylogenetic-aware data splitting ✅
- [x] Split train/val based on phylogenetic distance
- [x] Prevents data leakage from closely related sequences
- [x] File: `genomic_research/templates/prepare.py`

### T126: Add taxonomic hierarchy classification ✅
- [x] Multi-level classification: Family → Genus → Species
- [x] Hierarchical softmax or flat multi-task
- [x] File: `genomic_research/templates/train.py`

### T127: Add sequence alignment scoring ✅
- [x] Use trained embeddings for pairwise sequence similarity
- [x] Compare with BLAST alignment scores
- [x] File: new `genomic_research/templates/align.py`

### T128: Add variant effect prediction ✅
- [x] Given a mutation (position, ref, alt), predict effect
- [x] Binary classification: pathogenic vs benign
- [x] File: `genomic_research/templates/train.py`

### T129: Add promoter/enhancer prediction ✅
- [x] Per-position binary classification: regulatory vs non-regulatory
- [x] Use U-Net or per-position prediction head
- [x] File: `genomic_research/templates/train.py`

### T130: Add recombination detection ✅
- [x] Detect breakpoints where recombination occurred
- [x] Relevant for coronavirus evolution analysis
- [x] File: new analysis script

### T131: Add mutation rate estimation ✅
- [x] From pre-trained model, estimate per-position mutation rates
- [x] Compare with known mutation hotspots
- [x] File: new analysis script

### T132: Add multiple sequence alignment from embeddings ✅
- [x] Use model embeddings for guide tree construction
- [x] Compare with ClustalW/MUSCLE alignments
- [x] File: new analysis script

### T133: Add antimicrobial resistance gene detection ✅
- [x] Classification: resistance vs susceptible
- [x] Common downstream task for bacterial genomics
- [x] File: `genomic_research/templates/train.py`

### T134: Add host prediction for viruses ✅
- [x] Given a viral genome, predict the host organism
- [x] Multi-class classification with known host labels
- [x] File: `genomic_research/templates/train.py`

### T135: Add geographic origin prediction ✅
- [x] Predict geographic origin from sequence features
- [x] Multi-class (continent/country) or regression (lat/lon)
- [x] File: `genomic_research/templates/train.py`

---

## Phase 10: Production & Deployment (T136–T148)

Priority: LOW-MEDIUM — production readiness.

### T136: Add model quantization (INT8) ✅
- [x] Post-training quantization with `torch.quantization`
- [x] Measure speed improvement and accuracy degradation
- [x] Add `genomic-research export --quantize int8`
- [x] File: `genomic_research/cli.py`, `inference.py`

### T137: Add TorchScript compilation ✅
- [x] `torch.jit.script` or `torch.jit.trace` the model
- [x] Enables deployment without Python
- [x] File: `genomic_research/templates/inference.py`

### T138: Add torch.compile() support (PyTorch 2.0+) ✅
- [x] Wrap model with `torch.compile()` for faster training
- [x] Add `USE_COMPILE = True/False` hyperparameter
- [x] Only on supported platforms (CUDA + Triton)
- [x] File: `genomic_research/templates/train.py`

### T139: Create FastAPI inference server ✅
- [x] REST API: POST /predict with FASTA sequence → JSON response
- [x] Batch prediction support
- [x] Health check endpoint
- [x] File: new `genomic_research/serve.py`

### T140: Create Gradio demo interface ✅
- [x] Web UI: paste sequence → get prediction/embedding
- [x] Visualize attention weights
- [x] File: new `genomic_research/demo.py`

### T141: Add HuggingFace Hub integration ✅
- [x] Push trained model to HuggingFace Hub
- [x] Load pre-trained model from HuggingFace Hub
- [x] `genomic-research push --repo username/model-name`
- [x] File: `genomic_research/cli.py`

### T142: Add model card generation ✅
- [x] Auto-generate model card (HuggingFace format)
- [x] Include: architecture, training data, metrics, limitations
- [x] File: `genomic_research/templates/train.py`

### T143: Improve Dockerfile ✅
- [x] Multi-stage build (builder + runtime)
- [x] Add GPU support (NVIDIA runtime)
- [x] Add health check
- [x] Add docker-compose.yml
- [x] File: `Dockerfile`, `docker-compose.yml`

### T144: Add pip install validation test ✅
- [x] CI step: `pip install .` in clean virtualenv
- [x] Verify CLI entry point works
- [x] Verify all imports succeed
- [x] File: `.github/workflows/ci.yml`

### T145: Add PyPI publishing workflow ✅
- [x] GitHub Action to publish to PyPI on release tag
- [x] Semantic versioning
- [x] File: `.github/workflows/publish.yml` (new)

### T146: Add model compression via pruning ✅
- [x] Unstructured pruning (magnitude-based)
- [x] Structured pruning (remove attention heads)
- [x] Measure sparsity vs accuracy trade-off
- [x] File: `genomic_research/templates/train.py`

### T147: Add SafeTensors export ✅
- [x] Save model weights in SafeTensors format (safer than pickle)
- [x] Compatible with HuggingFace ecosystem
- [x] File: `genomic_research/templates/train.py`

### T148: Add batch inference optimization ✅
- [x] Dynamic batching for variable-length sequences
- [x] Key-value caching for CLM inference
- [x] Benchmark: sequences/second
- [x] File: `genomic_research/templates/inference.py`

---

## Phase 11: Experiment Management (T149–T160)

Priority: LOW-MEDIUM — better experiment tracking.

### T149: Add Weights & Biases integration ✅
- [x] Optional: `pip install wandb`
- [x] Log all metrics, hyperparameters, model architecture
- [x] Add `USE_WANDB = True/False` hyperparameter
- [x] File: `genomic_research/templates/train.py`

### T150: Add TensorBoard logging ✅
- [x] Log scalars (loss, lr, val_score) per step
- [x] Log model graph
- [x] Add `USE_TENSORBOARD = True/False`
- [x] File: `genomic_research/templates/train.py`

### T151: Add MLflow integration ✅
- [x] Optional experiment tracking with MLflow
- [x] Log metrics, params, artifacts
- [x] Add `USE_MLFLOW = True/False`
- [x] File: `genomic_research/templates/train.py`

### T152: Add hyperparameter search with Optuna ✅
- [x] Define search space in config
- [x] Run N trials within time budget
- [x] Save best hyperparameters
- [x] File: new `genomic_research/templates/search.py`

### T153: Add experiment comparison tool ✅
- [x] Read all results.tsv entries
- [x] Generate comparison table and charts
- [x] Rank experiments by val_score
- [x] File: `genomic_research/cli.py` or new script

### T154: Add experiment reproducibility checker ✅
- [x] Record all: code hash, data hash, library versions, seed
- [x] Verify same setup produces same results (±tolerance)
- [x] File: `genomic_research/templates/train.py`

### T155: Add config snapshot with each run ✅
- [x] Save full hyperparameter config as JSON alongside checkpoint
- [x] Include git commit hash, timestamp
- [x] File: `genomic_research/templates/train.py`

### T156: Add experiment tagging ✅
- [x] Tag experiments with descriptive labels
- [x] `EXPERIMENT_TAG = "baseline_transformer"` in results.tsv
- [x] File: `genomic_research/templates/train.py`

### T157: Add automatic best model selection ✅
- [x] After multiple experiments, find the best model across all runs
- [x] Copy best checkpoint to `best_overall/`
- [x] File: `genomic_research/cli.py`

### T158: Add experiment diff tool ✅
- [x] Compare train.py between two experiments
- [x] Show what hyperparameters changed and their effect
- [x] File: `genomic_research/cli.py`

### T159: Add results visualization dashboard ✅
- [x] Plot all experiments: val_score over time, parameter sweeps
- [x] Interactive HTML report
- [x] File: new script or integrate into CLI

### T160: Add experiment archiving ✅
- [x] `genomic-research archive` — save checkpoint, config, results as tar.gz
- [x] Include model card and metrics
- [x] File: `genomic_research/cli.py`

---

## Phase 12: Documentation & Education (T161–T170)

Priority: LOW — improve onboarding and understanding.

### T161: Add Jupyter tutorial notebook ✅
- [x] End-to-end walkthrough: data → prepare → train → evaluate
- [x] Include Coronaviridae example
- [x] File: `notebooks/tutorial.ipynb` (new)

### T162: Add architecture comparison notebook ✅
- [x] Train all architectures on same data
- [x] Compare: speed, memory, val_score, embedding quality
- [x] File: `notebooks/architecture_comparison.ipynb` (new)

### T163: Add API documentation ✅
- [x] Document all public functions in prepare.py
- [x] Document model classes in train.py
- [x] Document inference.py functions
- [x] Use Google-style docstrings
- [x] File: all template files

### T164: Add CHANGELOG.md ✅
- [x] Track version changes
- [x] Follow Keep a Changelog format
- [x] File: `CHANGELOG.md` (new)

### T165: Add architecture diagrams ✅
- [x] ASCII or Mermaid diagrams of each architecture
- [x] Data flow diagram: FASTA → tokens → model → predictions
- [x] File: `README.md` or `docs/` directory

### T166: Add hyperparameter tuning guide ✅
- [x] Recommended ranges for each hyperparameter
- [x] Common pitfalls and solutions
- [x] File: `docs/hyperparameter_guide.md` (new)

### T167: Add genomics background document ✅
- [x] Brief intro to DNA, genes, codons, mutations
- [x] Why pre-training on genomic data is useful
- [x] File: `docs/genomics_primer.md` (new)

### T168: Add troubleshooting guide ✅
- [x] Common errors and solutions
- [x] FAQ section
- [x] File: `docs/troubleshooting.md` (new)

### T169: Add benchmark results table ✅
- [x] Standard benchmarks: synthetic data, Coronaviridae
- [x] Compare all architectures and tokenizers
- [x] File: `docs/benchmarks.md` (new)

### T170: Improve program.md agent instructions ✅
- [x] Add more experiment strategies
- [x] Add genomics-specific optimization tips
- [x] Add common failure modes and recovery
- [x] File: `genomic_research/templates/program.md`

---

## Phase 13: Performance & Scalability (T171–T180)

Priority: LOW — optimize for large-scale experiments.

### T171: Profile and optimize data loading ✅
- [x] Identify bottlenecks in prepare.py
- [x] Optimize FASTA parsing for large files (>1GB)
- [x] Use memory-mapped files where possible
- [x] File: `genomic_research/templates/prepare.py`

### T172: Optimize tokenization speed ✅
- [x] Batch tokenization instead of per-sequence
- [x] Parallel tokenization with multiprocessing
- [x] Benchmark: tokens/second for each tokenizer
- [x] File: `genomic_research/templates/prepare.py`

### T173: Add mixed precision training with bfloat16 ✅
- [x] bfloat16 support for CUDA and MPS (Apple Silicon)
- [x] Better numerical stability than fp16
- [x] Add `AMP_DTYPE = "float16" | "bfloat16"` option
- [x] File: `genomic_research/templates/train.py`

### T174: Optimize memory for large models ✅
- [x] Add activation checkpointing per layer
- [x] Implement CPU offloading for optimizer states
- [x] File: `genomic_research/templates/train.py`

### T175: Add FSDP support (Fully Sharded Data Parallel) ✅
- [x] For multi-GPU training with large models
- [x] Alternative to DDP for memory-constrained setups
- [x] File: `genomic_research/templates/train.py`

### T176: Optimize inference batch processing ✅
- [x] Sort by length, batch similar lengths together
- [x] Reduces padding waste
- [x] File: `genomic_research/templates/inference.py`

### T177: Add persistent data caching with versioning ✅
- [x] Hash input file + parameters → cache key
- [x] Skip re-processing if cache is valid
- [x] File: `genomic_research/templates/prepare.py`

### T178: Optimize report generation speed ✅
- [x] Lazy import matplotlib (slow to import)
- [x] Use Agg backend for headless rendering
- [x] Parallel figure generation
- [x] File: `genomic_research/templates/prepare.py`

### T179: Add training throughput logging ✅
- [x] Log tokens/second, samples/second during training
- [x] Identify if data loading is the bottleneck
- [x] File: `genomic_research/templates/train.py`

### T180: Optimize model serialization ✅
- [x] Only save model weights, not full state (for inference)
- [x] Support incremental checkpoint saving
- [x] File: `genomic_research/templates/train.py`

---

## Task Summary

| Phase | Category | Tasks | Priority |
|-------|----------|-------|----------|
| 1 | Core Robustness & Bug Fixes | T1–T12 (12) | CRITICAL |
| 2 | Testing & Quality | T13–T30 (18) | HIGH |
| 3 | Advanced Architectures | T31–T50 (20) | HIGH |
| 4 | Training Optimization | T51–T68 (18) | MEDIUM-HIGH |
| 5 | Data Pipeline | T69–T82 (14) | MEDIUM |
| 6 | Data Augmentation | T83–T92 (10) | MEDIUM |
| 7 | Evaluation & Analysis | T93–T108 (16) | MEDIUM |
| 8 | CLI & UX | T109–T120 (12) | MEDIUM |
| 9 | Biological Domain | T121–T135 (15) | MEDIUM |
| 10 | Production & Deployment | T136–T148 (13) | LOW-MEDIUM |
| 11 | Experiment Management | T149–T160 (12) | LOW-MEDIUM |
| 12 | Documentation | T161–T170 (10) | LOW |
| 13 | Performance & Scalability | T171–T180 (10) | LOW |
| **Total** | | **180 tasks** | |

---

## Execution Strategy

### For autonomous 8-hour sessions:

1. **Start with Phase 1** (bug fixes) — establishes stability
2. **Move to Phase 2** (testing) — ensures future changes don't break things
3. **Alternate between Phase 3-4** (architectures + training) — core improvements
4. **Phase 5-6** (data pipeline + augmentation) — expands capabilities
5. **Phase 7-8** (evaluation + CLI) — polishes user experience
6. **Remaining phases** as time permits

### Per-task workflow:

```
1. Read the task description
2. Read relevant source files
3. Implement the change
4. Test: GENOMIC_TIME_BUDGET=15 python train.py  (or pytest)
5. Verify output/behavior
6. git add <files> && git commit -m "feat/fix: description"
7. Mark task [x] in OPENSPEC.md
8. Every 5 tasks: git push origin main
```

### Quick test commands:

```bash
# Pre-training smoke test
cd /tmp/genomic-test && cp genomic_research/templates/*.py . && \
python prepare.py --fasta data/coronaviridae.fasta --task pretrain --max-length 256 && \
GENOMIC_TIME_BUDGET=15 python train.py

# Classification smoke test
python prepare.py --csv data/coronaviridae.csv --seq-col sequence --task classify --label-col species --max-length 256 && \
GENOMIC_TIME_BUDGET=15 python train.py

# Unit tests
pytest tests/ -x -v

# Full test suite
GENOMIC_TIME_BUDGET=10 pytest tests/ -v
```
