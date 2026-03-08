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

### T13: Test classification end-to-end
- [ ] Add `TestEndToEnd.test_classify_e2e` using synthetic CSV with 3 classes
- [ ] Verify: prepare → train → reports with confusion matrix
- [ ] File: `tests/test_smoke.py`

### T14: Test regression end-to-end
- [ ] Add `TestEndToEnd.test_regress_e2e` using synthetic CSV with float labels
- [ ] Verify: prepare → train → reports with predicted_vs_actual
- [ ] File: `tests/test_smoke.py`

### T15: Test CLM objective
- [ ] Add test for causal language modeling objective
- [ ] Verify loss decreases and val_perplexity is reasonable
- [ ] File: `tests/test_smoke.py`

### T16: Test k-mer tokenizer with various k values
- [ ] Test k=3, k=4, k=6, k=8
- [ ] Verify vocab sizes are correct (5 + 4^k)
- [ ] Verify roundtrip encode/decode for each
- [ ] File: `tests/test_smoke.py`

### T17: Test inference.py
- [ ] Add `TestInference` class
- [ ] Test: load model from checkpoint, run prediction
- [ ] Test: extract embeddings, verify shape
- [ ] File: `tests/test_smoke.py`

### T18: Test CLI commands
- [ ] Add `TestCLI` class
- [ ] Test: `genomic-research list-models` outputs all architectures
- [ ] Test: `genomic-research status` when no cache exists
- [ ] Test: `genomic-research clean` removes cache
- [ ] File: `tests/test_smoke.py`

### T19: Test config override system
- [ ] Create test JSON config with custom hyperparameters
- [ ] Verify `_cfg()` correctly returns overridden values
- [ ] Verify non-overridden values use defaults
- [ ] File: `tests/test_smoke.py`

### T20: Test data augmentation functions
- [ ] Test `reverse_complement_tokens` with known inputs
- [ ] Test `span_mask_tokens` produces correct mask ratios
- [ ] Verify augmentation doesn't corrupt PAD tokens
- [ ] File: `tests/test_smoke.py`

### T21: Test checkpoint save/load roundtrip
- [ ] Save model → load model → verify outputs match
- [ ] Test with each architecture (transformer, cnn, lstm)
- [ ] File: `tests/test_smoke.py`

### T22: Test FASTQ quality filtering
- [ ] Generate synthetic FASTQ with quality scores
- [ ] Verify low-quality sequences are filtered
- [ ] File: `tests/test_smoke.py`

### T23: Test long sequence chunking
- [ ] Create sequence longer than max_length
- [ ] Verify correct number of chunks with overlap
- [ ] Verify chunk boundaries are correct
- [ ] File: `tests/test_smoke.py`

### T24: Test stratified split for classification
- [ ] Create imbalanced dataset (class A: 100, class B: 10, class C: 5)
- [ ] Verify each class appears in both train and val sets
- [ ] File: `tests/test_smoke.py`

### T25: Test class weights computation
- [ ] Verify weights are inversely proportional to class frequency
- [ ] Verify weights sum to n_classes (approximately)
- [ ] File: `tests/test_smoke.py`

### T26: Test report generation
- [ ] Verify all expected files are created for pretrain task
- [ ] Verify all expected files are created for classify task
- [ ] Verify metrics.json contains all expected fields
- [ ] File: `tests/test_smoke.py`

### T27: Add property-based tests for tokenizers
- [ ] Use hypothesis library to generate random sequences
- [ ] Property: `decode(encode(seq)) == seq` for all valid sequences
- [ ] Property: `len(encode(seq)) > 0` for non-empty sequences
- [ ] File: `tests/test_properties.py` (new)

### T28: Test multi-file input
- [ ] Create multiple FASTA files
- [ ] Test glob pattern input (if supported) or concatenation
- [ ] File: `tests/test_smoke.py`

### T29: Test edge cases
- [ ] Single sequence dataset
- [ ] Very short sequence (< max_length)
- [ ] Sequence with all N's
- [ ] Empty FASTA file
- [ ] CSV with missing values
- [ ] File: `tests/test_edge_cases.py` (new)

### T30: Add benchmark tests
- [ ] Test that training doesn't regress in speed (tokens/sec baseline)
- [ ] Test memory usage doesn't exceed expected bounds
- [ ] File: `tests/test_benchmark.py` (new)

---

## Phase 3: Advanced Model Architectures (T31–T50)

Priority: HIGH — expand model capabilities.

### T31: Implement Flash Attention support ✅
- [x] Use `torch.nn.functional.scaled_dot_product_attention` (PyTorch 2.0+ SDPA)
- [x] Falls back to manual attention for ALiBi (needs attn_bias)
- [x] File: `genomic_research/templates/train.py`

### T32: Implement Convolutional Transformer (hybrid)
- [ ] Conv layers for local features → Transformer for global context
- [ ] First N layers = Conv1d, remaining = Transformer
- [ ] Add `MODEL_TYPE = "conv_transformer"` option
- [ ] File: `genomic_research/templates/train.py`

### T33: Implement Perceiver-style architecture
- [ ] Use learned latent array (much smaller than sequence length)
- [ ] Cross-attention: latents attend to input
- [ ] Self-attention: latents attend to each other
- [ ] Good for very long sequences without O(n²)
- [ ] File: `genomic_research/templates/train.py`

### T34: Implement MoE (Mixture of Experts) layer
- [ ] Top-k routing (k=2 default)
- [ ] Expert load balancing loss
- [ ] Replace feed-forward layers with MoE
- [ ] Add `USE_MOE`, `N_EXPERTS`, `MOE_TOP_K` hyperparameters
- [ ] File: `genomic_research/templates/train.py`

### T35: Implement GRU variant for LSTM architecture ✅
- [x] `RNN_TYPE` hyperparameter + `model_type="gru"` shortcut
- [x] GenomicLSTM now uses `nn.GRU` or `nn.LSTM` based on config
- [x] File: `genomic_research/templates/train.py`

### T36: Implement dilated CNN ✅
- [x] `CNN_DILATION` hyperparameter for exponential dilation (1, 2, 4, 8...)
- [x] Dilated padding preserves sequence length
- [x] File: `genomic_research/templates/train.py`

### T37: Implement Reformer-style LSH attention
- [ ] Locality-Sensitive Hashing for approximate attention
- [ ] O(n log n) complexity
- [ ] Add as attention variant option
- [ ] File: `genomic_research/templates/train.py`

### T38: Implement relative position bias (T5-style)
- [ ] Learnable relative position buckets
- [ ] Add `POS_ENCODING = "relative"` option
- [ ] File: `genomic_research/templates/train.py`

### T39: Implement sliding window attention
- [ ] Fixed window size for local attention
- [ ] Optional global tokens (first/last positions)
- [ ] Add `ATTENTION_WINDOW = 128` hyperparameter
- [ ] Combine with Longformer-style global attention
- [ ] File: `genomic_research/templates/train.py`

### T40: Implement RWKV-style architecture
- [ ] Linear attention with time-decay
- [ ] O(n) complexity for inference
- [ ] Pure PyTorch implementation
- [ ] File: `genomic_research/templates/train.py`

### T41: Implement Hyena architecture
- [ ] Long convolution + gating mechanism
- [ ] Sub-quadratic complexity
- [ ] Excellent for genomic sequences (demonstrated in HyenaDNA paper)
- [ ] File: `genomic_research/templates/train.py`

### T42: Add multi-head classification
- [ ] Support multiple classification outputs simultaneously
- [ ] E.g., species + host + geographic origin
- [ ] Add `MULTI_LABEL = True/False` hyperparameter
- [ ] File: `genomic_research/templates/train.py`

### T43: Implement multi-scale CNN
- [ ] Parallel conv branches with different kernel sizes (3, 7, 15, 31)
- [ ] Concatenate features from all scales
- [ ] Inception-style architecture for genomics
- [ ] File: `genomic_research/templates/train.py`

### T44: Implement U-Net style architecture for sequence-level tasks
- [ ] Encoder-decoder with skip connections
- [ ] Good for per-position prediction tasks (e.g., secondary structure)
- [ ] File: `genomic_research/templates/train.py`

### T45: Implement Deep Sets for set-level features
- [ ] Permutation invariant architecture
- [ ] Useful when sequence order doesn't matter (e.g., metagenomics)
- [ ] File: `genomic_research/templates/train.py`

### T46: Add weight initialization strategies ✅
- [x] `_init_weights()` with trunc_normal for Linear/Embedding, kaiming for Conv1d, orthogonal for LSTM
- [x] Applied via `model.apply()` in `build_model()`
- [x] File: `genomic_research/templates/train.py`

### T47: Implement stochastic depth (layer drop) ✅
- [x] `DropPath` module with linearly increasing drop rate per layer
- [x] `STOCHASTIC_DEPTH` hyperparameter (0.0 = disabled)
- [x] Applied to both attention and FF residual connections
- [x] File: `genomic_research/templates/train.py`

### T48: Implement Pre-LayerNorm vs Post-LayerNorm toggle
- [ ] Current: Pre-norm (default for stability)
- [ ] Add `NORM_POSITION = "pre" | "post"` option
- [ ] File: `genomic_research/templates/train.py`

### T49: Implement model EMA (Exponential Moving Average) ✅
- [x] `ModelEMA` class with configurable decay, updated after each optimizer step
- [x] Final eval compares EMA vs best checkpoint, uses better one
- [x] `USE_EMA`, `EMA_DECAY` hyperparameters
- [x] File: `genomic_research/templates/train.py`

### T50: Implement DeepNorm for stable deep training
- [ ] Scale residual connection by α, scale initialization by β
- [ ] Allows training 1000+ layer Transformers
- [ ] Add `USE_DEEPNORM = True/False` option
- [ ] File: `genomic_research/templates/train.py`

---

## Phase 4: Training Optimization (T51–T68)

Priority: MEDIUM-HIGH — better training efficiency and convergence.

### T51: Implement additional LR schedules ✅
- [x] Added `linear`, `one_cycle`, `exponential` schedules
- [x] File: `genomic_research/templates/train.py`

### T52: Implement OneCycleLR schedule ✅
- [x] Merged into T51 as `one_cycle` schedule
- [x] File: `genomic_research/templates/train.py`

### T53: Implement AdamW with decoupled weight decay
- [ ] Current AdamW may not correctly decouple WD
- [ ] Verify implementation matches original paper
- [ ] File: `genomic_research/templates/train.py`

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

### T57: Implement curriculum learning
- [ ] Start with shorter sequences, gradually increase length
- [ ] Sort by sequence length and sample from easy → hard
- [ ] Add `USE_CURRICULUM = True/False` hyperparameter
- [ ] File: `genomic_research/templates/train.py`

### T58: Implement R-Drop regularization
- [ ] Train with dropout twice, minimize KL divergence between outputs
- [ ] Add `USE_RDROP = True/False`, `RDROP_ALPHA = 1.0`
- [ ] File: `genomic_research/templates/train.py`

### T59: Add layer-wise learning rate decay ✅
- [x] `LR_LAYER_DECAY` hyperparameter (1.0 = uniform, <1.0 = lower for earlier layers)
- [x] Auto-detects layer depth from parameter names
- [x] File: `genomic_research/templates/train.py`

### T60: Implement progressive resizing
- [ ] Start training with shorter max_length, increase over time
- [ ] Good for pre-training: 128 → 256 → 512
- [ ] Add `PROGRESSIVE_RESIZE = True/False`
- [ ] File: `genomic_research/templates/train.py`

### T61: Add gradient noise injection
- [ ] Add Gaussian noise to gradients during early training
- [ ] Helps escape sharp minima
- [ ] Add `GRAD_NOISE = 0.01` hyperparameter
- [ ] File: `genomic_research/templates/train.py`

### T62: Implement SWA (Stochastic Weight Averaging) ✅
- [x] `USE_SWA`, `SWA_LR` hyperparameters, collects in final 25%
- [x] Uses `torch.optim.swa_utils.AveragedModel`, updates BN stats
- [x] File: `genomic_research/templates/train.py`

### T63: Add dynamic batch sizing
- [ ] Start with small batch, increase during training
- [ ] Simulates LR warmup via batch size warmup
- [ ] Add `DYNAMIC_BATCH = True/False`
- [ ] File: `genomic_research/templates/train.py`

### T64: Implement SAM optimizer (Sharpness-Aware Minimization)
- [ ] Two forward passes per step (overhead, but better generalization)
- [ ] Add `USE_SAM = True/False`, `SAM_RHO = 0.05`
- [ ] File: `genomic_research/templates/train.py`

### T65: Implement multi-objective training
- [ ] Support MLM + contrastive loss simultaneously
- [ ] Weighted sum of losses with configurable weights
- [ ] Add `AUX_LOSSES = []` list configuration
- [ ] File: `genomic_research/templates/train.py`

### T66: Implement early stopping ✅
- [x] `EARLY_STOP_PATIENCE` hyperparameter (0 = disabled)
- [x] Stops training after N evals without improvement, still respects time budget
- [x] File: `genomic_research/templates/train.py`

### T67: Add training resume from checkpoint ✅
- [x] `RESUME_FROM` config option loads model + optimizer state
- [x] Checkpoint now saves optimizer_state_dict and step
- [x] File: `genomic_research/templates/train.py`

### T68: Implement knowledge distillation
- [ ] Large model (teacher) → small model (student)
- [ ] Soft label distillation loss
- [ ] Add `DISTILL_FROM = "path/to/teacher.pt"` config
- [ ] File: `genomic_research/templates/train.py`

---

## Phase 5: Data Pipeline Enhancement (T69–T82)

Priority: MEDIUM — expand data format and processing capabilities.

### T69: Support multi-file FASTA input ✅
- [x] `load_sequences()` accepts directory path or glob pattern
- [x] Recursively loads all .fasta/.fastq/.csv files from directory
- [x] File: `genomic_research/templates/prepare.py`

### T70: Support GenBank format input
- [ ] Parse .gb/.gbk files using BioPython
- [ ] Extract sequence + annotations (gene boundaries, CDS, etc.)
- [ ] File: `genomic_research/templates/prepare.py`

### T71: Support GFF/GTF annotation files
- [ ] Load gene annotations alongside sequences
- [ ] Use for per-position classification tasks
- [ ] File: `genomic_research/templates/prepare.py`

### T72: Add data statistics report ✅
- [x] GC content, N content, nucleotide counts, length stats
- [x] Saved as `data_report.json` in cache directory
- [x] File: `genomic_research/templates/prepare.py`

### T73: Implement streaming data loader for large datasets
- [ ] Use `IterableDataset` for datasets that don't fit in memory
- [ ] Stream from FASTA file directly
- [ ] Add `USE_STREAMING = True/False` when n_sequences > threshold
- [ ] File: `genomic_research/templates/prepare.py`

### T74: Add sequence deduplication ✅
- [x] MD5-based exact duplicate removal before tokenization
- [x] Reports count of removed duplicates
- [x] File: `genomic_research/templates/prepare.py`

### T75: Add reverse complement data doubling
- [ ] Option to add reverse complement of every sequence as additional training data
- [ ] Doubles dataset size for free
- [ ] Add `--rc-double` flag
- [ ] File: `genomic_research/templates/prepare.py`

### T76: Implement dynamic chunking strategies
- [ ] Current: fixed overlap (50%)
- [ ] Add options: no overlap, random offset, sliding window
- [ ] Add `--chunk-strategy fixed|random|slide` argument
- [ ] File: `genomic_research/templates/prepare.py`

### T77: Add data sampling for large datasets ✅
- [x] `--sample-n` and `--sample-frac` CLI args
- [x] Random subsampling with fixed seed for reproducibility
- [x] File: `genomic_research/templates/prepare.py`, `cli.py`

### T78: Support paired-end reads
- [ ] Load R1/R2 FASTQ pairs
- [ ] Merge or concatenate with separator token
- [ ] File: `genomic_research/templates/prepare.py`

### T79: Implement k-fold cross-validation
- [ ] Instead of single 80/20 split, support k-fold
- [ ] Add `--n-folds 5` argument
- [ ] Save each fold separately
- [ ] File: `genomic_research/templates/prepare.py`

### T80: Add sequence weighting by uniqueness
- [ ] Cluster sequences, weight inversely by cluster size
- [ ] Prevents over-representation of common sequences
- [ ] File: `genomic_research/templates/prepare.py`

### T81: Support protein sequences
- [ ] Detect amino acid sequences (not just DNA)
- [ ] Expand alphabet: 20 amino acids + special tokens
- [ ] Add `--seq-type dna|protein|auto` argument
- [ ] File: `genomic_research/templates/prepare.py`

### T82: Add VCF variant format support
- [ ] Load variant call files
- [ ] Generate sequences with variants applied to reference
- [ ] Useful for population genomics tasks
- [ ] File: `genomic_research/templates/prepare.py`

---

## Phase 6: Data Augmentation (T83–T92)

Priority: MEDIUM — improve model generalization.

### T83: Implement SNP noise injection
- [ ] Randomly substitute bases with configurable error rate (e.g., 1%)
- [ ] Simulates sequencing errors
- [ ] Add `USE_SNP_NOISE = True/False`, `SNP_RATE = 0.01`
- [ ] File: `genomic_research/templates/train.py`

### T84: Implement insertion/deletion noise
- [ ] Randomly insert or delete 1-3 bases
- [ ] Simulates indel errors
- [ ] Add `USE_INDEL_NOISE = True/False`, `INDEL_RATE = 0.005`
- [ ] File: `genomic_research/templates/train.py`

### T85: Implement sequence shuffling augmentation
- [ ] Shuffle within k-bp windows (preserves local composition)
- [ ] Add `USE_LOCAL_SHUFFLE = True/False`, `SHUFFLE_WINDOW = 10`
- [ ] File: `genomic_research/templates/train.py`

### T86: Implement random subsequence cropping
- [ ] Randomly crop a contiguous subsequence instead of using full chunk
- [ ] Good for pre-training with variable context lengths
- [ ] Add `USE_RANDOM_CROP = True/False`, `MIN_CROP_RATIO = 0.5`
- [ ] File: `genomic_research/templates/train.py`

### T87: Implement mixup augmentation for sequences
- [ ] Interpolate between two sequences' embeddings
- [ ] Mixup labels for classification
- [ ] Add `USE_MIXUP = True/False`, `MIXUP_ALPHA = 0.2`
- [ ] File: `genomic_research/templates/train.py`

### T88: Implement CutMix for sequences
- [ ] Replace random spans with spans from another sequence
- [ ] Mix labels proportionally
- [ ] Add `USE_CUTMIX = True/False`, `CUTMIX_ALPHA = 1.0`
- [ ] File: `genomic_research/templates/train.py`

### T89: Implement token dropout augmentation
- [ ] Randomly drop tokens (replace with PAD) during training
- [ ] Different from masking — no prediction target
- [ ] Add `TOKEN_DROPOUT = 0.0` (0 = disabled)
- [ ] File: `genomic_research/templates/train.py`

### T90: Implement whole-word masking for k-mer tokenizer
- [ ] When using k-mer tokenizer, mask whole k-mers not individual tokens
- [ ] More biologically meaningful masking
- [ ] File: `genomic_research/templates/train.py`

### T91: Implement denoising autoencoder objective
- [ ] Input = corrupted sequence, output = original sequence
- [ ] Corruption: random deletion + shuffling + insertion
- [ ] Add `OBJECTIVE = "denoise"` option
- [ ] File: `genomic_research/templates/train.py`

### T92: Implement contrastive learning augmentation
- [ ] Create positive pairs via augmentation (RC, crop, noise)
- [ ] Add contrastive loss (InfoNCE/SimCLR-style)
- [ ] Add `USE_CONTRASTIVE = True/False`, `CONTRASTIVE_TEMP = 0.07`
- [ ] File: `genomic_research/templates/train.py`

---

## Phase 7: Evaluation & Analysis (T93–T108)

Priority: MEDIUM — deeper understanding of model behavior.

### T93: Add attention weight visualization
- [ ] Extract attention weights from Transformer layers
- [ ] Plot attention heatmap for sample sequences
- [ ] Save as `reports/attention_map.png`
- [ ] File: `genomic_research/templates/train.py`, `prepare.py`

### T94: Add embedding PCA visualization
- [ ] Alternative to t-SNE: faster, linear
- [ ] Save as `reports/embedding_pca.png`
- [ ] Show explained variance ratio
- [ ] File: `genomic_research/templates/prepare.py`

### T95: Add UMAP embedding visualization
- [ ] Better structure preservation than t-SNE
- [ ] Optional: `pip install umap-learn`
- [ ] Save as `reports/embedding_umap.png`
- [ ] File: `genomic_research/templates/prepare.py`

### T96: Add per-position prediction accuracy
- [ ] For MLM: which positions are easiest/hardest to predict?
- [ ] Plot accuracy by position in sequence
- [ ] Save as `reports/position_accuracy.png`
- [ ] File: `genomic_research/templates/prepare.py`

### T97: Add nucleotide-level prediction analysis
- [ ] For each base (A/T/C/G/N), what's the prediction accuracy?
- [ ] Confusion matrix at the nucleotide level
- [ ] Save as `reports/nucleotide_confusion.png`
- [ ] File: `genomic_research/templates/prepare.py`

### T98: Add learning curve analysis
- [ ] Train with 10%, 25%, 50%, 75%, 100% of data
- [ ] Plot val_score vs data size
- [ ] Helps determine if more data would help
- [ ] File: `genomic_research/templates/train.py` or new script

### T99: Add model complexity analysis
- [ ] FLOPs per forward pass
- [ ] Memory footprint (model + activations)
- [ ] Throughput (sequences/second)
- [ ] Save as `reports/complexity.json`
- [ ] File: `genomic_research/templates/train.py`

### T100: Add GC content bias analysis
- [ ] Does the model perform differently on AT-rich vs GC-rich sequences?
- [ ] Bin sequences by GC content, compute val_score per bin
- [ ] Save as `reports/gc_bias.png`
- [ ] File: `genomic_research/templates/prepare.py`

### T101: Add sequence length bias analysis
- [ ] Does the model perform differently on short vs long sequences?
- [ ] Bin sequences by length, compute val_score per bin
- [ ] Save as `reports/length_bias.png`
- [ ] File: `genomic_research/templates/prepare.py`

### T102: Add ROC curve visualization
- [ ] Per-class ROC curves for classification
- [ ] Save as `reports/roc_curves.png`
- [ ] File: `genomic_research/templates/prepare.py`

### T103: Add precision-recall curve visualization
- [ ] Per-class PR curves for classification
- [ ] Save as `reports/pr_curves.png`
- [ ] File: `genomic_research/templates/prepare.py`

### T104: Add calibration plot for classification
- [ ] Expected vs observed probability
- [ ] Brier score
- [ ] Save as `reports/calibration.png`
- [ ] File: `genomic_research/templates/prepare.py`

### T105: Add gradient norm tracking
- [ ] Track gradient norms per layer during training
- [ ] Detect vanishing/exploding gradients
- [ ] Save as `reports/gradient_norms.png`
- [ ] File: `genomic_research/templates/train.py`

### T106: Add weight distribution analysis
- [ ] Histogram of weight values per layer
- [ ] Detect dead neurons
- [ ] Save as `reports/weight_distribution.png`
- [ ] File: `genomic_research/templates/train.py`

### T107: Add motif discovery from attention
- [ ] Extract high-attention regions from sequences
- [ ] Align and find consensus motifs
- [ ] Compare with known biological motifs
- [ ] File: `genomic_research/templates/train.py`

### T108: Add statistical significance testing
- [ ] When comparing two models, compute p-value (paired t-test on val scores)
- [ ] Bootstrap confidence intervals for metrics
- [ ] File: `genomic_research/templates/prepare.py`

---

## Phase 8: CLI & User Experience (T109–T120)

Priority: MEDIUM — better developer experience.

### T109: Add `genomic-research evaluate` command
- [ ] Load trained model and run evaluation on new data
- [ ] Support: `--checkpoint model.pt --fasta new_data.fasta`
- [ ] File: `genomic_research/cli.py`

### T110: Add `genomic-research predict` command
- [ ] Wrapper around inference.py
- [ ] `genomic-research predict --checkpoint model.pt --fasta query.fasta --output predictions.csv`
- [ ] File: `genomic_research/cli.py`

### T111: Add `genomic-research embed` command
- [ ] Extract embeddings and save as numpy/HDF5
- [ ] `genomic-research embed --checkpoint model.pt --fasta data.fasta --output embeddings.npy`
- [ ] File: `genomic_research/cli.py`

### T112: Add `genomic-research compare` command
- [ ] Compare two results.tsv files or two metrics.json files
- [ ] Show delta for each metric, highlight improvements
- [ ] File: `genomic_research/cli.py`

### T113: Add `genomic-research export` command
- [ ] Export model to ONNX, TorchScript, or SafeTensors
- [ ] `genomic-research export --checkpoint model.pt --format onnx --output model.onnx`
- [ ] File: `genomic_research/cli.py`

### T114: Add `genomic-research info` command
- [ ] Show model architecture, parameters, hyperparameters from checkpoint
- [ ] File: `genomic_research/cli.py`

### T115: Add `genomic-research search` command
- [ ] Search NCBI for sequences by organism, gene, etc.
- [ ] Download and prepare data in one step
- [ ] Uses BioPython Entrez API
- [ ] File: `genomic_research/cli.py`

### T116: Add progress bar to training
- [ ] Show real-time progress with tqdm or custom bar
- [ ] Display: step, loss, lr, ETA, val_score
- [ ] File: `genomic_research/templates/train.py`

### T117: Add colored terminal output
- [ ] Green for improvements, red for degradation
- [ ] Bold for important metrics
- [ ] Use ANSI escape codes (no external dependency)
- [ ] File: `genomic_research/templates/train.py`

### T118: Add `--dry-run` flag to train.py
- [ ] Print model architecture, parameter count, estimated time
- [ ] Don't actually train
- [ ] File: `genomic_research/templates/train.py`

### T119: Add `--profile` flag to train.py
- [ ] Run PyTorch profiler for first 10 steps
- [ ] Save Chrome trace to `reports/profile.json`
- [ ] File: `genomic_research/templates/train.py`

### T120: Add interactive experiment dashboard
- [ ] HTML report with all plots, metrics, and experiment history
- [ ] Auto-generated from results.tsv and reports/
- [ ] Single self-contained HTML file
- [ ] File: `genomic_research/templates/prepare.py` or new script

---

## Phase 9: Biological Domain Features (T121–T135)

Priority: MEDIUM — genomics-specific intelligence.

### T121: Add codon-aware tokenization
- [ ] Tokenize in reading frames (codons = 3 bases)
- [ ] Handles coding regions more naturally
- [ ] Add `--tokenizer codon` option
- [ ] File: `genomic_research/templates/prepare.py`

### T122: Add ORF detection and annotation
- [ ] Find open reading frames in sequences
- [ ] Mark coding vs non-coding regions
- [ ] Can use as auxiliary task for multi-task learning
- [ ] File: `genomic_research/templates/prepare.py`

### T123: Add GC content as auxiliary feature
- [ ] Compute local GC content in sliding windows
- [ ] Feed as additional input channel to model
- [ ] File: `genomic_research/templates/train.py`

### T124: Add k-mer frequency spectrum features
- [ ] Compute k-mer frequencies (k=2,3,4) per sequence
- [ ] Use as additional features for classification
- [ ] File: `genomic_research/templates/prepare.py`

### T125: Add phylogenetic-aware data splitting
- [ ] Split train/val based on phylogenetic distance
- [ ] Prevents data leakage from closely related sequences
- [ ] File: `genomic_research/templates/prepare.py`

### T126: Add taxonomic hierarchy classification
- [ ] Multi-level classification: Family → Genus → Species
- [ ] Hierarchical softmax or flat multi-task
- [ ] File: `genomic_research/templates/train.py`

### T127: Add sequence alignment scoring
- [ ] Use trained embeddings for pairwise sequence similarity
- [ ] Compare with BLAST alignment scores
- [ ] File: new `genomic_research/templates/align.py`

### T128: Add variant effect prediction
- [ ] Given a mutation (position, ref, alt), predict effect
- [ ] Binary classification: pathogenic vs benign
- [ ] File: `genomic_research/templates/train.py`

### T129: Add promoter/enhancer prediction
- [ ] Per-position binary classification: regulatory vs non-regulatory
- [ ] Use U-Net or per-position prediction head
- [ ] File: `genomic_research/templates/train.py`

### T130: Add recombination detection
- [ ] Detect breakpoints where recombination occurred
- [ ] Relevant for coronavirus evolution analysis
- [ ] File: new analysis script

### T131: Add mutation rate estimation
- [ ] From pre-trained model, estimate per-position mutation rates
- [ ] Compare with known mutation hotspots
- [ ] File: new analysis script

### T132: Add multiple sequence alignment from embeddings
- [ ] Use model embeddings for guide tree construction
- [ ] Compare with ClustalW/MUSCLE alignments
- [ ] File: new analysis script

### T133: Add antimicrobial resistance gene detection
- [ ] Classification: resistance vs susceptible
- [ ] Common downstream task for bacterial genomics
- [ ] File: `genomic_research/templates/train.py`

### T134: Add host prediction for viruses
- [ ] Given a viral genome, predict the host organism
- [ ] Multi-class classification with known host labels
- [ ] File: `genomic_research/templates/train.py`

### T135: Add geographic origin prediction
- [ ] Predict geographic origin from sequence features
- [ ] Multi-class (continent/country) or regression (lat/lon)
- [ ] File: `genomic_research/templates/train.py`

---

## Phase 10: Production & Deployment (T136–T148)

Priority: LOW-MEDIUM — production readiness.

### T136: Add model quantization (INT8)
- [ ] Post-training quantization with `torch.quantization`
- [ ] Measure speed improvement and accuracy degradation
- [ ] Add `genomic-research export --quantize int8`
- [ ] File: `genomic_research/cli.py`, `inference.py`

### T137: Add TorchScript compilation
- [ ] `torch.jit.script` or `torch.jit.trace` the model
- [ ] Enables deployment without Python
- [ ] File: `genomic_research/templates/inference.py`

### T138: Add torch.compile() support (PyTorch 2.0+)
- [ ] Wrap model with `torch.compile()` for faster training
- [ ] Add `USE_COMPILE = True/False` hyperparameter
- [ ] Only on supported platforms (CUDA + Triton)
- [ ] File: `genomic_research/templates/train.py`

### T139: Create FastAPI inference server
- [ ] REST API: POST /predict with FASTA sequence → JSON response
- [ ] Batch prediction support
- [ ] Health check endpoint
- [ ] File: new `genomic_research/serve.py`

### T140: Create Gradio demo interface
- [ ] Web UI: paste sequence → get prediction/embedding
- [ ] Visualize attention weights
- [ ] File: new `genomic_research/demo.py`

### T141: Add HuggingFace Hub integration
- [ ] Push trained model to HuggingFace Hub
- [ ] Load pre-trained model from HuggingFace Hub
- [ ] `genomic-research push --repo username/model-name`
- [ ] File: `genomic_research/cli.py`

### T142: Add model card generation
- [ ] Auto-generate model card (HuggingFace format)
- [ ] Include: architecture, training data, metrics, limitations
- [ ] File: `genomic_research/templates/train.py`

### T143: Improve Dockerfile
- [ ] Multi-stage build (builder + runtime)
- [ ] Add GPU support (NVIDIA runtime)
- [ ] Add health check
- [ ] Add docker-compose.yml
- [ ] File: `Dockerfile`, `docker-compose.yml`

### T144: Add pip install validation test
- [ ] CI step: `pip install .` in clean virtualenv
- [ ] Verify CLI entry point works
- [ ] Verify all imports succeed
- [ ] File: `.github/workflows/ci.yml`

### T145: Add PyPI publishing workflow
- [ ] GitHub Action to publish to PyPI on release tag
- [ ] Semantic versioning
- [ ] File: `.github/workflows/publish.yml` (new)

### T146: Add model compression via pruning
- [ ] Unstructured pruning (magnitude-based)
- [ ] Structured pruning (remove attention heads)
- [ ] Measure sparsity vs accuracy trade-off
- [ ] File: `genomic_research/templates/train.py`

### T147: Add SafeTensors export
- [ ] Save model weights in SafeTensors format (safer than pickle)
- [ ] Compatible with HuggingFace ecosystem
- [ ] File: `genomic_research/templates/train.py`

### T148: Add batch inference optimization
- [ ] Dynamic batching for variable-length sequences
- [ ] Key-value caching for CLM inference
- [ ] Benchmark: sequences/second
- [ ] File: `genomic_research/templates/inference.py`

---

## Phase 11: Experiment Management (T149–T160)

Priority: LOW-MEDIUM — better experiment tracking.

### T149: Add Weights & Biases integration
- [ ] Optional: `pip install wandb`
- [ ] Log all metrics, hyperparameters, model architecture
- [ ] Add `USE_WANDB = True/False` hyperparameter
- [ ] File: `genomic_research/templates/train.py`

### T150: Add TensorBoard logging
- [ ] Log scalars (loss, lr, val_score) per step
- [ ] Log model graph
- [ ] Add `USE_TENSORBOARD = True/False`
- [ ] File: `genomic_research/templates/train.py`

### T151: Add MLflow integration
- [ ] Optional experiment tracking with MLflow
- [ ] Log metrics, params, artifacts
- [ ] Add `USE_MLFLOW = True/False`
- [ ] File: `genomic_research/templates/train.py`

### T152: Add hyperparameter search with Optuna
- [ ] Define search space in config
- [ ] Run N trials within time budget
- [ ] Save best hyperparameters
- [ ] File: new `genomic_research/templates/search.py`

### T153: Add experiment comparison tool
- [ ] Read all results.tsv entries
- [ ] Generate comparison table and charts
- [ ] Rank experiments by val_score
- [ ] File: `genomic_research/cli.py` or new script

### T154: Add experiment reproducibility checker
- [ ] Record all: code hash, data hash, library versions, seed
- [ ] Verify same setup produces same results (±tolerance)
- [ ] File: `genomic_research/templates/train.py`

### T155: Add config snapshot with each run
- [ ] Save full hyperparameter config as JSON alongside checkpoint
- [ ] Include git commit hash, timestamp
- [ ] File: `genomic_research/templates/train.py`

### T156: Add experiment tagging
- [ ] Tag experiments with descriptive labels
- [ ] `EXPERIMENT_TAG = "baseline_transformer"` in results.tsv
- [ ] File: `genomic_research/templates/train.py`

### T157: Add automatic best model selection
- [ ] After multiple experiments, find the best model across all runs
- [ ] Copy best checkpoint to `best_overall/`
- [ ] File: `genomic_research/cli.py`

### T158: Add experiment diff tool
- [ ] Compare train.py between two experiments
- [ ] Show what hyperparameters changed and their effect
- [ ] File: `genomic_research/cli.py`

### T159: Add results visualization dashboard
- [ ] Plot all experiments: val_score over time, parameter sweeps
- [ ] Interactive HTML report
- [ ] File: new script or integrate into CLI

### T160: Add experiment archiving
- [ ] `genomic-research archive` — save checkpoint, config, results as tar.gz
- [ ] Include model card and metrics
- [ ] File: `genomic_research/cli.py`

---

## Phase 12: Documentation & Education (T161–T170)

Priority: LOW — improve onboarding and understanding.

### T161: Add Jupyter tutorial notebook
- [ ] End-to-end walkthrough: data → prepare → train → evaluate
- [ ] Include Coronaviridae example
- [ ] File: `notebooks/tutorial.ipynb` (new)

### T162: Add architecture comparison notebook
- [ ] Train all architectures on same data
- [ ] Compare: speed, memory, val_score, embedding quality
- [ ] File: `notebooks/architecture_comparison.ipynb` (new)

### T163: Add API documentation
- [ ] Document all public functions in prepare.py
- [ ] Document model classes in train.py
- [ ] Document inference.py functions
- [ ] Use Google-style docstrings
- [ ] File: all template files

### T164: Add CHANGELOG.md
- [ ] Track version changes
- [ ] Follow Keep a Changelog format
- [ ] File: `CHANGELOG.md` (new)

### T165: Add architecture diagrams
- [ ] ASCII or Mermaid diagrams of each architecture
- [ ] Data flow diagram: FASTA → tokens → model → predictions
- [ ] File: `README.md` or `docs/` directory

### T166: Add hyperparameter tuning guide
- [ ] Recommended ranges for each hyperparameter
- [ ] Common pitfalls and solutions
- [ ] File: `docs/hyperparameter_guide.md` (new)

### T167: Add genomics background document
- [ ] Brief intro to DNA, genes, codons, mutations
- [ ] Why pre-training on genomic data is useful
- [ ] File: `docs/genomics_primer.md` (new)

### T168: Add troubleshooting guide
- [ ] Common errors and solutions
- [ ] FAQ section
- [ ] File: `docs/troubleshooting.md` (new)

### T169: Add benchmark results table
- [ ] Standard benchmarks: synthetic data, Coronaviridae
- [ ] Compare all architectures and tokenizers
- [ ] File: `docs/benchmarks.md` (new)

### T170: Improve program.md agent instructions
- [ ] Add more experiment strategies
- [ ] Add genomics-specific optimization tips
- [ ] Add common failure modes and recovery
- [ ] File: `genomic_research/templates/program.md`

---

## Phase 13: Performance & Scalability (T171–T180)

Priority: LOW — optimize for large-scale experiments.

### T171: Profile and optimize data loading
- [ ] Identify bottlenecks in prepare.py
- [ ] Optimize FASTA parsing for large files (>1GB)
- [ ] Use memory-mapped files where possible
- [ ] File: `genomic_research/templates/prepare.py`

### T172: Optimize tokenization speed
- [ ] Batch tokenization instead of per-sequence
- [ ] Parallel tokenization with multiprocessing
- [ ] Benchmark: tokens/second for each tokenizer
- [ ] File: `genomic_research/templates/prepare.py`

### T173: Add mixed precision training with bfloat16
- [ ] bfloat16 support for CUDA and MPS (Apple Silicon)
- [ ] Better numerical stability than fp16
- [ ] Add `AMP_DTYPE = "float16" | "bfloat16"` option
- [ ] File: `genomic_research/templates/train.py`

### T174: Optimize memory for large models
- [ ] Add activation checkpointing per layer
- [ ] Implement CPU offloading for optimizer states
- [ ] File: `genomic_research/templates/train.py`

### T175: Add FSDP support (Fully Sharded Data Parallel)
- [ ] For multi-GPU training with large models
- [ ] Alternative to DDP for memory-constrained setups
- [ ] File: `genomic_research/templates/train.py`

### T176: Optimize inference batch processing
- [ ] Sort by length, batch similar lengths together
- [ ] Reduces padding waste
- [ ] File: `genomic_research/templates/inference.py`

### T177: Add persistent data caching with versioning
- [ ] Hash input file + parameters → cache key
- [ ] Skip re-processing if cache is valid
- [ ] File: `genomic_research/templates/prepare.py`

### T178: Optimize report generation speed
- [ ] Lazy import matplotlib (slow to import)
- [ ] Use Agg backend for headless rendering
- [ ] Parallel figure generation
- [ ] File: `genomic_research/templates/prepare.py`

### T179: Add training throughput logging
- [ ] Log tokens/second, samples/second during training
- [ ] Identify if data loading is the bottleneck
- [ ] File: `genomic_research/templates/train.py`

### T180: Optimize model serialization
- [ ] Only save model weights, not full state (for inference)
- [ ] Support incremental checkpoint saving
- [ ] File: `genomic_research/templates/train.py`

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
