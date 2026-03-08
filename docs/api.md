# API Reference

## prepare.py — Data Pipeline & Evaluation

### Data Loading

#### `load_sequences(path, seq_col=None, id_col=None)`
Load sequences from FASTA, FASTQ, GenBank, or CSV files.
- **path**: File path, directory, or glob pattern
- **seq_col**: Column name for sequences (CSV only)
- **id_col**: Column name for sequence IDs (CSV only)
- **Returns**: `list[(id, sequence)]`

#### `load_paired_sequences(r1_path, r2_path, merge="concatenate")`
Load paired-end FASTQ reads and merge them.
- **merge**: `"concatenate"` or `"interleave"`

#### `load_gff_annotations(gff_path, seq_dict=None)`
Parse GFF/GTF annotation file into feature records.

#### `load_vcf_variants(vcf_path, reference_seq=None)`
Parse VCF file for variant data.

### Tokenizers

#### `CharTokenizer`
Character-level tokenizer. 1 token per nucleotide.
- Vocab: `[PAD]=0, [MASK]=1, [CLS]=2, [SEP]=3, [UNK]=4, A=5, T=6, C=7, G=8, N=9`
- Methods: `encode(seq) -> list[int]`, `decode(ids) -> str`, `save(path)`, `load(path)`

#### `KmerTokenizer(k=6)`
K-mer tokenizer. Maps k consecutive nucleotides to one token.
- Vocab size: `4^k + NUM_SPECIAL`
- Same interface as CharTokenizer

#### `BPETokenizer(vocab_size=4096)`
Byte-Pair Encoding tokenizer (requires `tokenizers` package).
- Must call `train(sequences, vocab_size)` before use
- `pip install genomic-research[bpe]`

#### `CodonTokenizer`
Codon-level tokenizer (3 nucleotides per token). Best for coding regions.

#### `ProteinTokenizer`
Amino acid tokenizer for protein sequences.

### Data Preparation

#### `prepare_data(seq_path, task_type, tokenizer_type="char", ...)`
Main data preparation function. Tokenizes, chunks, splits, and saves data.
- **task_type**: `"pretrain"`, `"classify"`, or `"regress"`
- **tokenizer_type**: `"char"`, `"kmer"`, `"bpe"`, `"codon"`, `"protein"`
- **max_length**: Maximum sequence length in tokens (default: 512)
- **labels_path**: CSV file with labels (for classify/regress)
- Saves processed data to `~/.cache/genomic-research/`

#### `load_config() -> dict`
Load task configuration from cache.

#### `load_data(device) -> tuple`
Load preprocessed data tensors from cache.

### Evaluation

#### `evaluate(model, val_data, task_type, config, ...) -> dict`
Evaluate model on validation data. Returns metrics dict including `val_score`.

#### `generate_report(results, task_type, config, ...)`
Generate experiment report with plots and metrics.json.

### Analysis Functions

#### `phylogenetic_split(sequences, val_ratio=0.2, kmer_k=4)`
Split sequences by phylogenetic distance using k-mer clustering.

#### `detect_recombination_breakpoints(sequences, window_size=200)`
Detect recombination breakpoints using sliding window k-mer divergence.

#### `compute_kmer_spectrum(sequences, k_values=(2, 3, 4))`
Compute k-mer frequency spectrum for sequences.

#### `compute_gc_content_features(sequence, window_size=50)`
Compute sliding window GC content features.

#### `detect_orfs(sequence, min_length=100)`
Find open reading frames in a nucleotide sequence.

---

## train.py — Model & Training

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_TYPE` | `"transformer"` | Architecture choice |
| `D_MODEL` | `256` | Model dimension |
| `N_LAYERS` | `6` | Number of layers |
| `N_HEADS` | `8` | Attention heads (Transformer) |
| `D_FF` | `1024` | Feed-forward dimension |
| `LEARNING_RATE` | `1e-4` | Initial learning rate |
| `BATCH_SIZE` | `32` | Training batch size |
| `OBJECTIVE` | `"mlm"` | Pre-training objective (mlm/clm) |
| `MASK_RATIO` | `0.15` | MLM mask ratio |
| `DROPOUT` | `0.1` | Dropout rate |
| `USE_AMP` | `True` | Mixed precision training |

### Model Function

#### `build_model(model_type, vocab_size, max_len, ...) -> nn.Module`
Build model based on specified architecture and hyperparameters.

### Available Architectures
`transformer`, `cnn`, `lstm`, `gru`, `conv_transformer`, `perceiver`, `rwkv`, `hyena`, `reformer`, `unet`, `multiscale_cnn`, `deep_sets`, `moe`

---

## inference.py — Inference

#### `load_model(checkpoint_path, device) -> (model, config, task_config)`
Load trained model from checkpoint file.

#### `extract_embeddings(model, tokens, masks, device, batch_size=64) -> numpy.ndarray`
Extract mean-pooled sequence embeddings. Shape: `(N, D_MODEL)`.

#### `predict(model, tokens, masks, device, task_type, batch_size=64) -> Tensor`
Run inference on tokenized sequences.

---

## CLI Commands

```
genomic-research init          # Initialize experiment
genomic-research list-models   # List architectures
genomic-research status        # Show cache status
genomic-research clean         # Clear cache
genomic-research benchmark     # Compare architectures
genomic-research info          # Model checkpoint info
genomic-research evaluate      # Evaluate on new data
genomic-research predict       # Run predictions
genomic-research embed         # Extract embeddings
genomic-research export        # Export model (TorchScript/ONNX/etc.)
genomic-research compare       # Compare two experiments
genomic-research search        # Search NCBI
genomic-research dashboard     # HTML experiment dashboard
genomic-research serve         # FastAPI inference server
genomic-research demo          # Gradio demo interface
genomic-research push          # Push to HuggingFace Hub
genomic-research pull          # Pull from HuggingFace Hub
genomic-research hypersearch   # Optuna hyperparameter search
genomic-research leaderboard   # Ranked experiment table
genomic-research archive       # Archive experiment as tar.gz
genomic-research best-model    # Find best model from results
genomic-research experiment-diff  # Compare run configs
genomic-research model-card    # Generate model card
genomic-research align-score   # Embedding similarity scoring
genomic-research mutation-rate # Per-position mutation rates
genomic-research msa-embed     # Guide tree from embeddings
```
