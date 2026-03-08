# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- FastAPI inference server (`genomic-research serve`) with `/predict`, `/health`, `/info` endpoints
- Gradio demo interface (`genomic-research demo`) with sequence analysis and file upload
- HuggingFace Hub integration (`genomic-research push/pull`) for model sharing
- PyPI publishing workflow (`.github/workflows/publish.yml`)
- FSDP (Fully Sharded Data Parallel) support for multi-GPU training
- bfloat16 mixed precision training support
- Optuna-based hyperparameter search (`genomic-research hypersearch`)
- Experiment management: leaderboard, archive, best-model, experiment-diff
- W&B, TensorBoard, and MLflow integration for experiment tracking
- INT8 quantization, magnitude pruning, SafeTensors export formats
- Training throughput logging (samples/sec, tokens/sec)
- Weights-only checkpoint saving for smaller inference deployments
- Config snapshot with git hash for reproducibility
- Fast FASTA parser for large files (>50MB)
- Hash-based data caching to skip redundant preparation

### Changed
- Multi-stage Dockerfile with separate CPU and GPU stages
- Inference batch processing with length-based sorting for efficiency
- Batch tokenization with progress reporting for large datasets

## [0.1.0] - 2025-03-09

### Added
- Initial release
- Core framework: prepare.py (data pipeline) + train.py (model training)
- **Tokenizers**: char, kmer (k=3-8), BPE, codon, protein
- **Architectures**: Transformer, CNN, LSTM/GRU, Conv-Transformer, Perceiver, RWKV, Hyena, Reformer, U-Net, Multi-scale CNN, Deep Sets, Mixture of Experts
- **Tasks**: pre-training (MLM/CLM), classification, regression
- **Input formats**: FASTA, FASTQ, CSV, GenBank, GFF/GTF, VCF
- **Data features**: multi-file input, streaming loader, deduplication, reverse complement doubling, phylogenetic splitting
- **Data augmentation**: SNP noise, indel noise, random crop, token dropout, shuffle, mixup, cutmix
- **Optimizers**: AdamW, SGD, Adam, LAMB with cosine/step/exponential/plateau LR schedules
- **Training features**: gradient accumulation, gradient checkpointing, DDP, EMA, SWA, early stopping, curriculum learning, progressive resizing, knowledge distillation
- **Regularization**: dropout, weight decay, R-Drop, SAM optimizer, focal loss, label smoothing
- **Evaluation**: comprehensive metrics, confusion matrix, ROC/PR curves, calibration plots
- **Visualization**: attention heatmaps, UMAP embeddings, PCA, GC/length bias analysis, motif discovery
- **CLI commands**: init, list-models, status, clean, benchmark, info, evaluate, predict, embed, export, compare, search, dashboard, learning-curve, motif-discovery, align-score, mutation-rate, msa-embed, model-card
- **Export formats**: TorchScript, ONNX, INT8 quantized, pruned, SafeTensors
- CI/CD with GitHub Actions (Python 3.10-3.12)
- Docker support (CPU and GPU)
