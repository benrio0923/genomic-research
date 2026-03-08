"""
CLI entry point for genomic-research.

Usage:
    genomic-research init --fasta data.fasta --task pretrain
    genomic-research init --csv data.csv --seq-col sequence --task pretrain
    genomic-research list-models
    genomic-research status
    genomic-research clean
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent / "templates"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "genomic-research")

MODEL_ARCHITECTURES = {
    "transformer": {
        "description": "Standard Transformer Encoder (default)",
        "params": "d_model, n_heads, d_ff, n_layers",
        "pros": "Well-understood, flexible, pure PyTorch",
        "cons": "O(n²) attention, slower on very long sequences",
    },
    "mamba": {
        "description": "Mamba SSM (State Space Model)",
        "params": "d_model, d_state, d_conv, expand",
        "pros": "Linear complexity, excellent for long sequences",
        "cons": "Requires CUDA, pip install genomic-research[mamba]",
    },
    "cnn": {
        "description": "1D Convolutional Neural Network",
        "params": "channels, kernel_sizes, n_layers",
        "pros": "Fast training, good at local patterns",
        "cons": "Limited long-range dependencies",
    },
    "lstm": {
        "description": "Bidirectional LSTM/GRU",
        "params": "hidden_size, n_layers, bidirectional",
        "pros": "Sequential modeling, moderate complexity",
        "cons": "Slower training, gradient issues on very long sequences",
    },
    "conv_transformer": {
        "description": "Convolutional Transformer (hybrid)",
        "params": "d_model, n_heads, d_ff, n_layers, n_conv_layers",
        "pros": "Conv captures local patterns, Transformer adds global context",
        "cons": "More complex architecture, slightly more parameters",
    },
    "perceiver": {
        "description": "Perceiver-style (cross-attention latents)",
        "params": "d_model, n_heads, d_ff, n_layers, n_latents",
        "pros": "Sub-quadratic O(n*m), great for very long sequences",
        "cons": "Latent bottleneck may limit per-token prediction quality",
    },
    "rwkv": {
        "description": "RWKV (linear attention with time-decay)",
        "params": "d_model, d_ff, n_layers",
        "pros": "O(n) linear complexity, efficient inference",
        "cons": "May underperform Transformer on short sequences",
    },
    "multiscale_cnn": {
        "description": "Multi-Scale CNN (Inception-style)",
        "params": "d_model, d_ff, n_layers, kernel_sizes",
        "pros": "Captures patterns at multiple scales simultaneously",
        "cons": "More parameters than single-scale CNN",
    },
    "hyena": {
        "description": "Hyena (long convolution via FFT + gating)",
        "params": "d_model, d_ff, n_layers",
        "pros": "Sub-quadratic complexity, good for long sequences (HyenaDNA-inspired)",
        "cons": "Requires FFT, newer architecture with less community support",
    },
    "reformer": {
        "description": "Reformer (LSH Attention)",
        "params": "d_model, n_heads, d_ff, n_layers",
        "pros": "O(n log n) approximate attention, memory-efficient",
        "cons": "Approximate attention may lose precision on short sequences",
    },
    "unet": {
        "description": "U-Net (Encoder-Decoder with Skip Connections)",
        "params": "d_model, d_ff, n_layers",
        "pros": "Excellent for per-position prediction, skip connections preserve detail",
        "cons": "Requires power-of-2 friendly sequence lengths for best performance",
    },
    "deep_sets": {
        "description": "Deep Sets (Permutation Invariant)",
        "params": "d_model, d_ff, n_layers",
        "pros": "Order-invariant, good for metagenomics and set-level tasks",
        "cons": "Loses positional information by design",
    },
}


def cmd_init(args):
    """Initialize a genomic-research experiment in the current directory."""
    cwd = Path.cwd()

    # Copy template files
    files = ["prepare.py", "train.py", "program.md"]
    for f in files:
        src = TEMPLATES_DIR / f
        dst = cwd / f
        if dst.exists() and not args.force:
            print(f"  [skip] {f} already exists (use --force to overwrite)")
            continue
        shutil.copy2(src, dst)
        print(f"  [copy] {f}")

    # Initialize git if not already a repo
    if not (cwd / ".git").exists():
        subprocess.run(["git", "init"], cwd=cwd, capture_output=True)
        print("  [git]  initialized git repo")

    # Build prepare.py command
    print()
    prepare_cmd = [sys.executable, str(cwd / "prepare.py")]

    if args.fasta:
        prepare_cmd.extend(["--fasta", args.fasta])
    elif args.csv:
        prepare_cmd.extend(["--csv", args.csv])
    else:
        print("Error: must specify --fasta or --csv", file=sys.stderr)
        sys.exit(1)

    if args.task:
        prepare_cmd.extend(["--task", args.task])
    if args.tokenizer:
        prepare_cmd.extend(["--tokenizer", args.tokenizer])
    if args.kmer_size:
        prepare_cmd.extend(["--kmer-size", str(args.kmer_size)])
    if args.max_length:
        prepare_cmd.extend(["--max-length", str(args.max_length)])
    if args.seq_col:
        prepare_cmd.extend(["--seq-col", args.seq_col])
    if args.id_col:
        prepare_cmd.extend(["--id-col", args.id_col])
    if args.labels:
        prepare_cmd.extend(["--labels", args.labels])
    if args.label_col:
        prepare_cmd.extend(["--label-col", args.label_col])
    if args.sample_n:
        prepare_cmd.extend(["--sample-n", str(args.sample_n)])
    if args.sample_frac > 0:
        prepare_cmd.extend(["--sample-frac", str(args.sample_frac)])
    if args.rc_double:
        prepare_cmd.append("--rc-double")
    if args.chunk_strategy != "fixed":
        prepare_cmd.extend(["--chunk-strategy", args.chunk_strategy])
    if args.n_folds > 1:
        prepare_cmd.extend(["--n-folds", str(args.n_folds)])

    result = subprocess.run(prepare_cmd, cwd=cwd)
    if result.returncode != 0:
        print("\nError: data preparation failed. See output above.", file=sys.stderr)
        sys.exit(1)

    # Print next steps
    print()
    print("=" * 50)
    print("Ready! Next steps:")
    print()
    print("  1. Quick test:  GENOMIC_TIME_BUDGET=30 python train.py")
    print("  2. Full run:    python train.py")
    print("  3. Agent mode:  claude")
    print('     Then prompt: "Look at program.md and start experimenting"')
    print("=" * 50)


def cmd_list_models(args):
    """List available model architectures."""
    print("Available model architectures:")
    print()
    for name, info in MODEL_ARCHITECTURES.items():
        default = " (default)" if name == "transformer" else ""
        print(f"  {name}{default}")
        print(f"    {info['description']}")
        print(f"    Parameters: {info['params']}")
        print(f"    Pros: {info['pros']}")
        print(f"    Cons: {info['cons']}")
        print()
    print("To use a different architecture, modify MODEL_TYPE in train.py")
    print("or let the AI agent swap it during experimentation.")


def cmd_status(args):
    """Show current cache status."""
    config_path = os.path.join(CACHE_DIR, "task_config.json")
    if not os.path.exists(config_path):
        print("No data loaded. Run: genomic-research init --fasta <file> --task pretrain")
        return

    import json
    with open(config_path) as f:
        config = json.load(f)

    print(f"Cache:       {CACHE_DIR}")
    print(f"Source:      {config['source']}")
    print(f"Task:        {config['task_type']}")
    print(f"Tokenizer:   {config['tokenizer_type']} (vocab_size={config['vocab_size']})")
    print(f"Max length:  {config['max_length']} tokens")
    print(f"Sequences:   {config['n_sequences']}")
    print(f"Train:       {config['n_train']}")
    print(f"Val:         {config['n_val']}")
    if config['task_type'] == "classify":
        print(f"Classes:     {config.get('n_classes', 'N/A')}")


def cmd_clean(args):
    """Remove cached data."""
    if not os.path.exists(CACHE_DIR):
        print("Nothing to clean.")
        return

    shutil.rmtree(CACHE_DIR)
    print(f"Removed: {CACHE_DIR}")


def cmd_benchmark(args):
    """Run benchmark with synthetic data across architectures."""
    import tempfile
    import random

    time_budget = args.time or 30
    models = args.models.split(",") if args.models else ["transformer", "cnn", "lstm"]
    seq_length = args.seq_length or 500
    n_sequences = args.n_sequences or 100

    print(f"Benchmark: {len(models)} models, {n_sequences} sequences, {seq_length}bp each, {time_budget}s budget")
    print("=" * 60)

    # Generate synthetic FASTA
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        fasta_path = f.name
        random.seed(42)
        for i in range(n_sequences):
            seq = "".join(random.choices("ATCG", k=seq_length))
            f.write(f">seq_{i}\n{seq}\n")

    # Prepare data
    cwd = Path(tempfile.mkdtemp(prefix="genomic_bench_"))
    for template in ["prepare.py", "train.py", "program.md"]:
        shutil.copy2(TEMPLATES_DIR / template, cwd / template)

    prepare_cmd = [sys.executable, str(cwd / "prepare.py"),
                   "--fasta", fasta_path, "--task", "pretrain"]
    subprocess.run(prepare_cmd, cwd=cwd, capture_output=True)

    results = []
    for model_type in models:
        print(f"\n--- {model_type} ---")
        # Modify train.py for this model
        train_py = cwd / "train.py"
        content = train_py.read_text()
        content = content.replace('MODEL_TYPE = "transformer"', f'MODEL_TYPE = "{model_type}"')
        train_py.write_text(content)

        env = os.environ.copy()
        env["GENOMIC_TIME_BUDGET"] = str(time_budget)

        result = subprocess.run(
            [sys.executable, str(train_py)],
            cwd=cwd, capture_output=True, text=True, env=env,
        )

        # Parse results
        output = result.stdout + result.stderr
        metrics = {}
        for line in output.split("\n"):
            for key in ["val_score", "val_perplexity", "num_params", "num_steps", "training_seconds"]:
                if line.strip().startswith(f"{key}:"):
                    try:
                        metrics[key] = line.split(":")[1].strip().replace(",", "")
                    except (IndexError, ValueError):
                        pass

        metrics["model"] = model_type
        results.append(metrics)
        print(f"  val_score: {metrics.get('val_score', 'N/A')}")
        print(f"  perplexity: {metrics.get('val_perplexity', 'N/A')}")
        print(f"  params: {metrics.get('num_params', 'N/A')}")
        print(f"  steps: {metrics.get('num_steps', 'N/A')}")

        # Restore original
        content = content.replace(f'MODEL_TYPE = "{model_type}"', 'MODEL_TYPE = "transformer"')
        train_py.write_text(content)

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Model':<15} {'Perplexity':<15} {'Params':<15} {'Steps':<10}")
    print("-" * 55)
    for r in results:
        print(f"{r['model']:<15} {r.get('val_perplexity', 'N/A'):<15} {r.get('num_params', 'N/A'):<15} {r.get('num_steps', 'N/A'):<10}")

    # Cleanup
    os.unlink(fasta_path)
    shutil.rmtree(cwd, ignore_errors=True)


def cmd_info(args):
    """Show model architecture and hyperparameters from a checkpoint."""
    import torch
    import json

    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        print(f"Error: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    print(f"Checkpoint: {ckpt_path}")
    print()

    if "model_config" in ckpt:
        mc = ckpt["model_config"]
        print("Model Configuration:")
        for k, v in mc.items():
            print(f"  {k}: {v}")
        print()

    if "run_info" in ckpt:
        ri = ckpt["run_info"]
        print("Run Info:")
        for k, v in ri.items():
            print(f"  {k}: {v}")
        print()

    if "results" in ckpt:
        print("Results:")
        for k, v in ckpt["results"].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")
        print()

    # Parameter count
    if "model_state_dict" in ckpt:
        total = sum(v.numel() for v in ckpt["model_state_dict"].values())
        print(f"Total parameters: {total:,}")

    print(f"Keys in checkpoint: {list(ckpt.keys())}")


def cmd_evaluate(args):
    """Evaluate a trained model on new data."""
    import torch

    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        print(f"Error: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_config = ckpt.get("model_config", {})
    task_type = model_config.get("task_type", "pretrain")

    # We need prepare.py and train.py accessible
    # Use templates if not in current dir
    cwd = Path.cwd()
    for f in ["prepare.py", "train.py"]:
        if not (cwd / f).exists():
            src = TEMPLATES_DIR / f
            shutil.copy2(src, cwd / f)
            print(f"  [copy] {f}")

    # Prepare new data
    seq_path = args.fasta or args.csv
    if seq_path is None:
        print("Error: must specify --fasta or --csv for evaluation data", file=sys.stderr)
        sys.exit(1)

    prepare_cmd = [sys.executable, str(cwd / "prepare.py")]
    if args.fasta:
        prepare_cmd.extend(["--fasta", args.fasta])
    elif args.csv:
        prepare_cmd.extend(["--csv", args.csv])
    if args.seq_col:
        prepare_cmd.extend(["--seq-col", args.seq_col])
    prepare_cmd.extend(["--task", task_type])
    if args.labels:
        prepare_cmd.extend(["--labels", args.labels])
    if args.label_col:
        prepare_cmd.extend(["--label-col", args.label_col])

    result = subprocess.run(prepare_cmd, cwd=cwd)
    if result.returncode != 0:
        print("\nError: data preparation failed.", file=sys.stderr)
        sys.exit(1)

    # Import and run evaluation
    sys.path.insert(0, str(cwd))
    from prepare import load_config, load_data, evaluate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    data = load_data(device=device)

    # Rebuild model from checkpoint config
    from train import build_model
    model = build_model(**model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    results = evaluate(
        model, data, task_type, config,
        objective=ckpt.get("run_info", {}).get("objective", "mlm"),
        batch_size=32, device=device,
    )

    print("\n--- Evaluation Results ---")
    for k, v in results.items():
        if k in ("predictions", "targets", "probabilities", "confusion_matrix", "per_class"):
            continue
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")


def cmd_predict(args):
    """Run prediction on new sequences using a trained model."""
    import torch
    import csv

    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        print(f"Error: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_config = ckpt.get("model_config", {})
    task_type = model_config.get("task_type", "pretrain")

    if task_type == "pretrain":
        print("Note: predict command works best with classify/regress models.")
        print("For pre-trained models, use 'embed' command to extract embeddings.")

    cwd = Path.cwd()
    for f in ["prepare.py", "train.py"]:
        if not (cwd / f).exists():
            shutil.copy2(TEMPLATES_DIR / f, cwd / f)

    # Prepare data
    seq_path = args.fasta or args.csv
    if seq_path is None:
        print("Error: must specify --fasta or --csv", file=sys.stderr)
        sys.exit(1)

    prepare_cmd = [sys.executable, str(cwd / "prepare.py")]
    if args.fasta:
        prepare_cmd.extend(["--fasta", args.fasta])
    elif args.csv:
        prepare_cmd.extend(["--csv", args.csv])
    if args.seq_col:
        prepare_cmd.extend(["--seq-col", args.seq_col])
    prepare_cmd.extend(["--task", task_type])

    result = subprocess.run(prepare_cmd, cwd=cwd, capture_output=True)
    if result.returncode != 0:
        print("\nError: data preparation failed.", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, str(cwd))
    from prepare import load_config, load_data
    from train import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    data = load_data(device=device)

    model = build_model(**model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Run predictions
    all_tokens = data.get("val_tokens", data.get("train_tokens"))
    all_masks = data.get("val_mask", data.get("train_mask"))

    predictions = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(all_tokens), batch_size):
            batch_tokens = all_tokens[i:i + batch_size].to(device)
            batch_mask = all_masks[i:i + batch_size].to(device)
            output = model(batch_tokens, attention_mask=batch_mask)

            if task_type == "classify":
                preds = output.argmax(dim=-1).cpu().tolist()
                probs = torch.softmax(output, dim=-1).cpu().tolist()
                for j, (p, pr) in enumerate(zip(preds, probs)):
                    predictions.append({"index": i + j, "prediction": p, "confidence": max(pr)})
            elif task_type == "regress":
                preds = output.squeeze(-1).cpu().tolist()
                for j, p in enumerate(preds):
                    predictions.append({"index": i + j, "prediction": p})
            else:
                # Pre-train: skip per-sequence prediction
                break

    # Write output
    output_path = args.output or "predictions.csv"
    if predictions:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
            writer.writeheader()
            writer.writerows(predictions)
        print(f"Predictions saved to {output_path} ({len(predictions)} samples)")
    else:
        print("No predictions generated (model may be pre-train only).")


def cmd_embed(args):
    """Extract sequence embeddings and save to file."""
    import torch
    import numpy as np

    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        print(f"Error: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_config = ckpt.get("model_config", {})

    cwd = Path.cwd()
    for f in ["prepare.py", "train.py"]:
        if not (cwd / f).exists():
            shutil.copy2(TEMPLATES_DIR / f, cwd / f)

    seq_path = args.fasta or args.csv
    if seq_path is None:
        print("Error: must specify --fasta or --csv", file=sys.stderr)
        sys.exit(1)

    prepare_cmd = [sys.executable, str(cwd / "prepare.py")]
    if args.fasta:
        prepare_cmd.extend(["--fasta", args.fasta])
    elif args.csv:
        prepare_cmd.extend(["--csv", args.csv])
    if args.seq_col:
        prepare_cmd.extend(["--seq-col", args.seq_col])
    prepare_cmd.extend(["--task", "pretrain"])

    result = subprocess.run(prepare_cmd, cwd=cwd, capture_output=True)
    if result.returncode != 0:
        print("\nError: data preparation failed.", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, str(cwd))
    from prepare import load_config, load_data
    from train import build_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config()
    data = load_data(device=device)

    model = build_model(**model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Extract embeddings via mean pooling
    all_tokens = torch.cat([data["train_tokens"], data["val_tokens"]])
    all_masks = torch.cat([data["train_mask"], data["val_mask"]])

    embeddings = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(all_tokens), batch_size):
            batch_tokens = all_tokens[i:i + batch_size].to(device)
            batch_mask = all_masks[i:i + batch_size].to(device)
            x = model.embedding(batch_tokens)
            # Mean pooling over non-padding positions
            mask_exp = batch_mask.unsqueeze(-1).float()
            pooled = (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
            embeddings.append(pooled.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    output_path = args.output or "embeddings.npy"
    np.save(output_path, embeddings)
    print(f"Embeddings saved to {output_path} — shape: {embeddings.shape}")


def cmd_export(args):
    """Export a trained model to TorchScript or ONNX format."""
    import torch

    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        print(f"Error: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_config = ckpt.get("model_config", {})

    cwd = Path.cwd()
    for f in ["prepare.py", "train.py"]:
        if not (cwd / f).exists():
            shutil.copy2(TEMPLATES_DIR / f, cwd / f)

    sys.path.insert(0, str(cwd))
    from train import build_model

    model = build_model(**model_config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    fmt = args.format
    seq_len = model_config.get("max_length", 512)

    # Dummy input for tracing
    dummy_tokens = torch.randint(5, 100, (1, seq_len))
    dummy_mask = torch.ones(1, seq_len, dtype=torch.bool)

    if fmt == "torchscript":
        output_path = args.output or "model.pt"
        try:
            traced = torch.jit.trace(model, (dummy_tokens, dummy_mask))
            traced.save(output_path)
            print(f"TorchScript model saved to {output_path}")
        except Exception as e:
            print(f"TorchScript trace failed: {e}", file=sys.stderr)
            print("Trying torch.jit.script instead...")
            try:
                scripted = torch.jit.script(model)
                scripted.save(output_path)
                print(f"TorchScript (scripted) model saved to {output_path}")
            except Exception as e2:
                print(f"TorchScript script also failed: {e2}", file=sys.stderr)
                sys.exit(1)

    elif fmt == "onnx":
        output_path = args.output or "model.onnx"
        try:
            torch.onnx.export(
                model, (dummy_tokens, dummy_mask), output_path,
                input_names=["tokens", "attention_mask"],
                output_names=["output"],
                dynamic_axes={
                    "tokens": {0: "batch", 1: "seq_len"},
                    "attention_mask": {0: "batch", 1: "seq_len"},
                    "output": {0: "batch"},
                },
                opset_version=14,
            )
            print(f"ONNX model saved to {output_path}")
        except Exception as e:
            print(f"ONNX export failed: {e}", file=sys.stderr)
            print("Hint: pip install onnx onnxruntime")
            sys.exit(1)
    else:
        print(f"Unsupported format: {fmt}", file=sys.stderr)
        sys.exit(1)

    # Print export info
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Format:     {fmt}")
    print(f"  File size:  {file_size:.1f} MB")
    print(f"  Parameters: {total_params:,}")
    print(f"  Input:      tokens[batch, {seq_len}] + mask[batch, {seq_len}]")


def cmd_search(args):
    """Search NCBI for sequences and optionally download them."""
    try:
        from Bio import Entrez, SeqIO
    except ImportError:
        print("Error: BioPython required. Install with: pip install biopython")
        return

    Entrez.email = args.email or "genomic-research-user@example.com"
    db = args.database
    query = args.query
    max_results = args.max_results

    print(f"Searching NCBI {db} for: {query}")

    # Search
    handle = Entrez.esearch(db=db, term=query, retmax=max_results)
    results = Entrez.read(handle)
    handle.close()

    ids = results.get("IdList", [])
    total = results.get("Count", "?")
    print(f"Found {total} results, showing top {len(ids)}")

    if not ids:
        print("No results found.")
        return

    # Fetch summaries
    handle = Entrez.esummary(db=db, id=",".join(ids))
    summaries = Entrez.read(handle)
    handle.close()

    for i, doc in enumerate(summaries):
        title = doc.get("Title", "N/A")
        accession = doc.get("AccessionVersion", doc.get("Caption", "N/A"))
        length = doc.get("Length", doc.get("Slen", "?"))
        organism = doc.get("Organism", "N/A")
        print(f"\n  [{i+1}] {accession}")
        print(f"      {title[:80]}")
        print(f"      Organism: {organism} | Length: {length}")

    # Download if requested
    if args.output:
        print(f"\nDownloading {len(ids)} sequences to {args.output}...")
        handle = Entrez.efetch(db=db, id=",".join(ids), rettype="fasta", retmode="text")
        with open(args.output, "w") as f:
            f.write(handle.read())
        handle.close()
        print(f"Saved to {args.output}")
        print(f"Initialize with: genomic-research init --fasta {args.output} --task pretrain")


def cmd_dashboard(args):
    """Generate interactive HTML experiment dashboard."""
    import csv
    from pathlib import Path

    results_file = args.results or "results.tsv"
    report_dir = args.report_dir or "reports"
    output = args.output or "dashboard.html"

    # Load experiment results
    experiments = []
    if os.path.exists(results_file):
        with open(results_file, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                experiments.append(row)

    # Load metrics.json if available
    metrics = {}
    metrics_path = os.path.join(report_dir, "metrics.json")
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path) as f:
            metrics = json.load(f)

    # Find report images
    images = []
    if os.path.isdir(report_dir):
        import base64
        for fname in sorted(os.listdir(report_dir)):
            if fname.endswith(".png"):
                img_path = os.path.join(report_dir, fname)
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                images.append({"name": fname, "data": b64})

    # Generate HTML
    html_parts = [
        "<!DOCTYPE html><html><head>",
        "<meta charset='utf-8'>",
        "<title>genomic-research Dashboard</title>",
        "<style>",
        "body{font-family:system-ui,-apple-system,sans-serif;max-width:1200px;margin:0 auto;padding:20px;background:#f5f5f5}",
        "h1{color:#1a237e}h2{color:#283593;border-bottom:2px solid #e8eaf6;padding-bottom:8px}",
        ".card{background:white;border-radius:8px;padding:20px;margin:16px 0;box-shadow:0 2px 4px rgba(0,0,0,0.1)}",
        "table{border-collapse:collapse;width:100%;font-size:14px}",
        "th,td{padding:8px 12px;text-align:left;border-bottom:1px solid #e0e0e0}",
        "th{background:#e8eaf6;font-weight:600}tr:hover{background:#f5f5f5}",
        ".img-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:16px}",
        ".img-card{background:white;border-radius:8px;padding:12px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}",
        ".img-card img{width:100%;border-radius:4px}",
        ".metric{display:inline-block;background:#e8eaf6;padding:8px 16px;border-radius:20px;margin:4px;font-weight:600}",
        ".good{color:#2e7d32}.bad{color:#c62828}",
        "</style></head><body>",
        "<h1>genomic-research Experiment Dashboard</h1>",
    ]

    # Metrics summary
    if metrics:
        html_parts.append("<div class='card'><h2>Latest Results</h2><div>")
        for key in ["val_score", "val_perplexity", "val_loss", "val_token_accuracy",
                     "val_accuracy", "val_f1_macro", "val_mse", "val_r2"]:
            if key in metrics:
                val = metrics[key]
                cls = "good" if ("accuracy" in key or "f1" in key or "r2" in key) else ""
                html_parts.append(f"<span class='metric {cls}'>{key}: {val}</span>")
        html_parts.append("</div></div>")

    # Experiment history table
    if experiments:
        html_parts.append("<div class='card'><h2>Experiment History</h2><table><tr>")
        cols = list(experiments[0].keys())
        for c in cols:
            html_parts.append(f"<th>{c}</th>")
        html_parts.append("</tr>")
        for exp in experiments:
            html_parts.append("<tr>")
            for c in cols:
                html_parts.append(f"<td>{exp.get(c, '')}</td>")
            html_parts.append("</tr>")
        html_parts.append("</table></div>")

    # Images
    if images:
        html_parts.append("<div class='card'><h2>Report Visualizations</h2><div class='img-grid'>")
        for img in images:
            name = img["name"].replace(".png", "").replace("_", " ").title()
            html_parts.append(f"<div class='img-card'><h3>{name}</h3>")
            html_parts.append(f"<img src='data:image/png;base64,{img['data']}' alt='{name}'></div>")
        html_parts.append("</div></div>")

    html_parts.append(f"<p style='color:#999;text-align:center;margin-top:40px'>Generated by genomic-research</p>")
    html_parts.append("</body></html>")

    with open(output, "w") as f:
        f.write("\n".join(html_parts))
    print(f"Dashboard saved to {output}")
    print(f"Open in browser: file://{os.path.abspath(output)}")


def cmd_learning_curve(args):
    """Run learning curve analysis — train with different data fractions."""
    fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
    checkpoint = args.checkpoint or "checkpoints/best_model.pt"

    if not os.path.exists(checkpoint):
        print(f"Error: Checkpoint not found: {checkpoint}")
        return

    print("Learning Curve Analysis")
    print("=" * 40)
    print(f"Data fractions: {fractions}")
    print(f"This will run {len(fractions)} training runs.\n")
    print("To run: set GENOMIC_DATA_FRACTION environment variable before each training run.")
    print("Example:")
    for frac in fractions:
        print(f"  GENOMIC_DATA_FRACTION={frac} GENOMIC_TIME_BUDGET=60 python train.py")
    print(f"\nThen compare results in results.tsv or use: genomic-research dashboard")


def cmd_motif_discovery(args):
    """Discover motifs from attention weights of a trained model."""
    import torch

    checkpoint = args.checkpoint or "checkpoints/best_model.pt"
    if not os.path.exists(checkpoint):
        print(f"Error: Checkpoint not found: {checkpoint}")
        return

    top_k = args.top_k or 10
    window = args.window or 8

    # Load model
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mc = ckpt["model_config"]
    task_config = ckpt.get("task_config", {})

    sys.path.insert(0, ".")
    from train import build_model
    model = build_model(**mc)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load data
    from prepare import load_data, load_config
    config = load_config()
    data = load_data(device="cpu")

    # Extract attention for first N validation sequences
    n_samples = min(50, len(data["val_tokens"]))
    val_tokens = data["val_tokens"][:n_samples]
    val_mask = data["val_mask"][:n_samples]

    print(f"Extracting attention from {n_samples} sequences...")

    # Check if model has attention layers
    if not hasattr(model, 'layers'):
        print("Model does not have attention layers (requires Transformer).")
        return

    # Get attention weights from first layer
    motif_regions = []
    try:
        with torch.no_grad():
            x = model.embedding(val_tokens)
            if hasattr(model, 'emb_dropout'):
                x = model.emb_dropout(x)

            layer = model.layers[0]
            h = layer.norm1(x) if hasattr(layer, 'norm1') else x
            B, L, D = h.shape
            n_heads = layer.n_heads if hasattr(layer, 'n_heads') else 1
            head_dim = D // n_heads

            q = layer.q_proj(h).view(B, L, n_heads, head_dim).transpose(1, 2)
            k = layer.k_proj(h).view(B, L, n_heads, head_dim).transpose(1, 2)
            scale = head_dim ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            pad_mask = (val_mask == 0).unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(pad_mask, float("-inf"))
            attn = torch.softmax(attn, dim=-1)

            # Average over heads → (B, L, L), then sum over keys → per-position importance
            attn_avg = attn.mean(dim=1)  # (B, L, L)
            importance = attn_avg.sum(dim=-1)  # (B, L)

            # Find top-k high-attention regions per sequence
            # Decode tokens back to nucleotides
            nuc_map = {5: "A", 6: "T", 7: "C", 8: "G", 9: "N"}

            for i in range(B):
                seq_len = int(val_mask[i].sum().item())
                if seq_len < window + 2:
                    continue
                imp = importance[i, 1:seq_len-1]  # skip CLS/SEP
                tokens = val_tokens[i, 1:seq_len-1]

                # Sliding window: find highest-importance windows
                for start in range(0, len(imp) - window + 1):
                    score = imp[start:start+window].mean().item()
                    motif_seq = "".join(nuc_map.get(t.item(), "N") for t in tokens[start:start+window])
                    motif_regions.append((score, motif_seq))

    except Exception as e:
        print(f"Attention extraction failed: {e}")
        return

    if not motif_regions:
        print("No motif regions found.")
        return

    # Sort by attention score and find consensus
    motif_regions.sort(key=lambda x: -x[0])
    top_motifs = motif_regions[:top_k * 5]

    # Simple consensus: count nucleotide frequencies at each position
    from collections import Counter
    pos_counts = [Counter() for _ in range(window)]
    for _, motif in top_motifs[:top_k]:
        for j, c in enumerate(motif):
            pos_counts[j][c] += 1

    consensus = "".join(c.most_common(1)[0][0] for c in pos_counts)

    print(f"\nTop {min(top_k, len(motif_regions))} high-attention motifs:")
    for score, motif in motif_regions[:top_k]:
        print(f"  {motif}  (attention score: {score:.4f})")

    print(f"\nConsensus motif: {consensus}")

    # Save to file
    os.makedirs("reports", exist_ok=True)
    with open("reports/motifs.txt", "w") as f:
        f.write(f"Consensus: {consensus}\n\n")
        f.write("Top motifs by attention score:\n")
        for score, motif in motif_regions[:top_k * 3]:
            f.write(f"  {motif}\t{score:.4f}\n")
    print("Saved to reports/motifs.txt")


def cmd_align_score(args):
    """Score pairwise sequence similarity using trained model embeddings (T127)."""
    import torch

    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        print(f"Error: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    fasta_path = args.fasta
    if not os.path.exists(fasta_path):
        print(f"Error: FASTA file not found: {fasta_path}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    sys.path.insert(0, os.getcwd())
    from prepare import load_sequences, load_tokenizer
    from train import build_model

    seqs = load_sequences(fasta_path)
    if len(seqs) < 2:
        print("Error: need at least 2 sequences for pairwise comparison", file=sys.stderr)
        sys.exit(1)

    # Load tokenizer and model
    tok = load_tokenizer(os.path.join(os.path.expanduser("~"), ".cache", "genomic-research", "tokenizer.json"))
    model_cfg = ckpt.get("model_config", {})
    model = build_model(**model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    max_len = model_cfg.get("max_len", 512)
    top_k = min(args.top_k, len(seqs) * (len(seqs) - 1) // 2)

    # Embed all sequences
    embeddings = []
    ids = []
    with torch.no_grad():
        for seq_id, seq in seqs:
            token_ids = tok.encode(seq)[:max_len]
            token_ids += [0] * (max_len - len(token_ids))
            x = torch.tensor([token_ids], dtype=torch.long, device=device)
            out = model(x)
            # Use mean pooling for embedding
            emb = out.float().mean(dim=1).squeeze(0).cpu().numpy()
            embeddings.append(emb)
            ids.append(seq_id)

    import numpy as np
    embeddings = np.array(embeddings)

    # Compute cosine similarity matrix
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    normed = embeddings / norms
    sim_matrix = normed @ normed.T

    # Report top-k most similar pairs
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            pairs.append((sim_matrix[i, j], ids[i], ids[j]))
    pairs.sort(reverse=True)

    print(f"Pairwise similarity (top {top_k} of {len(pairs)} pairs):")
    print(f"{'Seq 1':<25} {'Seq 2':<25} {'Cosine Sim':>12}")
    print("-" * 64)
    for score, id1, id2 in pairs[:top_k]:
        print(f"{id1:<25} {id2:<25} {score:>12.4f}")


def cmd_mutation_rate(args):
    """Estimate per-position mutation rates from pre-trained model (T131)."""
    import torch

    ckpt_path = args.checkpoint
    fasta_path = args.fasta

    if not os.path.exists(ckpt_path):
        print(f"Error: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(fasta_path):
        print(f"Error: FASTA file not found: {fasta_path}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    sys.path.insert(0, os.getcwd())
    from prepare import load_sequences, load_tokenizer
    from train import build_model

    seqs = load_sequences(fasta_path)
    tok = load_tokenizer(os.path.join(os.path.expanduser("~"), ".cache", "genomic-research", "tokenizer.json"))
    model_cfg = ckpt.get("model_config", {})
    model = build_model(**model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    max_len = model_cfg.get("max_len", 512)
    import numpy as np

    # For each sequence, mask each position and measure model uncertainty (entropy)
    # High entropy = position is more variable = higher mutation rate
    out_path = args.output or "reports/mutation_rates.tsv"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w") as f:
        f.write("seq_id\tposition\tentropy\tpredicted_base\n")
        for seq_id, seq in seqs[:args.max_seqs]:
            token_ids = tok.encode(seq)[:max_len]
            orig_len = len(token_ids)
            token_ids += [0] * (max_len - len(token_ids))

            with torch.no_grad():
                for pos in range(min(orig_len, max_len)):
                    masked = list(token_ids)
                    masked[pos] = 1  # [MASK] token
                    x = torch.tensor([masked], dtype=torch.long, device=device)
                    logits = model(x)
                    probs = torch.softmax(logits[0, pos].float(), dim=-1)
                    entropy = -(probs * (probs + 1e-9).log()).sum().item()
                    pred_base = probs.argmax().item()
                    f.write(f"{seq_id}\t{pos}\t{entropy:.4f}\t{pred_base}\n")

    print(f"Mutation rates saved to {out_path}")
    print(f"Higher entropy = more variable position (potential mutation hotspot)")


def cmd_msa_embed(args):
    """Build guide tree from model embeddings for MSA (T132)."""
    import torch

    ckpt_path = args.checkpoint
    fasta_path = args.fasta

    if not os.path.exists(ckpt_path):
        print(f"Error: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(fasta_path):
        print(f"Error: FASTA file not found: {fasta_path}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    sys.path.insert(0, os.getcwd())
    from prepare import load_sequences, load_tokenizer
    from train import build_model
    import numpy as np

    seqs = load_sequences(fasta_path)
    tok = load_tokenizer(os.path.join(os.path.expanduser("~"), ".cache", "genomic-research", "tokenizer.json"))
    model_cfg = ckpt.get("model_config", {})
    model = build_model(**model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    max_len = model_cfg.get("max_len", 512)

    # Embed all sequences
    embeddings = []
    ids = []
    with torch.no_grad():
        for seq_id, seq in seqs:
            token_ids = tok.encode(seq)[:max_len]
            token_ids += [0] * (max_len - len(token_ids))
            x = torch.tensor([token_ids], dtype=torch.long, device=device)
            out = model(x)
            emb = out.float().mean(dim=1).squeeze(0).cpu().numpy()
            embeddings.append(emb)
            ids.append(seq_id)

    embeddings = np.array(embeddings)

    # Build distance matrix (cosine distance)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    normed = embeddings / norms
    sim_matrix = normed @ normed.T
    dist_matrix = 1.0 - sim_matrix

    # Build neighbor-joining guide tree (simple UPGMA)
    n = len(ids)
    clusters = {i: ids[i] for i in range(n)}
    dists = {(i, j): dist_matrix[i, j] for i in range(n) for j in range(i + 1, n)}

    newick_parts = {i: ids[i] for i in range(n)}
    next_id = n

    while len(clusters) > 1:
        # Find closest pair
        min_pair = min(dists, key=dists.get)
        i, j = min_pair
        d = dists[min_pair]

        # Merge
        new_name = f"({newick_parts[i]}:{d/2:.6f},{newick_parts[j]}:{d/2:.6f})"
        newick_parts[next_id] = new_name

        # Update distances (UPGMA average)
        new_dists = {}
        for k in clusters:
            if k == i or k == j:
                continue
            di = dists.get((min(i, k), max(i, k)), 0)
            dj = dists.get((min(j, k), max(j, k)), 0)
            new_dists[(min(next_id, k), max(next_id, k))] = (di + dj) / 2

        # Remove old entries
        to_remove = [(a, b) for a, b in dists if a in (i, j) or b in (i, j)]
        for key in to_remove:
            del dists[key]
        del clusters[i]
        del clusters[j]

        dists.update(new_dists)
        clusters[next_id] = new_name
        next_id += 1

    newick = list(newick_parts.values())[-1] + ";"

    out_path = args.output or "reports/guide_tree.nwk"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(newick + "\n")

    print(f"Guide tree saved to {out_path} (Newick format)")
    print(f"Use this tree with ClustalW/MUSCLE for embedding-guided MSA")


def cmd_model_card(args):
    """Generate model card in HuggingFace format (T142)."""
    import torch, json

    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        print(f"Error: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt.get("model_config", {})
    run_info = ckpt.get("run_info", {})

    # Load metrics if available
    metrics = {}
    metrics_path = "reports/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    model_type = model_cfg.get("model_type", "unknown")
    task_type = model_cfg.get("task_type", "pretrain")
    vocab_size = model_cfg.get("vocab_size", "?")
    d_model = model_cfg.get("d_model", "?")
    n_layers = model_cfg.get("n_layers", "?")
    n_params = sum(p.numel() for p in torch.nn.Module.__new__(torch.nn.Module).__class__.__mro__[0].__dict__.items()) if False else "?"
    try:
        sys.path.insert(0, os.getcwd())
        from train import build_model
        m = build_model(**model_cfg)
        n_params = sum(p.numel() for p in m.parameters())
    except Exception:
        n_params = "?"

    card = f"""---
language: en
tags:
- genomics
- {model_type}
- {task_type}
- dna
- biology
library_name: pytorch
---

# Genomic {model_type.title()} Model

## Model Description

This is a **{model_type}** model trained for **{task_type}** on genomic (DNA/RNA) sequences
using the [genomic-research](https://github.com/benrio0923/genomic-research) framework.

### Architecture

| Parameter | Value |
|-----------|-------|
| Model Type | {model_type} |
| Hidden Dimension | {d_model} |
| Layers | {n_layers} |
| Vocab Size | {vocab_size} |
| Parameters | {n_params:,} |
| Task | {task_type} |

### Training Details

"""
    if run_info:
        for k, v in run_info.items():
            card += f"- **{k}**: {v}\n"

    if metrics:
        card += "\n### Evaluation Results\n\n"
        card += "| Metric | Value |\n|--------|-------|\n"
        skip = {"predictions", "targets", "probabilities", "confusion_matrix",
                "per_class", "run_info", "metric_direction"}
        for k, v in metrics.items():
            if k in skip or not isinstance(v, (int, float)):
                continue
            card += f"| {k} | {v:.4f} |\n"

    card += """
## Usage

```python
import torch
from train import build_model

checkpoint = torch.load("best_model.pt", map_location="cpu")
model = build_model(**checkpoint["model_config"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

## Limitations

- Trained on specific genomic data; may not generalize to all organisms
- Performance depends on sequence similarity to training data
- Tokenization must match training configuration

## Framework

Built with [genomic-research](https://github.com/benrio0923/genomic-research)
"""

    out_path = args.output or "MODEL_CARD.md"
    with open(out_path, "w") as f:
        f.write(card)
    print(f"Model card saved to {out_path}")


def cmd_compare(args):
    """Compare two experiment results (metrics.json files)."""
    import json

    path1 = args.file1
    path2 = args.file2

    if not os.path.exists(path1):
        print(f"Error: file not found: {path1}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(path2):
        print(f"Error: file not found: {path2}", file=sys.stderr)
        sys.exit(1)

    with open(path1) as f:
        m1 = json.load(f)
    with open(path2) as f:
        m2 = json.load(f)

    # Collect all numeric metrics
    all_keys = sorted(set(list(m1.keys()) + list(m2.keys())))
    skip_keys = {"predictions", "targets", "probabilities", "confusion_matrix",
                 "per_class", "run_info", "metric_direction"}

    print(f"{'Metric':<30} {'File 1':>12} {'File 2':>12} {'Delta':>12} {'Change':>8}")
    print("-" * 76)
    for key in all_keys:
        if key in skip_keys:
            continue
        v1 = m1.get(key)
        v2 = m2.get(key)
        if not isinstance(v1, (int, float)) or not isinstance(v2, (int, float)):
            continue
        delta = v2 - v1
        pct = (delta / abs(v1) * 100) if v1 != 0 else 0
        sign = "+" if delta > 0 else ""
        # Green for improvements in val_score, red for degradation
        print(f"{key:<30} {v1:>12.4f} {v2:>12.4f} {sign}{delta:>11.4f} {sign}{pct:>6.1f}%")


def main():
    parser = argparse.ArgumentParser(
        prog="genomic-research",
        description="Genomic sequence AI research framework — train foundation models with AI agents",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init
    p_init = subparsers.add_parser("init", help="Initialize experiment in current directory")
    p_init.add_argument("--fasta", type=str, default=None, help="Path to FASTA/FASTQ file")
    p_init.add_argument("--csv", type=str, default=None, help="Path to CSV file")
    p_init.add_argument("--task", type=str, default="pretrain",
                        choices=["pretrain", "classify", "regress"],
                        help="Task type (default: pretrain)")
    p_init.add_argument("--tokenizer", type=str, default=None,
                        choices=["char", "kmer", "bpe"],
                        help="Tokenizer type (default: char)")
    p_init.add_argument("--kmer-size", type=int, default=None, help="K-mer size (default: 6)")
    p_init.add_argument("--max-length", type=int, default=None, help="Max sequence length in tokens")
    p_init.add_argument("--seq-col", type=str, default=None, help="Sequence column name (CSV)")
    p_init.add_argument("--id-col", type=str, default=None, help="ID column name")
    p_init.add_argument("--labels", type=str, default=None, help="Labels CSV file (classify/regress)")
    p_init.add_argument("--label-col", type=str, default=None, help="Label column name")
    p_init.add_argument("--sample-n", type=int, default=0, help="Subsample to N sequences")
    p_init.add_argument("--sample-frac", type=float, default=0.0, help="Subsample fraction (0-1)")
    p_init.add_argument("--rc-double", action="store_true", help="Double dataset with reverse complement")
    p_init.add_argument("--chunk-strategy", type=str, default="fixed",
                        choices=["fixed", "none", "random", "slide"],
                        help="Chunking strategy for long sequences (default: fixed)")
    p_init.add_argument("--n-folds", type=int, default=1,
                        help="Number of CV folds (1 = no CV)")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing files")

    # list-models
    subparsers.add_parser("list-models", help="List available model architectures")

    # status
    subparsers.add_parser("status", help="Show current data and cache info")

    # clean
    subparsers.add_parser("clean", help="Remove cached data")

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run benchmark across architectures")
    p_bench.add_argument("--models", type=str, default=None,
                         help="Comma-separated model types (default: transformer,cnn,lstm)")
    p_bench.add_argument("--time", type=int, default=30, help="Time budget per model in seconds")
    p_bench.add_argument("--seq-length", type=int, default=500, help="Synthetic sequence length")
    p_bench.add_argument("--n-sequences", type=int, default=100, help="Number of synthetic sequences")

    # info
    p_info = subparsers.add_parser("info", help="Show model info from checkpoint")
    p_info.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate trained model on new data")
    p_eval.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    p_eval.add_argument("--fasta", type=str, default=None, help="Path to FASTA/FASTQ file")
    p_eval.add_argument("--csv", type=str, default=None, help="Path to CSV file")
    p_eval.add_argument("--seq-col", type=str, default=None, help="Sequence column name (CSV)")
    p_eval.add_argument("--labels", type=str, default=None, help="Labels CSV (for computing metrics)")
    p_eval.add_argument("--label-col", type=str, default=None, help="Label column name")

    # predict
    p_pred = subparsers.add_parser("predict", help="Run prediction on new sequences")
    p_pred.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    p_pred.add_argument("--fasta", type=str, default=None, help="Path to FASTA/FASTQ file")
    p_pred.add_argument("--csv", type=str, default=None, help="Path to CSV file")
    p_pred.add_argument("--seq-col", type=str, default=None, help="Sequence column name (CSV)")
    p_pred.add_argument("--output", type=str, default="predictions.csv", help="Output file path")

    # embed
    p_embed = subparsers.add_parser("embed", help="Extract sequence embeddings")
    p_embed.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    p_embed.add_argument("--fasta", type=str, default=None, help="Path to FASTA/FASTQ file")
    p_embed.add_argument("--csv", type=str, default=None, help="Path to CSV file")
    p_embed.add_argument("--seq-col", type=str, default=None, help="Sequence column name (CSV)")
    p_embed.add_argument("--output", type=str, default="embeddings.npy", help="Output file (.npy)")

    # export
    p_export = subparsers.add_parser("export", help="Export model to TorchScript or ONNX")
    p_export.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    p_export.add_argument("--format", type=str, default="torchscript",
                          choices=["torchscript", "onnx"],
                          help="Export format (default: torchscript)")
    p_export.add_argument("--output", type=str, default=None, help="Output file path")

    # compare
    p_compare = subparsers.add_parser("compare", help="Compare two experiment results")
    p_compare.add_argument("file1", type=str, help="First metrics.json file")
    p_compare.add_argument("file2", type=str, help="Second metrics.json file")

    # search
    p_search = subparsers.add_parser("search", help="Search NCBI for sequences")
    p_search.add_argument("query", type=str, help="Search query (e.g., 'SARS-CoV-2[organism]')")
    p_search.add_argument("--database", "-d", type=str, default="nucleotide",
                          help="NCBI database (nucleotide, protein, gene)")
    p_search.add_argument("--max-results", "-n", type=int, default=10, help="Max results to show")
    p_search.add_argument("--output", "-o", type=str, default=None, help="Download FASTA to this file")
    p_search.add_argument("--email", type=str, default=None, help="Email for NCBI Entrez API")

    # dashboard
    p_dash = subparsers.add_parser("dashboard", help="Generate interactive HTML experiment dashboard")
    p_dash.add_argument("--results", type=str, default="results.tsv", help="Results TSV file")
    p_dash.add_argument("--report-dir", type=str, default="reports", help="Report directory with plots")
    p_dash.add_argument("--output", "-o", type=str, default="dashboard.html", help="Output HTML file")

    # learning-curve
    p_lc = subparsers.add_parser("learning-curve", help="Run learning curve analysis")
    p_lc.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Model checkpoint")

    # motif-discovery
    p_motif = subparsers.add_parser("motif-discovery", help="Discover motifs from attention weights")
    p_motif.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Model checkpoint")
    p_motif.add_argument("--top-k", type=int, default=10, help="Number of top motifs to show")
    p_motif.add_argument("--window", type=int, default=8, help="Motif window size")

    # align-score
    p_align = subparsers.add_parser("align-score", help="Score pairwise similarity using model embeddings")
    p_align.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    p_align.add_argument("--fasta", type=str, required=True, help="FASTA file with sequences to compare")
    p_align.add_argument("--top-k", type=int, default=20, help="Show top-k similar pairs")

    # mutation-rate
    p_mut = subparsers.add_parser("mutation-rate", help="Estimate per-position mutation rates from model")
    p_mut.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    p_mut.add_argument("--fasta", type=str, required=True, help="FASTA file")
    p_mut.add_argument("--output", type=str, default=None, help="Output TSV path")
    p_mut.add_argument("--max-seqs", type=int, default=5, help="Max sequences to analyze")

    # msa-embed
    p_msa = subparsers.add_parser("msa-embed", help="Build guide tree from model embeddings for MSA")
    p_msa.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    p_msa.add_argument("--fasta", type=str, required=True, help="FASTA file")
    p_msa.add_argument("--output", type=str, default=None, help="Output Newick tree path")

    # model-card
    p_card = subparsers.add_parser("model-card", help="Generate HuggingFace-format model card")
    p_card.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    p_card.add_argument("--output", type=str, default=None, help="Output path (default: MODEL_CARD.md)")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "list-models":
        cmd_list_models(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "clean":
        cmd_clean(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "embed":
        cmd_embed(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)
    elif args.command == "learning-curve":
        cmd_learning_curve(args)
    elif args.command == "motif-discovery":
        cmd_motif_discovery(args)
    elif args.command == "align-score":
        cmd_align_score(args)
    elif args.command == "mutation-rate":
        cmd_mutation_rate(args)
    elif args.command == "msa-embed":
        cmd_msa_embed(args)
    elif args.command == "model-card":
        cmd_model_card(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
