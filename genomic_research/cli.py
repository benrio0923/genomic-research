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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
