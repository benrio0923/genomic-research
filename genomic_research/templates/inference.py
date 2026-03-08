"""
Inference script for genomic-research models.

Usage:
    # Predict masked tokens (MLM)
    python inference.py --checkpoint checkpoints/best_model.pt --fasta query.fasta

    # Extract embeddings
    python inference.py --checkpoint checkpoints/best_model.pt --fasta query.fasta --embeddings out.npy

    # Classify sequences
    python inference.py --checkpoint checkpoints/best_model.pt --fasta query.fasta --classify
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# Add current dir to path for prepare.py imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prepare import (
    PAD_TOKEN_ID, MASK_TOKEN_ID, NUM_SPECIAL,
    load_sequences, CharTokenizer, KmerTokenizer, CACHE_DIR,
)


def load_model(checkpoint_path, device="cpu"):
    """Load a trained model from checkpoint."""
    # Import build_model from train.py
    import importlib.util
    train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    model_type = checkpoint.get("model_type", "transformer")

    # Build model using config
    # We need to construct the model manually since build_model depends on global vars
    from train import build_model
    model = build_model(
        model_type=model_type,
        **model_config,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, model_config, checkpoint.get("task_config", {})


def load_tokenizer(cache_dir=None):
    """Load tokenizer from cache."""
    cache_dir = cache_dir or CACHE_DIR
    tok_path = os.path.join(cache_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        print("Warning: No tokenizer found in cache, using CharTokenizer")
        return CharTokenizer()

    with open(tok_path) as f:
        tok_info = json.load(f)

    tok_type = tok_info.get("type", "char")
    if tok_type == "char":
        return CharTokenizer.load(tok_path)
    elif tok_type == "kmer":
        return KmerTokenizer.load(tok_path)
    else:
        try:
            from prepare import BPETokenizer
            return BPETokenizer.load(tok_path)
        except ImportError:
            print("BPE tokenizer requires 'tokenizers' package")
            sys.exit(1)


def tokenize_sequences(sequences, tokenizer, max_length, sort_by_length=True):
    """Tokenize and pad sequences. Sort by length for efficient batching (T176)."""
    indexed = []
    for idx, (sid, seq) in enumerate(sequences):
        tokens = tokenizer.encode(seq)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        indexed.append((idx, tokens))

    # Sort by token length to minimize padding waste within batches
    if sort_by_length:
        indexed.sort(key=lambda x: len(x[1]))

    sort_order = [i for i, _ in indexed]
    all_tokens = []
    all_masks = []

    for _, tokens in indexed:
        mask = [1] * len(tokens)
        pad_len = max_length - len(tokens)
        tokens = tokens + [PAD_TOKEN_ID] * pad_len
        mask = mask + [0] * pad_len
        all_tokens.append(tokens)
        all_masks.append(mask)

    return (
        torch.tensor(all_tokens, dtype=torch.long),
        torch.tensor(all_masks, dtype=torch.long),
        sort_order,
    )


@torch.no_grad()
def extract_embeddings(model, tokens, masks, device, batch_size=64):
    """Extract sequence embeddings (mean pooling over token embeddings)."""
    all_embeddings = []

    for i in range(0, len(tokens), batch_size):
        batch_tokens = tokens[i:i+batch_size].to(device)
        batch_masks = masks[i:i+batch_size].to(device)

        # Get embeddings from the model's embedding + encoder layers
        x = model.embedding(batch_tokens)

        # Apply position encoding if available
        if hasattr(model, 'pos_encoding') and not getattr(model, 'use_custom_layers', False):
            if getattr(model, 'pos_type', 'sinusoidal') == "learned":
                positions = torch.arange(batch_tokens.size(1), device=device)
                x = x + model.pos_encoding(positions)
            else:
                x = model.pos_encoding(x)

        # Run through encoder/layers
        if hasattr(model, 'encoder'):
            src_key_padding_mask = (batch_masks == 0)
            x = model.encoder(x, src_key_padding_mask=src_key_padding_mask)
        elif hasattr(model, 'layers') and hasattr(model, 'use_custom_layers'):
            for layer in model.layers:
                x = layer(x)
        elif hasattr(model, 'lstm'):
            x, _ = model.lstm(x)
        elif hasattr(model, 'convs') or hasattr(model, 'blocks'):
            x = x.transpose(1, 2)
            if hasattr(model, 'input_proj'):
                x = model.input_proj(x)
            blocks = getattr(model, 'blocks', [])
            for block in blocks:
                x = block(x) + x
            x = x.transpose(1, 2)

        x = model.ln(x)

        # Mean pooling
        mask_exp = batch_masks.unsqueeze(-1).float()
        pooled = (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        all_embeddings.append(pooled.cpu())

    return torch.cat(all_embeddings, dim=0).numpy()


@torch.no_grad()
def predict(model, tokens, masks, device, task_type, batch_size=64):
    """Run inference on sequences."""
    all_outputs = []

    for i in range(0, len(tokens), batch_size):
        batch_tokens = tokens[i:i+batch_size].to(device)
        batch_masks = masks[i:i+batch_size].to(device)

        output = model(batch_tokens, attention_mask=batch_masks)
        all_outputs.append(output.cpu())

    return torch.cat(all_outputs, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Inference with trained genomic model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--fasta", type=str, required=True, help="Input FASTA/FASTQ file")
    parser.add_argument("--embeddings", type=str, default=None,
                        help="Output path for embeddings (.npy)")
    parser.add_argument("--classify", action="store_true", help="Run classification")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, model_config, task_config = load_model(args.checkpoint, device)
    task_type = model_config.get("task_type", "pretrain")
    max_length = model_config.get("max_len", 512)
    print(f"Model type: {task_type} | Max length: {max_length}")

    # Load sequences
    print(f"Loading sequences from {args.fasta}...")
    sequences = load_sequences(args.fasta)
    print(f"Loaded {len(sequences)} sequences")

    # Tokenize (sorted by length for efficient batching)
    tokenizer = load_tokenizer()
    tokens, masks, sort_order = tokenize_sequences(sequences, tokenizer, max_length)
    # Build reverse index: original position -> sorted position
    unsort = [0] * len(sort_order)
    for sorted_idx, orig_idx in enumerate(sort_order):
        unsort[orig_idx] = sorted_idx

    # Extract embeddings
    if args.embeddings:
        print("Extracting embeddings...")
        embeddings = extract_embeddings(model, tokens, masks, device, args.batch_size)
        # Restore original order
        embeddings = embeddings[unsort]
        np.save(args.embeddings, embeddings)
        print(f"Embeddings saved to {args.embeddings} (shape: {embeddings.shape})")
        return

    # Run inference
    print("Running inference...")
    outputs = predict(model, tokens, masks, device, task_type, args.batch_size)

    if task_type == "classify":
        probs = torch.softmax(outputs, dim=-1)
        preds = outputs.argmax(dim=-1)
        target_names = task_config.get("target_names", [])

        for i, (sid, _) in enumerate(sequences):
            si = unsort[i]
            label = target_names[preds[si]] if preds[si] < len(target_names) else str(preds[si].item())
            conf = probs[si, preds[si]].item()
            print(f"{sid}\t{label}\t{conf:.4f}")

    elif task_type == "regress":
        for i, (sid, _) in enumerate(sequences):
            si = unsort[i]
            print(f"{sid}\t{outputs[si].item():.6f}")

    else:  # pretrain - show perplexity per sequence
        import math
        criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, reduction="none")
        vocab_size = model_config["vocab_size"]

        for i, (sid, _) in enumerate(sequences):
            si = unsort[i]
            logits = outputs[si]  # (L, vocab)
            target = tokens[si]  # (L,)
            mask = masks[si]
            valid = (mask == 1) & (target >= NUM_SPECIAL)
            if valid.sum() > 0:
                loss = criterion(logits[valid], target[valid]).mean().item()
                ppl = math.exp(min(loss, 20))
                print(f"{sid}\tperplexity={ppl:.4f}\tloss={loss:.6f}")

    if args.output:
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
