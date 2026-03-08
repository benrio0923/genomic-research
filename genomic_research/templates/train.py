"""
genomic-research training script — genomic foundation model pre-training.
Single-file, device-agnostic (CPU or CUDA).

Default: Transformer Encoder + MLM objective.
Agent can replace model architecture entirely (Mamba, CNN, LSTM, hybrid, etc.).

Usage: python train.py
"""

import math
import time

import torch
import torch.nn as nn

from prepare import (
    TIME_BUDGET, MASK_TOKEN_ID, PAD_TOKEN_ID, NUM_SPECIAL,
    load_config, load_data, make_dataloader, evaluate, generate_report,
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these — this is what the agent modifies)
# ---------------------------------------------------------------------------

MODEL_TYPE = "transformer"     # transformer, mamba, cnn, lstm (informational)
D_MODEL = 256                  # model dimension
N_LAYERS = 6                   # number of layers
N_HEADS = 8                    # attention heads (transformer only)
D_FF = 1024                    # feed-forward dimension
DROPOUT = 0.1                  # dropout rate
LEARNING_RATE = 1e-4           # learning rate
WEIGHT_DECAY = 1e-4            # L2 regularization
BATCH_SIZE = 32                # training batch size
LR_SCHEDULE = "cosine"         # lr schedule: constant, cosine, step
WARMUP_RATIO = 0.05            # fraction of training for LR warmup
OBJECTIVE = "mlm"              # mlm or clm
MASK_RATIO = 0.15              # fraction of tokens to mask (MLM only)

# ---------------------------------------------------------------------------
# Model — Transformer Encoder (agent can replace entirely)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=4096, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class GenomicTransformer(nn.Module):
    """Transformer Encoder for genomic sequence modeling."""

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len, dropout, task_type, n_classes=None):
        super().__init__()
        self.task_type = task_type
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN_ID)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)

        # Task-specific heads
        if task_type == "pretrain":
            self.head = nn.Linear(d_model, vocab_size)
        elif task_type == "classify":
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, n_classes),
            )
        elif task_type == "regress":
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)

        # Create padding mask for transformer (True = ignore)
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        # Causal mask for CLM
        if self.task_type == "pretrain" and hasattr(self, '_use_causal') and self._use_causal:
            sz = input_ids.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(sz, device=input_ids.device)
            x = self.encoder(x, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
        else:
            x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        x = self.ln(x)

        if self.task_type == "pretrain":
            return self.head(x)  # (B, L, vocab_size)
        else:
            # Use mean pooling over non-padded tokens
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)
            return self.head(x)  # (B, n_classes) or (B, 1)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed(42)
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

config = load_config()
task_type = config["task_type"]
vocab_size = config["vocab_size"]
max_length = config["max_length"]
print(f"Task: {task_type} | Vocab: {vocab_size} | Max length: {max_length}")
print(f"Train: {config['n_train']} | Val: {config['n_val']}")

data = load_data(device=device)

# Build model
n_classes = config.get("n_classes")
model = GenomicTransformer(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    d_ff=D_FF,
    n_layers=N_LAYERS,
    max_len=max_length,
    dropout=DROPOUT,
    task_type=task_type,
    n_classes=n_classes,
).to(device)

if OBJECTIVE == "clm":
    model._use_causal = True

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,}")

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

# Loss function
if task_type == "pretrain":
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)
elif task_type == "classify":
    criterion = nn.CrossEntropyLoss()
else:
    criterion = nn.MSELoss()

# DataLoader
actual_batch_size = min(BATCH_SIZE, config["n_train"] // 2) if config["n_train"] > 1 else 1
use_drop_last = config["n_train"] > actual_batch_size

if task_type in ("classify", "regress"):
    train_loader = make_dataloader(
        data["train_tokens"], data["train_mask"], actual_batch_size,
        shuffle=True, drop_last=use_drop_last, labels=data.get("train_labels"),
    )
else:
    train_loader = make_dataloader(
        data["train_tokens"], data["train_mask"], actual_batch_size,
        shuffle=True, drop_last=use_drop_last,
    )

print(f"Time budget: {TIME_BUDGET}s")
print(f"Batch size: {actual_batch_size}")
print(f"Model: {MODEL_TYPE} | d_model={D_MODEL} | layers={N_LAYERS} | heads={N_HEADS}")
print(f"Objective: {OBJECTIVE} | Mask ratio: {MASK_RATIO}")
print(f"LR: {LEARNING_RATE} | Weight decay: {WEIGHT_DECAY} | Schedule: {LR_SCHEDULE}")

# ---------------------------------------------------------------------------
# LR Schedule
# ---------------------------------------------------------------------------

def get_lr(elapsed_fraction):
    """Compute learning rate based on time-based progress."""
    progress = min(elapsed_fraction, 1.0)

    if progress < WARMUP_RATIO:
        warmup_factor = progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
        return LEARNING_RATE * warmup_factor

    adjusted_progress = (progress - WARMUP_RATIO) / (1.0 - WARMUP_RATIO)

    if LR_SCHEDULE == "cosine":
        return LEARNING_RATE * 0.5 * (1 + math.cos(math.pi * adjusted_progress))
    elif LR_SCHEDULE == "step":
        if adjusted_progress > 0.7:
            return LEARNING_RATE * 0.01
        elif adjusted_progress > 0.4:
            return LEARNING_RATE * 0.1
        return LEARNING_RATE
    else:
        return LEARNING_RATE


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
total_training_time = 0.0
step = 0
epoch = 0
smooth_train_loss = 0.0
best_val_score = None
best_state_dict = None

history = {
    "steps": [], "losses": [], "lrs": [],
    "eval_steps": [], "eval_scores": [],
    "eval_perplexities": [], "eval_token_accuracies": [],
}

eval_interval_seconds = max(TIME_BUDGET * 0.2, 10)
last_eval_time = 0.0

model.train()

while True:
    epoch += 1
    for batch in train_loader:
        t0 = time.time()

        if task_type in ("classify", "regress"):
            tokens_batch, mask_batch, labels_batch = batch
            tokens_batch = tokens_batch.to(device)
            mask_batch = mask_batch.to(device)
            labels_batch = labels_batch.to(device)

            output = model(tokens_batch, attention_mask=mask_batch)
            if task_type == "regress":
                output = output.squeeze(-1)
            loss = criterion(output, labels_batch)

        else:  # pretrain
            tokens_batch, mask_batch = batch
            tokens_batch = tokens_batch.to(device)
            mask_batch = mask_batch.to(device)

            if OBJECTIVE == "mlm":
                # Create MLM input
                input_ids = tokens_batch.clone()
                labels = tokens_batch.clone()

                mask_candidates = (mask_batch == 1) & (tokens_batch >= NUM_SPECIAL)
                rand = torch.rand_like(tokens_batch.float())
                mask_positions = mask_candidates & (rand < MASK_RATIO)

                input_ids[mask_positions] = MASK_TOKEN_ID
                labels[~mask_positions] = PAD_TOKEN_ID

                logits = model(input_ids, attention_mask=mask_batch)
                loss = criterion(logits.reshape(-1, vocab_size), labels.reshape(-1))

            else:  # CLM
                input_ids = tokens_batch[:, :-1]
                target_ids = tokens_batch[:, 1:]
                input_mask = mask_batch[:, :-1]

                logits = model(input_ids, attention_mask=input_mask)
                loss = criterion(logits.reshape(-1, vocab_size), target_ids.reshape(-1))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update LR
        elapsed_fraction = total_training_time / TIME_BUDGET if TIME_BUDGET > 0 else 1.0
        lr = get_lr(elapsed_fraction)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        t1 = time.time()
        dt = t1 - t0

        if step > 5:
            total_training_time += dt

        # Logging
        train_loss_f = loss.item()
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
        pct_done = 100 * min(total_training_time / TIME_BUDGET, 1.0)
        remaining = max(0, TIME_BUDGET - total_training_time)

        if step % 50 == 0:
            history["steps"].append(step)
            history["losses"].append(debiased_smooth_loss)
            history["lrs"].append(lr)
            print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lr: {lr:.2e} | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

        # Periodic evaluation
        if step > 5 and total_training_time - last_eval_time >= eval_interval_seconds:
            model.eval()
            eval_results = evaluate(
                model, data, task_type, config,
                objective=OBJECTIVE, batch_size=actual_batch_size * 2,
                device=device, mask_ratio=MASK_RATIO,
            )
            history["eval_steps"].append(step)
            history["eval_scores"].append(eval_results["val_score"])

            if task_type == "pretrain":
                history["eval_perplexities"].append(eval_results.get("val_perplexity", 0))
                if "val_token_accuracy" in eval_results:
                    history["eval_token_accuracies"].append(eval_results["val_token_accuracy"])

            if best_val_score is None or eval_results["val_score"] > best_val_score:
                best_val_score = eval_results["val_score"]
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            model.train()
            last_eval_time = total_training_time

        step += 1

        if step > 5 and total_training_time >= TIME_BUDGET:
            break

    if step > 5 and total_training_time >= TIME_BUDGET:
        break

print()

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

model.eval()
final_results = evaluate(
    model, data, task_type, config,
    objective=OBJECTIVE, batch_size=actual_batch_size * 2,
    device=device, mask_ratio=MASK_RATIO,
)

history["eval_steps"].append(step)
history["eval_scores"].append(final_results["val_score"])
if task_type == "pretrain":
    history["eval_perplexities"].append(final_results.get("val_perplexity", 0))
    if "val_token_accuracy" in final_results:
        history["eval_token_accuracies"].append(final_results["val_token_accuracy"])

if best_state_dict is not None and best_val_score is not None:
    if best_val_score > final_results["val_score"]:
        model.load_state_dict(best_state_dict)
        results = evaluate(
            model, data, task_type, config,
            objective=OBJECTIVE, batch_size=actual_batch_size * 2,
            device=device, mask_ratio=MASK_RATIO,
        )
    else:
        results = final_results
else:
    results = final_results

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

t_end = time.time()

if device.type == "cuda":
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
else:
    peak_vram_mb = 0.0

print("---")
print(f"task_type:        {task_type}")
print(f"objective:        {OBJECTIVE}")
print(f"val_score:        {results['val_score']:.6f}")

if task_type == "pretrain":
    print(f"val_perplexity:   {results.get('val_perplexity', 0):.4f}")
    print(f"val_loss:         {results.get('val_loss', 0):.6f}")
    if "val_token_accuracy" in results:
        print(f"val_token_acc:    {results['val_token_accuracy']:.4f}")
elif task_type == "classify":
    print(f"val_accuracy:     {results['val_accuracy']:.6f}")
    print(f"val_f1_macro:     {results['val_f1_macro']:.6f}")
    print(f"val_f1_weighted:  {results['val_f1_weighted']:.6f}")
    print(f"val_precision:    {results['val_precision_macro']:.6f}")
    print(f"val_recall:       {results['val_recall_macro']:.6f}")
else:
    print(f"val_mse:          {results['val_mse']:.6f}")
    print(f"val_rmse:         {results['val_rmse']:.6f}")
    print(f"val_mae:          {results['val_mae']:.6f}")
    print(f"val_r2:           {results['val_r2']:.6f}")

print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_epochs:       {epoch}")
print(f"num_params:       {num_params:,}")
print(f"model_type:       {MODEL_TYPE}")
print(f"device:           {device}")

# ---------------------------------------------------------------------------
# Generate report
# ---------------------------------------------------------------------------

run_info = {
    "model_type": MODEL_TYPE,
    "d_model": D_MODEL,
    "n_layers": N_LAYERS,
    "n_heads": N_HEADS,
    "d_ff": D_FF,
    "dropout": DROPOUT,
    "objective": OBJECTIVE,
    "mask_ratio": MASK_RATIO,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "batch_size": actual_batch_size,
    "lr_schedule": LR_SCHEDULE,
    "training_seconds": round(total_training_time, 1),
    "total_seconds": round(t_end - t_start, 1),
    "num_steps": step,
    "num_epochs": epoch,
    "num_params": num_params,
    "peak_vram_mb": round(peak_vram_mb, 1),
    "device": str(device),
}

generate_report(results, task_type, config, training_history=history,
                report_dir="reports", run_info=run_info)
