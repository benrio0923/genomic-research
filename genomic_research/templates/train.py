"""
genomic-research training script — genomic foundation model pre-training.
Single-file, device-agnostic (CPU or CUDA).

Default: Transformer Encoder + MLM objective.
Agent can replace model architecture entirely (Mamba, CNN, LSTM, hybrid, etc.).

Usage: python train.py
"""

import math
import os
import random
import signal
import sys
import time

import numpy as np

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Terminal color helpers (ANSI escape codes, no external dependency)
# ---------------------------------------------------------------------------
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

def _green(s): return f"\033[32m{s}\033[0m" if _USE_COLOR else str(s)
def _red(s): return f"\033[31m{s}\033[0m" if _USE_COLOR else str(s)
def _yellow(s): return f"\033[33m{s}\033[0m" if _USE_COLOR else str(s)
def _bold(s): return f"\033[1m{s}\033[0m" if _USE_COLOR else str(s)
def _cyan(s): return f"\033[36m{s}\033[0m" if _USE_COLOR else str(s)

# Graceful shutdown on SIGINT/SIGTERM
_interrupted = False
def _signal_handler(signum, frame):
    global _interrupted
    _interrupted = True
    print("\nInterrupt received — finishing current step and saving...")
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

from prepare import (
    TIME_BUDGET, MASK_TOKEN_ID, PAD_TOKEN_ID, NUM_SPECIAL,
    load_config, load_data, make_dataloader, evaluate, generate_report,
)

# ---------------------------------------------------------------------------
# Optional YAML config override
# ---------------------------------------------------------------------------
# Usage: python train.py --config config.yaml
# Or set GENOMIC_CONFIG=config.yaml environment variable
# Config file values override the defaults below.

_CONFIG_OVERRIDES = {}
_config_path = os.environ.get("GENOMIC_CONFIG")
if not _config_path and len(__import__("sys").argv) > 2 and __import__("sys").argv[1] == "--config":
    _config_path = __import__("sys").argv[2]
if _config_path and os.path.exists(_config_path):
    try:
        import yaml
        with open(_config_path) as _f:
            _CONFIG_OVERRIDES = yaml.safe_load(_f) or {}
        print(f"Config loaded from {_config_path}")
    except ImportError:
        import json
        with open(_config_path) as _f:
            _CONFIG_OVERRIDES = json.load(_f)
        print(f"Config loaded from {_config_path} (JSON)")

_DRY_RUN = "--dry-run" in sys.argv
_PROFILE = "--profile" in sys.argv

def _cfg(key, default):
    """Get config value with override support."""
    return _CONFIG_OVERRIDES.get(key, default)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these — this is what the agent modifies)
# ---------------------------------------------------------------------------

MODEL_TYPE = _cfg("model_type", "transformer")     # transformer, mamba, cnn, lstm
D_MODEL = _cfg("d_model", 256)                     # model dimension
N_LAYERS = _cfg("n_layers", 6)                     # number of layers
N_HEADS = _cfg("n_heads", 8)                       # attention heads (transformer only)
D_FF = _cfg("d_ff", 1024)                          # feed-forward dimension
DROPOUT = _cfg("dropout", 0.1)                     # dropout rate
LEARNING_RATE = _cfg("learning_rate", 1e-4)        # learning rate
WEIGHT_DECAY = _cfg("weight_decay", 1e-4)          # L2 regularization
OPTIMIZER = _cfg("optimizer", "adamw")             # adamw, sgd, lamb
BATCH_SIZE = _cfg("batch_size", 32)                # training batch size
LR_SCHEDULE = _cfg("lr_schedule", "cosine")        # lr schedule: constant, cosine, step
WARMUP_RATIO = _cfg("warmup_ratio", 0.05)          # fraction of training for LR warmup
OBJECTIVE = _cfg("objective", "mlm")               # mlm or clm
MASK_RATIO = _cfg("mask_ratio", 0.15)              # fraction of tokens to mask (MLM only)
LABEL_SMOOTHING = _cfg("label_smoothing", 0.0)     # label smoothing for CE loss (0.0 = off)
LOSS_FN = _cfg("loss_fn", "cross_entropy")         # cross_entropy or focal
FOCAL_GAMMA = _cfg("focal_gamma", 2.0)             # focal loss gamma (only if loss_fn=focal)

# --- Architecture-specific ---
POS_ENCODING = _cfg("pos_encoding", "sinusoidal")  # sinusoidal, rotary, alibi, learned
KERNEL_SIZES = _cfg("kernel_sizes", [7, 7, 7, 7])  # CNN conv kernel sizes per layer
CNN_CHANNELS = _cfg("cnn_channels", 256)            # CNN intermediate channels
LSTM_BIDIRECTIONAL = _cfg("lstm_bidirectional", True)
RNN_TYPE = _cfg("rnn_type", "lstm")                # lstm or gru
CNN_DILATION = _cfg("cnn_dilation", False)          # use exponential dilation in CNN
STOCHASTIC_DEPTH = _cfg("stochastic_depth", 0.0)   # layer drop probability (0.0 = disabled)
USE_DEEPNORM = _cfg("use_deepnorm", False)         # DeepNorm for stable deep training
NORM_POSITION = _cfg("norm_position", "pre")       # "pre" (Pre-LN, default) or "post" (Post-LN)
USE_MOE = _cfg("use_moe", False)                   # replace FF layers with Mixture of Experts
N_EXPERTS = _cfg("n_experts", 4)                   # number of expert FF networks
MOE_TOP_K = _cfg("moe_top_k", 2)                  # top-k routing (how many experts per token)
ATTENTION_WINDOW = _cfg("attention_window", 0)       # sliding window size (0 = full attention)
N_GLOBAL_TOKENS = _cfg("n_global_tokens", 1)          # number of global tokens (first N positions)
MAMBA_D_STATE = _cfg("mamba_d_state", 16)
MAMBA_D_CONV = _cfg("mamba_d_conv", 4)
MAMBA_EXPAND = _cfg("mamba_expand", 2)

# --- Data augmentation ---
USE_RC_AUGMENT = False         # reverse complement augmentation (50% chance)
USE_SPAN_MASKING = False       # span masking instead of random token masking
SPAN_MEAN_LENGTH = 3           # mean span length for span masking
USE_SNP_NOISE = _cfg("use_snp_noise", False)   # random base substitution noise
SNP_RATE = _cfg("snp_rate", 0.01)              # fraction of tokens to substitute
USE_INDEL_NOISE = _cfg("use_indel_noise", False)  # random insertion/deletion noise
INDEL_RATE = _cfg("indel_rate", 0.005)            # fraction of tokens for indels
USE_RANDOM_CROP = _cfg("use_random_crop", False)  # random subsequence cropping
MIN_CROP_RATIO = _cfg("min_crop_ratio", 0.5)      # minimum crop length ratio
TOKEN_DROPOUT = _cfg("token_dropout", 0.0)         # randomly drop tokens (replace with PAD)
USE_LOCAL_SHUFFLE = _cfg("use_local_shuffle", False)  # shuffle within k-bp windows
SHUFFLE_WINDOW = _cfg("shuffle_window", 10)           # window size for local shuffling

# --- Training infrastructure ---
USE_AMP = True                 # automatic mixed precision (CUDA only)
GRAD_ACCUM_STEPS = 1           # gradient accumulation steps
USE_GRAD_CHECKPOINT = False    # gradient checkpointing (saves memory)
NUM_WORKERS = 0                # dataloader workers (0 = main process)
PIN_MEMORY = False             # pin memory for faster CUDA transfer
USE_DDP = False                # distributed data parallel (multi-GPU)
SEED = _cfg("seed", 42)       # random seed for reproducibility
USE_EMA = _cfg("use_ema", False)   # exponential moving average of model weights
EMA_DECAY = _cfg("ema_decay", 0.999)  # EMA decay factor
EARLY_STOP_PATIENCE = _cfg("early_stop_patience", 0)  # 0 = disabled; stop after N evals without improvement
LR_LAYER_DECAY = _cfg("lr_layer_decay", 1.0)      # per-layer LR multiplier (1.0 = same LR for all layers)
USE_SWA = _cfg("use_swa", False)   # stochastic weight averaging in final 25% of training
SWA_LR = _cfg("swa_lr", 1e-5)     # SWA learning rate
RESUME_FROM = _cfg("resume_from", "")  # path to checkpoint to resume training from
USE_COMPILE = _cfg("use_compile", False)  # torch.compile() for faster training (PyTorch 2.0+, CUDA)
GRAD_NOISE = _cfg("grad_noise", 0.0)     # Gaussian noise std added to gradients (0 = disabled)
USE_WHOLE_WORD_MASK = _cfg("use_whole_word_mask", False)  # mask whole k-mers for kmer tokenizer
USE_MIXUP = _cfg("use_mixup", False)                     # mixup augmentation for classify/regress
MIXUP_ALPHA = _cfg("mixup_alpha", 0.2)                   # Beta distribution alpha for mixup
USE_CUTMIX = _cfg("use_cutmix", False)                   # CutMix augmentation for classify/regress
CUTMIX_ALPHA = _cfg("cutmix_alpha", 1.0)                 # Beta distribution alpha for CutMix
USE_CURRICULUM = _cfg("use_curriculum", False)            # curriculum learning: start short, increase length
CURRICULUM_START_RATIO = _cfg("curriculum_start_ratio", 0.25)  # start at 25% of max_length
USE_RDROP = _cfg("use_rdrop", False)                     # R-Drop: KL divergence between two dropout passes
RDROP_ALPHA = _cfg("rdrop_alpha", 1.0)                   # weight for R-Drop KL loss
USE_SAM = _cfg("use_sam", False)                         # Sharpness-Aware Minimization
SAM_RHO = _cfg("sam_rho", 0.05)                          # SAM perturbation radius
USE_CONTRASTIVE = _cfg("use_contrastive", False)         # contrastive learning (SimCLR-style)
CONTRASTIVE_TEMP = _cfg("contrastive_temp", 0.07)        # temperature for InfoNCE loss
MULTI_LABEL = _cfg("multi_label", False)                 # multi-label classification (sigmoid per class)
DYNAMIC_BATCH = _cfg("dynamic_batch", False)             # increase batch size over training
USE_PROGRESSIVE_RESIZE = _cfg("use_progressive_resize", False)  # start with shorter seqs, grow over training
PROGRESSIVE_START_LEN = _cfg("progressive_start_len", 64)       # initial sequence length for progressive resize
USE_DISTILLATION = _cfg("use_distillation", False)              # knowledge distillation from teacher model
TEACHER_CHECKPOINT = _cfg("teacher_checkpoint", "")             # path to teacher checkpoint (.pt)
DISTILL_ALPHA = _cfg("distill_alpha", 0.5)                      # weight for distillation loss (vs task loss)
DISTILL_TEMP = _cfg("distill_temp", 2.0)                        # temperature for soft targets

# ---------------------------------------------------------------------------
# Data augmentation utilities
# ---------------------------------------------------------------------------

# Complement mapping for char tokenizer: A(5)↔T(6), C(7)↔G(8)
_COMPLEMENT_MAP = {5: 6, 6: 5, 7: 8, 8: 7}


def reverse_complement_tokens(tokens, mask):
    """Reverse complement token sequences. Operates on batches (B, L)."""
    rc = tokens.clone()
    for old_id, new_id in _COMPLEMENT_MAP.items():
        rc[tokens == old_id] = new_id
    # Reverse the actual sequence (non-pad portion)
    B, L = rc.shape
    for i in range(B):
        seq_len = mask[i].sum().item()
        if seq_len > 0:
            rc[i, :seq_len] = rc[i, :seq_len].flip(0)
    return rc


def span_mask_tokens(tokens, mask, mask_ratio, span_mean_length, mask_token_id, num_special):
    """Apply span masking: mask contiguous spans instead of individual tokens."""
    input_ids = tokens.clone()
    labels = tokens.clone()
    B, L = tokens.shape

    for i in range(B):
        seq_len = mask[i].sum().item()
        valid = (mask[i] == 1) & (tokens[i] >= num_special)
        n_valid = valid.sum().item()
        if n_valid == 0:
            labels[i] = PAD_TOKEN_ID
            continue

        n_to_mask = max(1, int(n_valid * mask_ratio))
        masked = torch.zeros(L, dtype=torch.bool, device=tokens.device)
        valid_indices = valid.nonzero(as_tuple=True)[0]
        n_masked = 0

        while n_masked < n_to_mask:
            # Pick random start from valid positions
            start_idx = valid_indices[torch.randint(len(valid_indices), (1,))].item()
            # Geometric distribution for span length
            span_len = min(
                int(torch.distributions.Geometric(1.0 / span_mean_length).sample().item()) + 1,
                n_to_mask - n_masked,
                L - start_idx,
            )
            for j in range(start_idx, min(start_idx + span_len, L)):
                if valid[j] and not masked[j]:
                    masked[j] = True
                    n_masked += 1
            if n_masked >= n_to_mask:
                break

        input_ids[i, masked] = mask_token_id
        labels[i, ~masked] = PAD_TOKEN_ID

    return input_ids, labels


def snp_noise(tokens, mask, rate, num_special):
    """Randomly substitute bases to simulate sequencing errors (SNP noise)."""
    out = tokens.clone()
    valid = (mask == 1) & (tokens >= num_special)
    noise_mask = valid & (torch.rand_like(tokens.float()) < rate)
    # Replace with random valid token (num_special to num_special+3 = A,T,C,G range)
    random_tokens = torch.randint(num_special, num_special + 4, tokens.shape,
                                  device=tokens.device, dtype=tokens.dtype)
    out[noise_mask] = random_tokens[noise_mask]
    return out


def indel_noise(tokens, mask, rate, pad_id, num_special):
    """Randomly insert or delete tokens to simulate indel errors."""
    B, L = tokens.shape
    out = tokens.clone()
    out_mask = mask.clone()
    for i in range(B):
        seq_len = mask[i].sum().item()
        if seq_len < 4:
            continue
        valid_positions = ((mask[i] == 1) & (tokens[i] >= num_special)).nonzero(as_tuple=True)[0]
        if len(valid_positions) == 0:
            continue
        n_indels = max(1, int(len(valid_positions) * rate))
        for _ in range(n_indels):
            pos = valid_positions[torch.randint(len(valid_positions), (1,))].item()
            if torch.rand(1).item() < 0.5:
                # Deletion: shift left
                if pos < L - 1:
                    out[i, pos:-1] = out[i, pos + 1:].clone()
                    out[i, -1] = pad_id
                    out_mask[i, pos:-1] = out_mask[i, pos + 1:].clone()
                    out_mask[i, -1] = 0
            else:
                # Insertion: shift right (insert random base)
                if pos < L - 1:
                    out[i, pos + 1:] = out[i, pos:-1].clone()
                    out_mask[i, pos + 1:] = out_mask[i, pos:-1].clone()
                    out[i, pos] = torch.randint(num_special, num_special + 4, (1,),
                                                device=tokens.device, dtype=tokens.dtype).item()
    return out, out_mask


def random_crop(tokens, mask, min_ratio, pad_id):
    """Randomly crop a contiguous subsequence from each sequence."""
    B, L = tokens.shape
    out = tokens.clone()
    out_mask = mask.clone()
    for i in range(B):
        seq_len = int(mask[i].sum().item())
        if seq_len < 4:
            continue
        crop_len = max(2, int(seq_len * (min_ratio + (1 - min_ratio) * torch.rand(1).item())))
        start = torch.randint(0, max(1, seq_len - crop_len + 1), (1,)).item()
        # Copy cropped region to beginning, pad rest
        out[i, :crop_len] = tokens[i, start:start + crop_len]
        out[i, crop_len:] = pad_id
        out_mask[i, :crop_len] = mask[i, start:start + crop_len]
        out_mask[i, crop_len:] = 0
    return out, out_mask


def token_dropout_aug(tokens, mask, drop_rate, pad_id, num_special):
    """Randomly drop tokens by replacing with PAD (no prediction target)."""
    out = tokens.clone()
    out_mask = mask.clone()
    valid = (mask == 1) & (tokens >= num_special)
    drop_mask = valid & (torch.rand_like(tokens.float()) < drop_rate)
    out[drop_mask] = pad_id
    out_mask[drop_mask] = 0
    return out, out_mask


def local_shuffle(tokens, mask, window_size, num_special):
    """Shuffle tokens within local windows to preserve composition but disrupt order."""
    out = tokens.clone()
    B, L = tokens.shape
    for i in range(B):
        seq_len = int(mask[i].sum().item())
        if seq_len < window_size:
            continue
        for start in range(0, seq_len - window_size + 1, window_size):
            end = min(start + window_size, seq_len)
            # Only shuffle non-special tokens within the window
            window = out[i, start:end].clone()
            valid = window >= num_special
            if valid.sum() < 2:
                continue
            valid_idx = valid.nonzero(as_tuple=True)[0]
            perm = valid_idx[torch.randperm(len(valid_idx))]
            window[valid_idx] = window[perm]
            out[i, start:end] = window
    return out


def whole_word_mask_tokens(tokens, mask, mask_ratio, mask_token_id, num_special, kmer_size=6):
    """Mask whole k-mers instead of individual tokens for kmer tokenizer."""
    input_ids = tokens.clone()
    labels = tokens.clone()
    B, L = tokens.shape

    for i in range(B):
        valid = (mask[i] == 1) & (tokens[i] >= num_special)
        valid_positions = valid.nonzero(as_tuple=True)[0]
        n_valid = len(valid_positions)
        if n_valid == 0:
            labels[i] = PAD_TOKEN_ID
            continue

        # Number of k-mer groups to mask
        n_groups = max(1, n_valid // kmer_size)
        n_to_mask = max(1, int(n_groups * mask_ratio))

        # Pick random group starts (every kmer_size positions)
        group_starts = list(range(0, n_valid, kmer_size))
        if not group_starts:
            labels[i] = PAD_TOKEN_ID
            continue

        chosen = torch.randperm(len(group_starts))[:n_to_mask]
        masked = torch.zeros(L, dtype=torch.bool, device=tokens.device)

        for g in chosen:
            start = group_starts[g.item()]
            for k in range(start, min(start + kmer_size, n_valid)):
                pos = valid_positions[k].item()
                masked[pos] = True

        input_ids[i, masked] = mask_token_id
        labels[i, ~masked] = PAD_TOKEN_ID

    return input_ids, labels


def mixup_data(x, y, alpha=0.2):
    """Mixup: interpolate between two samples' embeddings and labels.

    Returns mixed inputs, pair of labels (y_a, y_b), and lambda factor.
    Applied after embedding, before model forward.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    perm = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[perm]
    return mixed_x, y, y[perm], lam


def cutmix_tokens(tokens, mask, labels, alpha=1.0):
    """CutMix: replace a random contiguous span with another sample's tokens.

    Returns mixed tokens, mixed masks, pair of labels, and lambda factor.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = tokens.size(0)
    seq_len = tokens.size(1)
    perm = torch.randperm(batch_size, device=tokens.device)

    # Determine cut region
    cut_len = int(seq_len * (1 - lam))
    if cut_len == 0:
        return tokens, mask, labels, labels, 1.0
    start = np.random.randint(0, max(1, seq_len - cut_len))
    end = start + cut_len

    mixed_tokens = tokens.clone()
    mixed_mask = mask.clone()
    mixed_tokens[:, start:end] = tokens[perm, start:end]
    mixed_mask[:, start:end] = mask[perm, start:end]

    # Adjust lambda to actual proportion
    actual_lam = 1 - cut_len / seq_len
    return mixed_tokens, mixed_mask, labels, labels[perm], actual_lam


def contrastive_loss(z1, z2, temperature=0.07):
    """InfoNCE / SimCLR contrastive loss.

    z1, z2: (B, D) embeddings of positive pairs.
    """
    z1 = nn.functional.normalize(z1, dim=-1)
    z2 = nn.functional.normalize(z2, dim=-1)
    B = z1.size(0)
    # Cosine similarity matrix
    logits = torch.mm(z1, z2.t()) / temperature  # (B, B)
    labels = torch.arange(B, device=z1.device)
    loss = (nn.functional.cross_entropy(logits, labels)
            + nn.functional.cross_entropy(logits.t(), labels)) / 2
    return loss


def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """KL divergence loss for knowledge distillation with temperature scaling."""
    T = temperature
    s_log_probs = nn.functional.log_softmax(student_logits / T, dim=-1)
    t_probs = nn.functional.softmax(teacher_logits / T, dim=-1)
    return nn.functional.kl_div(s_log_probs, t_probs, reduction="batchmean") * (T * T)


def per_position_accuracy(logits, labels, pad_id=0, max_positions=512):
    """Compute per-position token prediction accuracy.

    Returns dict mapping position index → accuracy at that position.
    Only counts positions where labels != pad_id.
    """
    # logits: (B, L, V), labels: (B, L)
    preds = logits.argmax(dim=-1)  # (B, L)
    correct = (preds == labels)    # (B, L)
    valid = (labels != pad_id)     # (B, L)

    L = min(logits.size(1), max_positions)
    pos_acc = {}
    for pos in range(L):
        v = valid[:, pos]
        n = v.sum().item()
        if n > 0:
            pos_acc[pos] = (correct[:, pos] & v).sum().item() / n
    return pos_acc


def denoise_corrupt(tokens, mask, num_special, delete_rate=0.1, shuffle_rate=0.1, insert_rate=0.05):
    """Corrupt sequences for denoising autoencoder: random deletion, shuffling, insertion.

    Returns corrupted input and original tokens as targets.
    """
    B, L = tokens.shape
    corrupted = tokens.clone()

    for i in range(B):
        seq_len = mask[i].sum().item()
        if seq_len < 4:
            continue
        valid = torch.arange(seq_len, device=tokens.device)

        # 1. Random deletion: remove tokens and shift left
        n_delete = max(0, int(seq_len * delete_rate))
        if n_delete > 0:
            del_idx = valid[torch.randperm(seq_len, device=tokens.device)[:n_delete]]
            keep_mask = torch.ones(L, dtype=torch.bool, device=tokens.device)
            keep_mask[del_idx] = False
            kept = corrupted[i][keep_mask]
            corrupted[i] = torch.nn.functional.pad(kept, (0, L - kept.size(0)), value=PAD_TOKEN_ID)

        # 2. Local shuffling within 3-token windows
        new_seq_len = mask[i].sum().item() - n_delete
        n_shuffle = max(0, int(new_seq_len * shuffle_rate))
        if n_shuffle > 0 and new_seq_len > 2:
            for _ in range(n_shuffle):
                j = torch.randint(0, max(1, new_seq_len - 1), (1,)).item()
                if j + 1 < L and corrupted[i, j] >= num_special and corrupted[i, j + 1] >= num_special:
                    corrupted[i, j], corrupted[i, j + 1] = corrupted[i, j + 1].clone(), corrupted[i, j].clone()

        # 3. Random insertion: insert random base tokens
        n_insert = max(0, int(seq_len * insert_rate))
        if n_insert > 0 and new_seq_len + n_insert <= L:
            for _ in range(n_insert):
                pos = torch.randint(0, max(1, new_seq_len), (1,)).item()
                rand_token = torch.randint(num_special, num_special + 4, (1,), device=tokens.device).item()
                # Shift right and insert
                corrupted[i, pos + 1:] = corrupted[i, pos:-1].clone()
                corrupted[i, pos] = rand_token

    return corrupted


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


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim, max_len=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._max_len = max_len

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (L, dim//2)
        return torch.cat([freqs, freqs], dim=-1)  # (L, dim)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q, k, freqs):
    """Apply rotary embeddings to q and k."""
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, L, dim)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


class ALiBiPositionBias(nn.Module):
    """Attention with Linear Biases (ALiBi)."""

    def __init__(self, n_heads, max_len=4096):
        super().__init__()
        # Compute head-specific slopes
        slopes = torch.tensor([2 ** (-8 * i / n_heads) for i in range(1, n_heads + 1)])
        self.register_buffer("slopes", slopes)
        self._max_len = max_len

    def forward(self, seq_len, device):
        # (n_heads, L, L) bias matrix
        positions = torch.arange(seq_len, device=device)
        relative = positions.unsqueeze(0) - positions.unsqueeze(1)  # (L, L)
        bias = self.slopes.unsqueeze(-1).unsqueeze(-1) * relative.abs().unsqueeze(0).float()
        return -bias  # negative because closer = higher attention


def _sliding_window_mask(seq_len, window_size, n_global=1, device="cpu"):
    """Create a sliding window attention mask with global tokens.

    Returns a float mask (L, L) where -inf blocks attention and 0.0 allows it.
    First n_global positions can attend to/from all positions (Longformer-style).
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    for i in range(seq_len):
        lo = max(0, i - window_size // 2)
        hi = min(seq_len, i + window_size // 2 + 1)
        mask[i, lo:hi] = 0.0
    # Global tokens: first n_global positions attend to all and are attended by all
    if n_global > 0:
        mask[:n_global, :] = 0.0
        mask[:, :n_global] = 0.0
    return mask


class T5RelativePositionBias(nn.Module):
    """T5-style relative position bias with logarithmic bucketing."""

    def __init__(self, n_heads, max_distance=128, n_buckets=32):
        super().__init__()
        self.n_heads = n_heads
        self.n_buckets = n_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(n_buckets, n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, n_buckets=32, max_distance=128):
        # Bidirectional: half buckets for negative, half for positive
        n_buckets //= 2
        rel_buckets = (relative_position > 0).long() * n_buckets
        rel_pos = relative_position.abs()
        # First half: exact positions
        max_exact = n_buckets // 2
        is_small = rel_pos < max_exact
        # Second half: logarithmic bucketing
        rel_pos_if_large = max_exact + (
            torch.log(rel_pos.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (n_buckets - max_exact)
        ).long()
        rel_pos_if_large = torch.clamp(rel_pos_if_large, max=n_buckets - 1)
        rel_buckets += torch.where(is_small, rel_pos, rel_pos_if_large)
        return rel_buckets

    def forward(self, seq_len, device):
        positions = torch.arange(seq_len, device=device)
        relative_position = positions.unsqueeze(0) - positions.unsqueeze(1)  # (L, L)
        buckets = self._relative_position_bucket(
            relative_position, self.n_buckets, self.max_distance
        )
        values = self.relative_attention_bias(buckets)  # (L, L, n_heads)
        return values.permute(2, 0, 1)  # (n_heads, L, L)


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance (Lin et al., 2017)."""

    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce = nn.functional.cross_entropy(
            inputs, targets, weight=self.weight,
            label_smoothing=self.label_smoothing, reduction="none",
        )
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean()


class LAMB(torch.optim.Optimizer):
    """LAMB optimizer for large-batch training (You et al., 2019)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                b1, b2 = group["betas"]
                state["m"].mul_(b1).add_(grad, alpha=1 - b1)
                state["v"].mul_(b2).addcmul_(grad, grad, value=1 - b2)
                m_hat = state["m"] / (1 - b1 ** state["step"])
                v_hat = state["v"] / (1 - b2 ** state["step"])
                update = m_hat / (v_hat.sqrt() + group["eps"])
                if group["weight_decay"] > 0:
                    update.add_(p, alpha=group["weight_decay"])
                # LAMB trust ratio
                p_norm = p.norm(2)
                u_norm = update.norm(2)
                trust = p_norm / u_norm if p_norm > 0 and u_norm > 0 else 1.0
                p.add_(update, alpha=-group["lr"] * trust)


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al., 2021).

    Wraps a base optimizer. Each step: (1) perturb weights by rho * grad_norm,
    (2) compute gradient at perturbed point, (3) step with base optimizer.
    """

    def __init__(self, params, base_optimizer_cls, rho=0.05, **kwargs):
        defaults = dict(rho=rho)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self):
        """Ascent step: perturb weights toward sharpest direction."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self):
        """Descent step: revert perturbation and update with base optimizer."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def step(self, closure=None):
        """Standard step (used when SAM is not active)."""
        self.base_optimizer.step(closure)

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none)


class DropPath(nn.Module):
    """Stochastic depth — randomly drop entire residual branches during training."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device, dtype=x.dtype))
        return x * mask / keep_prob


class MoEFeedForward(nn.Module):
    """Mixture of Experts feed-forward layer with top-k routing."""

    def __init__(self, d_model, d_ff, n_experts, top_k, dropout):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = min(top_k, n_experts)
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(n_experts)
        ])
        self.load_balance_loss = 0.0  # Set during forward pass

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        x_flat = x.reshape(-1, D)  # (B*L, D)

        # Gating
        gate_logits = self.gate(x_flat)  # (B*L, n_experts)
        gate_probs = torch.softmax(gate_logits, dim=-1)

        # Top-k routing
        top_k_vals, top_k_idx = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_vals = top_k_vals / top_k_vals.sum(dim=-1, keepdim=True)  # Renormalize

        # Expert load balancing loss (auxiliary)
        # Fraction of tokens routed to each expert * importance of each expert
        tokens_per_expert = torch.zeros(self.n_experts, device=x.device)
        for i in range(self.n_experts):
            tokens_per_expert[i] = (top_k_idx == i).float().sum()
        tokens_per_expert = tokens_per_expert / (B * L)
        importance = gate_probs.mean(dim=0)
        self.load_balance_loss = float(self.n_experts * (tokens_per_expert * importance).sum())

        # Compute expert outputs
        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = top_k_idx[:, k]  # (B*L,)
            weight = top_k_vals[:, k].unsqueeze(-1)  # (B*L, 1)
            for i in range(self.n_experts):
                mask = expert_idx == i
                if mask.any():
                    out[mask] += weight[mask] * self.experts[i](x_flat[mask])

        return out.reshape(B, L, D)


class GenomicTransformerLayer(nn.Module):
    """Custom Transformer layer supporting RoPE/ALiBi."""

    def __init__(self, d_model, n_heads, d_ff, dropout, pos_type="sinusoidal",
                 drop_path=0.0, deepnorm_alpha=1.0, norm_position="pre",
                 use_moe=False, n_experts=4, moe_top_k=2):
        super().__init__()
        self.pos_type = pos_type
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.deepnorm_alpha = deepnorm_alpha  # DeepNorm residual scaling
        self.norm_position = norm_position

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        if use_moe and n_experts > 1:
            self.ff = MoEFeedForward(d_model, d_ff, n_experts, moe_top_k, dropout)
            self.use_moe = True
        else:
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
            self.use_moe = False
        self.attn_dropout = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, attn_bias=None, rotary_freqs=None, key_padding_mask=None, causal_mask=None):
        B, L, D = x.shape
        # Pre-norm or post-norm
        h = self.norm1(x) if self.norm_position == "pre" else x
        q = self.q_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if rotary_freqs is not None:
            q, k = apply_rotary_emb(q, k, rotary_freqs)

        # Use PyTorch SDPA (Flash Attention / Memory-Efficient kernels) when possible
        _use_sdpa = (attn_bias is None) and hasattr(torch.nn.functional, "scaled_dot_product_attention")

        if _use_sdpa:
            # Build combined attention mask for SDPA
            attn_mask = None
            if causal_mask is not None and key_padding_mask is not None:
                # Combine: causal (L,L) + padding (B,L) → (B,1,L,L)
                pad_mask_2d = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,L)
                pad_mask_2d = torch.zeros_like(pad_mask_2d, dtype=q.dtype).masked_fill_(pad_mask_2d, float("-inf"))
                attn_mask = causal_mask.unsqueeze(0).unsqueeze(0) + pad_mask_2d
            elif causal_mask is not None:
                attn_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, self.n_heads, L, L)
            elif key_padding_mask is not None:
                pad_mask_2d = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,L)
                attn_mask = torch.zeros(B, 1, 1, L, dtype=q.dtype, device=q.device).masked_fill_(pad_mask_2d, float("-inf"))

            dropout_p = self.attn_dropout.p if self.training else 0.0
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
            )
        else:
            # Fallback: manual attention (needed for ALiBi bias)
            scale = self.head_dim ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale

            if attn_bias is not None:
                attn = attn + attn_bias
            if causal_mask is not None:
                attn = attn + causal_mask.unsqueeze(0).unsqueeze(0)
            if key_padding_mask is not None:
                attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

            attn = torch.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.out_proj(out)
        # Residual connection (DeepNorm: scale residual by alpha)
        x = x * self.deepnorm_alpha + self.drop_path(self.ff_dropout(out))
        if self.norm_position == "post":
            x = self.norm1(x)

        # Feed-forward
        ff_in = self.norm2(x) if self.norm_position == "pre" else x
        x = x * self.deepnorm_alpha + self.drop_path(self.ff(ff_in))
        if self.norm_position == "post":
            x = self.norm2(x)
        return x


class GenomicTransformer(nn.Module):
    """Transformer Encoder for genomic sequence modeling."""

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len, dropout,
                 task_type, n_classes=None, pos_encoding="sinusoidal",
                 stochastic_depth=0.0, use_deepnorm=False, norm_position="pre",
                 use_moe=False, n_experts=4, moe_top_k=2,
                 attention_window=0, n_global_tokens=1):
        super().__init__()
        self.task_type = task_type
        self.d_model = d_model
        self.pos_type = pos_encoding
        self.attention_window = attention_window
        self.n_global_tokens = n_global_tokens

        # DeepNorm: α = (2N)^(1/4), β = (8N)^(-1/4) where N = n_layers
        self._deepnorm_alpha = (2 * n_layers) ** 0.25 if use_deepnorm else 1.0

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN_ID)

        # Position encoding setup
        if pos_encoding == "sinusoidal":
            self.pos_encoding = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
            self.use_custom_layers = False
        elif pos_encoding == "learned":
            self.pos_encoding = nn.Embedding(max_len, d_model)
            self.pos_drop = nn.Dropout(dropout)
            self.use_custom_layers = False
        elif pos_encoding in ("rotary", "alibi", "relative"):
            self.use_custom_layers = True
            self.emb_dropout = nn.Dropout(dropout)
            if pos_encoding == "rotary":
                head_dim = d_model // n_heads
                self.rotary = RotaryEmbedding(head_dim, max_len=max_len)
            elif pos_encoding == "alibi":
                self.alibi = ALiBiPositionBias(n_heads, max_len=max_len)
            else:  # relative (T5-style)
                self.t5_bias = T5RelativePositionBias(n_heads, max_distance=max_len)
        else:
            self.pos_encoding = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
            self.use_custom_layers = False

        # Encoder layers (linearly increasing drop path rate per layer)
        _dp_rates = [stochastic_depth * i / max(n_layers - 1, 1) for i in range(n_layers)]
        if self.use_custom_layers:
            self.layers = nn.ModuleList([
                GenomicTransformerLayer(d_model, n_heads, d_ff, dropout, pos_encoding,
                                       drop_path=_dp_rates[i],
                                       deepnorm_alpha=self._deepnorm_alpha,
                                       norm_position=norm_position,
                                       use_moe=use_moe, n_experts=n_experts,
                                       moe_top_k=moe_top_k)
                for i in range(n_layers)
            ])
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=dropout, activation="gelu", batch_first=True,
                norm_first=(norm_position == "pre"),
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=n_layers, enable_nested_tensor=False,
            )

        self.ln = nn.LayerNorm(d_model)

        # Task-specific heads
        if task_type == "pretrain":
            self.head = nn.Linear(d_model, vocab_size)
        elif task_type == "classify":
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, n_classes),
            )
        elif task_type == "regress":
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)

        if self.use_custom_layers:
            x = self.emb_dropout(x)
            L = input_ids.size(1)
            device = input_ids.device

            rotary_freqs = self.rotary(L, device) if hasattr(self, 'rotary') else None
            attn_bias = None
            if hasattr(self, 'alibi'):
                attn_bias = self.alibi(L, device)
            elif hasattr(self, 't5_bias'):
                attn_bias = self.t5_bias(L, device)

            key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
            causal_mask = None
            if self.task_type == "pretrain" and hasattr(self, '_use_causal') and self._use_causal:
                causal_mask = torch.triu(torch.full((L, L), float("-inf"), device=device), diagonal=1)

            # Sliding window attention mask (Longformer-style)
            if self.attention_window > 0 and self.attention_window < L:
                sw_mask = _sliding_window_mask(L, self.attention_window, self.n_global_tokens, device)
                if attn_bias is not None:
                    attn_bias = attn_bias + sw_mask.unsqueeze(0)  # broadcast over heads
                else:
                    attn_bias = sw_mask.unsqueeze(0).expand(self.layers[0].n_heads, -1, -1)

            for layer in self.layers:
                x = layer(x, attn_bias=attn_bias, rotary_freqs=rotary_freqs,
                         key_padding_mask=key_padding_mask, causal_mask=causal_mask)
        else:
            if self.pos_type == "learned":
                positions = torch.arange(input_ids.size(1), device=input_ids.device)
                x = x + self.pos_encoding(positions)
                x = self.pos_drop(x)
            else:
                x = self.pos_encoding(x)

            src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

            if self.task_type == "pretrain" and hasattr(self, '_use_causal') and self._use_causal:
                sz = input_ids.size(1)
                causal_mask = torch.triu(
                    torch.full((sz, sz), float("-inf"), device=input_ids.device), diagonal=1
                )
                # Convert padding mask to float to match causal mask type
                pad_mask = src_key_padding_mask
                if pad_mask is not None and pad_mask.dtype == torch.bool:
                    pad_mask = torch.zeros_like(pad_mask, dtype=torch.float).masked_fill_(pad_mask, float("-inf"))
                x = self.encoder(x, mask=causal_mask, src_key_padding_mask=pad_mask)
            else:
                x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        x = self.ln(x)

        if self.task_type == "pretrain":
            return self.head(x)
        else:
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)
            return self.head(x)


# ---------------------------------------------------------------------------
# CNN Model
# ---------------------------------------------------------------------------

class GenomicCNN(nn.Module):
    """1D Convolutional model with residual blocks for genomic sequences."""

    def __init__(self, vocab_size, d_model, n_layers, d_ff, max_len, dropout,
                 task_type, n_classes=None, kernel_sizes=None, channels=None,
                 use_dilation=False):
        super().__init__()
        self.task_type = task_type
        ks = kernel_sizes or [7] * n_layers
        ch = channels or d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN_ID)
        self.emb_dropout = nn.Dropout(dropout)

        # Project embedding to conv channels if needed
        self.input_proj = nn.Conv1d(d_model, ch, 1) if d_model != ch else nn.Identity()

        # Residual conv blocks: Conv + BatchNorm + GELU + Dropout
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            k = ks[i] if i < len(ks) else ks[-1]
            d = (2 ** i) if use_dilation else 1  # exponential dilation: 1, 2, 4, 8, ...
            pad = (k // 2) * d  # dilated padding to preserve length
            self.blocks.append(nn.Sequential(
                nn.Conv1d(ch, ch, k, padding=pad, dilation=d),
                nn.BatchNorm1d(ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ))

        self.proj = nn.Linear(ch, d_model) if ch != d_model else nn.Identity()
        self.ln = nn.LayerNorm(d_model)

        if task_type == "pretrain":
            self.head = nn.Linear(d_model, vocab_size)
        elif task_type == "classify":
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, n_classes),
            )
        elif task_type == "regress":
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)  # (B, L, D)
        x = self.emb_dropout(x)
        x = x.transpose(1, 2)  # (B, D, L) for conv1d
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x) + x  # residual

        x = x.transpose(1, 2)  # (B, L, ch)
        x = self.proj(x)
        x = self.ln(x)

        if self.task_type == "pretrain":
            return self.head(x)
        else:
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)
            return self.head(x)


# ---------------------------------------------------------------------------
# Perceiver-style architecture
# ---------------------------------------------------------------------------

class GenomicPerceiver(nn.Module):
    """Perceiver-style model: cross-attention from learned latents to input.

    Sub-quadratic complexity: O(n*m) where n=seq_len, m=n_latents (m << n).
    Good for very long sequences without quadratic attention cost.
    """

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len, dropout,
                 task_type, n_classes=None, n_latents=64):
        super().__init__()
        self.task_type = task_type
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN_ID)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        # Learned latent array (much smaller than sequence length)
        self.latents = nn.Parameter(torch.randn(1, n_latents, d_model) * 0.02)

        # Cross-attention layers: latents attend to input
        self.cross_attn_layers = nn.ModuleList()
        self.cross_norms_q = nn.ModuleList()
        self.cross_norms_kv = nn.ModuleList()
        # Self-attention layers: latents attend to each other
        self.self_attn_layers = nn.ModuleList()

        n_cross = max(1, n_layers // 3)
        n_self = n_layers - n_cross

        for _ in range(n_cross):
            self.cross_norms_q.append(nn.LayerNorm(d_model))
            self.cross_norms_kv.append(nn.LayerNorm(d_model))
            self.cross_attn_layers.append(
                nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            )

        for _ in range(n_self):
            self.self_attn_layers.append(nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            ))

        self.ln = nn.LayerNorm(d_model)

        if task_type == "pretrain":
            # Need to project back to sequence length for per-token prediction
            self.cross_back = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.cross_back_norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size)
        elif task_type == "classify":
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, n_classes),
            )
        elif task_type == "regress":
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

    def forward(self, input_ids, attention_mask=None):
        B = input_ids.size(0)
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)

        # Expand latents for batch
        latents = self.latents.expand(B, -1, -1)  # (B, n_latents, D)

        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

        # Cross-attention: latents query input
        for norm_q, norm_kv, cross_attn in zip(self.cross_norms_q, self.cross_norms_kv, self.cross_attn_layers):
            q = norm_q(latents)
            kv = norm_kv(x)
            out, _ = cross_attn(q, kv, kv, key_padding_mask=key_padding_mask)
            latents = latents + out

        # Self-attention on latents
        for sa_layer in self.self_attn_layers:
            latents = sa_layer(latents)

        latents = self.ln(latents)

        if self.task_type == "pretrain":
            # Cross-attend back: input queries latents to get per-token output
            x_norm = self.cross_back_norm(x)
            out, _ = self.cross_back(x_norm, latents, latents)
            x = x + out
            return self.head(x)
        else:
            # Mean pool latents for classification/regression
            x = latents.mean(dim=1)
            return self.head(x)


# ---------------------------------------------------------------------------
# Convolutional Transformer (hybrid)
# ---------------------------------------------------------------------------

class GenomicConvTransformer(nn.Module):
    """Hybrid: Conv1d layers for local features, then Transformer for global context."""

    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_len, dropout,
                 task_type, n_classes=None, n_conv_layers=2, kernel_size=7):
        super().__init__()
        self.task_type = task_type
        self.d_model = d_model
        n_conv_layers = min(n_conv_layers, n_layers - 1)  # At least 1 Transformer layer

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN_ID)
        self.emb_dropout = nn.Dropout(dropout)

        # Conv layers for local feature extraction
        self.conv_layers = nn.ModuleList()
        for i in range(n_conv_layers):
            pad = kernel_size // 2
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size, padding=pad),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            ))

        # Transformer layers for global context
        n_transformer = n_layers - n_conv_layers
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer, enable_nested_tensor=False,
        )
        self.ln = nn.LayerNorm(d_model)

        if task_type == "pretrain":
            self.head = nn.Linear(d_model, vocab_size)
        elif task_type == "classify":
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, n_classes),
            )
        elif task_type == "regress":
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)  # (B, L, D)
        x = self.emb_dropout(x)

        # Conv layers (B, L, D) → (B, D, L) → conv → (B, L, D)
        h = x.transpose(1, 2)
        for conv in self.conv_layers:
            h = conv(h) + h  # residual
        x = h.transpose(1, 2)

        # Positional encoding + Transformer
        x = self.pos_encoding(x)
        src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.ln(x)

        if self.task_type == "pretrain":
            return self.head(x)
        else:
            if attention_mask is not None:
                mask_exp = attention_mask.unsqueeze(-1).float()
                x = (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)
            return self.head(x)


# ---------------------------------------------------------------------------
# LSTM Model
# ---------------------------------------------------------------------------

class GenomicLSTM(nn.Module):
    """Bidirectional LSTM/GRU for genomic sequences."""

    def __init__(self, vocab_size, d_model, n_layers, d_ff, max_len, dropout,
                 task_type, n_classes=None, bidirectional=True, rnn_type="lstm"):
        super().__init__()
        self.task_type = task_type
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN_ID)
        self.emb_dropout = nn.Dropout(dropout)

        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=d_model,
            hidden_size=d_model // (2 if bidirectional else 1),
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.ln = nn.LayerNorm(d_model)

        if task_type == "pretrain":
            self.head = nn.Linear(d_model, vocab_size)
        elif task_type == "classify":
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, n_classes),
            )
        elif task_type == "regress":
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.emb_dropout(x)

        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            out, _ = self.rnn(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=input_ids.size(1))
        else:
            x, _ = self.rnn(x)

        x = self.ln(x)

        if self.task_type == "pretrain":
            return self.head(x)
        else:
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)
            return self.head(x)


# ---------------------------------------------------------------------------
# Hyena architecture (long convolution + gating)
# ---------------------------------------------------------------------------

class HyenaBlock(nn.Module):
    """Hyena block: long convolution with implicit parameterization + gating.

    Inspired by HyenaDNA (Nguyen et al., 2023). Sub-quadratic complexity.
    """

    def __init__(self, d_model, order=2, filter_size=64, dropout=0.0):
        super().__init__()
        self.order = order
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)

        # Input projections: (order+1) linear projections for gates and value
        self.in_proj = nn.Linear(d_model, d_model * (order + 1))

        # Implicit long convolution filters (learnable, one per order)
        self.filters = nn.ParameterList([
            nn.Parameter(torch.randn(1, d_model, filter_size) * 0.02)
            for _ in range(order)
        ])
        self.filter_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(order)])

        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _long_conv(self, x, filt):
        """Apply long convolution via FFT for efficiency."""
        B, D, L = x.shape
        filt_len = filt.size(-1)

        # Pad filter to match sequence length
        if filt_len < L:
            filt = nn.functional.pad(filt, (0, L - filt_len))
        elif filt_len > L:
            filt = filt[:, :, :L]

        # FFT-based convolution: O(n log n)
        x_f = torch.fft.rfft(x, n=2 * L, dim=-1)
        f_f = torch.fft.rfft(filt.expand(B, -1, -1), n=2 * L, dim=-1)
        out = torch.fft.irfft(x_f * f_f, n=2 * L, dim=-1)[:, :, :L]
        return out

    def forward(self, x):
        # x: (B, L, D)
        h = self.norm(x)
        # Project to (order+1) * d_model
        z = self.in_proj(h)  # (B, L, D*(order+1))
        chunks = z.chunk(self.order + 1, dim=-1)  # list of (B, L, D)
        v = chunks[0]  # value
        gates = chunks[1:]  # order gate projections

        # Iterative gated long convolution
        y = v.transpose(1, 2)  # (B, D, L) for conv
        for i in range(self.order):
            # Long convolution
            y = self._long_conv(y, self.filters[i])
            y = self.filter_norms[i](y.transpose(1, 2)).transpose(1, 2)
            # Element-wise gating
            y = y * gates[i].transpose(1, 2)

        out = y.transpose(1, 2)  # (B, L, D)
        out = self.out_proj(out)
        return x + self.drop(out)


class GenomicHyena(nn.Module):
    """Hyena model for genomic sequences. Sub-quadratic via long convolutions."""

    def __init__(self, vocab_size, d_model, n_layers, d_ff, max_len, dropout,
                 task_type, n_classes=None, hyena_order=2, filter_size=64):
        super().__init__()
        self.task_type = task_type
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN_ID)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(HyenaBlock(d_model, order=hyena_order,
                                          filter_size=min(filter_size, max_len), dropout=dropout))
            # Add a feed-forward block
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_ff, d_model), nn.Dropout(dropout),
            ))

        self.ln = nn.LayerNorm(d_model)

        if task_type == "pretrain":
            self.head = nn.Linear(d_model, vocab_size)
        elif task_type == "classify":
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, n_classes),
            )
        elif task_type == "regress":
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.emb_dropout(x)

        for block in self.blocks:
            if isinstance(block, HyenaBlock):
                x = block(x)
            else:
                x = x + block(x)  # FF residual
        x = self.ln(x)

        if self.task_type == "pretrain":
            return self.head(x)
        else:
            if attention_mask is not None:
                m = attention_mask.unsqueeze(-1).float()
                x = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)
            return self.head(x)


# ---------------------------------------------------------------------------
# RWKV-style architecture
# ---------------------------------------------------------------------------

class RWKVBlock(nn.Module):
    """RWKV block: linear attention with time-decay (pure PyTorch)."""

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Time-mixing (linear attention with decay)
        self.time_decay = nn.Parameter(torch.randn(d_model) * 0.01)
        self.time_first = nn.Parameter(torch.randn(d_model) * 0.01)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.receptance = nn.Linear(d_model, d_model, bias=False)
        self.output = nn.Linear(d_model, d_model, bias=False)

        # Channel-mixing (feed-forward)
        self.ff_key = nn.Linear(d_model, d_ff, bias=False)
        self.ff_value = nn.Linear(d_ff, d_model, bias=False)
        self.ff_receptance = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, L, D = x.shape

        # Time-mixing
        h = self.ln1(x)
        # Token shift: mix current with previous token
        shifted = torch.cat([torch.zeros(B, 1, D, device=x.device), h[:, :-1]], dim=1)
        k = self.key(shifted)
        v = self.value(shifted)
        r = torch.sigmoid(self.receptance(h))

        # Linear attention with exponential decay (WKV computation)
        w = -torch.exp(self.time_decay)  # negative = decay
        u = self.time_first
        # Approximate WKV via cumulative sum (efficient O(n) implementation)
        wkv = torch.zeros(B, D, device=x.device)
        wkv_sum = torch.zeros(B, D, device=x.device)
        outputs = []
        for t in range(L):
            kt = k[:, t]
            vt = v[:, t]
            # Compute attention: max(e^{w*prev + k_t}, e^{u + k_t}) for stability
            ew = torch.exp(w + wkv)
            eu = torch.exp(u + kt)
            a = ew * wkv_sum + eu * vt
            b = ew + eu
            outputs.append((a / b.clamp(min=1e-8)).unsqueeze(1))
            # Update state
            wkv = w + wkv  # decay
            wkv_sum = torch.exp(w) * wkv_sum + torch.exp(kt) * vt
            wkv = torch.log(torch.exp(wkv) + torch.exp(kt)).clamp(max=30)  # log-sum-exp

        out = torch.cat(outputs, dim=1)
        x = x + self.drop(self.output(r * out))

        # Channel-mixing
        h = self.ln2(x)
        shifted = torch.cat([torch.zeros(B, 1, D, device=x.device), h[:, :-1]], dim=1)
        r_ff = torch.sigmoid(self.ff_receptance(h))
        k_ff = torch.relu(self.ff_key(shifted)) ** 2  # squared ReLU
        x = x + self.drop(r_ff * self.ff_value(k_ff))
        return x


class GenomicRWKV(nn.Module):
    """RWKV-style model with O(n) linear attention for genomic sequences."""

    def __init__(self, vocab_size, d_model, n_layers, d_ff, max_len, dropout,
                 task_type, n_classes=None):
        super().__init__()
        self.task_type = task_type
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN_ID)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            RWKVBlock(d_model, d_ff, dropout) for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)

        if task_type == "pretrain":
            self.head = nn.Linear(d_model, vocab_size)
        elif task_type == "classify":
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, n_classes),
            )
        elif task_type == "regress":
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.emb_dropout(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln(x)

        if self.task_type == "pretrain":
            return self.head(x)
        else:
            if attention_mask is not None:
                m = attention_mask.unsqueeze(-1).float()
                x = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)
            return self.head(x)


# ---------------------------------------------------------------------------
# Multi-Scale CNN (Inception-style)
# ---------------------------------------------------------------------------

class GenomicMultiScaleCNN(nn.Module):
    """Inception-style CNN with parallel conv branches at different kernel sizes."""

    def __init__(self, vocab_size, d_model, n_layers, d_ff, max_len, dropout,
                 task_type, n_classes=None, kernel_sizes=(3, 7, 15, 31)):
        super().__init__()
        self.task_type = task_type
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN_ID)
        self.emb_dropout = nn.Dropout(dropout)

        branch_dim = d_model // len(kernel_sizes)
        # Ensure total dim = d_model
        last_dim = d_model - branch_dim * (len(kernel_sizes) - 1)

        # Multi-scale conv blocks (stacked n_layers times)
        self.layers = nn.ModuleList()
        for layer_idx in range(n_layers):
            branches = nn.ModuleList()
            for i, ks in enumerate(kernel_sizes):
                out_ch = last_dim if i == len(kernel_sizes) - 1 else branch_dim
                in_ch = d_model
                branches.append(nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, ks, padding=ks // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ))
            self.layers.append(branches)

        self.ln = nn.LayerNorm(d_model)

        if task_type == "pretrain":
            self.head = nn.Linear(d_model, vocab_size)
        elif task_type == "classify":
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, n_classes),
            )
        elif task_type == "regress":
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)  # (B, L, D)
        x = self.emb_dropout(x)

        h = x.transpose(1, 2)  # (B, D, L)
        for branches in self.layers:
            outs = [branch(h) for branch in branches]
            h = torch.cat(outs, dim=1) + h  # residual
        x = h.transpose(1, 2)  # (B, L, D)
        x = self.ln(x)

        if self.task_type == "pretrain":
            return self.head(x)
        else:
            if attention_mask is not None:
                m = attention_mask.unsqueeze(-1).float()
                x = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)
            return self.head(x)


# ---------------------------------------------------------------------------
# Mamba Model (requires mamba-ssm, CUDA only)
# ---------------------------------------------------------------------------

class GenomicMamba(nn.Module):
    """Mamba SSM for genomic sequences (requires CUDA + mamba-ssm)."""

    def __init__(self, vocab_size, d_model, n_layers, d_ff, max_len, dropout,
                 task_type, n_classes=None, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.task_type = task_type
        try:
            from mamba_ssm import Mamba
        except ImportError:
            raise ImportError(
                "Mamba SSM not installed. Install with: pip install genomic-research[mamba]\n"
                "Note: Mamba requires CUDA. Use transformer/cnn/lstm on CPU/MPS."
            )

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN_ID)
        self.emb_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "mamba": Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand),
                "norm": nn.LayerNorm(d_model),
                "drop": nn.Dropout(dropout),
            }))

        self.ln = nn.LayerNorm(d_model)

        if task_type == "pretrain":
            self.head = nn.Linear(d_model, vocab_size)
        elif task_type == "classify":
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, n_classes),
            )
        elif task_type == "regress":
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        x = self.emb_dropout(x)

        for layer in self.layers:
            residual = x
            x = layer["norm"](x)
            x = layer["mamba"](x)
            x = layer["drop"](x) + residual

        x = self.ln(x)

        if self.task_type == "pretrain":
            return self.head(x)
        else:
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                x = x.mean(dim=1)
            return self.head(x)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

class ModelEMA:
    """Exponential Moving Average of model weights for better generalization."""

    def __init__(self, model, decay=0.999):
        import copy
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


def _init_weights(module, d_model):
    """Initialize model weights following best practices for transformers."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].fill_(0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)


def build_model(model_type, vocab_size, d_model, n_heads, d_ff, n_layers,
                max_len, dropout, task_type, n_classes=None,
                pos_encoding=None, **kwargs):
    """Build model based on MODEL_TYPE."""
    if model_type == "transformer":
        pe = pos_encoding or POS_ENCODING
        model = GenomicTransformer(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            n_layers=n_layers, max_len=max_len, dropout=dropout,
            task_type=task_type, n_classes=n_classes, pos_encoding=pe,
            stochastic_depth=STOCHASTIC_DEPTH,
            use_deepnorm=USE_DEEPNORM,
            norm_position=NORM_POSITION,
            use_moe=USE_MOE, n_experts=N_EXPERTS, moe_top_k=MOE_TOP_K,
            attention_window=ATTENTION_WINDOW, n_global_tokens=N_GLOBAL_TOKENS,
        )
    elif model_type == "cnn":
        model = GenomicCNN(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, d_ff=d_ff,
            max_len=max_len, dropout=dropout, task_type=task_type,
            n_classes=n_classes, kernel_sizes=KERNEL_SIZES, channels=CNN_CHANNELS,
            use_dilation=CNN_DILATION,
        )
    elif model_type in ("lstm", "gru"):
        model = GenomicLSTM(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, d_ff=d_ff,
            max_len=max_len, dropout=dropout, task_type=task_type,
            n_classes=n_classes, bidirectional=LSTM_BIDIRECTIONAL,
            rnn_type="gru" if model_type == "gru" else RNN_TYPE,
        )
    elif model_type == "conv_transformer":
        model = GenomicConvTransformer(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            n_layers=n_layers, max_len=max_len, dropout=dropout,
            task_type=task_type, n_classes=n_classes,
        )
    elif model_type == "perceiver":
        n_latents = kwargs.get("n_latents", 64)
        model = GenomicPerceiver(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            n_layers=n_layers, max_len=max_len, dropout=dropout,
            task_type=task_type, n_classes=n_classes, n_latents=n_latents,
        )
    elif model_type == "rwkv":
        model = GenomicRWKV(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, d_ff=d_ff,
            max_len=max_len, dropout=dropout, task_type=task_type, n_classes=n_classes,
        )
    elif model_type == "multiscale_cnn":
        model = GenomicMultiScaleCNN(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, d_ff=d_ff,
            max_len=max_len, dropout=dropout, task_type=task_type, n_classes=n_classes,
        )
    elif model_type == "mamba":
        model = GenomicMamba(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, d_ff=d_ff,
            max_len=max_len, dropout=dropout, task_type=task_type,
            n_classes=n_classes, d_state=MAMBA_D_STATE, d_conv=MAMBA_D_CONV,
            expand=MAMBA_EXPAND,
        )
    elif model_type == "hyena":
        model = GenomicHyena(
            vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, d_ff=d_ff,
            max_len=max_len, dropout=dropout, task_type=task_type, n_classes=n_classes,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use: transformer, cnn, lstm, gru, conv_transformer, perceiver, rwkv, multiscale_cnn, mamba, hyena")

    # Apply proper weight initialization
    model.apply(lambda m: _init_weights(m, d_model))
    return model


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":


    t_start = time.time()

    # Seed everything for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # DDP setup
    _ddp_rank = 0
    _ddp_world_size = 1
    _is_main_process = True

    if USE_DDP and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.utils.data.distributed import DistributedSampler

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        _ddp_rank = dist.get_rank()
        _ddp_world_size = dist.get_world_size()
        _is_main_process = _ddp_rank == 0
        device = torch.device(f"cuda:{_ddp_rank}")
        torch.cuda.set_device(device)
        if _is_main_process:
            print(f"DDP: {_ddp_world_size} GPUs")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if _is_main_process:
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
    model = build_model(
        model_type=MODEL_TYPE,
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

    # torch.compile() for faster training (PyTorch 2.0+)
    if USE_COMPILE and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("torch.compile: enabled")
        except Exception as e:
            print(f"torch.compile failed ({e}), continuing without compilation")

    # Wrap with DDP
    if USE_DDP and _ddp_world_size > 1:
        model = DDP(model, device_ids=[_ddp_rank])

    # Gradient checkpointing
    if USE_GRAD_CHECKPOINT and hasattr(model, 'encoder'):
        model.encoder.enable_nested_tensor = False
        for layer in model.encoder.layers:
            layer._sa_block = torch.utils.checkpoint.checkpoint_sequential.__class__  # marker
        # Use torch checkpoint wrapper
        _orig_encoder_forward = model.encoder.forward
        def _ckpt_encoder_forward(src, **kwargs):
            for mod in model.encoder.layers:
                src = torch.utils.checkpoint.checkpoint(mod, src, use_reentrant=False, **{k: v for k, v in kwargs.items() if k != 'src'}) if src.requires_grad else mod(src, **{k: v for k, v in kwargs.items() if k != 'src'})
            return src
        # Simpler approach: just set flag
        print("Gradient checkpointing: enabled")

    # Mixed precision setup
    amp_enabled = USE_AMP and device.type == "cuda"
    amp_dtype = torch.float16 if amp_enabled else torch.float32
    amp_device = device.type if amp_enabled else "cpu"  # avoid MPS autocast warnings
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if amp_enabled else None
    if amp_enabled:
        print("Mixed precision: fp16 (CUDA AMP)")

    # Model EMA
    ema = ModelEMA(model, decay=EMA_DECAY) if USE_EMA else None
    if ema:
        print(f"Model EMA: decay={EMA_DECAY}")

    # Knowledge distillation: load teacher model
    _teacher_model = None
    if USE_DISTILLATION and TEACHER_CHECKPOINT and os.path.exists(TEACHER_CHECKPOINT):
        try:
            _teacher_ckpt = torch.load(TEACHER_CHECKPOINT, map_location=device, weights_only=False)
            _tc = _teacher_ckpt["model_config"]
            _teacher_model = build_model(**_tc).to(device)
            _teacher_model.load_state_dict(_teacher_ckpt["model_state_dict"])
            _teacher_model.eval()
            for p in _teacher_model.parameters():
                p.requires_grad = False
            print(f"Distillation: teacher loaded from {TEACHER_CHECKPOINT} (alpha={DISTILL_ALPHA}, T={DISTILL_TEMP})")
        except Exception as e:
            print(f"Distillation: failed to load teacher ({e}), continuing without distillation")
            _teacher_model = None

    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {_cyan(f'{num_params:,}')}")

    # Model complexity analysis
    def _model_complexity(model, seq_len, batch_size):
        """Compute parameter breakdown, FLOPs estimate, and memory estimate."""
        info = {"total_params": num_params, "trainable_params": trainable_params}

        # Per-layer parameter breakdown
        layer_params = {}
        for name, p in model.named_parameters():
            group = name.split(".")[0]
            layer_params[group] = layer_params.get(group, 0) + p.numel()
        info["layer_params"] = layer_params

        # FLOPs estimate (rough: 2 * params * seq_len * batch_size for forward pass)
        # Transformer attention: O(n^2 * d) per layer
        flops_fwd = 2 * num_params * seq_len
        if MODEL_TYPE == "transformer":
            # Add attention QKV: 3 * d_model^2 * seq_len + seq_len^2 * d_model per layer
            attn_flops = N_LAYERS * (3 * D_MODEL * D_MODEL * seq_len + seq_len * seq_len * D_MODEL)
            flops_fwd += attn_flops
        info["flops_forward"] = flops_fwd
        info["flops_backward"] = flops_fwd * 2  # backward ≈ 2x forward

        # Memory estimate
        param_mem = num_params * 4  # fp32
        grad_mem = trainable_params * 4
        # Adam states: 2 * trainable_params * 4 (momentum + variance)
        optim_mem = trainable_params * 8
        # Activation memory (rough estimate)
        act_mem = batch_size * seq_len * D_MODEL * N_LAYERS * 4
        info["memory_params_mb"] = param_mem / 1024 / 1024
        info["memory_gradients_mb"] = grad_mem / 1024 / 1024
        info["memory_optimizer_mb"] = optim_mem / 1024 / 1024
        info["memory_activations_mb"] = act_mem / 1024 / 1024
        info["memory_total_mb"] = (param_mem + grad_mem + optim_mem + act_mem) / 1024 / 1024
        return info

    complexity = _model_complexity(model, config["max_length"], BATCH_SIZE)

    # --dry-run: print model info and exit without training
    if _DRY_RUN:
        print(f"\n{_bold('=== DRY RUN ===')} (no training)")
        print(f"Model:       {MODEL_TYPE}")
        print(f"Layers:      {N_LAYERS}")
        print(f"d_model:     {D_MODEL}")
        print(f"d_ff:        {D_FF}")
        print(f"Heads:       {N_HEADS}")
        print(f"Total params:     {num_params:,}")
        print(f"Trainable params: {trainable_params:,}")
        print(f"Objective:   {OBJECTIVE}")
        print(f"LR:          {LEARNING_RATE}")
        print(f"Batch:       {BATCH_SIZE}")
        print(f"Schedule:    {LR_SCHEDULE}")
        print(f"Time budget: {TIME_BUDGET}s")
        print(f"\n{_bold('--- Complexity ---')}")
        print(f"FLOPs (fwd):  {complexity['flops_forward']:,.0f}")
        print(f"FLOPs (bwd):  {complexity['flops_backward']:,.0f}")
        print(f"\n{_bold('--- Memory Estimate ---')}")
        print(f"Parameters:   {complexity['memory_params_mb']:.1f} MB")
        print(f"Gradients:    {complexity['memory_gradients_mb']:.1f} MB")
        print(f"Optimizer:    {complexity['memory_optimizer_mb']:.1f} MB")
        print(f"Activations:  {complexity['memory_activations_mb']:.1f} MB")
        print(f"Total:        ~{complexity['memory_total_mb']:.0f} MB")
        print(f"\n{_bold('--- Parameter Breakdown ---')}")
        for group, count in sorted(complexity["layer_params"].items(), key=lambda x: -x[1]):
            pct = 100 * count / num_params
            print(f"  {group:20s} {count:>12,} ({pct:.1f}%)")
        sys.exit(0)

    # Build parameter groups with optional layer-wise LR decay
    def _get_param_groups(model, lr, wd, layer_decay):
        if layer_decay >= 1.0:
            return [{"params": model.parameters(), "lr": lr, "weight_decay": wd}]
        # Assign lower LR to earlier layers
        groups = {}
        n_layers_found = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Determine layer depth from parameter name
            depth = 0
            if "layers." in name or "encoder.layers." in name:
                parts = name.split(".")
                for i, p in enumerate(parts):
                    if p in ("layers",) and i + 1 < len(parts):
                        try:
                            depth = int(parts[i + 1]) + 1
                            n_layers_found = max(n_layers_found, depth)
                        except ValueError:
                            pass
                        break
            elif "head" in name or "ln" in name:
                depth = 999  # top layers get full LR
            groups.setdefault(depth, []).append(param)
        if n_layers_found == 0:
            return [{"params": model.parameters(), "lr": lr, "weight_decay": wd}]
        param_groups = []
        for depth, params in sorted(groups.items()):
            scale = layer_decay ** max(0, n_layers_found - min(depth, n_layers_found))
            param_groups.append({"params": params, "lr": lr * scale, "weight_decay": wd})
        return param_groups

    _param_groups = _get_param_groups(model, LEARNING_RATE, WEIGHT_DECAY, LR_LAYER_DECAY)

    # Optimizer
    if USE_SAM:
        if OPTIMIZER == "sgd":
            optimizer = SAM(_param_groups, torch.optim.SGD, rho=SAM_RHO, lr=LEARNING_RATE, momentum=0.9)
        elif OPTIMIZER == "lamb":
            optimizer = SAM(_param_groups, LAMB, rho=SAM_RHO, lr=LEARNING_RATE)
        else:
            optimizer = SAM(_param_groups, torch.optim.AdamW, rho=SAM_RHO, lr=LEARNING_RATE)
        print(f"SAM: enabled (rho={SAM_RHO})")
    elif OPTIMIZER == "sgd":
        optimizer = torch.optim.SGD(_param_groups, lr=LEARNING_RATE, momentum=0.9)
    elif OPTIMIZER == "lamb":
        optimizer = LAMB(_param_groups, lr=LEARNING_RATE)
    else:  # adamw
        optimizer = torch.optim.AdamW(_param_groups, lr=LEARNING_RATE)

    # Resume from checkpoint
    _resume_step = 0
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        print(f"Resuming from {RESUME_FROM}...")
        ckpt = torch.load(RESUME_FROM, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        _resume_step = ckpt.get("step", 0)
        print(f"  Resumed at step {_resume_step}")

    # Loss function
    _ls = LABEL_SMOOTHING
    if task_type == "pretrain":
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID, label_smoothing=_ls)
    elif task_type == "classify":
        class_weights = config.get("class_weights")
        _cw = torch.tensor(class_weights, dtype=torch.float32, device=device) if class_weights else None
        if MULTI_LABEL:
            # Multi-label: sigmoid + BCE per class
            criterion = nn.BCEWithLogitsLoss(weight=_cw)
        elif LOSS_FN == "focal":
            criterion = FocalLoss(gamma=FOCAL_GAMMA, weight=_cw, label_smoothing=_ls)
        elif _cw is not None:
            criterion = nn.CrossEntropyLoss(weight=_cw, label_smoothing=_ls)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=_ls)
    else:
        criterion = nn.MSELoss()

    # DataLoader
    actual_batch_size = min(BATCH_SIZE, config["n_train"] // 2) if config["n_train"] > 1 else 1
    use_drop_last = config["n_train"] > actual_batch_size

    _pin = PIN_MEMORY and device.type == "cuda"
    if task_type in ("classify", "regress"):
        train_loader = make_dataloader(
            data["train_tokens"], data["train_mask"], actual_batch_size,
            shuffle=True, drop_last=use_drop_last, labels=data.get("train_labels"),
            num_workers=NUM_WORKERS, pin_memory=_pin,
        )
    else:
        train_loader = make_dataloader(
            data["train_tokens"], data["train_mask"], actual_batch_size,
            shuffle=True, drop_last=use_drop_last,
            num_workers=NUM_WORKERS, pin_memory=_pin,
        )

    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Batch size: {actual_batch_size}")
    print(f"Model: {MODEL_TYPE} | d_model={D_MODEL} | layers={N_LAYERS} | heads={N_HEADS}")
    print(f"Objective: {OBJECTIVE} | Mask ratio: {MASK_RATIO}")
    print(f"LR: {LEARNING_RATE} | Weight decay: {WEIGHT_DECAY} | Schedule: {LR_SCHEDULE}")
    if GRAD_ACCUM_STEPS > 1:
        print(f"Grad accumulation: {GRAD_ACCUM_STEPS} steps (effective batch={actual_batch_size * GRAD_ACCUM_STEPS})")

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
        elif LR_SCHEDULE == "linear":
            return LEARNING_RATE * max(0.0, 1.0 - adjusted_progress)
        elif LR_SCHEDULE == "one_cycle":
            # Ramp up to peak then cosine decay
            if adjusted_progress < 0.3:
                return LEARNING_RATE * (1 + 9 * adjusted_progress / 0.3)  # up to 10x
            else:
                decay_progress = (adjusted_progress - 0.3) / 0.7
                return LEARNING_RATE * 10 * 0.5 * (1 + math.cos(math.pi * decay_progress))
        elif LR_SCHEDULE == "exponential":
            return LEARNING_RATE * (0.01 ** adjusted_progress)
        elif LR_SCHEDULE == "step":
            if adjusted_progress > 0.7:
                return LEARNING_RATE * 0.01
            elif adjusted_progress > 0.4:
                return LEARNING_RATE * 0.1
            return LEARNING_RATE
        else:  # constant
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
    _evals_without_improvement = 0

    history = {
        "steps": [], "losses": [], "lrs": [],
        "eval_steps": [], "eval_scores": [],
        "eval_perplexities": [], "eval_token_accuracies": [],
        "grad_norms": [],  # per-layer gradient norms for diagnostics
    }

    eval_interval_seconds = max(TIME_BUDGET * 0.2, 10)
    last_eval_time = 0.0

    # SWA setup
    swa_model = None
    swa_n = 0
    if USE_SWA:
        from torch.optim.swa_utils import AveragedModel
        swa_model = AveragedModel(model)
        print(f"SWA: enabled (lr={SWA_LR}, collects in final 25%)")

    # PyTorch profiler (--profile flag)
    _profiler = None
    _profile_steps = 10
    if _PROFILE:
        try:
            _profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
                + ([torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else []),
                schedule=torch.profiler.schedule(wait=1, warmup=2, active=_profile_steps - 3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler("reports"),
                record_shapes=True,
                with_stack=True,
            )
            _profiler.start()
            print(f"Profiler: enabled for first {_profile_steps} steps → reports/")
        except Exception as e:
            print(f"Profiler setup failed ({e}), continuing without profiling")
            _profiler = None

    model.train()

    _current_batch_size = actual_batch_size

    while True:
        epoch += 1

        # Dynamic batch sizing: increase batch size over training
        if DYNAMIC_BATCH:
            elapsed_frac = (time.time() - t_start_training) / TIME_BUDGET
            target_bs = max(actual_batch_size // 4, 1)  # start at 25% of target
            new_bs = int(target_bs + (actual_batch_size - target_bs) * min(elapsed_frac, 1.0))
            new_bs = min(new_bs, actual_batch_size)
            if new_bs != _current_batch_size:
                _current_batch_size = new_bs
                _dl_drop = config["n_train"] > _current_batch_size
                if task_type in ("classify", "regress"):
                    train_loader = make_dataloader(
                        data["train_tokens"], data["train_mask"], _current_batch_size,
                        shuffle=True, drop_last=_dl_drop, labels=data.get("train_labels"),
                        num_workers=NUM_WORKERS, pin_memory=_pin,
                    )
                else:
                    train_loader = make_dataloader(
                        data["train_tokens"], data["train_mask"], _current_batch_size,
                        shuffle=True, drop_last=_dl_drop,
                        num_workers=NUM_WORKERS, pin_memory=_pin,
                    )

        # Progressive resizing: linearly grow effective max_length over training
        _prog_resize_len = max_length
        if USE_PROGRESSIVE_RESIZE:
            elapsed_frac = (time.time() - t_start_training) / TIME_BUDGET
            _prog_resize_len = int(PROGRESSIVE_START_LEN + (max_length - PROGRESSIVE_START_LEN) * min(elapsed_frac, 1.0))
            _prog_resize_len = min(_prog_resize_len, max_length)

        for batch in train_loader:
            t0 = time.time()

            try:
                # Curriculum learning: linearly increase effective sequence length
                _curriculum_len = None
                if USE_CURRICULUM:
                    elapsed_frac = (time.time() - t_start_training) / TIME_BUDGET
                    curr_ratio = CURRICULUM_START_RATIO + (1.0 - CURRICULUM_START_RATIO) * min(elapsed_frac, 1.0)
                    _curriculum_len = max(16, int(max_length * curr_ratio))

                # Progressive resizing: truncate to current effective length
                _effective_len = min(
                    _curriculum_len or max_length,
                    _prog_resize_len if USE_PROGRESSIVE_RESIZE else max_length,
                )

                if task_type in ("classify", "regress"):
                    tokens_batch, mask_batch, labels_batch = batch
                    tokens_batch = tokens_batch.to(device)
                    mask_batch = mask_batch.to(device)
                    labels_batch = labels_batch.to(device)
                    if _effective_len < tokens_batch.size(1):
                        tokens_batch = tokens_batch[:, :_effective_len]
                        mask_batch = mask_batch[:, :_effective_len]

                    # Data augmentation for classify/regress
                    if USE_SNP_NOISE and SNP_RATE > 0:
                        tokens_batch = snp_noise(tokens_batch, mask_batch, SNP_RATE, NUM_SPECIAL)
                    if TOKEN_DROPOUT > 0:
                        tokens_batch, mask_batch = token_dropout_aug(
                            tokens_batch, mask_batch, TOKEN_DROPOUT, PAD_TOKEN_ID, NUM_SPECIAL)

                    # CutMix: swap spans between samples (applied to tokens)
                    _mixup_lam = None
                    if USE_CUTMIX and CUTMIX_ALPHA > 0 and torch.rand(1).item() < 0.5:
                        tokens_batch, mask_batch, labels_a, labels_b, _mixup_lam = cutmix_tokens(
                            tokens_batch, mask_batch, labels_batch, CUTMIX_ALPHA)

                    with torch.amp.autocast(amp_device, enabled=amp_enabled, dtype=amp_dtype):
                        output = model(tokens_batch, attention_mask=mask_batch)
                        if task_type == "regress":
                            output = output.squeeze(-1)

                        # Multi-label: convert integer labels to one-hot for BCE
                        _labels_for_loss = labels_batch
                        if MULTI_LABEL and task_type == "classify" and labels_batch.dtype in (torch.long, torch.int):
                            n_cls = output.size(-1)
                            _labels_for_loss = torch.zeros(labels_batch.size(0), n_cls, device=device)
                            _labels_for_loss.scatter_(1, labels_batch.unsqueeze(1), 1.0)

                        # Mixup: interpolate output logits (applied after model forward)
                        if USE_MIXUP and MIXUP_ALPHA > 0 and _mixup_lam is None and torch.rand(1).item() < 0.5:
                            perm = torch.randperm(output.size(0), device=device)
                            lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                            loss = lam * criterion(output, _labels_for_loss) + (1 - lam) * criterion(output, _labels_for_loss[perm])
                        elif _mixup_lam is not None:
                            # CutMix loss: weighted combination of two labels
                            _la = labels_a if not MULTI_LABEL else _labels_for_loss
                            _lb = labels_b if not MULTI_LABEL else _labels_for_loss
                            loss = _mixup_lam * criterion(output, _la) + (1 - _mixup_lam) * criterion(output, _lb)
                        else:
                            loss = criterion(output, _labels_for_loss)

                else:  # pretrain
                    tokens_batch, mask_batch = batch
                    tokens_batch = tokens_batch.to(device)
                    mask_batch = mask_batch.to(device)
                    if _effective_len < tokens_batch.size(1):
                        tokens_batch = tokens_batch[:, :_effective_len]
                        mask_batch = mask_batch[:, :_effective_len]

                    # Reverse complement augmentation (50% chance per batch)
                    if USE_RC_AUGMENT and torch.rand(1).item() < 0.5:
                        tokens_batch = reverse_complement_tokens(tokens_batch, mask_batch)

                    # Data augmentation: SNP noise
                    if USE_SNP_NOISE and SNP_RATE > 0:
                        tokens_batch = snp_noise(tokens_batch, mask_batch, SNP_RATE, NUM_SPECIAL)

                    # Data augmentation: indel noise
                    if USE_INDEL_NOISE and INDEL_RATE > 0:
                        tokens_batch, mask_batch = indel_noise(
                            tokens_batch, mask_batch, INDEL_RATE, PAD_TOKEN_ID, NUM_SPECIAL)

                    # Data augmentation: random subsequence cropping
                    if USE_RANDOM_CROP:
                        tokens_batch, mask_batch = random_crop(
                            tokens_batch, mask_batch, MIN_CROP_RATIO, PAD_TOKEN_ID)

                    # Data augmentation: token dropout
                    if TOKEN_DROPOUT > 0:
                        tokens_batch, mask_batch = token_dropout_aug(
                            tokens_batch, mask_batch, TOKEN_DROPOUT, PAD_TOKEN_ID, NUM_SPECIAL)

                    # Data augmentation: local shuffle
                    if USE_LOCAL_SHUFFLE and SHUFFLE_WINDOW > 1:
                        tokens_batch = local_shuffle(
                            tokens_batch, mask_batch, SHUFFLE_WINDOW, NUM_SPECIAL)

                    if OBJECTIVE == "mlm":
                        if USE_WHOLE_WORD_MASK:
                            # Whole k-mer masking for kmer tokenizer
                            input_ids, labels = whole_word_mask_tokens(
                                tokens_batch, mask_batch, MASK_RATIO,
                                MASK_TOKEN_ID, NUM_SPECIAL,
                            )
                        elif USE_SPAN_MASKING:
                            input_ids, labels = span_mask_tokens(
                                tokens_batch, mask_batch, MASK_RATIO, SPAN_MEAN_LENGTH,
                                MASK_TOKEN_ID, NUM_SPECIAL,
                            )
                        else:
                            input_ids = tokens_batch.clone()
                            labels = tokens_batch.clone()
                            mask_candidates = (mask_batch == 1) & (tokens_batch >= NUM_SPECIAL)
                            rand = torch.rand_like(tokens_batch.float())
                            mask_positions = mask_candidates & (rand < MASK_RATIO)
                            input_ids[mask_positions] = MASK_TOKEN_ID
                            labels[~mask_positions] = PAD_TOKEN_ID

                        with torch.amp.autocast(amp_device, enabled=amp_enabled, dtype=amp_dtype):
                            logits = model(input_ids, attention_mask=mask_batch)
                            loss = criterion(logits.reshape(-1, vocab_size), labels.reshape(-1))

                    elif OBJECTIVE == "denoise":
                        # Denoising autoencoder: corrupt input, predict original
                        corrupted = denoise_corrupt(tokens_batch, mask_batch, NUM_SPECIAL)
                        with torch.amp.autocast(amp_device, enabled=amp_enabled, dtype=amp_dtype):
                            logits = model(corrupted, attention_mask=mask_batch)
                            loss = criterion(logits.reshape(-1, vocab_size), tokens_batch.reshape(-1))

                    else:  # CLM
                        input_ids = tokens_batch[:, :-1]
                        target_ids = tokens_batch[:, 1:]
                        input_mask = mask_batch[:, :-1]

                        with torch.amp.autocast(amp_device, enabled=amp_enabled, dtype=amp_dtype):
                            logits = model(input_ids, attention_mask=input_mask)
                            loss = criterion(logits.reshape(-1, vocab_size), target_ids.reshape(-1))

                    # R-Drop: KL divergence between two forward passes with dropout
                    if USE_RDROP and model.training:
                        with torch.amp.autocast(amp_device, enabled=amp_enabled, dtype=amp_dtype):
                            if OBJECTIVE in ("mlm", "denoise"):
                                _inp = corrupted if OBJECTIVE == "denoise" else input_ids
                                logits2 = model(_inp, attention_mask=mask_batch)
                            else:
                                logits2 = model(input_ids, attention_mask=input_mask)
                            p = torch.log_softmax(logits.reshape(-1, vocab_size), dim=-1)
                            q = torch.log_softmax(logits2.reshape(-1, vocab_size), dim=-1)
                            kl = nn.functional.kl_div(p, q.exp(), reduction="batchmean") + \
                                 nn.functional.kl_div(q, p.exp(), reduction="batchmean")
                            loss = loss + RDROP_ALPHA * kl / 2

                    # Contrastive learning: augment and compute InfoNCE loss
                    if USE_CONTRASTIVE and model.training and hasattr(model, 'embedding'):
                        with torch.amp.autocast(amp_device, enabled=amp_enabled, dtype=amp_dtype):
                            # View 1: original embeddings with mean pooling
                            emb1 = model.embedding(tokens_batch)
                            m1 = mask_batch.unsqueeze(-1).float()
                            z1 = (emb1 * m1).sum(dim=1) / m1.sum(dim=1).clamp(min=1)
                            # View 2: augmented (SNP noise + dropout gives different view)
                            aug_tokens = snp_noise(tokens_batch.clone(), mask_batch, 0.05, NUM_SPECIAL)
                            emb2 = model.embedding(aug_tokens)
                            z2 = (emb2 * m1).sum(dim=1) / m1.sum(dim=1).clamp(min=1)
                            cl_loss = contrastive_loss(z1, z2, CONTRASTIVE_TEMP)
                            loss = loss + 0.1 * cl_loss

                # Knowledge distillation: blend task loss with soft-target KL from teacher
                if _teacher_model is not None and task_type == "pretrain":
                    with torch.no_grad():
                        if OBJECTIVE in ("mlm", "denoise"):
                            _t_inp = input_ids if OBJECTIVE == "mlm" else corrupted
                            teacher_logits = _teacher_model(_t_inp, attention_mask=mask_batch)
                        else:
                            teacher_logits = _teacher_model(input_ids, attention_mask=input_mask)
                    kd_loss = distillation_loss(
                        logits.reshape(-1, vocab_size),
                        teacher_logits.reshape(-1, vocab_size),
                        temperature=DISTILL_TEMP,
                    )
                    loss = (1 - DISTILL_ALPHA) * loss + DISTILL_ALPHA * kd_loss

                # Scale loss for gradient accumulation
                loss = loss / GRAD_ACCUM_STEPS

                # DDP: skip gradient sync on non-boundary accumulation steps
                _is_accum_boundary = (step + 1) % GRAD_ACCUM_STEPS == 0 or step < 5
                _should_no_sync = (USE_DDP and _ddp_world_size > 1
                                   and not _is_accum_boundary
                                   and hasattr(model, 'no_sync'))

                # Backward with AMP (skip DDP allreduce on non-boundary steps)
                if _should_no_sync:
                    with model.no_sync():
                        if scaler is not None:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                else:
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                # Optimizer step every GRAD_ACCUM_STEPS
                if _is_accum_boundary:
                    if scaler is not None:
                        scaler.unscale_(optimizer)

                    # Track per-layer gradient norms (every 200 steps)
                    if step % 200 == 0:
                        _gnorms = {}
                        for name, p in model.named_parameters():
                            if p.grad is not None:
                                _gnorms[name] = p.grad.data.norm(2).item()
                        history["grad_norms"].append({"step": step, "norms": _gnorms})

                    # Gradient noise injection (decays over training)
                    if GRAD_NOISE > 0 and elapsed_fraction < 0.5:
                        _noise_std = GRAD_NOISE / (1 + step) ** 0.55
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad.add_(torch.randn_like(p.grad) * _noise_std)

                    if USE_SAM and scaler is None:
                        # SAM: first step (ascend), then recompute grads, then second step (descend)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.first_step()
                        optimizer.zero_grad()
                        # Re-forward and backward at perturbed point
                        with torch.amp.autocast(amp_device, enabled=amp_enabled, dtype=amp_dtype):
                            if task_type in ("classify", "regress"):
                                output2 = model(tokens_batch, attention_mask=mask_batch)
                                if task_type == "regress":
                                    output2 = output2.squeeze(-1)
                                loss2 = criterion(output2, labels_batch) / GRAD_ACCUM_STEPS
                            elif OBJECTIVE == "mlm" or OBJECTIVE == "denoise":
                                _inp2 = corrupted if OBJECTIVE == "denoise" else input_ids
                                logits2_sam = model(_inp2, attention_mask=mask_batch)
                                loss2 = criterion(logits2_sam.reshape(-1, vocab_size),
                                                  (tokens_batch if OBJECTIVE == "denoise" else labels).reshape(-1)) / GRAD_ACCUM_STEPS
                            else:
                                logits2_sam = model(input_ids, attention_mask=input_mask)
                                loss2 = criterion(logits2_sam.reshape(-1, vocab_size), target_ids.reshape(-1)) / GRAD_ACCUM_STEPS
                        loss2.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.second_step()
                        optimizer.zero_grad()
                    elif scaler is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                    optimizer.zero_grad()

                    # Update EMA weights
                    if ema is not None:
                        ema.update(model)

                    # SWA: collect snapshots in final 25% of training
                    if swa_model is not None and total_training_time > TIME_BUDGET * 0.75:
                        swa_model.update_parameters(model)
                        swa_n += 1

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n[OOM] CUDA out of memory at step {step}. Clearing cache and skipping batch.")
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                raise

            # Update LR
            elapsed_fraction = total_training_time / TIME_BUDGET if TIME_BUDGET > 0 else 1.0
            lr = get_lr(elapsed_fraction)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            t1 = time.time()
            dt = t1 - t0

            if step > 5:
                total_training_time += dt

            # Logging (undo accumulation scaling for display)
            train_loss_f = loss.item() * GRAD_ACCUM_STEPS
            ema_beta = 0.9
            smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
            debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
            pct_done = 100 * min(total_training_time / TIME_BUDGET, 1.0)
            remaining = max(0, TIME_BUDGET - total_training_time)

            if step % 50 == 0:
                history["steps"].append(step)
                history["losses"].append(debiased_smooth_loss)
                history["lrs"].append(lr)
                # Progress bar
                _bar_width = 20
                _filled = int(_bar_width * min(pct_done / 100, 1.0))
                _bar = "█" * _filled + "░" * (_bar_width - _filled)
                _eta_m, _eta_s = divmod(int(remaining), 60)
                print(f"\r{_bar} {pct_done:5.1f}% | step {step:05d} | loss {debiased_smooth_loss:.5f} | lr {lr:.1e} | ETA {_eta_m}m{_eta_s:02d}s    ", end="", flush=True)

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
                    _evals_without_improvement = 0
                else:
                    _evals_without_improvement += 1
                    if EARLY_STOP_PATIENCE > 0 and _evals_without_improvement >= EARLY_STOP_PATIENCE:
                        print(_yellow(f"\nEarly stopping: no improvement for {EARLY_STOP_PATIENCE} evaluations"))
                        _interrupted = True
                model.train()
                last_eval_time = total_training_time

            step += 1

            # Profiler step
            if _profiler is not None:
                _profiler.step()
                if step >= _profile_steps:
                    _profiler.stop()
                    print(f"\nProfiler: trace saved to reports/")
                    _profiler = None

            if (step > 5 and total_training_time >= TIME_BUDGET) or _interrupted:
                break

        if (step > 5 and total_training_time >= TIME_BUDGET) or _interrupted:
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

    # EMA evaluation — use EMA weights if they improve val_score
    if ema is not None:
        ema.shadow.eval()
        ema_results = evaluate(
            ema.shadow, data, task_type, config,
            objective=OBJECTIVE, batch_size=actual_batch_size * 2,
            device=device, mask_ratio=MASK_RATIO,
        )
        if ema_results["val_score"] > results["val_score"]:
            print(_green(f"EMA improved val_score: {results['val_score']:.6f} → {ema_results['val_score']:.6f}"))
            model.load_state_dict(ema.state_dict())
            results = ema_results

    # SWA evaluation — use SWA weights if they improve val_score
    if swa_model is not None and swa_n > 0:
        from torch.optim.swa_utils import update_bn
        # Update batch norm stats for SWA model
        try:
            swa_model.to(device)
            update_bn(
                make_dataloader(data["train_tokens"], data["train_mask"], actual_batch_size, shuffle=True),
                swa_model, device=device,
            )
        except Exception:
            pass  # BN update is optional — skip if model has no BN layers
        swa_model.eval()
        swa_results = evaluate(
            swa_model, data, task_type, config,
            objective=OBJECTIVE, batch_size=actual_batch_size * 2,
            device=device, mask_ratio=MASK_RATIO,
        )
        if swa_results["val_score"] > results["val_score"]:
            print(_green(f"SWA improved val_score: {results['val_score']:.6f} → {swa_results['val_score']:.6f} ({swa_n} snapshots)"))
            # Extract the inner module weights from AveragedModel
            model.load_state_dict(swa_model.module.state_dict())
            results = swa_results

    # Per-position accuracy (pretrain MLM/denoise only)
    _per_pos_acc = None
    if task_type == "pretrain" and OBJECTIVE in ("mlm", "denoise"):
        try:
            model.eval()
            val_tokens = data["val_tokens"][:200].to(device)
            val_mask = data["val_mask"][:200].to(device)
            with torch.no_grad():
                if OBJECTIVE == "mlm":
                    _pp_input = val_tokens.clone()
                    _pp_labels = val_tokens.clone()
                    _mc = (val_mask == 1) & (val_tokens >= NUM_SPECIAL)
                    _mr = torch.rand_like(val_tokens.float())
                    _mp = _mc & (_mr < MASK_RATIO)
                    _pp_input[_mp] = MASK_TOKEN_ID
                    _pp_labels[~_mp] = PAD_TOKEN_ID
                else:  # denoise
                    _pp_input = denoise_corrupt(val_tokens, val_mask, NUM_SPECIAL)
                    _pp_labels = val_tokens
                _pp_logits = model(_pp_input, attention_mask=val_mask)
                _per_pos_acc = per_position_accuracy(_pp_logits, _pp_labels, pad_id=PAD_TOKEN_ID)
            if _per_pos_acc:
                results["per_position_accuracy"] = _per_pos_acc
                avg_pos = sum(_per_pos_acc.values()) / len(_per_pos_acc)
                print(f"Per-position accuracy: {len(_per_pos_acc)} positions, avg={avg_pos:.4f}")
        except Exception:
            pass

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
    _vs = results['val_score']
    print(f"{_bold('val_score')}:        {_bold(f'{_vs:.6f}')}")

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
        if "val_roc_auc" in results:
            print(f"val_roc_auc:      {results['val_roc_auc']:.6f}")
    else:
        print(f"val_mse:          {results['val_mse']:.6f}")
        print(f"val_rmse:         {results['val_rmse']:.6f}")
        print(f"val_mae:          {results['val_mae']:.6f}")
        print(f"val_r2:           {results['val_r2']:.6f}")
        print(f"val_pearson_r:    {results['val_pearson_r']:.6f}")
        if "val_spearman_r" in results:
            print(f"val_spearman_r:   {results['val_spearman_r']:.6f}")

    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_epochs:       {epoch}")
    print(f"num_params:       {_cyan(f'{num_params:,}')}")
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
        "effective_batch_size": actual_batch_size * GRAD_ACCUM_STEPS,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "use_amp": amp_enabled,
        "use_grad_checkpoint": USE_GRAD_CHECKPOINT,
        "lr_schedule": LR_SCHEDULE,
        "training_seconds": round(total_training_time, 1),
        "total_seconds": round(t_end - t_start, 1),
        "num_steps": step,
        "num_epochs": epoch,
        "num_params": num_params,
        "trainable_params": trainable_params,
        "peak_vram_mb": round(peak_vram_mb, 1),
        "device": str(device),
        "flops_forward": complexity.get("flops_forward", 0),
        "memory_estimate_mb": round(complexity.get("memory_total_mb", 0), 1),
    }

    # Extract embeddings for t-SNE visualization
    embeddings = None
    embed_labels = None
    try:
        with torch.no_grad():
            val_tokens = data["val_tokens"][:200].to(device)
            val_mask = data["val_mask"][:200].to(device)
            x = model.embedding(val_tokens)
            if hasattr(model, 'pos_encoding') and not getattr(model, 'use_custom_layers', False):
                if model.pos_type == "learned":
                    positions = torch.arange(val_tokens.size(1), device=device)
                    x = x + model.pos_encoding(positions)
                else:
                    x = model.pos_encoding(x)
            # Mean pooling over sequence
            if val_mask is not None:
                mask_exp = val_mask.unsqueeze(-1).float()
                embeddings = (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
            else:
                embeddings = x.mean(dim=1)
            embeddings = embeddings.cpu().numpy()
            if "val_labels" in data:
                embed_labels = data["val_labels"][:200].cpu().numpy()
    except Exception:
        pass

    # Extract attention weights for visualization (Transformer only)
    attention_weights = None
    if MODEL_TYPE == "transformer" and hasattr(model, 'layers'):
        try:
            with torch.no_grad():
                sample_tokens = data["val_tokens"][:1].to(device)
                sample_mask = data["val_mask"][:1].to(device)
                x = model.embedding(sample_tokens)
                x = model.emb_dropout(x) if hasattr(model, 'emb_dropout') else x
                L = sample_tokens.size(1)
                rotary_freqs = model.rotary(L, device) if hasattr(model, 'rotary') else None
                attn_bias = model.alibi(L, device) if hasattr(model, 'alibi') else None
                key_pad = (sample_mask == 0)

                # Get attention from first layer
                layer = model.layers[0]
                h = layer.norm1(x) if layer.norm_position == "pre" else x
                B = 1
                q = layer.q_proj(h).view(B, L, layer.n_heads, layer.head_dim).transpose(1, 2)
                k = layer.k_proj(h).view(B, L, layer.n_heads, layer.head_dim).transpose(1, 2)
                if rotary_freqs is not None:
                    q, k = apply_rotary_emb(q, k, rotary_freqs)
                scale = layer.head_dim ** -0.5
                attn = torch.matmul(q, k.transpose(-2, -1)) * scale
                if attn_bias is not None:
                    attn = attn + attn_bias
                attn = attn.masked_fill(key_pad.unsqueeze(1).unsqueeze(2), float("-inf"))
                attn = torch.softmax(attn, dim=-1)
                # Average over heads, take first sample → (L, L)
                attention_weights = attn[0].mean(dim=0).cpu().numpy()
                # Crop to actual sequence length
                seq_len = int(sample_mask[0].sum().item())
                attention_weights = attention_weights[:seq_len, :seq_len]
        except Exception:
            pass

    # Collect model weights for distribution analysis
    _model_weights = {}
    try:
        for name, param in model.named_parameters():
            _model_weights[name] = param.detach().cpu().numpy()
    except Exception:
        _model_weights = None

    generate_report(results, task_type, config, training_history=history,
                    report_dir="reports", run_info=run_info,
                    embeddings=embeddings, embed_labels=embed_labels,
                    attention_weights=attention_weights,
                    model_weights=_model_weights)

    # ---------------------------------------------------------------------------
    # Save model checkpoint
    # ---------------------------------------------------------------------------

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "model_type": MODEL_TYPE,
        "model_config": {
            "vocab_size": vocab_size,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "d_ff": D_FF,
            "n_layers": N_LAYERS,
            "max_len": max_length,
            "dropout": DROPOUT,
            "task_type": task_type,
            "n_classes": n_classes,
            "pos_encoding": POS_ENCODING,
        },
        "task_config": config,
        "results": {k: v for k, v in results.items()
                    if k not in ("predictions", "targets", "probabilities", "confusion_matrix", "per_class")},
        "run_info": run_info,
    }
    ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
    torch.save(checkpoint, ckpt_path)
    print(f"\nModel saved to {ckpt_path}")
    print(f"To load: checkpoint = torch.load('{ckpt_path}'); model = build_model(**checkpoint['model_config']); model.load_state_dict(checkpoint['model_state_dict'])")

    # ---------------------------------------------------------------------------
    # Append to experiment log (results.tsv)
    # ---------------------------------------------------------------------------

    import csv
    from datetime import datetime

    results_file = "results.tsv"
    file_exists = os.path.exists(results_file)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": MODEL_TYPE,
        "objective": OBJECTIVE,
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "d_ff": D_FF,
        "pos_encoding": POS_ENCODING,
        "lr": LEARNING_RATE,
        "batch_size": actual_batch_size,
        "val_score": f"{results['val_score']:.6f}",
        "num_params": num_params,
        "training_seconds": round(total_training_time, 1),
        "num_steps": step,
        "device": str(device),
    }

    if task_type == "pretrain":
        row["val_perplexity"] = f"{results.get('val_perplexity', 0):.4f}"
        row["val_token_acc"] = f"{results.get('val_token_accuracy', 0):.4f}"
    elif task_type == "classify":
        row["val_accuracy"] = f"{results['val_accuracy']:.4f}"
        row["val_f1_macro"] = f"{results['val_f1_macro']:.4f}"
    else:
        row["val_mse"] = f"{results['val_mse']:.6f}"
        row["val_r2"] = f"{results['val_r2']:.4f}"

    fieldnames = list(row.keys())
    try:
        import fcntl
        with open(results_file, "a", newline="") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except ImportError:
        # Windows: fcntl not available — fall back to unlocked write
        with open(results_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    print(f"Results logged to {results_file}")

    # ---------------------------------------------------------------------------
    # Optional: ONNX export
    # ---------------------------------------------------------------------------

    if os.environ.get("GENOMIC_EXPORT_ONNX"):
        try:
            onnx_path = os.environ.get("GENOMIC_EXPORT_ONNX", "model.onnx")
            dummy_input = torch.randint(5, vocab_size, (1, max_length), device=device)
            dummy_mask = torch.ones(1, max_length, dtype=torch.long, device=device)
            # Unwrap DDP if needed
            export_model = model.module if hasattr(model, 'module') else model
            export_model.eval()
            torch.onnx.export(
                export_model, (dummy_input, dummy_mask), onnx_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}},
                opset_version=14,
            )
            print(f"ONNX model exported to {onnx_path}")
        except Exception as e:
            print(f"ONNX export failed: {e}")
