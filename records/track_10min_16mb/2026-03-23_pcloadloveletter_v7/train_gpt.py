"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
import glob
import io
import math
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

try:
    import zstandard as zstd
    _HAS_ZSTD = True
except ImportError:
    import zlib
    _HAS_ZSTD = False

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# pcloadloveletter v6:
# - 11 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and MLP hidden 1500
# - BigramHash + optional TrigramHash embeddings
# - vocab size 1024, sequence length 2048, tied embeddings
# - 786,432 train tokens per step for 20,000 iterations with a ~10 minute cap
# v5 base: Partial RoPE, LN Scale, XSA4, Late QAT, novel compression
# v6 additions:
# - EMA (decay=0.997) replacing SWA — smoother model averaging
# - Value Residual (arXiv:2410.17897) — cache layer-0 V, blend via learned lambda
# - Gated Attention (arXiv:2505.06708) — per-head sigmoid gate after SDPA
# - AdamW TTT — test-time training with cosine schedule and per-layer lr
# - TrigramHash — optional 3-token context hash embedding
# - Eval stride 32 (from 64) — more context per scored token

class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 3000))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 786_432))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 2048))
    progressive_seq = bool(int(os.environ.get("PROGRESSIVE_SEQ", "0")))
    progressive_seq_schedule = os.environ.get("PROGRESSIVE_SEQ_SCHEDULE", "512:0.25,1024:0.5,2048:1.0")
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 12))  # v8: overnight sweep showed 12L optimal
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 3))
    mlp_hidden = int(os.environ.get("MLP_HIDDEN", 1300))  # v8: overnight sweep, narrower MLP fits 12L budget
    num_loops = int(os.environ.get("NUM_LOOPS", 1))
    lora_rank = int(os.environ.get("LORA_RANK", 0))
    qat = bool(int(os.environ.get("QAT", "0")))
    eval_stride = int(os.environ.get("EVAL_STRIDE", 64))  # v8: PR #700 showed same BPB quality, 2x faster eval
    eval_batch_seqs = int(os.environ.get("EVAL_BATCH_SEQS", 64))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    rope_base = float(os.environ.get("ROPE_BASE", 50000.0))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    bigram_vocab_size = int(os.environ.get("BIGRAM_VOCAB_SIZE", 8192))  # v8: overnight sweep, 4x larger hash table
    bigram_dim = int(os.environ.get("BIGRAM_DIM", 128))

    # v5: Partial RoPE — only first 16 of 64 head dims get positional encoding
    rope_dims = int(os.environ.get("ROPE_DIMS", 16))
    # v5: LN Scale — progressive layer norm damping
    ln_scale = bool(int(os.environ.get("LN_SCALE", "1")))
    # v5: XSA — exclusive self-attention on last N layers
    xsa_layers = int(os.environ.get("XSA_LAYERS", 12))  # v8.1: XSA on ALL 12 layers
    # v5: Late QAT — STE int6 fake-quantization when lr_scale < threshold
    late_qat = bool(int(os.environ.get("LATE_QAT", "1")))
    late_qat_threshold = float(os.environ.get("LATE_QAT_THRESHOLD", 0.1))
    # v7: QAT mode — "ste" (v5 original) or "soft_round" (PR #606, differentiable rounding)
    qat_mode = os.environ.get("QAT_MODE", "ste")  # v8: soft_round proven useless at ~500 QAT steps, revert to STE
    qat_alpha_start = float(os.environ.get("QAT_ALPHA_START", 1.0))
    qat_alpha_end = float(os.environ.get("QAT_ALPHA_END", 16.0))

    # v6: EMA replacing SWA
    ema_enabled = bool(int(os.environ.get("EMA_ENABLED", "1")))
    ema_decay = float(os.environ.get("EMA_DECAY", 0.997))
    # v6: Value Residual (arXiv:2410.17897)
    value_residual = bool(int(os.environ.get("VALUE_RESIDUAL", "1")))
    # v6: Gated Attention (arXiv:2505.06708)
    gated_attention = bool(int(os.environ.get("GATED_ATTENTION", "1")))
    # v6: TrigramHash (optional, default off for int6+zstd size budget)
    trigram_vocab_size = int(os.environ.get("TRIGRAM_VOCAB_SIZE", 4096))
    trigram_dim = int(os.environ.get("TRIGRAM_DIM", 128))
    use_trigram = bool(int(os.environ.get("USE_TRIGRAM", "0")))
    # v7: QuadgramHash
    quadgram_vocab_size = int(os.environ.get("QUADGRAM_VOCAB_SIZE", 1024))
    quadgram_dim = int(os.environ.get("QUADGRAM_DIM", 32))
    use_quadgram = bool(int(os.environ.get("USE_QUADGRAM", "1")))
    # v7: Shared Sparse Sidecar (PR #555)
    sparse_start_layer = int(os.environ.get("SPARSE_START_LAYER", 8))
    sparse_hidden_dim = int(os.environ.get("SPARSE_HIDDEN_DIM", 48))
    # v7: LoRA TTT mode
    ttt_mode = os.environ.get("TTT_MODE", "lora")  # 'lora' or 'legacy'
    ttt_lora_rank = int(os.environ.get("TTT_LORA_RANK", 8))
    ttt_lora_lr = float(os.environ.get("TTT_LORA_LR", 0.01))
    ttt_eval_seq_len = int(os.environ.get("TTT_EVAL_SEQ_LEN", 1024))
    # v6: AdamW TTT — test-time training
    ttt_enabled = bool(int(os.environ.get("TTT_ENABLED", "1")))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))  # v7: per-document, 3 epochs (PR #503 recipe)
    ttt_lr = float(os.environ.get("TTT_LR", 1e-4))  # v7: 0.0001 per fbe dev (PR #503)
    ttt_wd = float(os.environ.get("TTT_WD", 0.0))
    ttt_grad_clip = float(os.environ.get("TTT_GRAD_CLIP", 1.0))
    ttt_lr_output_mult = float(os.environ.get("TTT_LR_OUTPUT_MULT", 3.0))
    ttt_lr_mlp_fc_mult = float(os.environ.get("TTT_LR_MLP_FC_MULT", 0.5))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", 16))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 9))  # v7: freeze first 9 blocks (PR #503)
    ttt_min_doc_len = int(os.environ.get("TTT_MIN_DOC_LEN", 512))  # v7: skip short docs
    ttt_chunk_size = int(os.environ.get("TTT_CHUNK_SIZE", 256))  # v7: per-document chunk size
    ttt_min_nll = bool(int(os.environ.get("TTT_MIN_NLL", "1")))  # v7b: min-NLL epoch selection (PR #611)
    ttt_k_lora = bool(int(os.environ.get("TTT_K_LORA", "1")))  # v7b: K-projection LoRA (PR #611)
    ttt_k_lora_lr_mult = float(os.environ.get("TTT_K_LORA_LR_MULT", 0.3))  # v7b: K-LoRA lr multiplier

    # v8: STP (Semantic Tube Prediction) auxiliary loss (arXiv:2602.22617)
    stp_lambda = float(os.environ.get("STP_LAMBDA", 0.0))  # 0.0 = disabled, try 0.02
    # v8: CROWN-Q — quantization-aware regularizer during warmdown (encourages flat basins)
    crown_q_lambda = float(os.environ.get("CROWN_Q_LAMBDA", 0.01))  # 0.0 = disabled

    # v8: N-gram eval interpolation (PRs #702, #706) — blend n-gram stats with neural logits at eval time
    ngram_eval = bool(int(os.environ.get("NGRAM_EVAL", "1")))
    ngram_max_order = int(os.environ.get("NGRAM_MAX_ORDER", 5))
    ngram_alpha = float(os.environ.get("NGRAM_ALPHA", 0.1))  # weight for n-gram component (0.1 = 90% neural)
    ngram_smoothing = float(os.environ.get("NGRAM_SMOOTHING", 0.01))  # add-k smoothing constant

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.04))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.02))  # v8: overnight sweep, lower lr for 12L stability
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.02))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.99))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.92))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 1500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    lora_lr = float(os.environ.get("LORA_LR", 0.01))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.3))
    swa_enabled = bool(int(os.environ.get("SWA_ENABLED", "0")))  # v6: EMA replaces SWA by default
    swa_start_frac = float(os.environ.get("SWA_START_FRAC", 0.2))  # tight SWA: only last 20% of warmdown
    swa_every = int(os.environ.get("SWA_EVERY", 50))  # collect checkpoint every N steps
    muon_weight_decay = float(os.environ.get("MUON_WEIGHT_DECAY", 0.04))

# v8: Progressive Sequence Length — parse schedule and compute current seq_len

def _parse_progressive_schedule(schedule_str: str) -> list[tuple[int, float]]:
    """Parse 'len:frac,len:frac,...' into sorted list of (seq_len, fraction) tuples."""
    pairs = []
    for entry in schedule_str.split(","):
        sl, frac = entry.strip().split(":")
        pairs.append((int(sl), float(frac)))
    return sorted(pairs, key=lambda x: x[1])

def get_progressive_seq_len(step: int, total_steps: int, schedule: list[tuple[int, float]], fallback: int) -> int:
    """Return the seq_len for the current step based on progressive schedule."""
    frac = step / max(total_steps, 1)
    for seq_len, threshold in schedule:
        if frac < threshold:
            return seq_len
    return fallback  # at or past the last threshold, use full length

# NorMuon optimizer (arXiv:2510.05491, PR #89)

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """NorMuon: Muon with per-row second-moment normalization.
    Drop-in replacement for the original Muon optimizer."""
    def __init__(self, params, lr: float, momentum: float, backend_steps: int,
                 nesterov: bool = True, beta2: float = 0.95, weight_decay: float = 0.0):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                 nesterov=nesterov, beta2=beta2, weight_decay=weight_decay),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            beta2 = group["beta2"]
            weight_decay = group["weight_decay"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                        state["second_momentum"] = torch.zeros(
                            g.size(0), 1, device=g.device, dtype=torch.float32
                        )
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g = g.to(dtype=p.grad.dtype)
                    # NorMuon: per-row second-moment normalization
                    vnorm = g.norm(dim=(-2, -1), keepdim=True)
                    v_mean = torch.mean(g * g, dim=-1, keepdim=True)
                    state["second_momentum"].lerp_(v_mean, 1 - beta2)
                    step_size = 1.0 / state["second_momentum"].sqrt().add_(1e-10)
                    g.mul_(step_size)
                    vnorm_new = g.norm(dim=(-2, -1), keepdim=True)
                    g.mul_(vnorm / vnorm_new.add_(1e-10))
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1).bfloat16()
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# Tokenizer-agnostic BPB evaluation (bring your own tokenizer)

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss = model(x, y).detach()
            batch_token_count = float(y.numel())
            val_loss_sum += batch_loss.to(torch.float64) * batch_token_count
            val_token_count += batch_token_count
            prev_ids = x.reshape(-1)
            tgt_ids = y.reshape(-1)
            token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# v8: N-gram eval interpolation (PRs #702, #706)
# Blends neural model log-probs with n-gram statistics at eval time.
# Zero training cost, zero artifact cost — tables built on-the-fly from scored tokens.

class NgramEvalMixer:
    """Eval-time n-gram interpolation. Maintains count tables built causally
    from already-scored tokens and blends n-gram log-probs with neural log-probs."""

    # Fixed interpolation weights per order (1-gram through 5-gram), a la PR #706.
    # These weight how much each n-gram order contributes to the n-gram distribution.
    ORDER_WEIGHTS = [0.6, 0.15, 0.1, 0.08, 0.07]

    def __init__(self, vocab_size: int = 1024, max_order: int = 5,
                 alpha: float = 0.1, smoothing: float = 0.01):
        self.vocab_size = vocab_size
        self.max_order = min(max_order, 5)
        self.alpha = alpha        # blend weight for n-gram component
        self.smoothing = smoothing  # add-k smoothing

        # Count tables — CPU only, transferred to GPU for blending
        self.unigram_counts = torch.zeros(vocab_size, dtype=torch.float32)
        self.bigram_counts: dict[int, torch.Tensor] = {}      # prev -> counts[V]
        self.trigram_counts: dict[int, torch.Tensor] = {}      # hash(prev2,prev1) -> counts[V]
        # 4-gram and 5-gram: hashed into fixed-size buckets
        self._4gram_counts: dict[int, torch.Tensor] = {}       # hash key -> counts[V]
        self._5gram_counts: dict[int, torch.Tensor] = {}       # hash key -> counts[V]
        self.total_tokens = 0

    def reset(self) -> None:
        """Clear all count tables (e.g. between documents)."""
        self.unigram_counts.zero_()
        self.bigram_counts.clear()
        self.trigram_counts.clear()
        self._4gram_counts.clear()
        self._5gram_counts.clear()
        self.total_tokens = 0

    @staticmethod
    def _hash_context(*tokens: int) -> int:
        """Simple hash for n-gram context tuples."""
        h = 0
        for t in tokens:
            h = ((h * 1000003) ^ t) & 0x7FFFFFFF
        return h

    def update(self, tokens: torch.Tensor) -> None:
        """Update n-gram tables with newly scored tokens. tokens: 1D CPU int tensor."""
        toks = tokens.cpu().tolist() if tokens.is_cuda else tokens.tolist()
        n = len(toks)
        for i in range(n):
            tid = toks[i]
            # Unigram
            self.unigram_counts[tid] += 1
            self.total_tokens += 1
            # Bigram
            if i >= 1 and self.max_order >= 2:
                key = toks[i - 1]
                if key not in self.bigram_counts:
                    self.bigram_counts[key] = torch.zeros(self.vocab_size, dtype=torch.float32)
                self.bigram_counts[key][tid] += 1
            # Trigram
            if i >= 2 and self.max_order >= 3:
                key = self._hash_context(toks[i - 2], toks[i - 1])
                if key not in self.trigram_counts:
                    self.trigram_counts[key] = torch.zeros(self.vocab_size, dtype=torch.float32)
                self.trigram_counts[key][tid] += 1
            # 4-gram
            if i >= 3 and self.max_order >= 4:
                key = self._hash_context(toks[i - 3], toks[i - 2], toks[i - 1])
                if key not in self._4gram_counts:
                    self._4gram_counts[key] = torch.zeros(self.vocab_size, dtype=torch.float32)
                self._4gram_counts[key][tid] += 1
            # 5-gram
            if i >= 4 and self.max_order >= 5:
                key = self._hash_context(toks[i - 4], toks[i - 3], toks[i - 2], toks[i - 1])
                if key not in self._5gram_counts:
                    self._5gram_counts[key] = torch.zeros(self.vocab_size, dtype=torch.float32)
                self._5gram_counts[key][tid] += 1

    def get_ngram_log_probs(self, context_tokens: list[int]) -> torch.Tensor:
        """Get smoothed n-gram log-probabilities using weighted backoff across orders.
        context_tokens: list of preceding token IDs (most recent last).
        Returns: log_probs tensor [vocab_size] on CPU."""
        k = self.smoothing
        V = self.vocab_size
        combined = torch.zeros(V, dtype=torch.float32)
        weight_sum = 0.0

        for order in range(1, self.max_order + 1):
            w = self.ORDER_WEIGHTS[order - 1] if order <= len(self.ORDER_WEIGHTS) else 0.0
            if w <= 0:
                continue
            counts = None
            total = 0.0

            if order == 1:
                counts = self.unigram_counts
                total = float(self.total_tokens)
            elif order == 2 and len(context_tokens) >= 1:
                key = context_tokens[-1]
                counts = self.bigram_counts.get(key)
                total = float(counts.sum().item()) if counts is not None else 0.0
            elif order == 3 and len(context_tokens) >= 2:
                key = self._hash_context(context_tokens[-2], context_tokens[-1])
                counts = self.trigram_counts.get(key)
                total = float(counts.sum().item()) if counts is not None else 0.0
            elif order == 4 and len(context_tokens) >= 3:
                key = self._hash_context(context_tokens[-3], context_tokens[-2], context_tokens[-1])
                counts = self._4gram_counts.get(key)
                total = float(counts.sum().item()) if counts is not None else 0.0
            elif order == 5 and len(context_tokens) >= 4:
                key = self._hash_context(context_tokens[-4], context_tokens[-3],
                                         context_tokens[-2], context_tokens[-1])
                counts = self._5gram_counts.get(key)
                total = float(counts.sum().item()) if counts is not None else 0.0

            if counts is not None and total > 0:
                # Add-k smoothed probabilities
                probs = (counts + k) / (total + k * V)
                combined += w * probs
                weight_sum += w
            else:
                # Fall back to uniform for this order
                combined += w * (1.0 / V)
                weight_sum += w

        if weight_sum > 0:
            combined /= weight_sum
        else:
            combined.fill_(1.0 / V)

        # Clamp to avoid log(0)
        combined.clamp_(min=1e-10)
        return combined.log()

    def blend_logits(self, neural_logits: torch.Tensor, context_tokens_batch: list[list[int]],
                     device: torch.device) -> torch.Tensor:
        """Blend neural logits with n-gram log-probs for a batch of positions.
        neural_logits: [batch, vocab] on GPU.
        context_tokens_batch: list of context token lists, one per batch element.
        Returns: blended log-probs [batch, vocab] on device."""
        neural_lp = F.log_softmax(neural_logits.float(), dim=-1)

        bsz = neural_logits.size(0)
        ngram_lp_batch = torch.zeros(bsz, self.vocab_size, dtype=torch.float32)
        for b in range(bsz):
            ngram_lp_batch[b] = self.get_ngram_log_probs(context_tokens_batch[b])
        ngram_lp_batch = ngram_lp_batch.to(device)

        # Log-domain interpolation: log(alpha * exp(ngram_lp) + (1-alpha) * exp(neural_lp))
        combined = torch.logaddexp(
            ngram_lp_batch + math.log(self.alpha),
            neural_lp + math.log(1 - self.alpha),
        )
        return combined

    def blend_per_token_loss(self, neural_logits: torch.Tensor, targets: torch.Tensor,
                             all_tokens: torch.Tensor, score_start: int, score_len: int,
                             device: torch.device) -> torch.Tensor:
        """Compute blended per-token NLL for a scored region.
        neural_logits: [seq_len, vocab] — logits for the full window.
        targets: [seq_len] — target token IDs for the full window.
        all_tokens: 1D — the full token stream (for context lookup).
        score_start: absolute position in all_tokens where scoring starts.
        score_len: number of tokens to score.
        Returns: per-token NLL [score_len] on device."""
        # Get neural log-probs for scored region
        scored_logits = neural_logits[-score_len:]  # [score_len, V]
        scored_targets = targets[-score_len:]        # [score_len]

        # Build context for each scored position
        all_toks_list = all_tokens.cpu().tolist() if all_tokens.is_cuda else all_tokens.tolist()
        ctx_list = []
        for i in range(score_len):
            abs_pos = score_start + i
            ctx_start = max(0, abs_pos - (self.max_order - 1))
            ctx_list.append(all_toks_list[ctx_start:abs_pos])

        blended_lp = self.blend_logits(scored_logits, ctx_list, device)
        # Per-token NLL from blended log-probs
        nll = -blended_lp[torch.arange(score_len, device=device), scored_targets]
        return nll


# Post-training int6 quantization + zstd compression

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0

# Int6 quantization: values in [-32, 31] stored in int8 containers.
# "Late-K passthrough" keeps last 2 layers' K projections in fp16.
INT6_RANGE_MAX = 31
INT6_RANGE_MIN = -31  # symmetric with INT6_RANGE_MAX
INT6_LATE_K_FP16_PATTERNS = ("blocks.9.attn.c_k.weight", "blocks.10.attn.c_k.weight")

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t

def quantize_float_tensor(t: Tensor, bits: int = 6) -> tuple[Tensor, Tensor]:
    """Per-row quantization at specified bit width with outlier clipping.
    Matches PR #114's proven approach. Stored in int8 containers."""
    max_val = (2 ** (bits - 1)) - 1  # 31 for int6, 127 for int8
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / float(max_val)).clamp_min(1e-12)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -max_val, max_val).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / float(max_val) if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -max_val, max_val).to(torch.int8).contiguous()
    return q, scale

def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        # tok_emb.weight always kept in fp16 (embedding table quality matters).
        # Late-K passthrough: last 2 layers' K projections kept in fp16.
        force_fp16 = (name == "tok_emb.weight" or any(p in name for p in INT6_LATE_K_FP16_PATTERNS))
        if force_fp16:
            kept = t.to(dtype=torch.float16).contiguous()
            passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int6_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        if qmeta.get(name, {}).get("scheme") == "per_row" or s.ndim > 0:
            s = s.to(dtype=torch.float32)
            # Broadcast the saved row scale back across trailing dimensions.
            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out



# ============================================================
# v5: Novel compression pipeline — per-tensor codebook + Huffman
# ============================================================
import heapq
import struct
import json as _json

# Codebook levels per tensor type (from 40 experiments on beastmode)
CODEBOOK_MLP = int(os.environ.get("CODEBOOK_MLP", 48))
CODEBOOK_ATTN_QKV = int(os.environ.get("CODEBOOK_ATTN_QKV", 80))
CODEBOOK_ATTN_PROJ = int(os.environ.get("CODEBOOK_ATTN_PROJ", 64))
USE_NOVEL_COMPRESSION = bool(int(os.environ.get("USE_NOVEL_COMPRESSION", "1")))
# v8: Codebook-GPTQ hybrid — Hessian-weighted error compensation after codebook quantization
USE_GPTQ = bool(int(os.environ.get("USE_GPTQ", "1")))
GPTQ_CALIB_SEQS = int(os.environ.get("GPTQ_CALIB_SEQS", 256))  # calibration sequences for Hessian


def _get_codebook_levels(name: str) -> int:
    """Determine codebook size based on tensor type."""
    if "mlp.fc" in name or "mlp.proj" in name:
        return CODEBOOK_MLP
    elif "attn.c_q" in name or "attn.c_k" in name or "attn.c_v" in name:
        return CODEBOOK_ATTN_QKV
    elif "attn.proj" in name:
        return CODEBOOK_ATTN_PROJ
    return 64  # default


def codebook_quantize_tensor(t: Tensor, n_levels: int, max_iter: int = 20) -> tuple[Tensor, Tensor]:
    """Per-tensor k-means codebook quantization. Returns (indices, codebook)."""
    flat = t.float().flatten()
    n = flat.numel()
    vmin, vmax = float(flat.min()), float(flat.max())
    if vmin == vmax:
        return torch.zeros(t.shape, dtype=torch.int8), torch.full((n_levels,), vmin, dtype=torch.float16)
    codebook = torch.linspace(vmin, vmax, n_levels, device=flat.device)
    for _ in range(max_iter):
        diffs = (flat.unsqueeze(1) - codebook.unsqueeze(0)).abs()
        indices = diffs.argmin(dim=1)
        new_codebook = torch.zeros(n_levels, device=flat.device)
        for j in range(n_levels):
            mask = indices == j
            if mask.any():
                new_codebook[j] = flat[mask].mean()
            else:
                new_codebook[j] = codebook[j]
        if torch.allclose(codebook, new_codebook, atol=1e-8):
            break
        codebook = new_codebook
    diffs = (flat.unsqueeze(1) - codebook.unsqueeze(0)).abs()
    indices = diffs.argmin(dim=1).to(torch.int8).reshape(t.shape)
    return indices, codebook.to(torch.float16)


def codebook_dequantize_tensor(indices: Tensor, codebook: Tensor, dtype: torch.dtype) -> Tensor:
    """Reverse codebook quantization."""
    return codebook.float()[indices.long()].to(dtype)


# v8: Codebook-GPTQ hybrid compression
def codebook_gptq_quantize(W: Tensor, codebook: Tensor, H: Tensor) -> Tensor:
    """Column-by-column GPTQ with codebook centroids.

    After k-means finds the codebook, GPTQ processes columns left-to-right:
    1. Quantize column j to nearest codebook centroid
    2. Compute quantization error
    3. Distribute error to remaining columns weighted by Hessian structure

    Args:
        W: Weight tensor [out_features, in_features], float32
        codebook: Codebook centroids [n_levels], float32
        H: Approximate Hessian (X^T @ X) [in_features, in_features], float32

    Returns:
        W_q: GPTQ-adjusted quantized weight tensor [out_features, in_features], float32
    """
    W_q = W.clone()
    n_cols = W_q.shape[1]
    # Add small damping to Hessian diagonal for numerical stability
    diag = H.diag()
    damp = 0.01 * diag.mean()
    H_damped_diag = diag + damp

    for col in range(n_cols):
        # Quantize column to nearest codebook centroid
        col_vals = W_q[:, col]
        diffs = (col_vals.unsqueeze(1) - codebook.unsqueeze(0)).abs()
        nearest_idx = diffs.argmin(dim=1)
        W_q_col = codebook[nearest_idx]

        # Compute per-element quantization error
        error = col_vals - W_q_col

        # Distribute error to remaining columns via Hessian weights
        if col + 1 < n_cols:
            h_diag = H_damped_diag[col]
            h_row = H[col, col + 1:]  # [remaining_cols]
            scale = h_row / (h_diag + 1e-6)  # [remaining_cols]
            # error: [out_features], scale: [remaining_cols] -> outer product update
            W_q[:, col + 1:] += error.unsqueeze(1) * scale.unsqueeze(0)

        W_q[:, col] = W_q_col

    return W_q


def collect_gptq_hessians(
    model: nn.Module,
    val_tokens: Tensor,
    device: torch.device,
    n_seqs: int = 256,
    seq_len: int = 2048,
) -> dict[str, Tensor]:
    """Collect approximate Hessians (X^T @ X) for each CastedLinear layer.

    Runs calibration sequences through the model and accumulates input
    activations per linear layer. Returns dict mapping state_dict weight
    name -> Hessian on CPU in float32.

    Memory-safe: processes one layer's Hessian at a time via hooks.
    """
    # Build map from nn.Module -> state_dict name for CastedLinear weights
    linear_names: dict[nn.Module, str] = {}
    for name, module in model.named_modules():
        if isinstance(module, CastedLinear) and module.weight.ndim == 2 and module.weight.numel() > INT8_KEEP_FLOAT_MAX_NUMEL:
            sd_name = name + ".weight"
            # Skip passthrough tensors (tok_emb, late-K, small tensors)
            if sd_name == "tok_emb.weight" or any(p in sd_name for p in INT6_LATE_K_FP16_PATTERNS):
                continue
            linear_names[module] = sd_name

    if not linear_names:
        return {}

    # Prepare calibration data from validation tokens
    total_tokens = val_tokens.numel() - 1
    usable_seqs = total_tokens // seq_len
    actual_seqs = min(n_seqs, usable_seqs)
    if actual_seqs <= 0:
        return {}

    # Accumulate H = X^T @ X for each linear layer
    hessians: dict[str, Tensor] = {}
    sample_counts: dict[str, int] = {}
    hooks = []

    def make_hook(sd_name: str, in_features: int):
        def hook_fn(module, input, output):
            x = input[0].detach().float()  # [bsz, seq, in_features]
            x_2d = x.reshape(-1, in_features)  # [bsz*seq, in_features]
            xtx = x_2d.T @ x_2d  # [in_features, in_features]
            if sd_name not in hessians:
                hessians[sd_name] = torch.zeros(in_features, in_features, dtype=torch.float32, device=device)
                sample_counts[sd_name] = 0
            hessians[sd_name] += xtx
            sample_counts[sd_name] += x_2d.shape[0]
        return hook_fn

    # Register hooks
    for module, sd_name in linear_names.items():
        in_features = module.weight.shape[1]
        h = module.register_forward_hook(make_hook(sd_name, in_features))
        hooks.append(h)

    # Run calibration sequences through model
    model.eval()
    calib_batch_size = min(8, actual_seqs)  # small batches to save memory
    with torch.inference_mode():
        for batch_start in range(0, actual_seqs, calib_batch_size):
            batch_end = min(batch_start + calib_batch_size, actual_seqs)
            x_list = []
            y_list = []
            for i in range(batch_start, batch_end):
                start = i * seq_len
                chunk = val_tokens[start:start + seq_len + 1].to(dtype=torch.int64, device=device)
                x_list.append(chunk[:-1])
                y_list.append(chunk[1:])
            x_batch = torch.stack(x_list)
            y_batch = torch.stack(y_list)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                model(x_batch, y_batch)
            del x_batch, y_batch, x_list, y_list

    # Remove hooks
    for h in hooks:
        h.remove()

    # Normalize and move to CPU
    result: dict[str, Tensor] = {}
    for sd_name, H in hessians.items():
        n = sample_counts[sd_name]
        if n > 0:
            result[sd_name] = (H / n).cpu()
        del H
    del hessians
    torch.cuda.empty_cache()

    model.train()
    return result


class _HuffNode:
    def __init__(self, v=None, f=0, l=None, r=None):
        self.v, self.f, self.l, self.r = v, f, l, r
    def __lt__(self, o): return self.f < o.f


def huffman_encode_tensor(values: Tensor) -> tuple[bytes, dict, int]:
    """Huffman-encode a flat int tensor. Returns (encoded_bytes, code_table, n_bits)."""
    flat = values.flatten().tolist()
    freq = {}
    for v in flat:
        freq[v] = freq.get(v, 0) + 1
    heap = [_HuffNode(v=v, f=f) for v, f in freq.items() if f > 0]
    heapq.heapify(heap)
    if len(heap) == 1:
        n = heapq.heappop(heap)
        root = _HuffNode(l=n, r=_HuffNode(v=-999, f=0), f=n.f)
    else:
        while len(heap) > 1:
            l, r = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(heap, _HuffNode(l=l, r=r, f=l.f + r.f))
        root = heap[0]
    codes = {}
    def _build(node, prefix=""):
        if node.v is not None:
            codes[node.v] = prefix or "0"
            return
        if node.l: _build(node.l, prefix + "0")
        if node.r: _build(node.r, prefix + "1")
    _build(root)
    bits = "".join(codes[v] for v in flat)
    n_bits = len(bits)
    pad = (8 - n_bits % 8) % 8
    bits += "0" * pad
    encoded = bytearray(int(bits[i:i+8], 2) for i in range(0, len(bits), 8))
    return bytes(encoded), {str(k): v for k, v in codes.items()}, n_bits


def huffman_decode_tensor(data: bytes, codes: dict, n_bits: int, shape: list) -> Tensor:
    """Decode Huffman-encoded data back to int tensor."""
    decode_table = {v: int(k) for k, v in codes.items()}
    bits = "".join(f"{b:08b}" for b in data)[:n_bits]
    values = []
    current = ""
    for bit in bits:
        current += bit
        if current in decode_table:
            values.append(decode_table[current])
            current = ""
    return torch.tensor(values, dtype=torch.int8).reshape(shape)


PCLL_MAGIC = b"PCLL"


def save_novel_format(state_dict: dict[str, Tensor], output_path: str, hessians: dict[str, Tensor] | None = None) -> int:
    """Save model using our novel compression pipeline."""
    header = {"version": 1, "tensors": {}}
    data_parts = []
    offset = 0

    for name, tensor in state_dict.items():
        t = tensor.detach().cpu()
        # Passthrough small/non-float tensors
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL or t.ndim < 2 or not t.is_floating_point():
            force_fp16 = (name == "tok_emb.weight" or any(p in name for p in INT6_LATE_K_FP16_PATTERNS))
            if force_fp16 or any(p in name for p in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
                raw = t.float().numpy().tobytes() if any(p in name for p in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS) else t.half().numpy().tobytes()
            else:
                raw = t.half().numpy().tobytes()
            entry = {"method": "passthrough", "offset": offset, "len": len(raw),
                     "shape": list(t.shape), "dtype": "float32" if any(p in name for p in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS) else "float16",
                     "numel": t.numel()}
            header["tensors"][name] = entry
            data_parts.append(raw)
            offset += len(raw)
            continue

        # Codebook quantize
        n_levels = _get_codebook_levels(name)
        indices, codebook = codebook_quantize_tensor(t, n_levels)

        # v8: GPTQ error compensation — adjust weights using Hessian before final index assignment
        if USE_GPTQ and hessians is not None and name in hessians:
            W_gptq = codebook_gptq_quantize(t.float(), codebook.float(), hessians[name])
            # Re-compute indices after GPTQ adjustment (centroids stay the same)
            flat_gptq = W_gptq.flatten()
            diffs = (flat_gptq.unsqueeze(1) - codebook.float().unsqueeze(0)).abs()
            indices = diffs.argmin(dim=1).to(torch.int8).reshape(t.shape)
            del W_gptq, flat_gptq, diffs

        encoded, codes, n_bits = huffman_encode_tensor(indices)
        cb_bytes = codebook.numpy().tobytes()
        ht_bytes = _json.dumps(codes, separators=(",", ":")).encode("utf-8")

        entry = {
            "method": "codebook_huffman", "offset": offset,
            "data_len": len(encoded), "cb_len": len(cb_bytes), "ht_len": len(ht_bytes),
            "n_bits": n_bits, "n_levels": n_levels,
            "shape": list(t.shape), "dtype": str(t.dtype).replace("torch.", ""),
            "numel": t.numel(),
        }
        blob = encoded + cb_bytes + ht_bytes
        entry["total_len"] = len(blob)
        header["tensors"][name] = entry
        data_parts.append(blob)
        offset += len(blob)

    header_bytes = _json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(output_path, "wb") as f:
        f.write(PCLL_MAGIC)
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)
        for part in data_parts:
            f.write(part)
    return os.path.getsize(output_path)


def load_novel_format(input_path: str) -> dict[str, Tensor]:
    """Load model from our novel compression format."""
    with open(input_path, "rb") as f:
        magic = f.read(4)
        assert magic == PCLL_MAGIC, f"Invalid magic: {magic}"
        header_len = struct.unpack("<I", f.read(4))[0]
        header = _json.loads(f.read(header_len).decode("utf-8"))
        data = f.read()

    state_dict = {}
    for name, entry in header["tensors"].items():
        if entry["method"] == "passthrough":
            dtype = getattr(torch, entry["dtype"])
            raw = data[entry["offset"]:entry["offset"] + entry["len"]]
            arr = np.frombuffer(raw, dtype=np.float16 if entry["dtype"] == "float16" else np.float32)
            state_dict[name] = torch.from_numpy(arr.copy()).reshape(entry["shape"]).to(dtype)
        elif entry["method"] == "codebook_huffman":
            base = entry["offset"]
            encoded = data[base:base + entry["data_len"]]
            cb_raw = data[base + entry["data_len"]:base + entry["data_len"] + entry["cb_len"]]
            ht_raw = data[base + entry["data_len"] + entry["cb_len"]:base + entry["total_len"]]
            codebook = torch.from_numpy(np.frombuffer(cb_raw, dtype=np.float16).copy())
            codes = _json.loads(ht_raw.decode("utf-8"))
            indices = huffman_decode_tensor(encoded, codes, entry["n_bits"], entry["shape"])
            dtype = getattr(torch, entry["dtype"])
            state_dict[name] = codebook_dequantize_tensor(indices, codebook, dtype)
    return state_dict


def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


def fake_quantize_int8_per_row(w: Tensor) -> Tensor:
    """Simulate per-row int8 quantization with STE."""
    scale = w.detach().abs().amax(dim=-1, keepdim=True).div_(127.0).clamp_(min=1.0 / 127.0)
    w_deq = (w / scale).round().clamp_(-127, 127) * scale
    return w + (w_deq - w).detach()


def fake_quantize_int6_per_row(w: Tensor) -> Tensor:
    """v5: Simulate per-row int6 quantization with STE for Late QAT."""
    with torch.no_grad():
        w32 = w.float()
        row_max = w32.abs().amax(dim=-1, keepdim=True)
        scale = (row_max / 31.0).clamp_min(1e-12)
        w_q = torch.clamp(torch.round(w32 / scale), -31, 31) * scale
    return w + (w_q - w).detach()  # STE: forward uses quantized, backward flows through original


def soft_round(x: Tensor, alpha: float) -> Tensor:
    """Differentiable rounding via tanh approximation (PR #606, EthanYangTW).
    When alpha is small (~1), gradients flow smoothly through rounding.
    When alpha is large (~16), converges to standard round()."""
    floor_x = x.floor()
    r = x - floor_x  # fractional part, in [0, 1)
    # tanh(alpha/2) is a scalar constant for the given alpha
    tanh_half = math.tanh(alpha / 2.0)
    return floor_x + 0.5 * torch.tanh(alpha * (r - 0.5)) / tanh_half + 0.5


def fake_quantize_int6_soft_round(w: Tensor, alpha: float) -> Tensor:
    """v7: Simulate per-row int6 quantization with differentiable soft-round for QAT."""
    w32 = w.float()
    with torch.no_grad():
        row_max = w32.abs().amax(dim=-1, keepdim=True)
        scale = (row_max / 31.0).clamp_min(1e-12)
    # Scale to quantization grid, apply soft_round (differentiable), clamp, scale back
    y = w32 / scale
    y_q = soft_round(y, alpha)
    y_q = torch.clamp(y_q, -31, 31)
    return (y_q * scale).to(w.dtype)


class CastedLinear(nn.Linear):
    _qat: bool = False
    _qat_enabled: bool = False  # v5: toggled by training loop when lr_scale < threshold
    _qat_mode: str = "soft_round"  # v7: "ste" or "soft_round"
    _qat_alpha: float = 1.0  # v7: current soft-round alpha (annealed by training loop)

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        if self._qat and self.training:
            w = fake_quantize_int8_per_row(w)
        elif CastedLinear._qat_enabled and self.training and w.ndim == 2 and w.numel() > 65536:
            if CastedLinear._qat_mode == "soft_round":
                w = fake_quantize_int6_soft_round(w, CastedLinear._qat_alpha)
            else:
                w = fake_quantize_int6_per_row(w)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w.to(x.dtype), bias)


def compute_crown_q_loss(model: nn.Module, clip_range: int = 15) -> Tensor:
    """CROWN-Q: quantization-aware regularizer that encourages flat weight basins.

    Penalizes mean(w^2) * delta^2 / 12 per row, where delta = row_max / clip_range.
    Uses clip_range=15 (coarser than int6's 31) to over-penalize into flatter basins.
    Only considers CastedLinear weights with >1000 elements."""
    penalties = []
    for module in model.modules():
        if isinstance(module, CastedLinear) and module.weight.numel() > 1000:
            w = module.weight
            with torch.no_grad():
                row_max = w.detach().abs().amax(dim=1)
                delta = row_max / clip_range  # coarser than actual int6 quantizer
            # w^2 needs gradients; delta is detached
            penalty = (w * w).mean(dim=1) * (delta * delta) / 12.0
            penalties.append(penalty.mean())
    if not penalties:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return torch.stack(penalties).mean()


class AttentionLoRA(nn.Module):
    """Per-iteration LoRA adapters for attention projections (zero-init B matrices)."""
    def __init__(self, dim: int, kv_dim: int, rank: int):
        super().__init__()
        self.q_A = nn.Parameter(torch.empty(dim, rank))
        self.q_B = nn.Parameter(torch.zeros(rank, dim))
        self.k_A = nn.Parameter(torch.empty(dim, rank))
        self.k_B = nn.Parameter(torch.zeros(rank, kv_dim))
        self.v_A = nn.Parameter(torch.empty(dim, rank))
        self.v_B = nn.Parameter(torch.zeros(rank, kv_dim))
        self.proj_A = nn.Parameter(torch.empty(dim, rank))
        self.proj_B = nn.Parameter(torch.zeros(rank, dim))
        self._init_lora()

    def _init_lora(self) -> None:
        for name in ("q_A", "k_A", "v_A", "proj_A"):
            nn.init.kaiming_uniform_(getattr(self, name), a=math.sqrt(5))


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, rope_dims: int = 0):
        super().__init__()
        # v5: Partial RoPE — only first rope_dims dimensions get positional encoding
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    """Apply rotary embeddings. If rope_dims < x.size(-1), only rotate first rope_dims dims."""
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope = x[..., :rope_dims]
        x_pass = x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rotated = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rotated, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        rope_dims: int = 0,
        use_xsa: bool = False,
        use_gated_attention: bool = False,
        use_value_residual: bool = False,
        layer_idx: int = 0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.rope_dims = rope_dims
        self.use_xsa = use_xsa
        self._layer_idx = layer_idx
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base, rope_dims=rope_dims)
        # v6: Gated Attention — per-head sigmoid gate after SDPA
        self.use_gated_attention = use_gated_attention
        if use_gated_attention:
            self.attn_gate = CastedLinear(dim, num_heads, bias=True)
            nn.init.zeros_(self.attn_gate.weight)
            nn.init.constant_(self.attn_gate.bias, 4.0)  # near-open at init
        # v6: Value Residual — learned blend with layer-0 V
        self.use_value_residual = use_value_residual
        if use_value_residual and layer_idx > 0:
            self.v_lambda = nn.Parameter(torch.tensor([0.5, 0.5], dtype=torch.float32))

    def _xsa_subtract(self, y: Tensor, v: Tensor) -> Tensor:
        """XSA: subtract self-value projection. GQA-aware, zero-alloc."""
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x: Tensor, lora: AttentionLoRA | None = None,
                q_delta: Tensor | None = None, v_delta: Tensor | None = None,
                k_delta: Tensor | None = None) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)
        if lora is not None:
            q = q + (x @ lora.q_A) @ lora.q_B
            k = k + (x @ lora.k_A) @ lora.k_B
            v = v + (x @ lora.v_A) @ lora.v_B
        # v7: LoRA TTT deltas (pre-computed, per-batch-element)
        if q_delta is not None:
            q = q + q_delta
        if k_delta is not None:
            k = k + k_delta
        if v_delta is not None:
            v = v + v_delta
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # v6: Value Residual — cache v0 from layer 0, blend in subsequent layers
        if self.use_value_residual and hasattr(self, '_gpt_ref') and self._gpt_ref is not None:
            if self._layer_idx == 0:
                self._gpt_ref._v0_cache = v.detach()
            elif self._gpt_ref._v0_cache is not None and hasattr(self, 'v_lambda'):
                lam = self.v_lambda.to(dtype=v.dtype)
                v = lam[0] * self._gpt_ref._v0_cache.to(dtype=v.dtype) + lam[1] * v
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, rope_dims=self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, rope_dims=self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )
        # v6: Gated Attention — per-head sigmoid gate
        if self.use_gated_attention:
            gate = torch.sigmoid(self.attn_gate(x))  # [B, T, num_heads]
            gate = gate.transpose(1, 2).unsqueeze(-1)  # [B, num_heads, T, 1]
            y = y * gate
        # v5: XSA — subtract self-value projection to remove self-bias
        if self.use_xsa:
            y_bhsd = y.transpose(1, 2)  # [B, T, H, D]
            v_bhsd = v.transpose(1, 2)  # [B, T, Hkv, D]
            y_xsa = self._xsa_subtract(y_bhsd, v_bhsd)
            y = y_xsa.reshape(bsz, seqlen, dim)
        else:
            y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        out = self.proj(y)
        if lora is not None:
            out = out + (y @ lora.proj_A) @ lora.proj_B
        return out


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, mlp_hidden: int = 0):
        super().__init__()
        hidden = mlp_hidden if mlp_hidden > 0 else mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True

    def forward(self, x: Tensor) -> Tensor:
        x = F.leaky_relu(self.fc(x), negative_slope=0.5)  # v6b: LeakyReLU(0.5)² per PR #518
        return self.proj(x.square())


class SmearGate(nn.Module):
    """Learned temporal smoothing: x = (1-gate)*x + gate*x_prev, gate=sigmoid(param)."""
    def __init__(self, dim: int):
        super().__init__()
        self.gate_logit = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate_logit.to(dtype=x.dtype))[None, None, :]
        x_prev = F.pad(x[:, :-1, :], (0, 0, 1, 0))  # shift right, pad with zeros
        return (1 - gate) * x + gate * x_prev


class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(bigram_dim, model_dim, bias=False)
        self.proj._zero_init = True
        nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens):
        t = tokens.to(torch.int32)
        mod = self.bigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod
        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod
        return out.long()

    def forward(self, token_ids):
        h = self.embed(self.bigram_hash(token_ids))
        h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class TrigramHashEmbedding(nn.Module):
    """v6: 3-token context hash embedding, extending BigramHash to trigrams."""
    def __init__(self, trigram_vocab_size: int, trigram_dim: int, model_dim: int):
        super().__init__()
        self.trigram_vocab_size = trigram_vocab_size
        self.embed = nn.Embedding(trigram_vocab_size, trigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(trigram_dim, model_dim, bias=False)
        self.proj._zero_init = True
        nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def trigram_hash(self, tokens):
        t = tokens.to(torch.int32)
        mod = self.trigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = mod  # sentinel for position 0
        out[..., 1] = mod  # sentinel for position 1
        out[..., 2:] = (
            torch.bitwise_xor(
                torch.bitwise_xor(36313 * t[..., 2:], 27191 * t[..., 1:-1]),
                51497 * t[..., :-2],
            )
            % mod
        )
        return out.long()

    def forward(self, token_ids):
        h = self.embed(self.trigram_hash(token_ids))
        h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class QuadgramHashEmbedding(nn.Module):
    """v7: 4-token context hash embedding, extending TrigramHash."""
    def __init__(self, quadgram_vocab_size: int, quadgram_dim: int, model_dim: int):
        super().__init__()
        self.quadgram_vocab_size = quadgram_vocab_size
        self.embed = nn.Embedding(quadgram_vocab_size, quadgram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(quadgram_dim, model_dim, bias=False)
        self.proj._zero_init = True
        nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def quadgram_hash(self, tokens):
        t = tokens.to(torch.int32)
        mod = self.quadgram_vocab_size
        out = torch.full_like(t, mod - 1)  # sentinel for positions 0-2
        if t.shape[-1] > 3:
            out[..., 3:] = (104723 * t[..., :-3] + 9533 * t[..., 1:-2] + 97 * t[..., 2:-1] + t[..., 3:]) % mod
        return out.long()

    def forward(self, token_ids):
        h = self.embed(self.quadgram_hash(token_ids))
        h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class SharedSparseSidecar(nn.Module):
    """v7: Lightweight shared refinement module applied at late layers.
    Zero-initialized so it starts as a no-op and learns its contribution.
    Based on PR #555 (ymrohit, 1.0916 BPB)."""
    def __init__(self, dim: int, hidden_dim: int, num_sites: int):
        super().__init__()
        self.gate = CastedLinear(dim, hidden_dim, bias=False)
        self.value = CastedLinear(dim, hidden_dim, bias=False)
        self.local = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, groups=hidden_dim, bias=False)
        self.proj = CastedLinear(hidden_dim, dim, bias=False)
        self.proj._zero_init = True
        self.site_gate = nn.Embedding(num_sites, hidden_dim)
        nn.init.zeros_(self.site_gate.weight)

    def forward(self, x: Tensor, site_idx: int) -> Tensor:
        gate_pre = self.gate(x) + self.site_gate.weight[site_idx].to(dtype=x.dtype)[None, None, :]
        gate = F.silu(gate_pre)
        value = self.value(x).transpose(1, 2)
        value = self.local(F.pad(value, (2, 0))).transpose(1, 2)
        return self.proj(gate * value)


BOS_ID = 1  # SentencePiece BOS token ID


class BatchedLinearLoRA(nn.Module):
    """Per-batch-element LoRA adapter for TTT. Delta = x @ A^T @ B^T.
    B is zero-initialized so delta starts at zero."""
    def __init__(self, bsz: int, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.in_features = in_features
        self.A = nn.Parameter(torch.empty(bsz, rank, in_features))
        self.B = nn.Parameter(torch.zeros(bsz, out_features, rank))
        self.reset()

    def forward(self, x: Tensor) -> Tensor:
        # Cast A/B to input dtype (handles bfloat16 model + float32 LoRA)
        return (x @ self.A.to(dtype=x.dtype).transpose(1, 2)) @ self.B.to(dtype=x.dtype).transpose(1, 2)

    def reset(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad():
            self.A.uniform_(-bound, bound)
            self.B.zero_()


class BatchedTTTLoRA(nn.Module):
    """All LoRA adapters for one batch of documents: Q + V + optional K per block + LM head."""
    def __init__(self, bsz: int, model: nn.Module, rank: int, use_k_lora: bool = False):
        super().__init__()
        self.use_k_lora = use_k_lora
        dim = model.tok_emb.embedding_dim
        vocab = model.tok_emb.num_embeddings
        self.lm_head_lora = BatchedLinearLoRA(bsz, dim, vocab, rank)
        self.q_loras = nn.ModuleList()
        self.v_loras = nn.ModuleList()
        self.k_loras = nn.ModuleList() if use_k_lora else None
        for block in model.blocks:
            q_out = block.attn.c_q.weight.shape[0]
            v_out = block.attn.c_v.weight.shape[0]
            k_out = block.attn.c_k.weight.shape[0]
            self.q_loras.append(BatchedLinearLoRA(bsz, dim, q_out, rank))
            self.v_loras.append(BatchedLinearLoRA(bsz, dim, v_out, rank))
            if use_k_lora:
                self.k_loras.append(BatchedLinearLoRA(bsz, dim, k_out, rank))

    def reset(self) -> None:
        for m in self.modules():
            if isinstance(m, BatchedLinearLoRA):
                m.reset()


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        mlp_hidden: int = 0,
        layer_idx: int = 0,
        rope_dims: int = 0,
        use_xsa: bool = False,
        ln_scale: bool = False,
        use_gated_attention: bool = False,
        use_value_residual: bool = False,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init,
            rope_dims=rope_dims, use_xsa=use_xsa,
            use_gated_attention=use_gated_attention,
            use_value_residual=use_value_residual,
            layer_idx=layer_idx,
        )
        self.mlp = MLP(dim, mlp_mult, mlp_hidden=mlp_hidden)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        # v5: LN Scale — deeper layers get smaller norm scale
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

    def forward(self, x: Tensor, x0: Tensor, lora: AttentionLoRA | None = None,
                q_delta_fn=None, v_delta_fn=None, k_delta_fn=None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_input = self.attn_norm(x)
        if self.ln_scale_factor != 1.0:
            attn_input = attn_input * self.ln_scale_factor
        # v7: LoRA TTT deltas computed from normed attention input
        qd = q_delta_fn(attn_input) if q_delta_fn is not None else None
        vd = v_delta_fn(attn_input) if v_delta_fn is not None else None
        kd = k_delta_fn(attn_input) if k_delta_fn is not None else None
        attn_out = self.attn(attn_input, lora=lora, q_delta=qd, v_delta=vd, k_delta=kd)
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        mlp_input = self.mlp_norm(x)
        if self.ln_scale_factor != 1.0:
            mlp_input = mlp_input * self.ln_scale_factor
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(mlp_input)
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        mlp_hidden: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        num_loops: int = 1,
        lora_rank: int = 0,
        bigram_vocab_size: int = 2048,
        bigram_dim: int = 128,
        rope_dims: int = 0,
        xsa_layers: int = 0,
        ln_scale: bool = False,
        value_residual: bool = False,
        gated_attention: bool = False,
        use_trigram: bool = False,
        trigram_vocab_size: int = 4096,
        trigram_dim: int = 128,
        use_quadgram: bool = False,
        quadgram_vocab_size: int = 1024,
        quadgram_dim: int = 32,
        sparse_start_layer: int = 8,
        sparse_hidden_dim: int = 48,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.num_unique_layers = num_layers
        self.num_loops = num_loops
        effective_depth = num_layers * num_loops
        self.value_residual = value_residual
        self._v0_cache = None  # v6: mutable cache for value residual
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim)
        self.trigram = TrigramHashEmbedding(trigram_vocab_size, trigram_dim, model_dim) if use_trigram else None
        self.quadgram = QuadgramHashEmbedding(quadgram_vocab_size, quadgram_dim, model_dim) if use_quadgram else None
        self.num_encoder_layers = effective_depth // 2
        self.num_decoder_layers = effective_depth - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    mlp_hidden=mlp_hidden,
                    layer_idx=i,
                    rope_dims=rope_dims,
                    use_xsa=(i >= num_layers - xsa_layers),  # XSA on last N layers
                    ln_scale=ln_scale,
                    use_gated_attention=gated_attention,
                    use_value_residual=value_residual,
                )
                for i in range(num_layers)
            ]
        )
        # Per-(loop, block) LoRA adapters for attention projections.
        # Only created when num_loops > 1 and lora_rank > 0.
        kv_dim = num_kv_heads * (model_dim // num_heads)
        if lora_rank > 0 and num_loops > 1:
            self.lora_adapters = nn.ModuleList(
                [
                    nn.ModuleList(
                        [AttentionLoRA(model_dim, kv_dim, lora_rank) for _ in range(num_layers)]
                    )
                    for _ in range(num_loops)
                ]
            )
        else:
            self.lora_adapters = None
        self.smear_gate = SmearGate(model_dim)
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        # v7: Shared Sparse Sidecar (applied at late layers)
        self.sparse_layer_indices = tuple(
            i for i in range(num_layers)
            if sparse_hidden_dim > 0 and i >= sparse_start_layer
        )
        self.sparse_site_lut = {idx: site for site, idx in enumerate(self.sparse_layer_indices)}
        self.shared_sparse = (
            SharedSparseSidecar(model_dim, sparse_hidden_dim, len(self.sparse_layer_indices))
            if self.sparse_layer_indices else None
        )
        self.shared_sparse_norm = RMSNorm() if self.sparse_layer_indices else None
        self.shared_sparse_scale = (
            nn.Parameter(torch.zeros(len(self.sparse_layer_indices), model_dim, dtype=torch.float32))
            if self.sparse_layer_indices else None
        )
        # v6: set up back-references for value residual v0 cache
        # Use object.__setattr__ to bypass nn.Module registration (avoids circular module tree)
        if value_residual:
            for block in self.blocks:
                object.__setattr__(block.attn, '_gpt_ref', self)
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _apply_sidecar(self, x: Tensor, eff_idx: int) -> Tensor:
        """Apply shared sparse sidecar if this layer is a sidecar site."""
        site_idx = self.sparse_site_lut.get(eff_idx)
        if site_idx is None or self.shared_sparse is None:
            return x
        sidecar_out = self.shared_sparse(self.shared_sparse_norm(x), site_idx)
        return x + self.shared_sparse_scale[site_idx].to(dtype=x.dtype)[None, None, :] * sidecar_out

    def forward(self, input_ids: Tensor, target_ids: Tensor, ttt_lora=None, stp_lambda: float = 0.0,
                return_logits: bool = False) -> Tensor:
        """Forward pass. When ttt_lora is provided, returns per-token loss [bsz, seq] for LoRA TTT.
        If return_logits=True with ttt_lora, returns logits [bsz, seq, vocab] instead of loss."""
        self._v0_cache = None
        x = self.tok_emb(input_ids)
        x = x + self.bigram(input_ids)
        if self.trigram is not None:
            x = x + self.trigram(input_ids)
        if self.quadgram is not None:
            x = x + self.quadgram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear_gate(x)
        x0 = x
        skips: list[Tensor] = []

        eff_idx = 0
        for loop_idx in range(self.num_loops):
            for block_idx in range(self.num_unique_layers):
                lora = self.lora_adapters[loop_idx][block_idx] if self.lora_adapters is not None else None
                # v7: LoRA TTT delta functions for this block
                qd_fn = ttt_lora.q_loras[block_idx] if ttt_lora is not None else None
                vd_fn = ttt_lora.v_loras[block_idx] if ttt_lora is not None else None
                kd_fn = ttt_lora.k_loras[block_idx] if (ttt_lora is not None and ttt_lora.k_loras is not None) else None
                if eff_idx < self.num_encoder_layers:
                    x = self.blocks[block_idx](x, x0, lora=lora, q_delta_fn=qd_fn, v_delta_fn=vd_fn, k_delta_fn=kd_fn)
                    skips.append(x)
                else:
                    dec_idx = eff_idx - self.num_encoder_layers
                    if dec_idx < self.num_skip_weights and skips:
                        x = x + self.skip_weights[dec_idx].to(dtype=x.dtype)[None, None, :] * skips.pop()
                    x = self.blocks[block_idx](x, x0, lora=lora, q_delta_fn=qd_fn, v_delta_fn=vd_fn, k_delta_fn=kd_fn)
                # v7: Sidecar after each block
                x = self._apply_sidecar(x, eff_idx)
                eff_idx += 1

        x_norm = self.final_norm(x)

        # v8: STP auxiliary loss — enforce locally linear hidden state trajectories
        # Use fixed positions (25%, 50%, 75%) to avoid data-dependent indexing that breaks torch.compile
        self._stp_loss = None
        if stp_lambda > 0 and x_norm.shape[1] >= 4:
            seq_len = x_norm.shape[1]
            h_s = x_norm[:, seq_len // 4]
            h_r = x_norm[:, seq_len // 2]
            h_t = x_norm[:, (3 * seq_len) // 4]
            stp_loss = 1 - F.cosine_similarity(h_t - h_r, h_r - h_s, dim=-1).mean()
            self._stp_loss = stp_loss.detach()

        if self.tie_embeddings:
            logits_proj = F.linear(x_norm, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x_norm)

        # v7: LoRA TTT path — add LM head delta, return per-token loss (or logits if requested)
        if ttt_lora is not None:
            logits_proj = logits_proj + ttt_lora.lm_head_lora(x_norm)
            logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
            if return_logits:
                return logits
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                target_ids.reshape(-1),
                reduction="none",
            ).reshape(input_ids.shape)

        # Standard training path: scalar mean loss
        logits = self.logit_softcap * torch.tanh(logits_proj.reshape(-1, logits_proj.size(-1)) / self.logit_softcap)
        ce_loss = F.cross_entropy(logits.float(), target_ids.reshape(-1), reduction="mean")
        if stp_lambda > 0 and x_norm.shape[1] >= 3:
            return ce_loss + stp_lambda * stp_loss
        return ce_loss

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Return logits (bsz, seq_len, vocab) without computing loss."""
        self._v0_cache = None
        x = self.tok_emb(input_ids)
        x = x + self.bigram(input_ids)
        if self.trigram is not None:
            x = x + self.trigram(input_ids)
        if self.quadgram is not None:
            x = x + self.quadgram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear_gate(x)
        x0 = x
        skips: list[Tensor] = []
        eff_idx = 0
        for loop_idx in range(self.num_loops):
            for block_idx in range(self.num_unique_layers):
                lora = self.lora_adapters[loop_idx][block_idx] if self.lora_adapters is not None else None
                if eff_idx < self.num_encoder_layers:
                    x = self.blocks[block_idx](x, x0, lora=lora)
                    skips.append(x)
                else:
                    dec_idx = eff_idx - self.num_encoder_layers
                    if dec_idx < self.num_skip_weights and skips:
                        x = x + self.skip_weights[dec_idx].to(dtype=x.dtype)[None, None, :] * skips.pop()
                    x = self.blocks[block_idx](x, x0, lora=lora)
                x = self._apply_sidecar(x, eff_idx)
                eff_idx += 1
        x = self.final_norm(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)


def eval_val_sliding(
    args: Hyperparameters,
    base_model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    stride: int,
    batch_seqs: int = 32,
    ngram_mixer: "NgramEvalMixer | None" = None,
) -> tuple[float, float]:
    """Sliding window evaluation: each token scored with maximum context.

    Windows of train_seq_len advance by `stride`. Only the last `stride` tokens
    per window contribute to the score (first window scores all). Windows are
    batched and distributed across ranks.

    If ngram_mixer is provided, blends n-gram log-probs with neural logits and
    updates the mixer's count tables with each scored token (causal).
    """
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1

    # Build windows; skip any too short to score a full stride
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if min(ws + seq_len, total_tokens) - ws >= stride]
    total_windows = len(window_starts)

    # Distribute across ranks
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    # Pre-cache val_tokens as CPU list for n-gram context lookups
    use_ngram = ngram_mixer is not None
    if use_ngram:
        val_tokens_cpu = val_tokens.cpu()
        val_tokens_list = val_tokens_cpu.tolist()  # pre-compute once for fast context slicing

    base_model.eval()
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi + batch_seqs]
            bsz = len(batch_ws)

            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens: list[int] = []

            for i, ws in enumerate(batch_ws):
                end = min(ws + seq_len, total_tokens)
                wlen = end - ws
                wlens.append(wlen)
                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]
                y_batch[i, :wlen] = chunk[1:]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = base_model.forward_logits(x_batch)

            if not use_ngram:
                # Standard path: cross-entropy from neural logits only
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1),
                    reduction="none",
                ).reshape(bsz, seq_len)
            else:
                # N-gram blended path: compute per-token NLL from blended log-probs
                nll = torch.zeros(bsz, seq_len, device=device, dtype=torch.float32)

            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else wlen - stride

                if use_ngram:
                    # Blend n-gram with neural for scored positions
                    score_len = wlen - s
                    scored_logits = logits[i, s:wlen]  # [score_len, V]
                    scored_targets = y_batch[i, s:wlen]  # [score_len]

                    # Build context list for each scored position
                    ctx_list = []
                    for j in range(score_len):
                        abs_pos = ws + s + j + 1  # +1 because targets are shifted by 1
                        ctx_start = max(0, abs_pos - (ngram_mixer.max_order - 1))
                        ctx_list.append(val_tokens_list[ctx_start:abs_pos])

                    blended_lp = ngram_mixer.blend_logits(scored_logits, ctx_list, device)
                    scored_nll = -blended_lp[
                        torch.arange(score_len, device=device), scored_targets
                    ].to(torch.float64)

                    # Update n-gram tables with newly scored tokens (causal)
                    ngram_mixer.update(val_tokens_cpu[ws + s + 1: ws + wlen + 1])
                else:
                    scored_nll = nll[i, s:wlen].to(torch.float64)

                loss_sum += scored_nll.sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]
                prev = x_batch[i, s:wlen]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()

            # Progress (rank 0 only)
            if rank == 0 and (bi // batch_seqs) % 50 == 0:
                done = min(bi + batch_seqs, len(my_windows))
                pct = done / len(my_windows) * 100
                running_bpb = 0.0
                if token_count.item() > 0:
                    rl = (loss_sum / token_count).item()
                    running_bpb = rl / math.log(2.0) * (token_count.item() / byte_count.item())
                ngram_tag = " +ngram" if use_ngram else ""
                print(f"  sliding_eval{ngram_tag} [{pct:5.1f}%] {done}/{len(my_windows)} windows running_bpb={running_bpb:.6f}", flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / token_count).item()
    bits_per_token = val_loss / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    base_model.train()
    return val_loss, bits_per_token * tokens_per_byte


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        mlp_hidden=args.mlp_hidden,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        num_loops=args.num_loops,
        lora_rank=args.lora_rank,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        rope_dims=args.rope_dims,
        xsa_layers=args.xsa_layers,
        ln_scale=args.ln_scale,
        value_residual=args.value_residual,
        gated_attention=args.gated_attention,
        use_trigram=args.use_trigram,
        trigram_vocab_size=args.trigram_vocab_size,
        trigram_dim=args.trigram_dim,
        use_quadgram=args.use_quadgram,
        quadgram_vocab_size=args.quadgram_vocab_size,
        quadgram_dim=args.quadgram_dim,
        sparse_start_layer=args.sparse_start_layer,
        sparse_hidden_dim=args.sparse_hidden_dim,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, (CastedLinear, AttentionLoRA)):
            module.float()
        if isinstance(module, CastedLinear) and args.qat:
            module._qat = True
    restore_low_dim_params_to_fp32(base_model)
    # Orthogonal initialization for large 2D CastedLinear weights
    for name, module in base_model.named_modules():
        if isinstance(module, CastedLinear):
            if getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)
            elif module.weight.ndim == 2 and min(module.weight.shape) >= 64:
                nn.init.orthogonal_(module.weight, gain=1.0)
    log0(f"qat:{args.qat} late_qat:{args.late_qat} qat_mode:{args.qat_mode} alpha:[{args.qat_alpha_start},{args.qat_alpha_end}]")
    if args.stp_lambda > 0:
        log0(f"stp: enabled lambda={args.stp_lambda}")
    if args.crown_q_lambda > 0:
        log0(f"crown_q: enabled lambda={args.crown_q_lambda}")
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    block_named_params = list(base_model.blocks.named_parameters())
    matrix_params = [
        p
        for name, p in block_named_params
        if p.ndim == 2 and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in block_named_params
        if p.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    # Bigram module: embed -> token optimizer, proj -> Muon, scale -> scalar
    matrix_params.append(base_model.bigram.proj.weight)
    scalar_params.append(base_model.bigram.scale)
    if base_model.skip_weights.numel() > 0:
        scalar_params.append(base_model.skip_weights)
    scalar_params.append(base_model.smear_gate.gate_logit)
    # v6: Trigram params
    token_embed_params = [base_model.tok_emb.weight, base_model.bigram.embed.weight]
    if base_model.trigram is not None:
        matrix_params.append(base_model.trigram.proj.weight)
        scalar_params.append(base_model.trigram.scale)
        token_embed_params.append(base_model.trigram.embed.weight)
    # v7: QuadgramHash params
    if base_model.quadgram is not None:
        matrix_params.append(base_model.quadgram.proj.weight)
        scalar_params.append(base_model.quadgram.scale)
        token_embed_params.append(base_model.quadgram.embed.weight)
    # v7: Sidecar params
    if base_model.shared_sparse is not None:
        matrix_params.append(base_model.shared_sparse.gate.weight)
        matrix_params.append(base_model.shared_sparse.value.weight)
        matrix_params.append(base_model.shared_sparse.proj.weight)
        scalar_params.append(base_model.shared_sparse.local.weight)
        scalar_params.append(base_model.shared_sparse.site_gate.weight)
        scalar_params.append(base_model.shared_sparse_scale)
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.Adam(
        [{"params": token_embed_params, "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_weight_decay,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.Adam(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lora_adapters is not None:
        lora_params = list(base_model.lora_adapters.parameters())
        optimizer_lora = torch.optim.Adam(
            [{"params": lora_params, "lr": args.lora_lr, "base_lr": args.lora_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.append(optimizer_lora)
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.Adam(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    n_lora = sum(p.numel() for p in base_model.lora_adapters.parameters()) if base_model.lora_adapters is not None else 0
    effective_depth = args.num_layers * args.num_loops
    log0(f"model_params:{n_params} (unique_layers:{args.num_layers} loops:{args.num_loops} effective_depth:{effective_depth} lora_rank:{args.lora_rank} lora_params:{n_lora})")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
    )
    log0(f"seed:{args.seed}")

    # v8: Progressive Sequence Length schedule
    prog_seq_schedule: list[tuple[int, float]] | None = None
    prog_seq_prev_len = 0  # track changes for logging
    if args.progressive_seq:
        prog_seq_schedule = _parse_progressive_schedule(args.progressive_seq_schedule)
        log0(f"progressive_seq: schedule={prog_seq_schedule}")
        # Increase dynamo cache limit to accommodate multiple seq lengths
        torch._dynamo.config.cache_size_limit = 32

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    training_time_ms = 0.0
    stop_after_step: int | None = None
    swa_state: dict[str, Tensor] | None = None
    swa_count = 0
    # v7: Soft-Round QAT tracking
    CastedLinear._qat_mode = args.qat_mode
    qat_start_step: int | None = None  # step when QAT first activated
    qat_step_count = 0  # steps since QAT activated (for alpha annealing)
    # v6: EMA state
    ema_state: dict[str, Tensor] | None = None
    if args.ema_enabled:
        ema_state = {name: t.detach().clone() for name, t in base_model.state_dict().items()}
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        # v5: Late QAT — enable fake-quantization when LR scale drops below threshold
        # v7: Soft-Round QAT with alpha annealing (PR #606)
        if args.late_qat and scale < args.late_qat_threshold:
            if not CastedLinear._qat_enabled:
                # First QAT step: record start for alpha annealing
                qat_start_step = step
                # Estimate total QAT steps from remaining iterations
                remaining = (stop_after_step or args.iterations) - step
                qat_total_steps = max(remaining, 1)
                CastedLinear._qat_enabled = True
                log0(f"qat_activated: mode={args.qat_mode} step={step} est_total_qat_steps={qat_total_steps}")
            # Anneal alpha for soft-round QAT
            if args.qat_mode == "soft_round":
                qat_frac = min(qat_step_count / max(qat_total_steps, 1), 1.0)
                CastedLinear._qat_alpha = args.qat_alpha_start + qat_frac * (args.qat_alpha_end - args.qat_alpha_start)
                qat_step_count += 1
        zero_grad_all()
        # v8: Progressive Sequence Length — use shorter seqs early for more steps/sec
        cur_seq_len = args.train_seq_len
        if prog_seq_schedule is not None:
            total_est = stop_after_step or args.iterations
            cur_seq_len = get_progressive_seq_len(step, total_est, prog_seq_schedule, args.train_seq_len)
            if cur_seq_len != prog_seq_prev_len:
                log0(f"progressive_seq: step={step} seq_len={cur_seq_len}")
                prog_seq_prev_len = cur_seq_len
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, cur_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y, stp_lambda=args.stp_lambda)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        # v8: CROWN-Q — quantization-aware regularizer (active during warmdown with QAT)
        crown_q_val = None
        if args.crown_q_lambda > 0 and CastedLinear._qat_enabled:
            crown_q = compute_crown_q_loss(base_model)
            crown_q_val = crown_q.detach().item()
            (args.crown_q_lambda * crown_q).backward()

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()

        # v6: EMA update every step
        if ema_state is not None:
            with torch.no_grad():
                for name, param in base_model.state_dict().items():
                    ema_state[name].lerp_(param, 1.0 - args.ema_decay)

        step += 1

        # SWA: collect weight snapshots during second half of warmdown
        if args.swa_enabled and scale < args.swa_start_frac and step % args.swa_every == 0:
            if swa_state is None:
                swa_state = {name: t.detach().cpu().clone() for name, t in base_model.state_dict().items()}
                swa_count = 1
            else:
                for name, t in base_model.state_dict().items():
                    swa_state[name] += t.detach().cpu()
                swa_count += 1

        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            qat_info = ""
            if CastedLinear._qat_enabled and args.qat_mode == "soft_round":
                qat_info = f" qat_alpha:{CastedLinear._qat_alpha:.2f}"
            stp_info = ""
            if args.stp_lambda > 0 and base_model._stp_loss is not None:
                stp_info = f" stp_loss:{base_model._stp_loss.item():.4f}"
            crown_q_info = ""
            if crown_q_val is not None:
                crown_q_info = f" crown_q:{crown_q_val:.6f}"
            seq_info = f" seq_len:{cur_seq_len}" if prog_seq_schedule is not None else ""
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
                f"{qat_info}{stp_info}{crown_q_info}{seq_info}"
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )

    # -----------------------------
    # SWA: APPLY AVERAGED WEIGHTS
    # -----------------------------
    if args.swa_enabled and swa_state is not None and swa_count > 1:
        log0(f"swa: averaging {swa_count} checkpoints")
        avg_state = {name: (t / swa_count).to(dtype=base_model.state_dict()[name].dtype)
                     for name, t in swa_state.items()}
        base_model.load_state_dict(avg_state, strict=True)
        del swa_state, avg_state

    # v6: EMA — apply EMA weights (overrides SWA if both enabled)
    if ema_state is not None:
        log0(f"ema: applying EMA weights (decay={args.ema_decay})")
        ema_cast = {name: t.to(dtype=base_model.state_dict()[name].dtype)
                    for name, t in ema_state.items()}
        base_model.load_state_dict(ema_cast, strict=True)
        del ema_state, ema_cast

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")

    artifact_path = "final_model.pcll" if USE_NOVEL_COMPRESSION else "final_model.int6.ptz"

    if USE_NOVEL_COMPRESSION:
        # v8: Codebook-GPTQ hybrid — collect Hessians for error compensation
        gptq_hessians = None
        if USE_GPTQ:
            log0(f"gptq: collecting Hessians from {GPTQ_CALIB_SEQS} calibration sequences...")
            t_hessian = time.perf_counter()
            gptq_hessians = collect_gptq_hessians(
                base_model, val_tokens, device,
                n_seqs=GPTQ_CALIB_SEQS, seq_len=args.train_seq_len,
            )
            log0(f"gptq: collected {len(gptq_hessians)} Hessians in {1000*(time.perf_counter()-t_hessian):.0f}ms")

        # v5+v8: Novel compression — per-tensor codebook + GPTQ + Huffman + zstd
        log0(f"compression: novel pipeline (codebook + {'gptq + ' if USE_GPTQ else ''}huffman + zstd)")
        t_compress = time.perf_counter()
        raw_size = save_novel_format(base_model.state_dict(), "final_model.pcll.raw", hessians=gptq_hessians)
        del gptq_hessians  # free memory
        # Apply zstd on top
        with open("final_model.pcll.raw", "rb") as f:
            raw_blob = f.read()
        if _HAS_ZSTD:
            cctx = zstd.ZstdCompressor(level=22)
            compressed = cctx.compress(raw_blob)
        else:
            compressed = zlib.compress(raw_blob, level=9)
        if master_process:
            with open(artifact_path, "wb") as f:
                f.write(compressed)
            artifact_bytes = os.path.getsize(artifact_path)
            log0(f"Novel compression: {artifact_bytes} bytes ({artifact_bytes/1024/1024:.2f} MB) "
                 f"raw:{raw_size} bytes, compress_time:{1000*(time.perf_counter()-t_compress):.0f}ms")
            log0(f"Total submission size novel: {artifact_bytes + code_bytes} bytes")
    else:
        # Fallback: original int6+zstd pipeline
        log0("compression: baseline (int6 + torch.save + zstd)")
        quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())
        quant_buf = io.BytesIO()
        torch.save(quant_obj, quant_buf)
        quant_raw = quant_buf.getvalue()
        if _HAS_ZSTD:
            cctx = zstd.ZstdCompressor(level=22)
            quant_blob = cctx.compress(quant_raw)
        else:
            quant_blob = zlib.compress(quant_raw, level=9)
        if master_process:
            with open(artifact_path, "wb") as f:
                f.write(quant_blob)
            quant_file_bytes = os.path.getsize(artifact_path)
            log0(f"Serialized model int6+zstd: {quant_file_bytes} bytes")
            log0(f"Total submission size int6+zstd: {quant_file_bytes + code_bytes} bytes")

    if distributed:
        dist.barrier()

    # Roundtrip validation: decompress, load, eval
    if USE_NOVEL_COMPRESSION:
        with open(artifact_path, "rb") as f:
            blob = f.read()
        if _HAS_ZSTD:
            dctx = zstd.ZstdDecompressor()
            decompressed = dctx.decompress(blob)
        else:
            decompressed = zlib.decompress(blob)
        # Write temp file for load_novel_format (rank-specific to avoid race condition)
        tmp_path = f"_tmp_roundtrip_rank{rank}.pcll"
        with open(tmp_path, "wb") as f:
            f.write(decompressed)
        restored_state = load_novel_format(tmp_path)
        base_model.load_state_dict(restored_state, strict=True)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists("final_model.pcll.raw"):
            os.remove("final_model.pcll.raw")
    else:
        with open(artifact_path, "rb") as f:
            quant_blob_disk = f.read()
        if _HAS_ZSTD:
            dctx = zstd.ZstdDecompressor()
            quant_decompressed = dctx.decompress(quant_blob_disk)
        else:
            quant_decompressed = zlib.decompress(quant_blob_disk)
        quant_state = torch.load(io.BytesIO(quant_decompressed), map_location="cpu", weights_only=False)
        base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)

    # v8: N-gram eval mixer — built causally from scored tokens, zero artifact cost
    ngram_mixer = None
    if args.ngram_eval:
        ngram_mixer = NgramEvalMixer(
            vocab_size=args.vocab_size,
            max_order=args.ngram_max_order,
            alpha=args.ngram_alpha,
            smoothing=args.ngram_smoothing,
        )
        log0(f"ngram_eval: enabled max_order={args.ngram_max_order} "
             f"alpha={args.ngram_alpha} smoothing={args.ngram_smoothing}")

    # -----------------------------
    # v7: BATCHED LORA TTT (legal per-document adaptation)
    # Per-document LoRA rank-8 on Q, V, LM head. 64 docs batched per GPU.
    # Score every chunk, train on non-final chunks. Reset LoRA between docs.
    # Legal per Issue #402 (Case 3) + leloy! [OAI] ruling.
    # CRITICAL: Use fresh UNCOMPILED model — torch.compile ignores LoRA deltas.
    # Falls back to standard sliding window eval if TTT disabled.
    # -----------------------------
    if args.ttt_enabled and args.ttt_mode == "lora":
        torch.cuda.empty_cache()
        ttt_start = time.perf_counter()
        log0(f"lora_ttt: starting rank={args.ttt_lora_rank} lr={args.ttt_lora_lr} "
             f"epochs={args.ttt_epochs} chunk={args.ttt_chunk_size} "
             f"batch={args.ttt_batch_seqs} min_doc={args.ttt_min_doc_len} "
             f"min_nll={args.ttt_min_nll} k_lora={args.ttt_k_lora}")

        # Find document boundaries
        bos_positions = (val_tokens == BOS_ID).nonzero(as_tuple=True)[0].cpu().numpy()
        docs = []
        for i in range(len(bos_positions)):
            start = int(bos_positions[i])
            end = int(bos_positions[i + 1]) + 1 if i + 1 < len(bos_positions) else val_tokens.numel()
            if end - start >= 2:
                docs.append((start, end - start))

        # Shard docs across ranks
        rank_docs = docs[(len(docs) * rank) // world_size: (len(docs) * (rank + 1)) // world_size]
        short_docs = [d for d in rank_docs if d[1] < args.ttt_min_doc_len]
        long_docs = [d for d in rank_docs if d[1] >= args.ttt_min_doc_len]
        log0(f"lora_ttt: {len(docs)} total docs, rank{rank}: "
             f"{len(long_docs)} long + {len(short_docs)} short")

        # Create fresh UNCOMPILED model for TTT
        if torch.cuda.is_available():
            torch._dynamo.reset()
        ttt_model = copy.deepcopy(base_model)
        ttt_model.eval()
        for p in ttt_model.parameters():
            p.requires_grad_(False)

        loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        byte_sum = torch.zeros((), device=device, dtype=torch.float64)
        token_count = torch.zeros((), device=device, dtype=torch.float64)

        # Score short docs without TTT
        with torch.no_grad():
            for ds, dl in short_docs:
                x = val_tokens[ds: ds + dl - 1].to(device=device, dtype=torch.int64).unsqueeze(0)
                y = val_tokens[ds + 1: ds + dl].to(device=device, dtype=torch.int64).unsqueeze(0)
                if ngram_mixer is not None:
                    # N-gram blended scoring for short docs
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = ttt_model.forward_logits(x)
                    scored_logits = logits[0]  # [seq, V]
                    scored_targets = y[0]       # [seq]
                    n_tok = dl - 1
                    ctx_list = []
                    for j in range(n_tok):
                        abs_pos = ds + j + 1
                        ctx_start = max(0, abs_pos - (ngram_mixer.max_order - 1))
                        ctx_list.append(val_tokens[ctx_start:abs_pos].cpu().tolist())
                    blended_lp = ngram_mixer.blend_logits(scored_logits, ctx_list, device)
                    doc_nll = -blended_lp[
                        torch.arange(n_tok, device=device), scored_targets
                    ].to(torch.float64)
                    loss_sum += doc_nll.sum()
                    ngram_mixer.update(val_tokens[ds + 1: ds + dl].cpu())
                else:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        mean_loss = ttt_model(x, y)
                    loss_sum += mean_loss.to(torch.float64) * (dl - 1)
                token_count += dl - 1
                tgt = y.reshape(-1)
                px = x.reshape(-1)
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[px]).to(torch.float64)
                byte_sum += tb.sum()

        # Sort long docs by length for efficient batching
        chunk_size = args.ttt_chunk_size
        eval_seq = args.ttt_eval_seq_len
        long_docs.sort(key=lambda d: (d[1] - 2) // chunk_size)
        batch_size = args.ttt_batch_seqs

        # v7b: helper to build Adam with separate K-LoRA param group (0.3x lr)
        def _make_ttt_opt(lora_module):
            if lora_module.k_loras is not None:
                k_params = set(lora_module.k_loras.parameters())
                qv_params = [p for p in lora_module.parameters() if p not in k_params]
                param_groups = [
                    {"params": qv_params, "lr": args.ttt_lora_lr},
                    {"params": list(k_params), "lr": args.ttt_lora_lr * args.ttt_k_lora_lr_mult},
                ]
                return torch.optim.Adam(param_groups, betas=(args.beta1, args.beta2), eps=1e-10)
            return torch.optim.Adam(lora_module.parameters(), lr=args.ttt_lora_lr,
                                    betas=(args.beta1, args.beta2), eps=1e-10)

        # Create reusable LoRA
        if long_docs and batch_size > 0:
            lora = BatchedTTTLoRA(batch_size, ttt_model, args.ttt_lora_rank,
                                  use_k_lora=args.ttt_k_lora).to(device)
            opt = _make_ttt_opt(lora)

        for bi in range(0, len(long_docs), batch_size):
            batch = long_docs[bi: bi + batch_size]
            bsz = len(batch)
            if bsz == batch_size:
                cur_lora = lora
                cur_lora.reset()
                for g in opt.param_groups:
                    for p in g["params"]:
                        s = opt.state.get(p)
                        if s:
                            s["exp_avg"].zero_()
                            s["exp_avg_sq"].zero_()
                            s["step"].fill_(0)
                cur_opt = opt
            else:
                cur_lora = BatchedTTTLoRA(bsz, ttt_model, args.ttt_lora_rank,
                                          use_k_lora=args.ttt_k_lora).to(device)
                cur_opt = _make_ttt_opt(cur_lora)

            pred_lens = [dl - 1 for _, dl in batch]
            num_chunks = [(pl + chunk_size - 1) // chunk_size for pl in pred_lens]
            max_nc = max(num_chunks)

            # v7b: min-NLL epoch selection — track best scores per document across epochs
            use_min_nll = args.ttt_min_nll and args.ttt_epochs > 1
            if use_min_nll:
                best_avg_nll = [float("inf")] * bsz
                best_loss_sum_doc = [0.0] * bsz
                best_tok_cnt_doc = [0] * bsz
                best_byte_cnt_doc = [0.0] * bsz

            for epoch in range(args.ttt_epochs):
                # v7b: per-epoch accumulators when min-NLL enabled
                if use_min_nll:
                    ep_loss = [0.0] * bsz
                    ep_toks = [0] * bsz
                    ep_bytes = [0.0] * bsz

                for ci in range(max_nc):
                    active = [ci < nc for nc in num_chunks]
                    # Build batch tensors
                    chunk_end_ref = min((ci + 1) * chunk_size, max(pred_lens))
                    win_start_ref = max(0, chunk_end_ref - eval_seq)
                    wl_ref = chunk_end_ref - win_start_ref
                    x = torch.zeros(bsz, wl_ref, dtype=torch.int64, device=device)
                    y = torch.zeros(bsz, wl_ref, dtype=torch.int64, device=device)
                    chunk_offsets = []
                    chunk_lens = []

                    for b in range(bsz):
                        if not active[b]:
                            chunk_offsets.append(0)
                            chunk_lens.append(0)
                            continue
                        ds, dl = batch[b]
                        pl = pred_lens[b]
                        ce = min((ci + 1) * chunk_size, pl)
                        cs = ci * chunk_size
                        ws = max(0, ce - eval_seq)
                        wl = ce - ws
                        co = cs - ws
                        cl = ce - cs
                        toks = val_tokens[ds + ws: ds + ws + wl + 1].to(dtype=torch.int64, device=device)
                        actual_wl = min(wl, wl_ref)
                        x[b, :actual_wl] = toks[:actual_wl]
                        y[b, :actual_wl] = toks[1:actual_wl + 1]
                        chunk_offsets.append(co)
                        chunk_lens.append(cl)

                    is_last_chunk = all(ci >= nc - 1 for nc in num_chunks)

                    if not is_last_chunk:
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            ptl = ttt_model(x, y, ttt_lora=cur_lora)
                    else:
                        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            ptl = ttt_model(x, y, ttt_lora=cur_lora)

                    # v7b: score every epoch when min-NLL enabled, else final epoch only
                    should_score = use_min_nll or (epoch == args.ttt_epochs - 1)
                    if should_score:
                        # v8: get logits for n-gram blending if enabled
                        ttt_logits = None
                        if ngram_mixer is not None:
                            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                ttt_logits = ttt_model(x, y, ttt_lora=cur_lora, return_logits=True)

                        with torch.no_grad():
                            for b in range(bsz):
                                if not active[b]:
                                    continue
                                co, cl = chunk_offsets[b], chunk_lens[b]
                                if cl <= 0:
                                    continue

                                if ngram_mixer is not None and ttt_logits is not None:
                                    # N-gram blended scoring
                                    ds_b, dl_b = batch[b]
                                    cs_b = ci * chunk_size
                                    scored_logits = ttt_logits[b, co: co + cl]  # [cl, V]
                                    scored_targets = y[b, co: co + cl]           # [cl]
                                    # Build context for each position
                                    ctx_list = []
                                    for j in range(cl):
                                        abs_pos = ds_b + cs_b + j + 1  # +1 for target shift
                                        ctx_start = max(0, abs_pos - (ngram_mixer.max_order - 1))
                                        ctx_list.append(val_tokens[ctx_start:abs_pos].cpu().tolist())
                                    blended_lp = ngram_mixer.blend_logits(scored_logits, ctx_list, device)
                                    chunk_nll_t = -blended_lp[
                                        torch.arange(cl, device=device), scored_targets
                                    ].to(torch.float64)
                                    chunk_nll = chunk_nll_t.sum().item()
                                    # Update n-gram tables with scored tokens (causal)
                                    ngram_mixer.update(val_tokens[ds_b + cs_b + 1: ds_b + cs_b + cl + 1].cpu())
                                else:
                                    chunk_nll = ptl[b, co: co + cl].to(torch.float64).sum().item()

                                tgt = y[b, co: co + cl]
                                px = x[b, co: co + cl]
                                tb = base_bytes_lut[tgt].to(torch.float64)
                                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[px]).to(torch.float64)
                                chunk_bytes = tb.sum().item()
                                if use_min_nll:
                                    ep_loss[b] += chunk_nll
                                    ep_toks[b] += cl
                                    ep_bytes[b] += chunk_bytes
                                else:
                                    loss_sum += chunk_nll
                                    token_count += cl
                                    byte_sum += chunk_bytes

                    # Train on non-final chunks
                    if not is_last_chunk:
                        train_loss = torch.zeros(1, device=device)
                        for b in range(bsz):
                            if ci >= num_chunks[b] - 1:
                                continue
                            co, cl = chunk_offsets[b], chunk_lens[b]
                            if cl > 0:
                                train_loss = train_loss + ptl[b, co: co + cl].mean()
                        cur_opt.zero_grad()
                        train_loss.backward()
                        cur_opt.step()

                # v7b: end of epoch — keep best per-doc scores (lowest avg NLL)
                if use_min_nll:
                    for b in range(bsz):
                        if ep_toks[b] == 0:
                            continue
                        avg = ep_loss[b] / ep_toks[b]
                        if avg < best_avg_nll[b]:
                            best_avg_nll[b] = avg
                            best_loss_sum_doc[b] = ep_loss[b]
                            best_tok_cnt_doc[b] = ep_toks[b]
                            best_byte_cnt_doc[b] = ep_bytes[b]

            # v7b: accumulate best-epoch scores into global totals
            if use_min_nll:
                for b in range(bsz):
                    loss_sum += best_loss_sum_doc[b]
                    token_count += best_tok_cnt_doc[b]
                    byte_sum += best_byte_cnt_doc[b]

            if rank == 0 and (bi + batch_size) % (batch_size * 10) == 0:
                elapsed = time.perf_counter() - ttt_start
                avg_l = loss_sum.item() / max(token_count.item(), 1)
                log0(f"lora_ttt: batch {bi // batch_size + 1}/"
                     f"{(len(long_docs) + batch_size - 1) // batch_size} "
                     f"time={elapsed:.1f}s avg_loss={avg_l:.4f}")

        # All-reduce across ranks
        if distributed:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(byte_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(token_count, op=dist.ReduceOp.SUM)

        del ttt_model
        torch.cuda.empty_cache()

        q_val_loss = float(loss_sum.item() / max(token_count.item(), 1))
        q_val_bpb = float((loss_sum.item() / math.log(2.0)) / max(byte_sum.item(), 1))
        compression_tag = "novel_roundtrip" if USE_NOVEL_COMPRESSION else "int6_zstd_roundtrip"
        log0(f"final_{compression_tag}_lora_ttt val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
             f"eval_time:{1000.0 * (time.perf_counter() - ttt_start):.0f}ms")
        log0(f"final_{compression_tag}_lora_ttt_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    else:
        # Standard sliding window eval (no TTT or legacy mode)
        pass
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    compression_tag = "novel_roundtrip" if USE_NOVEL_COMPRESSION else "int6_zstd_roundtrip"
    if args.eval_stride > 0 and args.eval_stride < args.train_seq_len:
        # v8: fresh n-gram mixer for final sliding eval (separate from TTT)
        final_ngram_mixer = None
        if args.ngram_eval:
            final_ngram_mixer = NgramEvalMixer(
                vocab_size=args.vocab_size,
                max_order=args.ngram_max_order,
                alpha=args.ngram_alpha,
                smoothing=args.ngram_smoothing,
            )
        ngram_tag = " +ngram" if final_ngram_mixer else ""
        log0(f"final_eval_mode:sliding_window{ngram_tag} stride:{args.eval_stride} batch_seqs:{args.eval_batch_seqs}")
        q_val_loss, q_val_bpb = eval_val_sliding(
            args,
            base_model,
            rank,
            world_size,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            stride=args.eval_stride,
            batch_seqs=args.eval_batch_seqs,
            ngram_mixer=final_ngram_mixer,
        )
    else:
        log0("final_eval_mode:standard")
        q_val_loss, q_val_bpb = eval_val(
            args,
            model,
            rank,
            world_size,
            device,
            grad_accum_steps,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
    torch.cuda.synchronize()
    log0(
        f"final_{compression_tag} val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(f"final_{compression_tag}_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
