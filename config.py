"""
Configuration dataclasses for TinyGPT.

ModelConfig defines the architecture; TrainConfig controls the optimiser and
logging.  Both are plain Python dataclasses so they serialise cleanly to/from
dicts (json.dumps(vars(cfg))).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Transformer architecture hyper-parameters."""

    vocab_size: int = 65       # Set to tokenizer.vocab_size before building model
    context_len: int = 256     # Maximum sequence length (T)
    d_model: int = 384         # Residual stream / embedding dimension
    n_heads: int = 6           # Number of attention heads (d_model must be divisible)
    n_layers: int = 6          # Number of TransformerBlocks stacked
    d_ff: int = 1536           # FFN inner dimension (GPT-2 uses 4 × d_model)
    dropout: float = 0.1       # Dropout probability (set 0.0 at inference)


@dataclass
class TrainConfig:
    """Training-loop hyper-parameters."""

    # ── Paths ──────────────────────────────────────────────────────────────
    data_path: str = "data/corpus.txt"
    tokenizer_path: str = "checkpoints/tokenizer.json"
    checkpoint_dir: str = "checkpoints"

    # ── Data ───────────────────────────────────────────────────────────────
    train_split: float = 0.9        # Fraction of tokens used for training

    # ── Optimisation ───────────────────────────────────────────────────────
    batch_size: int = 32
    max_steps: int = 5000
    grad_accum_steps: int = 1       # Effective batch = batch_size × grad_accum_steps
    lr: float = 3e-4
    min_lr_ratio: float = 0.1       # LR floor = lr × min_lr_ratio (cosine tail)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 200

    # ── Evaluation & checkpointing ─────────────────────────────────────────
    eval_interval: int = 500        # Run validation every N optimiser steps
    n_eval_batches: int = 20        # Val batches per evaluation pass
    log_interval: int = 50          # Print train loss every N steps

    # ── Runtime ────────────────────────────────────────────────────────────
    device: str = "auto"            # "auto" | "cpu" | "cuda" | "xpu" | "mps"
    compile_model: bool = False     # torch.compile (PyTorch ≥ 2.0; skip on XPU)
