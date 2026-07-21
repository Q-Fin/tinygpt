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

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.context_len <= 0:
            raise ValueError(f"context_len must be positive, got {self.context_len}")
        if self.d_model <= 0 or self.n_heads <= 0 or self.n_layers <= 0 or self.d_ff <= 0:
            raise ValueError("d_model, n_heads, n_layers, and d_ff must all be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


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
    seed: int = 1337                # RNG seed for data shuffling (Trainer) — see train.py
                                     # for seeding model init before construction.

    def __post_init__(self) -> None:
        if not 0.0 < self.train_split < 1.0:
            raise ValueError(f"train_split must be in (0, 1), got {self.train_split}")
        if self.batch_size <= 0 or self.grad_accum_steps <= 0:
            raise ValueError("batch_size and grad_accum_steps must be positive")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {self.max_steps}")
        if self.warmup_steps < 0 or self.warmup_steps >= self.max_steps:
            raise ValueError(
                f"warmup_steps ({self.warmup_steps}) must be in [0, max_steps={self.max_steps})"
            )
        if self.eval_interval <= 0 or self.log_interval <= 0:
            raise ValueError("eval_interval and log_interval must be positive (used as a modulus)")
        if self.n_eval_batches <= 0:
            raise ValueError(f"n_eval_batches must be positive, got {self.n_eval_batches}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}")
