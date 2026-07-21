"""
TinyGPT: Decoder-only transformer language model, built entirely from scratch.

Architecture follows GPT-2 (Radford et al. 2019) with:
  • Learned positional embeddings    (as opposed to sinusoidal; Vaswani et al. 2017)
  • Pre-LayerNorm in each block      (Xiong et al. 2020, arXiv:2002.04745)
  • Causal multi-head self-attention with fused QKV projection
  • Position-wise FFN with GELU      (Hendrycks & Gimpel 2016)
  • Weight-tied lm_head ↔ token_emb  (Press & Wolf 2017, arXiv:1608.05859)
  • Scaled residual-projection init  (GPT-2 §2.3)

Default config (vocab=65, d=384, L=6, d_ff=1536, T=256) → ~10.75 M parameters.

References:
  Vaswani et al. (2017) "Attention Is All You Need"         arXiv:1706.03762
  Radford et al. (2019) "Language Models are Unsupervised Multitask Learners"
  Ba et al.      (2016) "Layer Normalization"               arXiv:1607.06450
  Karpathy       (2022)  nanoGPT                 github.com/karpathy/nanoGPT
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


# ─────────────────────────────────────────────────────────────────────────────
# Primitive building blocks
# ─────────────────────────────────────────────────────────────────────────────


class LayerNorm(nn.Module):
    """
    Layer Normalisation (Ba et al. 2016).

        LN(x) = γ · (x − μ) / √(σ² + ε) + β

    Implemented from scratch (not via nn.LayerNorm) for full transparency.
    Uses population variance (unbiased=False), matching GPT-2's behaviour.

    Parameters
    ----------
    d_model : residual stream dimension
    eps     : numerical stability floor added before square-root
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))   # γ  (gain)
        self.bias   = nn.Parameter(torch.zeros(d_model))  # β  (shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu    = x.mean(dim=-1, keepdim=True)
        sigma2 = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mu) / (sigma2 + self.eps).sqrt()
        return self.weight * x_hat + self.bias


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (autoregressive) self-attention.

        Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) · V

    Implementation notes
    --------------------
    • Q, K, V are computed in a single fused projection for cache efficiency.
    • The causal mask is stored as a non-persistent buffer (not saved in
      state_dict) and sliced to the actual sequence length at runtime.
    • No bias in projections — GPT-2 convention.

    Parameter count per block
    -------------------------
      qkv_proj : 3 · d_model²  (e.g. 3 × 384² = 442 368)
      out_proj :     d_model²  (e.g.     384² = 147 456)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(
                f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})"
            )
        self.d_model = cfg.d_model
        self.n_heads = cfg.n_heads
        self.d_k     = cfg.d_model // cfg.n_heads   # per-head dimension

        # Fused QKV + output projections (no bias)
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj  = nn.Linear(cfg.d_model,     cfg.d_model, bias=False)

        self.attn_drop  = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # Lower-triangular causal mask — shape (1, 1, T, T)
        # persistent=False: excluded from state_dict, re-created on load
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.context_len, cfg.context_len))
                  .view(1, 1, cfg.context_len, cfg.context_len),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, d_model

        # ── Project to Q, K, V ───────────────────────────────────────────
        q, k, v = self.qkv_proj(x).split(self.d_model, dim=-1)  # each (B, T, C)

        # Reshape to (B, n_heads, T, d_k) for batched matrix multiply
        def to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        # ── Scaled dot-product attention ─────────────────────────────────
        scale  = 1.0 / math.sqrt(self.d_k)
        scores = (q @ k.transpose(-2, -1)) * scale          # (B, H, T, T)

        # Mask out future positions → −∞ → 0 after softmax
        scores = scores.masked_fill(
            self.mask[:, :, :T, :T] == 0, float("-inf")
        )
        weights = self.attn_drop(F.softmax(scores, dim=-1)) # (B, H, T, T)

        # ── Weighted aggregation of values ───────────────────────────────
        out = weights @ v                                    # (B, H, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_drop(self.out_proj(out))


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (Vaswani et al. §3.3).

        FFN(x) = GELU( x W₁ ) W₂

    d_ff = 4 · d_model is the canonical expansion ratio (both in the original
    paper and in GPT-2).  GELU replaces the original ReLU as in GPT/GPT-2.

    No bias terms — consistent with GPT-2.

    Parameter count per block
    -------------------------
      fc1 : d_model × d_ff   (e.g. 384 × 1536 = 589 824)
      fc2 : d_ff × d_model   (e.g. 1536 × 384 = 589 824)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.fc1  = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.fc2  = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """
    GPT-2 style transformer block with Pre-LayerNorm.

        x ← x + Attention( LN₁(x) )   # self-attention sub-layer
        x ← x + FFN(       LN₂(x) )   # feed-forward sub-layer

    Pre-LN (normalising before the sub-layer, not after) was shown by
    Xiong et al. (2020) to produce more stable gradients than the original
    post-LN formulation and to reduce the need for careful learning-rate
    warm-up.

    Parameter count per block (d=384, H=6, d_ff=1536)
    --------------------------------------------------
      LN₁ + LN₂  :   2 × 768 =   1 536
      Attention   :             589 824
      FFN         :           1 179 648
      ─────────────────────────────────
      Total       :           1 771 008
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1  = LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = LayerNorm(cfg.d_model)
        self.ff   = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(  self.ln2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────


class TinyGPT(nn.Module):
    """
    Decoder-only GPT-style language model.

    Default parameter budget  (vocab=65, d=384, L=6, d_ff=1536, T=256)
    ────────────────────────────────────────────────────────────────────
    Component                           Params
    ─────────────────────────────────── ──────────
    Token embedding  (65 × 384)          24 960
    Position embedding (256 × 384)       98 304
    6 × TransformerBlock             10 626 048
      └ per block (1 771 008):
          LN×2          1 536
          QKV proj    442 368
          Out proj    147 456
          FFN fc1     589 824
          FFN fc2     589 824
    Final LayerNorm                         768
    LM head  (weight-tied → 0 extra)          0
    ─────────────────────────────────── ─────────
    TOTAL                            10 750 080
    ────────────────────────────────────────────

    Memory in float32 training  (weights + grads + Adam m/v):
      10.75M × 4 bytes × 4 ≈ 172 MB  — fits comfortably on CPU / shared RAM.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # Input embeddings
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb   = nn.Embedding(cfg.context_len, cfg.d_model)
        self.emb_drop  = nn.Dropout(cfg.dropout)

        # Transformer stack
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

        # Final normalisation
        self.ln_f = LayerNorm(cfg.d_model)

        # Language model head — no bias; weight tied to token_emb
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # Press & Wolf 2017

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────

    def _init_weights(self) -> None:
        """
        GPT-2 weight initialisation (§2.3 of Radford et al. 2019):

          1. All weights initialised from N(0, 0.02).
          2. Residual projection outputs scaled down by 1/√(2L) so that
             the variance of the residual stream stays O(1) regardless of
             depth L.  Without this, the variance grows linearly with depth
             and the first few training steps are numerically unstable.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        # Scale residual projections (out_proj in attention, fc2 in FFN)
        residual_std = 0.02 / math.sqrt(2 * self.cfg.n_layers)
        for name, p in self.named_parameters():
            if "out_proj.weight" in name or "fc2.weight" in name:
                nn.init.normal_(p, mean=0.0, std=residual_std)

    # ── Forward pass ──────────────────────────────────────────────────────

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        idx     : (B, T) long tensor — token indices
        targets : (B, T) long tensor — next-token labels (optional)

        Returns
        -------
        logits : (B, T, vocab_size) — unnormalised log-probabilities
        loss   : scalar cross-entropy, or None if targets is None
        """
        B, T = idx.shape
        if T > self.cfg.context_len:
            raise ValueError(
                f"Input length {T} exceeds context_len {self.cfg.context_len}"
            )

        pos = torch.arange(T, device=idx.device)  # (T,)
        x   = self.emb_drop(self.token_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.ln_f(x))   # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten to (B*T, V) vs (B*T,); ignore_index=-1 for future masking
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    # ── Text generation ───────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Autoregressive token generation.

        Parameters
        ----------
        idx           : (1, T) seed token indices (on correct device)
        max_new_tokens: number of tokens to sample
        temperature   : logit scale; < 1 → sharper dist; > 1 → flatter dist
        top_k         : if set, restrict sampling to top-k logits
        top_p         : nucleus sampling — smallest set with cumulative P ≥ top_p

        Returns
        -------
        (1, T + max_new_tokens) token indices
        """
        was_training = self.training  # restored in `finally`, even on exception
        self.eval()

        try:
            for _ in range(max_new_tokens):
                # Crop context to context_len
                ctx     = idx[:, -self.cfg.context_len:]
                logits, _ = self(ctx)
                logits  = logits[:, -1, :] / temperature   # (1, V)

                # ── Top-k filtering ──────────────────────────────────────
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    threshold, _ = torch.topk(logits, k)
                    logits[logits < threshold[:, -1:]] = float("-inf")

                # ── Nucleus (top-p) filtering ─────────────────────────────
                if top_p is not None:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    # Shift right so the token that crosses the threshold is kept
                    remove = (cum_probs - F.softmax(sorted_logits, dim=-1)) > top_p
                    sorted_logits[remove] = float("-inf")
                    logits = torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)

                probs     = F.softmax(logits, dim=-1)
                next_tok  = torch.multinomial(probs, num_samples=1)  # (1, 1)
                idx       = torch.cat([idx, next_tok], dim=1)
        finally:
            self.train(was_training)

        return idx

    # ── Utilities ─────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """
        Count unique trainable parameters.

        Weight-tied tensors (token_emb.weight ≡ lm_head.weight) are counted
        once via data_ptr() deduplication.
        """
        seen: set = set()
        total = 0
        for p in self.parameters():
            ptr = p.data_ptr()
            if ptr not in seen:
                seen.add(ptr)
                total += p.numel()
        return total
