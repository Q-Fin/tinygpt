"""
Training loop for TinyGPT.

Optimisation design:

  Optimiser : AdamW (Loshchilov & Hutter 2019, arXiv:1711.05101)
                betas=(0.9, 0.95), eps=1e-8
                Weight decay applied only to 2-D weight matrices;
                embeddings, LayerNorm parameters, and biases are exempt.

  LR schedule: Linear warm-up for warmup_steps, then cosine annealing
                down to lr × min_lr_ratio at max_steps.
                Following Hoffmann et al. (2022, Chinchilla) and GPT-3.

  Gradient clipping: global L2 norm clipped to grad_clip (default 1.0)
                     Pascanu et al. (2013), "On the difficulty of training RNNs".

  Gradient accumulation: optional micro-batching for larger effective batches.

  Checkpoints: ckpt_best.pt    — lowest validation loss seen
               ckpt_latest.pt  — most recent completed step
               Both store model_cfg so generate.py can reconstruct the model.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .config import ModelConfig, TrainConfig
from .dataset import TokenDataset, make_loader
from .model import TinyGPT


# ─────────────────────────────────────────────────────────────────────────────
# Device resolution
# ─────────────────────────────────────────────────────────────────────────────


def resolve_device(spec: str) -> torch.device:
    """
    Resolve a device string to a torch.device.

    "auto" priority: Intel XPU (via IPEX) → CUDA → Apple MPS → CPU.

    Intel Iris Xe users: install intel-extension-for-pytorch for XPU support.
    Without it, the model trains on CPU using system RAM (shared with GPU).
    """
    if spec != "auto":
        return torch.device(spec)

    # 1. Intel XPU (Intel Extension for PyTorch)
    try:
        import intel_extension_for_pytorch  # type: ignore  # noqa: F401
        if torch.xpu.is_available():        # type: ignore[attr-defined]
            return torch.device("xpu")
    except ImportError:
        pass

    # 2. NVIDIA CUDA
    if torch.cuda.is_available():
        return torch.device("cuda")

    # 3. Apple MPS (M-series)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")

    # 4. CPU (always available)
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────


class Trainer:
    """
    Self-contained training loop.

    Usage
    -----
        trainer = Trainer(model, train_ds, val_ds, cfg)
        trainer.train()                     # full run
        trainer.load("latest")              # resume
    """

    def __init__(
        self,
        model:    TinyGPT,
        train_ds: TokenDataset,
        val_ds:   TokenDataset,
        cfg:      TrainConfig,
    ) -> None:
        self.cfg    = cfg
        self.device = resolve_device(cfg.device)
        print(f"[Trainer] device = {self.device}")

        # Move model to device; optionally compile
        self.model = model.to(self.device)
        if cfg.compile_model:
            try:
                self.model = torch.compile(self.model)  # type: ignore[assignment]
                print("[Trainer] torch.compile() active.")
            except Exception as exc:
                print(f"[Trainer] torch.compile() skipped ({exc}).")

        # Data loaders
        self.train_loader = make_loader(train_ds, cfg.batch_size, shuffle=True)
        self.val_loader   = make_loader(val_ds,   cfg.batch_size, shuffle=False)

        # Optimiser and LR schedule
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # State
        self.step          = 0
        self.epoch         = 0
        self.best_val_loss = float("inf")

    # ── Optimiser ─────────────────────────────────────────────────────────

    def _build_optimizer(self) -> torch.optim.AdamW:
        """
        AdamW with separate weight-decay / no-weight-decay parameter groups.

        Rule: apply weight decay to 2-D weight matrices (Linear weights).
              Exempt:  embeddings, LayerNorm γ/β, Linear biases.

        Weight-tied parameters (token_emb.weight ≡ lm_head.weight) are deduplicated via data_ptr() so they appear in exactly one group.
        """
        decay:    list = []
        no_decay: list = []
        seen_ptrs: set = set()

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            ptr = p.data_ptr()
            if ptr in seen_ptrs:
                continue          # skip tied / shared tensor
            seen_ptrs.add(ptr)

            # 2-D weight matrices that are NOT embeddings or LayerNorm
            if (
                p.ndim >= 2
                and "emb"  not in name
                and "ln"   not in name.lower()
            ):
                decay.append(p)
            else:
                no_decay.append(p)

        return torch.optim.AdamW(
            [
                {"params": decay,    "weight_decay": self.cfg.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.cfg.lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """
        Cosine decay with linear warm-up.

            0 → warmup_steps   : LR ramps linearly from 0 to cfg.lr
            warmup → max_steps : LR decays via cosine to cfg.lr × min_lr_ratio

        The cosine schedule was shown to be nearly optimal by Hoffmann et al. (2022) and is standard in GPT-family training runs.
        """
        wu    = self.cfg.warmup_steps
        total = self.cfg.max_steps
        floor = self.cfg.min_lr_ratio

        def lr_fn(step: int) -> float:
            if step < wu:
                return step / max(1, wu)
            t = (step - wu) / max(1, total - wu)
            return max(floor, 0.5 * (1.0 + math.cos(math.pi * t)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_fn)

    # ── Evaluation ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate(self) -> float:
        """Average cross-entropy loss on `n_eval_batches` validation batches."""
        self.model.eval()
        total, count = 0.0, 0
        for x, y in self.val_loader:
            if count >= self.cfg.n_eval_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            total += loss.item()
            count += 1
        self.model.train()
        return total / max(count, 1)

    # ── Checkpointing ──────────────────────────────────────────────────────

    def _save(self, tag: str) -> None:
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"ckpt_{tag}.pt"
        torch.save(
            {
                "step":            self.step,
                "epoch":           self.epoch,
                "best_val_loss":   self.best_val_loss,
                # model_cfg stored so generate.py can rebuild without the config file
                "model_cfg":       vars(self.model.cfg),
                "model_state":     self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
            },
            path,
        )
        print(f"[Trainer] checkpoint → {path}")

    def load(self, tag: str = "latest") -> None:
        """Resume training from a saved checkpoint."""
        path = Path(self.cfg.checkpoint_dir) / f"ckpt_{tag}.pt"
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.step          = ckpt["step"]
        self.epoch         = ckpt["epoch"]
        self.best_val_loss = ckpt["best_val_loss"]
        print(f"[Trainer] resumed from step {self.step}  (best_val={self.best_val_loss:.4f})")

    # ── Main loop ──────────────────────────────────────────────────────────

    def train(self) -> None:
        """
        Run the main training loop from self.step to cfg.max_steps.

        Supports resumption: call load() before train() to continue.
        """
        cfg      = self.cfg
        log_path = Path(cfg.checkpoint_dir) / "log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        n_params  = self.model.count_parameters()
        eff_batch = cfg.batch_size * cfg.grad_accum_steps

        print(
            f"\n{'='*58}\n"
            f"  TinyGPT — training\n"
            f"  parameters     : {n_params:,}\n"
            f"  device         : {self.device}\n"
            f"  batch size     : {cfg.batch_size}"
            f"  ×  {cfg.grad_accum_steps} accum  =  {eff_batch} effective\n"
            f"  steps          : {self.step} → {cfg.max_steps}\n"
            f"  lr             : {cfg.lr:.1e}  (warmup {cfg.warmup_steps},"
            f" floor ×{cfg.min_lr_ratio})\n"
            f"  eval every     : {cfg.eval_interval} steps\n"
            f"{'='*58}\n"
        )

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        train_iter    = iter(self.train_loader)
        t0            = time.perf_counter()
        running_loss  = 0.0
        n_since_log   = 0

        while self.step < cfg.max_steps:

            # ── Micro-batch accumulation loop ──────────────────────────────
            step_loss = 0.0
            for _ in range(cfg.grad_accum_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    self.epoch += 1
                    train_iter = iter(self.train_loader)
                    x, y = next(train_iter)

                x, y = x.to(self.device), y.to(self.device)
                _, loss = self.model(x, y)

                # Scale loss before backward so gradient magnitude is
                # independent of accumulation count
                (loss / cfg.grad_accum_steps).backward()
                step_loss += loss.item() / cfg.grad_accum_steps

            # ── Optimiser step ─────────────────────────────────────────────
            nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.step   += 1
            running_loss += step_loss
            n_since_log  += 1

            # ── Logging ────────────────────────────────────────────────────
            if self.step % cfg.log_interval == 0:
                avg_loss = running_loss / n_since_log
                elapsed  = time.perf_counter() - t0
                lr       = self.scheduler.get_last_lr()[0]

                print(
                    f"step {self.step:>5d}/{cfg.max_steps}"
                    f"  loss={avg_loss:.4f}"
                    f"  lr={lr:.2e}"
                    f"  {elapsed:6.1f}s"
                )
                with open(log_path, "a") as fh:
                    fh.write(json.dumps({
                        "step": self.step, "train_loss": avg_loss,
                        "lr": lr, "elapsed": elapsed,
                    }) + "\n")

                running_loss = 0.0
                n_since_log  = 0

            # ── Validation & checkpointing ─────────────────────────────────
            if self.step % cfg.eval_interval == 0:
                val_loss = self._evaluate()
                ppl      = math.exp(min(val_loss, 20.0))

                is_best = val_loss < self.best_val_loss
                marker  = " ← best" if is_best else ""
                print(
                    f"  ┌ val_loss={val_loss:.4f}"
                    f"  ppl={ppl:.2f}{marker}\n"
                    f"  └ (prev best={self.best_val_loss:.4f})"
                )

                with open(log_path, "a") as fh:
                    fh.write(json.dumps({
                        "step": self.step, "val_loss": val_loss, "ppl": ppl,
                    }) + "\n")

                if is_best:
                    self.best_val_loss = val_loss
                    self._save("best")

                self._save("latest")

        total_elapsed = time.perf_counter() - t0
        print(
            f"\n[Trainer] done.  {cfg.max_steps} steps in {total_elapsed:.0f}s"
            f"  best_val_loss={self.best_val_loss:.4f}"
        )
