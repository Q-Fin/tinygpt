"""
Train TinyGPT.

Quick start (after running prepare_data.py):
    python scripts/train.py

Resume from latest checkpoint:
    python scripts/train.py --resume

Key flags:
    --steps 5000          optimizer steps  (5 000 ≈ 30–90 min on CPU)
    --batch-size 32
    --lr 3e-4
    --device auto         auto-detect XPU / CUDA / MPS / CPU
    --grad-accum 4        effective batch = batch_size × grad_accum
    --compile             torch.compile (PyTorch ≥ 2.0; do NOT use on XPU)

Model size flags (defaults → ~10.75 M parameters with char tokenizer):
    --d-model 384   --n-heads 6   --n-layers 6   --context-len 256

For a ~13.8 M parameter model with BPE (vocab=4096):
    First run prepare_data.py --tokenizer bpe --vocab-size 4096 then  train.py  (d-model / n-layers flags unchanged)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinygpt.config   import ModelConfig, TrainConfig
from tinygpt.dataset  import prepare_datasets
from tinygpt.model    import TinyGPT
from tinygpt.tokenizer import load_tokenizer
from tinygpt.trainer  import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TinyGPT")

    # ── Paths ─────────────────────────────────────────────────────────────
    p.add_argument("--data-path",      default="data/corpus.txt")
    p.add_argument("--tokenizer-path", default="checkpoints/tokenizer.json")
    p.add_argument("--checkpoint-dir", default="checkpoints")

    # ── Optimisation ──────────────────────────────────────────────────────
    p.add_argument("--steps",        type=int,   default=5000)
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--grad-accum",   type=int,   default=1,
                   help="Gradient accumulation steps (effective batch = batch × accum)")
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--warmup",       type=int,   default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip",    type=float, default=1.0)
    p.add_argument("--dropout",      type=float, default=0.1)

    # ── Evaluation ────────────────────────────────────────────────────────
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--log-interval",  type=int, default=50)
    p.add_argument("--n-eval-batches",type=int, default=20)

    # ── Model architecture ────────────────────────────────────────────────
    p.add_argument("--d-model",    type=int, default=384)
    p.add_argument("--n-heads",    type=int, default=6)
    p.add_argument("--n-layers",   type=int, default=6)
    p.add_argument("--context-len",type=int, default=256)

    # ── Runtime ───────────────────────────────────────────────────────────
    p.add_argument("--device",  default="auto",
                   help="auto | cpu | cuda | xpu | mps")
    p.add_argument("--resume",  action="store_true",
                   help="Resume from checkpoints/ckpt_latest.pt")
    p.add_argument("--compile", action="store_true",
                   help="Enable torch.compile (PyTorch >= 2.0; skip on XPU)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tok = load_tokenizer(args.tokenizer_path)
    print(f"[Main] tokenizer loaded  vocab_size={tok.vocab_size}")

    # ── Configs ───────────────────────────────────────────────────────────
    mcfg = ModelConfig(
        vocab_size  = tok.vocab_size,
        context_len = args.context_len,
        d_model     = args.d_model,
        n_heads     = args.n_heads,
        n_layers    = args.n_layers,
        d_ff        = 4 * args.d_model,
        dropout     = args.dropout,
    )

    tcfg = TrainConfig(
        data_path       = args.data_path,
        tokenizer_path  = args.tokenizer_path,
        checkpoint_dir  = args.checkpoint_dir,
        batch_size      = args.batch_size,
        max_steps       = args.steps,
        grad_accum_steps= args.grad_accum,
        lr              = args.lr,
        warmup_steps    = args.warmup,
        weight_decay    = args.weight_decay,
        grad_clip       = args.grad_clip,
        eval_interval   = args.eval_interval,
        log_interval    = args.log_interval,
        n_eval_batches  = args.n_eval_batches,
        device          = args.device,
        compile_model   = args.compile,
    )

    # ── Data ──────────────────────────────────────────────────────────────
    train_ds, val_ds = prepare_datasets(args.data_path, tok, mcfg.context_len)

    # ── Model ─────────────────────────────────────────────────────────────
    model = TinyGPT(mcfg)
    print(f"[Main] model built  params={model.count_parameters():,}")

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = Trainer(model, train_ds, val_ds, tcfg)
    if args.resume:
        trainer.load("latest")
    trainer.train()


if __name__ == "__main__":
    main()
