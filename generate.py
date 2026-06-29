"""
Generate text from a trained TinyGPT checkpoint.

Usage
-----
    python scripts/generate.py --prompt "HAMLET:"
    python scripts/generate.py --prompt "To be or" --max-tokens 400 --temperature 0.9
    python scripts/generate.py --prompt "" --top-k 50 --top-p 0.95

Sampling strategies (can be combined):
    --temperature  float   Logit scale (default 0.8).
                           < 1.0 → more deterministic / repetitive
                           > 1.0 → more random / creative
    --top-k        int     Restrict sampling to top-k logits (default 40).
                           Set 0 to disable.
    --top-p        float   Nucleus sampling: keep smallest set summing to ≥ p.
                           Typical value 0.9–0.95.  Set None to disable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinygpt.config    import ModelConfig
from tinygpt.model     import TinyGPT
from tinygpt.tokenizer import load_tokenizer
from tinygpt.trainer   import resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text with TinyGPT")

    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--checkpoint",     default="best", choices=["best", "latest"],
                   help="Which checkpoint to load (default: best)")
    p.add_argument("--prompt",         default="",
                   help="Seed text.  Empty string = unconditional generation.")
    p.add_argument("--max-tokens",     type=int,   default=300,
                   help="Number of new tokens to generate (default: 300)")
    p.add_argument("--temperature",    type=float, default=0.8)
    p.add_argument("--top-k",         type=int,   default=40,
                   help="Top-k filter (0 = disabled)")
    p.add_argument("--top-p",         type=float, default=None,
                   help="Nucleus sampling threshold (None = disabled)")
    p.add_argument("--device",        default="auto")

    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = resolve_device(args.device)

    ckpt_dir = Path(args.checkpoint_dir)
    tok_path  = ckpt_dir / "tokenizer.json"
    ckpt_path = ckpt_dir / f"ckpt_{args.checkpoint}.pt"

    # ── Load tokenizer ────────────────────────────────────────────────────
    if not tok_path.exists():
        sys.exit(f"[Generate] tokenizer not found: {tok_path}\n"
                 "  Run scripts/prepare_data.py first.")
    tok = load_tokenizer(str(tok_path))

    # ── Load checkpoint ───────────────────────────────────────────────────
    if not ckpt_path.exists():
        sys.exit(f"[Generate] checkpoint not found: {ckpt_path}\n"
                 "  Run scripts/train.py first.")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # ── Rebuild model from saved config ───────────────────────────────────
    # model_cfg is stored in the checkpoint by Trainer._save()
    mcfg         = ModelConfig(**ckpt["model_cfg"])
    mcfg.dropout = 0.0   # no dropout at inference

    model = TinyGPT(mcfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(
        f"[Generate] loaded ckpt_{args.checkpoint}.pt"
        f"  step={ckpt['step']}"
        f"  val_loss={ckpt['best_val_loss']:.4f}"
        f"  params={model.count_parameters():,}"
        f"  device={device}"
    )

    # ── Encode prompt ─────────────────────────────────────────────────────
    prompt_ids = tok.encode(args.prompt) if args.prompt else [0]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    top_k = args.top_k if args.top_k > 0 else None

    # ── Generate ──────────────────────────────────────────────────────────
    out = model.generate(
        idx,
        max_new_tokens = args.max_tokens,
        temperature    = args.temperature,
        top_k          = top_k,
        top_p          = args.top_p,
    )

    new_ids   = out[0, len(prompt_ids):].tolist()
    generated = tok.decode(new_ids)

    print("\n" + "─" * 64)
    print(args.prompt + generated)
    print("─" * 64)


if __name__ == "__main__":
    main()
