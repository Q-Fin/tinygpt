"""
Download a training corpus and train the tokenizer.

By default downloads TinyShakespeare (~1.1 MB) and trains a character-level tokenizer (instant).  Pass --tokenizer bpe for byte-level BPE.

Usage
-----
    # Character-level (default; instant)
    python scripts/prepare_data.py

    # Byte-level BPE (better compression, ~2–5 min on TinyShakespeare)
    python scripts/prepare_data.py --tokenizer bpe --vocab-size 4096

    # Use your own corpus
    python scripts/prepare_data.py --corpus /path/to/corpus.txt
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinygpt.tokenizer import BPETokenizer, CharTokenizer

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn"
    "/master/data/tinyshakespeare/input.txt"
)


def download_shakespeare(data_dir: str) -> str:
    out = Path(data_dir) / "corpus.txt"
    if out.exists():
        print(f"[Prepare] corpus already present: {out}")
        return str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print("[Prepare] downloading TinyShakespeare …")
    urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, out)
    print(f"[Prepare] {out.stat().st_size / 1024:.0f} KB saved → {out}")
    return str(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare corpus and tokenizer for TinyGPT")
    p.add_argument("--data-dir",        default="data",
                   help="Directory to store the corpus (default: data/)")
    p.add_argument("--corpus",          default=None,
                   help="Path to an existing text corpus (skips download)")
    p.add_argument("--tokenizer-path",  default="checkpoints/tokenizer.json",
                   help="Where to save the trained tokenizer")
    p.add_argument("--tokenizer",       choices=["char", "bpe"], default="char",
                   help="Tokenizer type: char (default) or bpe")
    p.add_argument("--vocab-size",      type=int, default=4096,
                   help="BPE vocabulary size (ignored for char)")
    args = p.parse_args()

    # ── Corpus ───────────────────────────────────────────────────────────
    if args.corpus:
        corpus_path = args.corpus
        print(f"[Prepare] using existing corpus: {corpus_path}")
    else:
        corpus_path = download_shakespeare(args.data_dir)

    text = Path(corpus_path).read_text(encoding="utf-8")
    print(f"[Prepare] corpus: {len(text):,} characters, {len(text.encode())//1024} KB")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    if args.tokenizer == "char":
        tok = CharTokenizer()
        tok.train(text)
    else:
        tok = BPETokenizer()
        tok.train(text, vocab_size=args.vocab_size)

    tok.save(args.tokenizer_path)

    # ── Round-trip sanity check ───────────────────────────────────────────
    sample    = text[:200]
    enc       = tok.encode(sample)
    dec       = tok.decode(enc)
    match     = sample == dec
    ratio     = len(enc) / max(1, len(sample))

    print(f"\n[Prepare] encode / decode round-trip (first 200 chars):")
    print(f"  original  : {repr(sample[:70])}")
    print(f"  decoded   : {repr(dec[:70])}")
    print(f"  lossless  : {match}")
    print(f"  ratio     : {ratio:.2f} tokens / char")
    print(f"  vocab     : {tok.vocab_size}")
    print(f"\n[Prepare] tokenizer saved → {args.tokenizer_path}")
    print("[Prepare] ready to train:  python scripts/train.py")


if __name__ == "__main__":
    main()
