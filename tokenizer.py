"""
Tokenizers built entirely from scratch.

Two implementations are provided:

  CharTokenizer  — character-level, O(N) training, instant.
                   Vocabulary = sorted unique chars in corpus.
                   Good default for quick experiments on small corpora.

  BPETokenizer   — byte-level BPE (Sennrich et al. 2016 / GPT-2 variant).
                   Starts with 256 single-byte tokens; learns merge rules by iteratively fusing the most frequent adjacent pair.
                   Better compression; takes O(N·M) time to train.

References:
  Sennrich, Haddow & Birch (2016) "Neural Machine Translation of Rare Words with Subword Units"  arXiv:1508.07909
  Radford et al. (2019) GPT-2 — byte-level BPE with no <unk>.
  Karpathy (2023) minBPE  github.com/karpathy/minbpe
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _apply_merge(ids: List[int], a: int, b: int, new_id: int) -> List[int]:
    """
    Replace every non-overlapping occurrence of the bigram (a, b) in `ids`
    with `new_id`.  Single left-to-right pass: O(N).

    Example
    -------
    _apply_merge([1, 2, 3, 1, 2], 1, 2, 99) → [99, 3, 99]
    """
    out: List[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Character-level tokenizer
# ─────────────────────────────────────────────────────────────────────────────


class CharTokenizer:
    """
    Trivial character-level tokenizer.

    Vocabulary is the sorted set of unique characters found in the training corpus.  Encode/decode are O(N) with no pre-processing overhead.

    Limitations: unknown characters at inference time are mapped to token 0. For Shakespeare this is fine; the full corpus is used for training.

    Serialisation format (JSON):
        { "type": "char", "stoi": { "<char>": <id>, ... } }
    """

    def __init__(self) -> None:
        self._stoi: Dict[str, int] = {}
        self._itos: Dict[int, str] = {}
        self._trained: bool = False

    # ── Training ──────────────────────────────────────────────────────────

    def train(self, text: str) -> None:
        """Build vocabulary from all unique characters in `text`."""
        chars = sorted(set(text))
        self._stoi = {c: i for i, c in enumerate(chars)}
        self._itos = dict(enumerate(chars))
        self._trained = True
        print(f"[CharTokenizer] vocab_size={len(chars)}")

    # ── Encode / Decode ───────────────────────────────────────────────────

    def encode(self, text: str) -> List[int]:
        """Map each character to its integer ID.  Unknown chars → 0."""
        if not self._trained:
            raise RuntimeError("CharTokenizer is not trained. Call train() or load() first.")
        return [self._stoi.get(c, 0) for c in text]

    def decode(self, ids: List[int]) -> str:
        """Reconstruct text from a list of token IDs.  Unknown IDs → '?'."""
        return "".join(self._itos.get(i, "?") for i in ids)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "char", "stoi": self._stoi}, f, ensure_ascii=False)
        print(f"[CharTokenizer] saved → {path}")

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("type") != "char":
            raise ValueError(f"Not a CharTokenizer file: {path!r} (type={data.get('type')!r})")
        tok = cls()
        tok._stoi = data["stoi"]
        tok._itos = {v: k for k, v in tok._stoi.items()}
        tok._trained = True
        return tok

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self._stoi)

    def __len__(self) -> int:
        return self.vocab_size


# ─────────────────────────────────────────────────────────────────────────────
# Byte-level BPE tokenizer
# ─────────────────────────────────────────────────────────────────────────────


class BPETokenizer:
    """
    Byte-level Byte-Pair Encoding tokenizer.

    Algorithm (Sennrich et al. 2016, adapted to bytes as in GPT-2):

      1. Initialise vocabulary with 256 single-byte tokens (IDs 0–255). Every possible byte value is covered → no <unk> token needed.

      2. Encode the training corpus as a flat list of byte IDs.

      3. Repeat `vocab_size − 256` times:
           a. Count all adjacent token pairs.
           b. Select the most frequent pair (a, b).
           c. Assign a new token ID `new_id = 256 + step`.
           d. Record the merge rule: merges[(a, b)] = new_id.
           e. Replace every (a, b) in the corpus with new_id.

      4. To encode new text at inference: convert to bytes, then greedily apply the learnt merge rules in the order they were learned (lowest merge ID first = earliest / most frequent).

    Serialisation format (JSON):
        {
          "type": "bpe",
          "vocab":  { "<id>": [<byte>, ...], ... },
          "merges": [[a, b, new_id], ...]
        }

    Complexity
    ----------
    Training : O(N · M)  where N = |corpus bytes|, M = n_merges
    Encoding : O(N · M)  per string (worst case; typically much less)
    """

    def __init__(self) -> None:
        self.vocab:  Dict[int, bytes]            = {}   # id  → bytes
        self.merges: Dict[Tuple[int, int], int]  = {}   # (a,b) → new_id
        self._trained: bool = False

    # ── Training ──────────────────────────────────────────────────────────

    def train(self, text: str, vocab_size: int = 4096, verbose: bool = True) -> None:
        """
        Learn BPE merge rules from raw text.

        Parameters
        ----------
        text       : full training corpus as a plain string
        vocab_size : target vocabulary size (must be > 256)
        verbose    : print progress every 500 merges
        """
        if vocab_size <= 256:
            raise ValueError(
                f"vocab_size must be > 256 (256 base bytes required), got {vocab_size}"
            )
        n_merges = vocab_size - 256

        # ── Initialise base vocabulary ────────────────────────────────────
        self.vocab  = {i: bytes([i]) for i in range(256)}
        self.merges = {}

        # ── Encode corpus as byte IDs ─────────────────────────────────────
        ids: List[int] = list(text.encode("utf-8"))
        if verbose:
            print(f"[BPE] corpus: {len(text):,} chars → {len(ids):,} bytes")
            print(f"[BPE] learning {n_merges:,} merge rules …")

        for step in range(n_merges):
            # Count all adjacent bigrams
            stats: Dict[Tuple[int, int], int] = defaultdict(int)
            prev = ids[0]
            for cur in ids[1:]:
                stats[(prev, cur)] += 1
                prev = cur

            if not stats:
                if verbose:
                    print(f"[BPE] no more pairs at step {step}. stopping.")
                break

            # Select most frequent pair
            best = max(stats, key=stats.__getitem__)
            a, b = best
            new_id = 256 + step
            merged_bytes = self.vocab[a] + self.vocab[b]

            # Register new token
            self.vocab[new_id] = merged_bytes
            self.merges[best] = new_id

            # Replace all occurrences in corpus
            ids = _apply_merge(ids, a, b, new_id)

            if verbose and (step + 1) % 500 == 0:
                pct = 100.0 * (step + 1) / n_merges
                txt = merged_bytes.decode("utf-8", "replace")
                print(
                    f"[BPE]  step {step+1:>5}/{n_merges}  ({pct:4.0f}%)"
                    f"  {txt!r:24s}  id={new_id}  freq={stats[best]}"
                )

        self._trained = True
        if verbose:
            print(f"[BPE] done. vocab_size={len(self.vocab)}")

    # ── Encode / Decode ───────────────────────────────────────────────────

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.

        Converts text to UTF-8 bytes, then applies merge rules greedily in the order they were learned (lowest new_id = earliest merge first).
        
        This is equivalent to the priority-based BPE decoding.
        """
        if not self._trained:
            raise RuntimeError("BPETokenizer is not trained. Call train() or load() first.")
        ids: List[int] = list(text.encode("utf-8"))

        while True:
            # Find the applicable merge with the smallest new_id
            # (earliest learned = highest frequency = should be applied first)
            best_new_id = float("inf")
            best_pos    = -1
            for i in range(len(ids) - 1):
                mid = self.merges.get((ids[i], ids[i + 1]))
                if mid is not None and mid < best_new_id:
                    best_new_id = mid
                    best_pos    = i

            if best_pos == -1:
                break  # No applicable merges remain

            a, b = ids[best_pos], ids[best_pos + 1]
            ids  = _apply_merge(ids, a, b, int(best_new_id))

        return ids

    def decode(self, ids: List[int]) -> str:
        """Reconstruct text from token IDs via byte concatenation."""
        raw = b"".join(self.vocab.get(i, b"\xef\xbf\xbd") for i in ids)  # U+FFFD for unknown
        return raw.decode("utf-8", errors="replace")

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Serialise to compact JSON (merges as [a, b, new_id] triples)."""
        data = {
            "type":   "bpe",
            "vocab":  {str(k): list(v) for k, v in self.vocab.items()},
            "merges": [[a, b, c] for (a, b), c in self.merges.items()],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        print(f"[BPE] saved → {path}")

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("type") != "bpe":
            raise ValueError(f"Not a BPETokenizer file: {path!r} (type={data.get('type')!r})")
        tok = cls()
        tok.vocab   = {int(k): bytes(v) for k, v in data["vocab"].items()}
        tok.merges  = {(int(a), int(b)): int(c) for a, b, c in data["merges"]}
        tok._trained = True
        return tok

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __len__(self) -> int:
        return self.vocab_size


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────


def load_tokenizer(path: str):
    """
    Load either a CharTokenizer or BPETokenizer from a saved JSON file.

    Reads the "type" field to dispatch to the correct class.
    """
    with open(path, encoding="utf-8") as f:
        tok_type = json.load(f).get("type")

    if tok_type == "char":
        return CharTokenizer.load(path)
    elif tok_type == "bpe":
        return BPETokenizer.load(path)
    else:
        raise ValueError(
            f"Unknown tokenizer type {tok_type!r} in {path!r}. "
            "Expected 'char' or 'bpe'."
        )
