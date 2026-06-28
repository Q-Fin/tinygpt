"""
Dataset utilities for causal language model pre-training.

The standard GPT pre-training objective is next-token prediction:
Given a sequence of T tokens, predict token t+1 from tokens 0..t for all positions t simultaneously.

This module provides:
  TokenDataset     — sliding-window (x, y) dataset backed by a 1-D LongTensor
  prepare_datasets — tokenise a text file and split into train / val
  make_loader      — thin wrapper around DataLoader with sensible defaults
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

from .tokenizer import BPETokenizer, CharTokenizer

Tokenizer = Union[CharTokenizer, BPETokenizer]


class TokenDataset(Dataset):
    """
    Fixed-length sliding-window next-token-prediction dataset.

    Given a flat token sequence of length N, the dataset contains N − context_len − 1 samples.  Each sample is a pair:

        x = tokens[i   : i + context_len]        input
        y = tokens[i+1 : i + context_len + 1]    targets (x shifted right by 1)

    Prediction at position t uses only context tokens 0..t (causal LM).

    Memory layout
    -------------
    The full token sequence is stored as a single contiguous 1-D LongTensor.
    Items are created with .clone() to avoid issues with pinned-memory DataLoaders and multi-process workers that may share the underlying tensor.
    """

    def __init__(self, tokens: list, context_len: int) -> None:
        self.data        = torch.tensor(tokens, dtype=torch.long)
        self.context_len = context_len

    def __len__(self) -> int:
        # Last valid start index: len(data) - context_len - 1
        return max(0, len(self.data) - self.context_len - 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.context_len + 1]
        return chunk[:-1].clone(), chunk[1:].clone()


def prepare_datasets(
    text_path: str,
    tokenizer: Tokenizer,
    context_len: int,
    train_split: float = 0.9,
) -> Tuple[TokenDataset, TokenDataset]:
    """
    Load a UTF-8 text file, tokenise it, and return train/val datasets.

    The split is performed on the flat token sequence (not on document
    boundaries), which is standard practice for single-file corpora.

    Parameters
    ----------
    text_path   : path to raw UTF-8 text corpus
    tokenizer   : trained CharTokenizer or BPETokenizer
    context_len : model's maximum sequence length
    train_split : fraction of tokens for training (remainder → validation)

    Returns
    -------
    train_dataset, val_dataset
    """
    text   = Path(text_path).read_text(encoding="utf-8")
    tokens = tokenizer.encode(text)
    n      = len(tokens)
    split  = int(n * train_split)

    print(
        f"[Data] {n:,} tokens total"
        f" — train: {split:,}  val: {n - split:,}"
        f"  (context_len={context_len})"
    )

    if n - split < context_len + 1:
        print("[Data] WARNING: validation set is smaller than one context window.")

    return (
        TokenDataset(tokens[:split],  context_len),
        TokenDataset(tokens[split:],  context_len),
    )


def make_loader(
    dataset: TokenDataset,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """
    Build a DataLoader with sensible defaults for LM training.

    num_workers=0 is intentional: avoids fork-related issues on Windows, macOS MPS, and Intel XPU.  For large corpora on Linux you can increase this, but for our dataset sizes (< 10 MB) it makes no measurable difference.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,   # enable if using CUDA; can hurt XPU/MPS
        drop_last=True,     # ensures all batches are the same size
    )
