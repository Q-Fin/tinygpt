# TinyGPT

A **~10–14 M parameter GPT-style decoder-only transformer** built entirely from
scratch in PyTorch. No pre-trained weights, no Hugging Face, no shortcuts.

Designed to train comfortably on a modest CPU or an Intel Iris Xe integrated
GPU with shared system RAM.

---

## Architecture

The model implements the GPT-2 decoder-only transformer (Radford et al. 2019)
with Pre-LayerNorm (Xiong et al. 2020).

```
TinyGPT
├── Token Embedding      E_tok  ∈ ℝ^{V × d}
├── Position Embedding   E_pos  ∈ ℝ^{T × d}   (learned, not sinusoidal)
├── Dropout
│
├── N × TransformerBlock
│   ├── LayerNorm₁
│   ├── CausalSelfAttention
│   │     Fused QKV projection  (d → 3d, no bias)
│   │     Reshape to H heads, d_k = d/H per head
│   │     scores = QKᵀ / √d_k
│   │     causal mask (lower-triangular)
│   │     softmax → dropout → weighted sum of V
│   │     Output projection  (d → d, no bias)
│   │     Residual dropout
│   ├── LayerNorm₂
│   └── FeedForward
│         fc1 : d → 4d  (GELU)
│         fc2 : 4d → d
│         Dropout
│
├── Final LayerNorm
└── LM Head  (d → V, no bias, weight-tied to E_tok)
```

### Parameter budget

Default configuration: `vocab=65, d=384, H=6, L=6, d_ff=1536, T=256`

| Component                      | Params      |
|--------------------------------|-------------|
| Token embedding  (65 × 384)   |      24,960 |
| Position embedding (256 × 384) |      98,304 |
| **6 × TransformerBlock**       | **10,626,048** |
| · LN×2 per block (2 × 768)    |       1,536 |
| · QKV proj (384 × 1152)       |     442,368 |
| · Out proj (384 × 384)        |     147,456 |
| · FFN fc1  (384 × 1536)       |     589,824 |
| · FFN fc2  (1536 × 384)       |     589,824 |
| Final LayerNorm                |         768 |
| LM Head (weight-tied → 0)     |           0 |
| **Total**                      | **10,750,080** |

With BPE tokenizer (`vocab=4096`) the total rises to **~12.3 M**.

Memory in float32 training (weights + gradients + Adam m/v):
```
10.75 M × 4 bytes × 4 ≈ 172 MB
```
This fits in shared system RAM on any modern machine.

---

## Design decisions

| Choice | Rationale |
|---|---|
| Pre-LayerNorm | More stable gradients than post-LN (Xiong et al. 2020) |
| No bias in QKV/FFN | Follows GPT-2; fewer params, comparable performance |
| Weight tying (lm\_head ↔ token\_emb) | Press & Wolf 2017; saves V×d params, regularises |
| GELU activation | Better than ReLU for language tasks (GPT, BERT) |
| Learned positional embeddings | Simpler than sinusoidal; works for fixed T |
| AdamW betas (0.9, 0.95) | Recommended for large-batch LM (GPT-3, Chinchilla) |
| Cosine LR decay + warmup | Near-optimal schedule (Hoffmann et al. 2022) |
| Weight decay on 2-D weights only | Embeddings and LN params are exempt (standard practice) |
| Scaled residual init | Prevents variance blow-up with depth (GPT-2 §2.3) |

---

## References

| Paper | Notes |
|---|---|
| Vaswani et al. (2017) "Attention Is All You Need" arXiv:1706.03762 | Core attention mechanism |
| Radford et al. (2019) "Language Models are Unsupervised Multitask Learners" (GPT-2) | Architecture, init, weight tying |
| Ba et al. (2016) "Layer Normalization" arXiv:1607.06450 | LayerNorm formulation |
| Xiong et al. (2020) "On Layer Normalization in the Transformer Architecture" arXiv:2002.04745 | Pre-LN motivation |
| Sennrich et al. (2016) "Neural Machine Translation of Rare Words with Subword Units" arXiv:1508.07909 | BPE algorithm |
| Press & Wolf (2017) "Using the Output Embedding to Improve Language Models" arXiv:1608.05859 | Weight tying |
| Loshchilov & Hutter (2019) "Decoupled Weight Decay Regularization" arXiv:1711.05101 | AdamW |
| Pascanu et al. (2013) "On the difficulty of training recurrent neural networks" | Gradient clipping |
| Karpathy (2022) nanoGPT github.com/karpathy/nanoGPT | Reference implementation |

---

## Setup

```bash
pip install torch>=2.0.0
```

**Intel Iris Xe / Intel GPU or whataver your hardware is (mine is the Intel Iris, but this is still optional, for XPU acceleration):**
```bash
pip install intel-extension-for-pytorch
```
Without IPEX the trainer automatically falls back to CPU, which trains this
model comfortably using shared system RAM.

---

## Quick start

```bash
# 1. Download TinyShakespeare + train tokenizer (a few seconds)
python scripts/prepare_data.py

# 2. Train (5 000 steps; ~30–90 min on CPU depending on hardware)
python scripts/train.py

# 3. Generate text
python scripts/generate.py --prompt "HAMLET:"
```

---

## Directory structure

```
tinygpt/
├── tinygpt/               Python package (the model itself)
│   ├── __init__.py        public API
│   ├── config.py          ModelConfig, TrainConfig dataclasses
│   ├── model.py           LayerNorm, CausalSelfAttention, FeedForward,
│   │                      TransformerBlock, TinyGPT
│   ├── tokenizer.py       CharTokenizer, BPETokenizer, load_tokenizer
│   ├── dataset.py         TokenDataset, prepare_datasets, make_loader
│   └── trainer.py         Trainer (AdamW, cosine schedule, checkpointing)
│
├── scripts/
│   ├── prepare_data.py    download corpus + train tokenizer
│   ├── train.py           training entry point
│   └── generate.py        inference / text generation
│
├── data/
│   └── corpus.txt         created by prepare_data.py
│
├── checkpoints/
│   ├── tokenizer.json     saved tokenizer
│   ├── ckpt_best.pt       checkpoint with lowest val loss
│   ├── ckpt_latest.pt     most recent checkpoint
│   └── log.jsonl          step-by-step training log (JSON-lines)
│
├── requirements.txt
└── README.md
```

---

## Configuration reference

### `prepare_data.py`

| Flag | Default | Description |
|---|---|---|
| `--tokenizer` | `char` | `char` (instant) or `bpe` (better compression) |
| `--vocab-size` | `4096` | BPE vocabulary size (ignored for char) |
| `--corpus` | (download) | Path to your own corpus instead of Shakespeare |
| `--tokenizer-path` | `checkpoints/tokenizer.json` | Output path |

### `train.py`

| Flag | Default | Description |
|---|---|---|
| `--steps` | `5000` | Total optimiser steps |
| `--batch-size` | `32` | Mini-batch size |
| `--grad-accum` | `1` | Gradient accumulation (eff. batch = batch × accum) |
| `--lr` | `3e-4` | Peak learning rate |
| `--warmup` | `200` | Linear LR warmup steps |
| `--d-model` | `384` | Embedding / hidden dimension |
| `--n-heads` | `6` | Attention heads |
| `--n-layers` | `6` | Transformer blocks |
| `--context-len` | `256` | Max sequence length |
| `--device` | `auto` | `auto` \| `cpu` \| `cuda` \| `xpu` \| `mps` |
| `--resume` | off | Resume from `ckpt_latest.pt` |
| `--compile` | off | `torch.compile` (PyTorch ≥ 2.0; skip on XPU) |

### `generate.py`

| Flag | Default | Description |
|---|---|---|
| `--prompt` | `""` | Seed text |
| `--max-tokens` | `300` | Tokens to generate |
| `--temperature` | `0.8` | Logit scale (< 1 = sharper, > 1 = more random) |
| `--top-k` | `40` | Top-k filter (0 = disabled) |
| `--top-p` | `None` | Nucleus sampling threshold |
| `--checkpoint` | `best` | `best` or `latest` |

---

## Intel Iris Xe notes (my own hardware, you should look at your CPU/GPU documentation)

Intel Iris Xe integrated graphics do **not** have CUDA support.  Three training
paths are available:

### Path 1: CPU (no extra packages, recommended)
Training happens on CPU using system RAM (typically 8–32 GB shared).
The model weights + Adam states for 10.75 M params fit in ~200 MB.

### Path 2: Intel XPU via IPEX
```bash
# Check compatible versions at: https://github.com/intel/intel-extension-for-pytorch
pip install intel-extension-for-pytorch
python scripts/train.py --device xpu
# Do NOT pass --compile on XPU (torch.compile has limited XPU support)
```
The trainer auto-detects XPU when IPEX is installed.

### Path 3: DirectML on Windows (community)
Install `torch-directml` and pass `--device privateuseone`.

### Recommended training settings for Iris Xe / CPU
```bash
python scripts/train.py \
    --steps 20000 \
    --batch-size 16 \
    --grad-accum 2 \
    --lr 3e-4 \
    --context-len 256
```
The effective batch size is 32 (16 × 2 accum), matching the default but using
half the peak memory.

---

## Expected training behaviour (char-level, TinyShakespeare)

| Step | Train loss | Val loss | PPL | Notes |
|---|---|---|---|---|
| 0 | ~4.17 | ~4.17 | ~65 | Random baseline: log(65) |
| 200 | ~2.5 | ~2.5 | ~12 | After warmup |
| 1 000 | ~1.8 | ~1.9 | ~6.7 | Sentence structure forming |
| 5 000 | ~1.4 | ~1.5 | ~4.5 | Recognisable Shakespearean text |
| 20 000 | ~1.2 | ~1.3 | ~3.7 | Better character consistency |

> nanoGPT achieves ~1.47 bits/char (≈ 1.02 nats/char) on Shakespeare with a
> similar-scale model.  My target is ~1.3–1.5 nats/char after 20 000 steps.

---

## Monitoring training

The trainer writes a JSON-lines log to `checkpoints/log.jsonl`:

```jsonl
{"step": 50,  "train_loss": 3.812, "lr": 7.5e-05, "elapsed": 12.3}
{"step": 100, "train_loss": 3.201, "lr": 1.5e-04, "elapsed": 24.7}
{"step": 500, "val_loss": 2.431, "ppl": 11.38}
```

Plot with Python:

```python
import json, matplotlib.pyplot as plt

train, val = [], []
with open("checkpoints/log.jsonl") as f:
    for line in f:
        d = json.loads(line)
        if "train_loss" in d: train.append((d["step"], d["train_loss"]))
        if "val_loss"   in d: val.append(  (d["step"], d["val_loss"]))

plt.plot(*zip(*train), label="train")
plt.plot(*zip(*val),   label="val", marker="o")
plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.show()
```

---

## Using your own corpus

```bash
# Point prepare_data.py at any UTF-8 text file
python scripts/prepare_data.py --corpus /path/to/mybook.txt

# Train as usual
python scripts/train.py
```

For corpora > 50 MB, consider BPE tokenisation (`--tokenizer bpe`) to reduce
sequence lengths and speed up training, at the cost of a ~2–5 minute
one-time tokenizer training step.

---

## Programmatic usage

```python
from tinygpt import TinyGPT, ModelConfig, CharTokenizer, Trainer, TrainConfig
from tinygpt import prepare_datasets

# Build tokenizer
tok = CharTokenizer()
tok.train(open("data/corpus.txt").read())
tok.save("checkpoints/tokenizer.json")

# Build model
cfg   = ModelConfig(vocab_size=tok.vocab_size)
model = TinyGPT(cfg)
print(f"{model.count_parameters():,} parameters")   # 10,750,080

# Prepare data
train_ds, val_ds = prepare_datasets("data/corpus.txt", tok, cfg.context_len)

# Train
tcfg    = TrainConfig(max_steps=5000)
trainer = Trainer(model, train_ds, val_ds, tcfg)
trainer.train()

# Generate
import torch
prompt = torch.tensor([tok.encode("HAMLET:")], dtype=torch.long)
out    = model.generate(prompt, max_new_tokens=200, temperature=0.8, top_k=40)
print(tok.decode(out[0].tolist()))
```
