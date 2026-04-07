# Hippo-encoder

Minimal framework for distilling a text embedding teacher into a small language model that learns the teacher's semantic space.

## What This Does

The current training stack freezes a text encoder teacher and trains a small text model to predict:

- the teacher text embedding for the input text
- the teacher hidden-state summary

This is the simplest path toward a tiny model that starts to internalize another encoder's representation geometry without bringing in image data yet.

## Dataset Format

Provide a local JSONL file where each row looks like:

```json
{"text":"a red chair near the window"}
```

- `text` is the caption, query, or short passage you want to encode

## Quick Start

Install dependencies:

```bash
pip install -e .
```

Run training:

```bash
python -m hippo_encoder.train --config configs/distill_clip_tiny.json
```

Bootstrap a few thousand public text rows from Hugging Face:

```bash
python scripts/prepare_text_dataset.py \
  --dataset ag_news \
  --split train \
  --text-column text \
  --limit 5000 \
  --prefix "passage: " \
  --output data/train.jsonl
```

For Google Colab, use [Hippo_Encoder_Colab.ipynb](/home/cameron/projects/Hippo-encoder/Hippo_Encoder_Colab.ipynb).

## Default Setup

- Teacher: `intfloat/e5-base-v2`
- Student: `distilgpt2`
- Losses: cosine matching to teacher text embeddings plus a contrastive term

## Repo Layout

- `configs/distill_clip_tiny.json`: example config
- `scripts/prepare_text_dataset.py`: download and export public text data to JSONL
- `src/hippo_encoder/data.py`: local text-only dataset loader
- `src/hippo_encoder/teacher.py`: frozen text teacher wrapper
- `src/hippo_encoder/student.py`: small language model encoder with projection heads
- `src/hippo_encoder/losses.py`: distillation objectives
- `src/hippo_encoder/train.py`: training entrypoint

## Next Steps

Useful upgrades from here:

- swap `distilgpt2` for a stronger sub-1B student
- add richer teacher targets from intermediate layers instead of only pooled outputs
- add a second positional scheme if you want to experiment with dual-RoPE token structure
- add evaluation for retrieval recall and nearest-neighbor alignment
