# Hippo-encoder

Minimal framework for distilling an open-source CLIP teacher into a small language model that learns the teacher's embedding space.

## What This Does

The current training stack freezes a CLIP teacher and trains a small text model to predict:

- the teacher image embedding for a paired caption
- the teacher text embedding for that same caption
- the teacher text hidden state summary

This is a practical first step toward a tiny model that "understands" the encoder geometry instead of trying to reason over raw encoder weights directly.

## Dataset Format

Provide a local JSONL file where each row looks like:

```json
{"image":"example.jpg","text":"a red chair near the window"}
```

- `image` is relative to `image_root`
- `text` is the caption or query text

## Quick Start

Install dependencies:

```bash
pip install -e .
```

Run training:

```bash
python -m hippo_encoder.train --config configs/distill_clip_tiny.json
```

For Google Colab, use [Hippo_Encoder_Colab.ipynb](/home/cameron/projects/Hippo-encoder/Hippo_Encoder_Colab.ipynb).

## Default Setup

- Teacher: `openai/clip-vit-base-patch32`
- Student: `distilgpt2`
- Losses: cosine matching to teacher image/text embeddings plus a contrastive term

## Repo Layout

- `configs/distill_clip_tiny.json`: example config
- `src/hippo_encoder/data.py`: local image-text dataset loader
- `src/hippo_encoder/teacher.py`: frozen CLIP teacher wrapper
- `src/hippo_encoder/student.py`: small language model encoder with projection heads
- `src/hippo_encoder/losses.py`: distillation objectives
- `src/hippo_encoder/train.py`: training entrypoint

## Next Steps

Useful upgrades from here:

- swap `distilgpt2` for a stronger sub-1B student
- add region or patch-level targets instead of only pooled teacher outputs
- add a second positional scheme if you want to experiment with dual-RoPE token structure
- add evaluation for retrieval recall and nearest-neighbor alignment
