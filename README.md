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

Benchmark region-style IN membership:

```bash
python scripts/benchmark_region_membership.py \
  --cases benchmarks/sample_region_cases.json \
  --student-checkpoint /path/to/checkpoint/epoch-2
```

This benchmark does not test top-k ranking. It tests whether a query-derived hypercube-style region includes the positives and excludes the negatives in each case.

The region representation itself now uses two sparse arrays:

- `minus`: per-dimension lower slack relative to the anchor embedding
- `plus`: per-dimension upper slack relative to the anchor embedding

These are stored as sparse range operations and hydrated into full 768-dimensional bounds at query time.

## Default Setup

- Teacher: `intfloat/e5-base-v2`
- Student: `BAAI/bge-small-en-v1.5`
- Losses: cosine matching to teacher text embeddings plus a contrastive term

## Repo Layout

- `configs/distill_clip_tiny.json`: example config
- `benchmarks/sample_region_cases.json`: sample IN/OUT benchmark cases
- `scripts/benchmark_region_membership.py`: region-membership benchmark for query/positive/negative cases
- `scripts/prepare_text_dataset.py`: download and export public text data to JSONL
- `src/hippo_encoder/region.py`: sparse two-array region program, hydration, and scoring
- `src/hippo_encoder/data.py`: local text-only dataset loader
- `src/hippo_encoder/teacher.py`: frozen text teacher wrapper
- `src/hippo_encoder/student.py`: small language model encoder with projection heads
- `src/hippo_encoder/losses.py`: distillation objectives
- `src/hippo_encoder/train.py`: training entrypoint

## Next Steps

Useful upgrades from here:

- compare `BAAI/bge-small-en-v1.5` against other small students
- add richer teacher targets from intermediate layers instead of only pooled outputs
- add a second positional scheme if you want to experiment with dual-RoPE token structure
- add evaluation for retrieval recall and nearest-neighbor alignment
