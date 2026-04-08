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

Build region-training cases from a text JSONL file:

```bash
python scripts/prepare_region_cases.py \
  --input-jsonl data/train.jsonl \
  --output data/region_cases.json \
  --num-cases 2000
```

This embeds the text rows with the teacher, picks near neighbors as positives, and samples hard plus easy negatives to produce `query / positives / negatives` supervision for the formula head.

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

Estimate how verbose the region program would be to generate:

```bash
python scripts/benchmark_region_program_size.py \
  --cases benchmarks/sample_region_cases.json
```

This compares a raw dense dump of all `minus/plus` dimension values against the sparse range-op format and reports rough token-count savings.

Benchmark a grouped anchor-preserving region program:

```bash
python scripts/benchmark_group_region.py \
  --cases benchmarks/sample_region_cases.json \
  --student-checkpoint /path/to/checkpoint/epoch-2 \
  --group-size 16
```

This keeps the encoder vector as the anchor and replaces raw per-dimension bounds with grouped `minus/plus` arrays over fixed blocks of dimensions.

Benchmark a ranged formula-based region program:

```bash
python scripts/benchmark_formula_region.py \
  --cases benchmarks/sample_region_cases.json \
  --max-terms-per-side 24
```

This fits a compact local-formula DSL to the `minus/plus` arrays with bounded terms such as constants, ramps, and Gaussian bumps over explicit dimension ranges. This is closer to "draw the shape you want" than the earlier global-only formula.

For direct model generation, use [ranged_formula_region_prompt.md](/home/cameron/projects/Hippo-encoder/prompts/ranged_formula_region_prompt.md) as the starting prompt/spec. It explicitly encourages broad blanket margins plus local refinements instead of overly conservative shapes.

Benchmark direct model-generated formula regions:

```bash
python scripts/benchmark_generate_formula_region.py \
  --cases benchmarks/sample_region_cases.json \
  --generator-model Qwen/Qwen2.5-1.5B-Instruct
```

This uses an instruction model to emit the ranged-formula JSON directly, validates it, hydrates it, and scores it on the same IN/OUT benchmark cases.

Train a native student formula head from query/positive/negative cases:

```bash
python scripts/train_student_formula_head.py \
  --student-checkpoint /path/to/distill-bge-small/epoch-2 \
  --cases benchmarks/sample_region_cases.json \
  --output-dir /tmp/hippo_formula_head \
  --freeze-backbone
```

Train the formula head directly against region behavior:

```bash
python scripts/train_student_formula_behavior.py \
  --student-checkpoint /path/to/distill-bge-small/epoch-2 \
  --cases data/region_cases.json \
  --output-dir /tmp/hippo_formula_behavior \
  --freeze-backbone
```

Benchmark the student-owned formula head:

```bash
python scripts/benchmark_student_formula_region.py \
  --student-checkpoint /tmp/hippo_formula_head/epoch-199 \
  --cases benchmarks/sample_region_cases.json
```

## Default Setup

- Teacher: `intfloat/e5-base-v2`
- Student: `BAAI/bge-small-en-v1.5`
- Losses: cosine matching to teacher text embeddings plus a contrastive term

## Repo Layout

- `configs/distill_clip_tiny.json`: example config
- `benchmarks/sample_region_cases.json`: sample IN/OUT benchmark cases
- `scripts/benchmark_region_membership.py`: region-membership benchmark for query/positive/negative cases
- `scripts/benchmark_formula_region.py`: formula-based region benchmark
- `scripts/benchmark_generate_formula_region.py`: direct model-generated formula benchmark
- `scripts/train_student_formula_head.py`: train a native student-owned formula head from region cases
- `scripts/train_student_formula_behavior.py`: train the native formula head directly on positive/negative region behavior
- `scripts/benchmark_student_formula_region.py`: benchmark a student-owned formula head
- `scripts/benchmark_group_region.py`: grouped anchor-preserving region benchmark
- `scripts/benchmark_region_program_size.py`: dense-vs-sparse program size benchmark
- `scripts/prepare_text_dataset.py`: download and export public text data to JSONL
- `scripts/prepare_region_cases.py`: build region training cases from text JSONL via teacher-space neighbors
- `prompts/ranged_formula_region_prompt.md`: prompt/spec for direct ranged-formula program generation
- `src/hippo_encoder/formula_region.py`: compact formula-based plus/minus region program
- `src/hippo_encoder/group_region.py`: grouped two-array region program
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
