# Hippo-encoder

Minimal framework for the original tiny-LLM idea:

- freeze a strong teacher encoder
- train a small language model to internalize that encoder's semantic space
- evaluate whether the tiny model preserves the teacher geometry well enough to act like a cheap encoder

## Core Idea

The primary path in this repo freezes a text encoder teacher and trains a small text model to predict:

- the teacher text embedding for the input text
- the teacher hidden-state summary

This is the reset architecture. The tiny model itself is the thing being trained to act like the teacher encoder.

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
python -m hippo_encoder.train --config configs/tiny_llm_reset.json
```

Bootstrap a few thousand plain text rows from Hugging Face:

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

Evaluate a trained tiny encoder checkpoint:

```bash
python scripts/eval_student_encoder.py \
  --student-checkpoint /path/to/checkpoint/epoch-2
```

This reports teacher-student cosine agreement, nearest-neighbor overlap, and a few paraphrase checks.

For a stronger training signal than `ag_news`, build pair/triplet supervision:

```bash
python scripts/prepare_pair_dataset.py \
  --source msmarco_triplet \
  --limit 50000 \
  --output data/train_pairs.jsonl
```

Supported sources:

- `msmarco_triplet`: retrieval-style query/positive/negative supervision
- `all_nli_pair`: semantic pair supervision
- `all_nli_triplet`: semantic triplet supervision

Then train with the pair-aware reset config:

```bash
python -m hippo_encoder.train --config configs/tiny_llm_pair_reset.json
```

## Experimental Region Work

The repo also contains region / delta / formula experiments. Those are exploratory and are not the primary reset architecture.

Benchmark the simplest direct delta representation on top of a trained checkpoint:

```bash
python scripts/benchmark_direct_delta_region.py \
  --cases benchmarks/sample_region_cases.json \
  --student-checkpoint /path/to/checkpoint/epoch-2
```

This uses the simplest anchored representation:

- derive dense `minus/plus` deltas from teacher positive spread
- keep those deltas fixed
- swap only the anchor from teacher to student
- measure how well the student anchor preserves IN/OUT behavior

Train a prompt-conditioned dense delta head on top of the best tiny encoder checkpoint:

```bash
python scripts/train_prompt_dense_delta.py \
  --student-checkpoint /path/to/checkpoint/epoch-2 \
  --cases benchmarks/sample_region_cases.json \
  --output-dir /tmp/prompt_dense_delta \
  --freeze-backbone
```

Benchmark the prompt-conditioned dense delta head:

```bash
python scripts/benchmark_prompt_dense_delta.py \
  --student-checkpoint /tmp/prompt_dense_delta/epoch-9 \
  --cases benchmarks/sample_region_cases.json
```

This is the first faithful prompted-latent version in the repo:

- prompt + query go into the tiny model
- the tiny model predicts dense `minus/plus` deltas
- the deltas are evaluated directly by positive/negative region behavior

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

Benchmark a dual-rope point region program across budgets:

```bash
python scripts/benchmark_rope_region.py \
  --cases benchmarks/sample_region_cases.json \
  --student-checkpoint /path/to/checkpoint/epoch-2 \
  --program-type formula \
  --budgets 16 32 64 128
```

This maps dimensions onto two deterministic ropes. Each region op is:

- `x`
- `y`
- `rope` (`0` or `1`)
- `value`

and the program applies exact points, local shapes, or local formula terms on that two-rope layout. The benchmark reports how teacher/student behavior changes as the term budget increases.

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

Benchmark Hippo-5-style soft-box retrieval with larger distractor pools:

```bash
python scripts/benchmark_hippo5_softbox_retrieval.py \
  --cases data/region_cases.json \
  --student-checkpoint /path/to/distill-bge-small/epoch-3 \
  --case-limit 100 \
  --distractors-per-case 1000 \
  --output /tmp/hippo5_softbox_retrieval.json
```

This reports retrieval-facing metrics such as top-1 accuracy, precision@K, recall@K, MRR, and score-threshold precision/recall using the same weighted soft-box violation formula as Hippo-5 `SearchSoftBox`.

Benchmark LoCoMo long-term memory evidence retrieval through the real Hippo-5 Go database:

```bash
python scripts/benchmark_locomo_hippo5_retrieval.py \
  --download \
  --student-checkpoint /path/to/distill-bge-small/epoch-3 \
  --hippo5-path /path/to/Hippo-5 \
  --output /tmp/locomo_hippo5_retrieval.json
```

This is a fairer memory-system test than region benchmarks: it indexes only conversation turns, queries with the LoCoMo question text, and scores whether the annotated evidence dialog IDs are retrieved at top-k.

## Model Artifact

The current baseline model is documented in `models/hippoencoder-bge-small-all-nli-pair-500k-c025-epoch3/`.

The full checkpoint is 130 MB, with a 127 MB `model.safetensors` file. That is larger than GitHub's normal 100 MB file limit, so distribute it as a GitHub Release asset or with Git LFS rather than committing raw weights to `main`.

Verify and unpack the artifact:

```bash
python scripts/verify_model_artifact.py \
  --manifest models/hippoencoder-bge-small-all-nli-pair-500k-c025-epoch3/manifest.json \
  --artifact /path/to/hippoencoder-bge-small-all-nli-pair-500k-c025-epoch3.tar.gz \
  --extract-to /tmp/hippoencoder-model
```

## Default Setup

- Teacher: `intfloat/e5-base-v2`
- Student: `BAAI/bge-small-en-v1.5`
- Losses: cosine matching to teacher text embeddings plus hidden-state matching

## Repo Layout

- `configs/tiny_llm_reset.json`: reset config for plain text distillation
- `configs/tiny_llm_pair_reset.json`: reset config for pair/triplet distillation
- `scripts/eval_student_encoder.py`: clean held-out evaluation for the distilled student encoder
- `scripts/prepare_pair_dataset.py`: prepare pair/triplet JSONL from stronger embedding datasets
- `scripts/benchmark_direct_delta_region.py`: simplest anchored dense delta benchmark
- `scripts/train_prompt_dense_delta.py`: train a prompt-conditioned dense delta head
- `scripts/benchmark_prompt_dense_delta.py`: benchmark a prompt-conditioned dense delta head
- `benchmarks/sample_region_cases.json`: sample IN/OUT benchmark cases
- `scripts/benchmark_region_membership.py`: region-membership benchmark for query/positive/negative cases
- `scripts/benchmark_formula_region.py`: formula-based region benchmark
- `scripts/benchmark_generate_formula_region.py`: direct model-generated formula benchmark
- `scripts/train_student_formula_head.py`: train a native student-owned formula head from region cases
- `scripts/train_student_formula_behavior.py`: train the native formula head directly on positive/negative region behavior
- `scripts/benchmark_student_formula_region.py`: benchmark a student-owned formula head
- `scripts/benchmark_group_region.py`: grouped anchor-preserving region benchmark
- `scripts/benchmark_rope_region.py`: two-rope point-region benchmark across budgets
- `scripts/benchmark_hippo5_softbox_retrieval.py`: Hippo-5-style soft-box retrieval benchmark with large distractor pools
- `scripts/benchmark_locomo_hippo5_retrieval.py`: LoCoMo long-term memory evidence retrieval through the real Hippo-5 Go database
- `scripts/verify_model_artifact.py`: verify and unpack the released Hippo-encoder checkpoint artifact
- `scripts/benchmark_region_program_size.py`: dense-vs-sparse program size benchmark
- `scripts/prepare_text_dataset.py`: download and export public text data to JSONL
- `scripts/prepare_region_cases.py`: build region training cases from text JSONL via teacher-space neighbors
- `prompts/ranged_formula_region_prompt.md`: prompt/spec for direct ranged-formula program generation
- `src/hippo_encoder/formula_region.py`: compact formula-based plus/minus region program
- `src/hippo_encoder/group_region.py`: grouped two-array region program
- `src/hippo_encoder/rope_region.py`: two-rope point/rectangle region programs
- `src/hippo_encoder/region.py`: sparse two-array region program, hydration, and scoring
- `src/hippo_encoder/data.py`: local text-only dataset loader
- `src/hippo_encoder/teacher.py`: frozen text teacher wrapper
- `src/hippo_encoder/student.py`: small language model encoder with projection heads
- `src/hippo_encoder/losses.py`: distillation objectives
- `src/hippo_encoder/train.py`: training entrypoint

## Next Steps

Useful upgrades from the reset path:

- use pair/triplet supervision instead of generic plain text whenever possible
- compare `BAAI/bge-small-en-v1.5` against other small students
- add richer teacher targets from intermediate layers instead of only pooled outputs
- expand held-out evaluation coverage and retrieval diagnostics
- only revisit delta / region generation after the tiny encoder itself is clearly strong
