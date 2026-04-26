# Hippoencoder 500k C025 Epoch 3

This is the current baseline Hippo-encoder checkpoint used for the Hippo-5 retrieval and LoCoMo memory tests.

The weight artifact is not committed directly to normal Git because `backbone/model.safetensors` is 127 MB, which exceeds GitHub's 100 MB normal file limit. Store the tarball as a GitHub Release asset or via Git LFS.

## Artifact

- File: `hippoencoder-bge-small-all-nli-pair-500k-c025-epoch3.tar.gz`
- Colab/Drive path: `/content/drive/MyDrive/hippo_encoder_runs/hippoencoder-bge-small-all-nli-pair-500k-c025-epoch3.tar.gz`
- Size: `124506335` bytes
- SHA256: `629a57921357a1bf62646ec4ced78d01f3914e299ab118dedf54c7e0cb691837`

An `xz -9e` artifact is also available at `/content/drive/MyDrive/hippo_encoder_runs/hippoencoder-bge-small-all-nli-pair-500k-c025-epoch3.tar.xz`. It is `120507304` bytes with SHA256 `22ca9b219497fb8e7ea9c45df6d30608137fd2a8e50db3d290ddea8bd8a438d0`, so stronger compression still does not fit under GitHub's normal 100 MB limit.

## Use

After downloading or copying the artifact locally:

```bash
python scripts/verify_model_artifact.py \
  --manifest models/hippoencoder-bge-small-all-nli-pair-500k-c025-epoch3/manifest.json \
  --artifact /path/to/hippoencoder-bge-small-all-nli-pair-500k-c025-epoch3.tar.gz \
  --extract-to /tmp/hippoencoder-model
```

Then load the extracted checkpoint:

```python
import torch
from hippo_encoder.student import TinyEncoderStudent

model = TinyEncoderStudent.load_checkpoint(
    "/tmp/hippoencoder-model/hippoencoder-bge-small-all-nli-pair-500k-c025-epoch3",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
```

## Key Results

- Synthetic Hippo-5 softbox retrieval, 100 queries, 20 positives/query, 1000 distractors/query: Top-20 precision `82.55%`, Top-20 recall `82.55%`.
- LoCoMo evidence retrieval through actual Hippo-5 Go DB: Top-5 hit rate `54.41%`, Top-20 hit rate `73.09%`, Top-50 hit rate `84.13%`.
