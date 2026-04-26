from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_file(artifact: Path, output_dir: Path, chunk_size: int) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = artifact.name
    parts = []
    with artifact.open("rb") as handle:
        index = 0
        while True:
            data = handle.read(chunk_size)
            if not data:
                break
            index += 1
            part_name = f"{stem}.part{index:03d}"
            part_path = output_dir / part_name
            part_path.write_bytes(data)
            parts.append(
                {
                    "index": index,
                    "path": str(part_path.relative_to(output_dir.parent)),
                    "filename": part_name,
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a model artifact into GitHub-sized chunks.")
    parser.add_argument("--artifact", required=True, help="Artifact file to split.")
    parser.add_argument("--output-dir", required=True, help="Directory where chunks should be written.")
    parser.add_argument("--manifest", required=True, help="Chunk manifest JSON to write.")
    parser.add_argument("--chunk-size-mb", type=int, default=45, help="Chunk size in MiB.")
    args = parser.parse_args()

    artifact = Path(args.artifact)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest)
    chunk_size = args.chunk_size_mb * 1024 * 1024

    parts = split_file(artifact, output_dir, chunk_size)
    payload = {
        "artifact_filename": artifact.name,
        "artifact_size_bytes": artifact.stat().st_size,
        "artifact_sha256": sha256_file(artifact),
        "chunk_size_bytes": chunk_size,
        "part_count": len(parts),
        "parts": parts,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
