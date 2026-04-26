from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in tar.getmembers():
        target = (destination / member.name).resolve()
        if destination != target and destination not in target.parents:
            raise ValueError(f"Unsafe tar member path: {member.name}")
    tar.extractall(destination)


def verify_files(root: Path, manifest: dict) -> None:
    artifact_name = manifest["name"]
    checkpoint_root = root / artifact_name
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Expected extracted checkpoint directory: {checkpoint_root}")
    for entry in manifest["files"]:
        path = checkpoint_root / entry["path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint file: {path}")
        size = path.stat().st_size
        if size != entry["size_bytes"]:
            raise ValueError(f"Size mismatch for {path}: got {size}, expected {entry['size_bytes']}")
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise ValueError(f"SHA256 mismatch for {path}: got {digest}, expected {entry['sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and optionally extract a Hippo-encoder model artifact.")
    parser.add_argument("--manifest", required=True, help="Path to model manifest JSON.")
    parser.add_argument("--artifact", required=True, help="Path to checkpoint tar.gz artifact.")
    parser.add_argument("--extract-to", default=None, help="Directory to extract into after checksum verification.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    artifact_path = Path(args.artifact)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    artifact = manifest["artifact"]
    size = artifact_path.stat().st_size
    if size != artifact["size_bytes"]:
        raise ValueError(f"Artifact size mismatch: got {size}, expected {artifact['size_bytes']}")
    digest = sha256_file(artifact_path)
    if digest != artifact["sha256"]:
        raise ValueError(f"Artifact SHA256 mismatch: got {digest}, expected {artifact['sha256']}")

    print(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "size_bytes": size,
                "sha256": digest,
                "artifact_verified": True,
            },
            indent=2,
        )
    )

    if args.extract_to:
        destination = Path(args.extract_to)
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(artifact_path, "r:gz") as tar:
            safe_extract(tar, destination)
        verify_files(destination, manifest)
        print(
            json.dumps(
                {
                    "extracted_to": str(destination),
                    "checkpoint": str(destination / manifest["name"]),
                    "files_verified": len(manifest["files"]),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
