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


def main() -> None:
    parser = argparse.ArgumentParser(description="Reassemble a split Hippo-encoder model artifact.")
    parser.add_argument("--manifest", required=True, help="Path to chunks.json.")
    parser.add_argument("--output", default=None, help="Output artifact path. Defaults next to manifest.")
    parser.add_argument("--extract-to", default=None, help="Optional directory to extract the artifact into.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent
    output = Path(args.output) if args.output else base_dir / manifest["artifact_filename"]

    with output.open("wb") as out:
        for part in sorted(manifest["parts"], key=lambda item: item["index"]):
            path = base_dir / part["filename"]
            if not path.exists():
                # Support manifests written with paths relative to the model directory.
                path = manifest_path.parent.parent / part["path"]
            data = path.read_bytes()
            size = len(data)
            if size != part["size_bytes"]:
                raise ValueError(f"Size mismatch for {path}: got {size}, expected {part['size_bytes']}")
            digest = hashlib.sha256(data).hexdigest()
            if digest != part["sha256"]:
                raise ValueError(f"SHA256 mismatch for {path}: got {digest}, expected {part['sha256']}")
            out.write(data)

    size = output.stat().st_size
    if size != manifest["artifact_size_bytes"]:
        raise ValueError(f"Artifact size mismatch: got {size}, expected {manifest['artifact_size_bytes']}")
    digest = sha256_file(output)
    if digest != manifest["artifact_sha256"]:
        raise ValueError(f"Artifact SHA256 mismatch: got {digest}, expected {manifest['artifact_sha256']}")

    print(
        json.dumps(
            {
                "artifact": str(output),
                "size_bytes": size,
                "sha256": digest,
                "verified": True,
            },
            indent=2,
        )
    )

    if args.extract_to:
        destination = Path(args.extract_to)
        destination.mkdir(parents=True, exist_ok=True)
        mode = "r:xz" if output.suffix == ".xz" else "r:gz"
        with tarfile.open(output, mode) as tar:
            safe_extract(tar, destination)
        print(json.dumps({"extracted_to": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
