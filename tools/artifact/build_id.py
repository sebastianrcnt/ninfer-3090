"""Content identity for a written artifact.

`model_id` and `weights_id` name a *target* — the architecture and the quantization
recipe — and are compile-time constants of the converter. Two artifacts built from
different checkpoints by the same converter therefore carry identical identity, which
is not enough to tell their weights apart. `build_id` closes that gap: it is a digest
of the payload itself, so it moves whenever a single weight byte does.

The digest covers the payload region only, never the JSON directory, because stamping
the value into the directory would otherwise change what the digest is taken over.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

from .container import (
    MAGIC,
    PAYLOAD_ALIGNMENT,
    PREFIX,
    PREFIX_BYTES,
    ArtifactError,
)
from .layouts import align_up

BUILD_ID_CHUNK_BYTES = 8 << 20


def _read_prefix(handle) -> tuple[bytes, int]:
    handle.seek(0)
    prefix = handle.read(PREFIX_BYTES)
    if len(prefix) != PREFIX_BYTES:
        raise ArtifactError("artifact is shorter than its prefix")
    magic, json_bytes = PREFIX.unpack(prefix)
    if magic != MAGIC:
        raise ArtifactError("artifact magic is not NInfer v2")
    if json_bytes == 0:
        raise ArtifactError("json_bytes must be positive")
    return magic, json_bytes


def compute_build_id(path: str | Path) -> str:
    """Digest the payload region of a finished artifact."""

    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        _, json_bytes = _read_prefix(handle)
        payload_offset = align_up(PREFIX_BYTES + json_bytes, PAYLOAD_ALIGNMENT)
        size = target.stat().st_size
        if payload_offset > size:
            raise ArtifactError("declared payload start extends beyond the file")
        handle.seek(payload_offset)
        remaining = size - payload_offset
        while remaining > 0:
            chunk = handle.read(min(BUILD_ID_CHUNK_BYTES, remaining))
            if not chunk:
                raise ArtifactError("artifact payload ended before its declared length")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def stamp_build_id(path: str | Path) -> str:
    """Compute the payload digest and record it in the directory, in place.

    The directory is followed by zero padding out to the payload alignment, so the
    added member almost always fits without moving the payload. Rewriting the payload
    to gain a few dozen bytes would mean copying the whole artifact, so a genuinely
    full directory is reported rather than silently rebuilt.
    """

    target = Path(path)
    build_id = compute_build_id(target)
    with target.open("r+b") as handle:
        _, json_bytes = _read_prefix(handle)
        payload_offset = align_up(PREFIX_BYTES + json_bytes, PAYLOAD_ALIGNMENT)
        handle.seek(PREFIX_BYTES)
        directory = json.loads(handle.read(json_bytes).decode("utf-8"))
        identity = directory.get("identity")
        if not isinstance(identity, dict):
            raise ArtifactError("artifact directory has no identity object")
        existing = identity.get("build_id")
        if existing == build_id:
            return build_id
        identity["build_id"] = build_id
        encoded = json.dumps(
            directory, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        available = payload_offset - PREFIX_BYTES
        if len(encoded) > available:
            raise ArtifactError(
                f"stamped directory needs {len(encoded)} bytes but only {available} "
                "are reserved before the payload; rebuild the artifact instead"
            )
        handle.seek(0)
        handle.write(PREFIX.pack(MAGIC, len(encoded)))
        handle.write(encoded)
        handle.write(b"\x00" * (available - len(encoded)))
        handle.flush()
    return build_id


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="print the computed digest without writing it",
    )
    args = parser.parse_args()
    if args.check:
        print(compute_build_id(args.artifact))
    else:
        print(stamp_build_id(args.artifact))


if __name__ == "__main__":
    main()
