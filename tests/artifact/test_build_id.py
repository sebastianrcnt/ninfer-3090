from __future__ import annotations

from pathlib import Path

import pytest

from tools.artifact.build_id import compute_build_id, stamp_build_id
from tools.artifact.container import (
    PREFIX,
    PREFIX_BYTES,
    RAW_BYTES_V1,
    ArtifactError,
    ArtifactIdentity,
    ResourceSpec,
    parse_directory,
    write_artifact,
)


def _write(path: Path, first: bytes, second: bytes = b"world") -> None:
    write_artifact(
        path,
        ArtifactIdentity("m", "w"),
        [
            (ResourceSpec("frontend/a", RAW_BYTES_V1, len(first)), first),
            (ResourceSpec("frontend/b", RAW_BYTES_V1, len(second)), second),
        ],
    )


def _identity(path: Path) -> ArtifactIdentity:
    with path.open("rb") as handle:
        _magic, json_bytes = PREFIX.unpack(handle.read(PREFIX_BYTES))
        identity, _objects = parse_directory(handle.read(json_bytes))
    return identity


def test_stamping_records_the_payload_digest(tmp_path: Path) -> None:
    artifact = tmp_path / "a.ninfer"
    _write(artifact, b"hello")
    expected = compute_build_id(artifact)

    assert stamp_build_id(artifact) == expected
    assert _identity(artifact).build_id == expected


def test_stamping_leaves_the_payload_and_its_digest_untouched(tmp_path: Path) -> None:
    # The digest covers the payload only, so writing it into the directory must not change
    # what a later recomputation sees; otherwise stamping could never be verified.
    artifact = tmp_path / "a.ninfer"
    _write(artifact, b"hello")
    before = compute_build_id(artifact)
    stamp_build_id(artifact)

    assert compute_build_id(artifact) == before
    assert stamp_build_id(artifact) == before


def test_a_single_changed_payload_byte_changes_the_digest(tmp_path: Path) -> None:
    # This is the property the slot guard depends on: two artifacts of one target that differ
    # only in weights must not share an identity.
    first = tmp_path / "a.ninfer"
    second = tmp_path / "b.ninfer"
    _write(first, b"hello")
    _write(second, b"hellp")

    assert compute_build_id(first) != compute_build_id(second)


def test_identical_payloads_share_a_digest(tmp_path: Path) -> None:
    first = tmp_path / "a.ninfer"
    second = tmp_path / "b.ninfer"
    _write(first, b"hello")
    _write(second, b"hello")

    assert compute_build_id(first) == compute_build_id(second)


def test_an_unstamped_artifact_reports_no_build_id(tmp_path: Path) -> None:
    artifact = tmp_path / "a.ninfer"
    _write(artifact, b"hello")

    assert _identity(artifact).build_id == ""


def test_a_foreign_identity_member_is_still_refused(tmp_path: Path) -> None:
    artifact = tmp_path / "a.ninfer"
    _write(artifact, b"hello")
    with artifact.open("r+b") as handle:
        _magic, json_bytes = PREFIX.unpack(handle.read(PREFIX_BYTES))
        directory = handle.read(json_bytes).decode("utf-8")
    tampered = directory.replace('"weights_id":"w"', '"weights_id":"w","surprise":"x"', 1)
    assert tampered != directory
    with pytest.raises(ArtifactError):
        parse_directory(tampered.encode("utf-8"))
