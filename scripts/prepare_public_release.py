#!/usr/bin/env python3
"""Copy an explicit publication file list into an isolated, hashed export.

The source worktree and Git index are read only. No directories, globs, data,
checkpoints, or private research material are implicitly included. Dry runs
validate and hash the same files without creating an output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


MANIFEST_NAME = "_release_manifest.json"
DEFAULT_FILE_LIST = "configs/icaise2026/publication_files.json"
BLOCKED_DIRECTORIES = frozenset(
    {
        ".git", ".git_old", ".ssh", "data", "vitaldb_data", "paper",
        "paper_studies", "papers", "research_sources_continuous",
        "existing_research", "excluded_methods", "venv", ".venv",
    }
)
BLOCKED_SUFFIXES = frozenset(
    {".npz", ".npy", ".pt", ".pth", ".ckpt", ".pkl", ".pickle",
     ".vital", ".tar", ".gz", ".zip", ".pem", ".key"}
)


class ReleaseError(RuntimeError):
    """Publication inputs or output placement are invalid."""


def _relative_path(value: object, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ReleaseError(f"{label} must be a nonempty POSIX relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or path.as_posix() != value
        or any(char in value for char in "*?[]")
    ):
        raise ReleaseError(f"{label} must be a canonical relative path without globs: {value}")
    return path


def _reject_symlinks(path: Path) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ReleaseError(f"symlink paths are not supported: {component}")


def _blocked_path(path: PurePosixPath) -> bool:
    parts = path.parts
    if any(
        part in BLOCKED_DIRECTORIES
        or part.startswith("federated_data")
        or part == "runs"
        or part.startswith("runs_")
        or part.startswith("outputs_")
        or part == ".env"
        or part.startswith(".env.")
        for part in parts
    ):
        return True
    name = path.name.lower()
    return (
        path.suffix.lower() in BLOCKED_SUFFIXES
        or name in {MANIFEST_NAME, "clinical_data.csv", "data_inventory.json"}
        or name.startswith(("research_notes_", "deep-research-"))
        or (name.startswith("case_") and path.suffix.lower() == ".csv")
        or "review form" in name
        or "notification acceptance" in name
    )


def _digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _output_directory(source_root: Path, out_dir: Path) -> Path:
    output = Path(os.path.abspath(out_dir))
    _reject_symlinks(output)
    output = output.resolve()
    if output == source_root or output in source_root.parents:
        raise ReleaseError("output must not be the source root or one of its parents")
    if output.is_relative_to(source_root):
        staging = source_root / "runs" / "public_release"
        if not output.is_relative_to(staging):
            raise ReleaseError("output inside the source root must be under runs/public_release")
    if output.exists():
        if not output.is_dir():
            raise ReleaseError(f"output is not a directory: {output}")
        if any(output.iterdir()):
            raise ReleaseError(f"output directory must be empty; nothing was overwritten: {output}")
    return output


def _load_file_list(source_root: Path, file_list: Path) -> tuple[dict[str, Any], str]:
    path = file_list if file_list.is_absolute() else source_root / file_list
    path = Path(os.path.abspath(path))
    _reject_symlinks(path)
    if not path.is_relative_to(source_root) or not path.is_file():
        raise ReleaseError("publication file list must be a regular file inside the source root")
    try:
        data = path.read_bytes()
        document = json.loads(data)
    except (OSError, ValueError) as exc:
        raise ReleaseError(f"cannot read publication file list: {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ReleaseError("publication file list must have schema_version: 1")
    if not isinstance(document.get("files"), list) or not document["files"]:
        raise ReleaseError("publication file list must contain a nonempty files array")
    if not isinstance(document.get("excluded_context", []), list):
        raise ReleaseError("excluded_context must be an array of relative paths")
    if not isinstance(document.get("source_overrides", {}), dict):
        raise ReleaseError("source_overrides must map selected paths to relative source files")
    return document, hashlib.sha256(data).hexdigest()


def prepare_release(
    source_root: Path,
    file_list: Path,
    out_dir: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate all selections before copying; refuse any nonempty output."""
    root = Path(os.path.abspath(source_root))
    _reject_symlinks(root)
    root = root.resolve()
    if not root.is_dir():
        raise ReleaseError(f"source root is not a directory: {root}")
    output = _output_directory(root, out_dir)
    document, file_list_hash = _load_file_list(root, file_list)
    excluded = [
        _relative_path(value, label="excluded_context entry")
        for value in document.get("excluded_context", [])
    ]
    overrides = document.get("source_overrides", {})
    for target, source in overrides.items():
        _relative_path(target, label="source_overrides target")
        _relative_path(source, label="source_overrides source")
        if target not in document["files"]:
            raise ReleaseError(f"source_overrides target is not in files: {target}")
    selected: list[tuple[Path, dict[str, Any]]] = []
    seen: set[str] = set()
    for value in document["files"]:
        relative = _relative_path(value, label="files entry")
        if value in seen:
            raise ReleaseError(f"duplicate publication file: {value}")
        seen.add(value)
        if _blocked_path(relative):
            raise ReleaseError(f"data, checkpoint, or private-context path cannot be exported: {value}")
        if any(relative == prefix or prefix in relative.parents for prefix in excluded):
            raise ReleaseError(f"publication file conflicts with excluded_context: {value}")
        source_relative = _relative_path(overrides.get(value, value), label="source file")
        if _blocked_path(source_relative):
            raise ReleaseError(f"data, checkpoint, or private-context source cannot be exported: {source_relative}")
        if any(source_relative == prefix or prefix in source_relative.parents for prefix in excluded):
            raise ReleaseError(f"source file conflicts with excluded_context: {source_relative}")
        source = root.joinpath(*source_relative.parts)
        _reject_symlinks(source)
        if not source.is_relative_to(root) or not source.is_file():
            raise ReleaseError(f"publication source must exist as a regular file: {source_relative}")
        digest, size = _digest(source)
        selected.append((source, {"path": value, "source_path": source_relative.as_posix(), "sha256": digest, "bytes": size}))

    total_bytes = sum(record["bytes"] for _, record in selected)
    manifest = {
        "schema_version": 1,
        "source": "current_worktree",
        "file_list_sha256": file_list_hash,
        "file_count": len(selected),
        "total_bytes": total_bytes,
        "files": [record for _, record in selected],
    }
    if not dry_run:
        # A partial export is retained on an I/O failure, so it cannot be silently
        # replaced on the next invocation. A complete export always has a manifest.
        output.mkdir(parents=True, exist_ok=True)
        _output_directory(root, output)
        for source, record in selected:
            destination = output / record["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            _reject_symlinks(destination)
            with source.open("rb") as src, destination.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            shutil.copymode(source, destination)
            copied_digest, copied_size = _digest(destination)
            if copied_digest != record["sha256"] or copied_size != record["bytes"]:
                raise ReleaseError(f"source changed during export; no release manifest written: {record['path']}")
        with (output / MANIFEST_NAME).open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return {
        "mode": "dry_run" if dry_run else "export",
        "out_dir": str(output),
        "file_count": len(selected),
        "total_bytes": total_bytes,
        "manifest_written": not dry_run,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--file-list", "--manifest", dest="file_list", type=Path, default=Path(DEFAULT_FILE_LIST))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = prepare_release(args.source_root, args.file_list, args.out_dir, dry_run=args.dry_run)
    except (ReleaseError, OSError) as exc:
        print(f"public release error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
