from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autofmu import __version__


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _public_config(config: dict) -> dict:
    return {
        key: value
        for key, value in config.items()
        if not str(key).startswith("_")
    }


def _config_sha256(config: dict) -> str:
    payload = json.dumps(
        _public_config(config),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def current_git_commit(root: Any = None) -> str:
    cwd = Path(root).resolve() if root else None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def dependency_versions() -> dict:
    versions = {}
    for distribution in ("numpy", "pandas", "PyYAML", "fmpy", "scipy", "openpyxl"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


class RunManifest:
    """Append-only run manifest with stage status and artifact hashes."""

    def __init__(self, path: Path, run_id: str) -> None:
        self.path = Path(path)
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data: dict = {
                "run_id": run_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "stages": {},
                "artifacts": [],
                "warnings": [],
            }

    def record_stage(self, stage: str, status: str = "completed") -> None:
        self.data["stages"][stage] = {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def bind_run(self, config: dict, git_commit: str = None) -> None:
        """Bind a run directory to one project config and code revision.

        A later stage may reopen the manifest, but it must present the same
        resolved public configuration and Git revision. This prevents a reused
        run ID from mixing datasets or results produced by different code.
        """
        commit = (
            str(git_commit)
            if git_commit is not None
            else current_git_commit(config.get("_root"))
        )
        identity = {
            "dataset": str(config.get("dataset", "")),
            "config_sha256": _config_sha256(config),
            "git_commit": commit,
            "autofmu_version": __version__,
            "python_version": platform.python_version(),
            "dependencies": dependency_versions(),
        }
        existing = self.data.get("identity")
        if existing:
            if existing.get("config_sha256") != identity["config_sha256"]:
                raise ValueError(
                    "run-id is already bound to a different project config; "
                    "choose a new run-id"
                )
            old_commit = str(existing.get("git_commit") or "")
            if old_commit and commit and old_commit != commit:
                raise ValueError(
                    "run-id is already bound to a different code revision; "
                    "choose a new run-id"
                )
            return
        if self.data.get("artifacts") or self.data.get("stages"):
            raise ValueError(
                "existing legacy run has no config identity; choose a new run-id"
            )
        self.data["identity"] = identity

    def add_warning(self, message: str) -> None:
        if message not in self.data["warnings"]:
            self.data["warnings"].append(message)

    def add_artifact(self, path: Path, root: Path) -> None:
        path = Path(path)
        record = {
            "path": str(path.resolve().relative_to(Path(root).resolve())),
            "sha256": sha256_file(path),
        }
        for index, artifact in enumerate(self.data["artifacts"]):
            if artifact["path"] == record["path"]:
                self.data["artifacts"][index] = record
                break
        else:
            self.data["artifacts"].append(record)

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
