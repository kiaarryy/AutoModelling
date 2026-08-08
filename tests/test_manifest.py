from __future__ import annotations

import pytest

from autofmu.manifest import RunManifest, sha256_file


def test_add_artifact_refreshes_hash_for_existing_path(tmp_path):
    artifact = tmp_path / "result.csv"
    manifest = RunManifest(tmp_path / "manifest.json", "run-1")
    artifact.write_text("old", encoding="utf-8")
    manifest.add_artifact(artifact, tmp_path)
    old_hash = manifest.data["artifacts"][0]["sha256"]

    artifact.write_text("new", encoding="utf-8")
    manifest.add_artifact(artifact, tmp_path)

    assert len(manifest.data["artifacts"]) == 1
    assert manifest.data["artifacts"][0]["sha256"] != old_hash
    assert manifest.data["artifacts"][0]["sha256"] == sha256_file(artifact)


def test_add_artifact_keeps_distinct_paths(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    manifest = RunManifest(tmp_path / "manifest.json", "run-1")

    manifest.add_artifact(first, tmp_path)
    manifest.add_artifact(second, tmp_path)

    assert [row["path"] for row in manifest.data["artifacts"]] == [
        "first.csv",
        "second.csv",
    ]


def test_bind_run_rejects_different_config(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = RunManifest(path, "run-1")
    manifest.bind_run(
        {"dataset": "site_a", "devices": [{"id": "A"}]},
        git_commit="abc",
    )
    manifest.write()

    loaded = RunManifest(path, "run-1")
    with pytest.raises(ValueError, match="different project config"):
        loaded.bind_run(
            {"dataset": "tencent", "devices": [{"id": "B"}]},
            git_commit="abc",
        )


def test_bind_run_accepts_same_config_with_different_key_order(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = RunManifest(path, "run-1")
    manifest.bind_run(
        {"dataset": "site_a", "devices": [{"type": "pump", "id": "P1"}]},
        git_commit="abc",
    )
    manifest.write()

    loaded = RunManifest(path, "run-1")
    loaded.bind_run(
        {"devices": [{"id": "P1", "type": "pump"}], "dataset": "site_a"},
        git_commit="abc",
    )

    assert loaded.data["identity"]["dataset"] == "site_a"
    assert loaded.data["identity"]["git_commit"] == "abc"
    assert set(loaded.data["identity"]["dependencies"]) >= {
        "numpy",
        "pandas",
        "PyYAML",
    }


def test_bind_run_rejects_different_code_revision(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = RunManifest(path, "run-1")
    manifest.bind_run({"dataset": "site_a", "devices": []}, git_commit="abc")
    manifest.write()

    loaded = RunManifest(path, "run-1")
    with pytest.raises(ValueError, match="different code revision"):
        loaded.bind_run(
            {"dataset": "site_a", "devices": []},
            git_commit="def",
        )
