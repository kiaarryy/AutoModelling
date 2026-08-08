"""Config portability: external roots are overridable by environment variable
and `${VAR}` expands, so a clone runs without editing any YAML."""
from __future__ import annotations

import pytest

from autofmu.config import (
    ENV_DATA_ROOT,
    ENV_FMU_ROOT,
    data_root,
    expand,
    fmu_root_from_env,
    load_project,
    run_dir,
)


def test_fmu_root_env_overrides_config_value(monkeypatch):
    monkeypatch.setenv(ENV_FMU_ROOT, "/opt/fmus")
    assert fmu_root_from_env("E:/whatever") == "/opt/fmus"


def test_fmu_root_falls_back_to_config_when_env_unset(monkeypatch):
    monkeypatch.delenv(ENV_FMU_ROOT, raising=False)
    assert fmu_root_from_env("E:/default") == "E:/default"


def test_expand_resolves_environment_variables(monkeypatch):
    monkeypatch.setenv("AUTOFMU_TEST_ROOT", "/data")
    assert expand("${AUTOFMU_TEST_ROOT}/site_a") == "/data/site_a"


def test_data_root_env_override(monkeypatch, tmp_path):
    # a minimal project config; the data dir need not exist for data_root() to resolve
    cfg_path = tmp_path / "project.yaml"
    cfg_path.write_text("data_root: /config/value\ndevices: []\n", encoding="utf-8")
    config = load_project(cfg_path)
    monkeypatch.setenv(ENV_DATA_ROOT, str(tmp_path / "from_env"))
    assert data_root(config) == (tmp_path / "from_env").resolve()
    monkeypatch.delenv(ENV_DATA_ROOT, raising=False)
    assert str(data_root(config)).replace("\\", "/").endswith("/config/value")


@pytest.mark.parametrize(
    "unsafe_run_id",
    ["../escape", r"..\escape", "/tmp/escape", "C:/escape", "a/b", ""],
)
def test_run_dir_rejects_unsafe_run_id(tmp_path, unsafe_run_id):
    config = {"_root": tmp_path, "outputs_dir": "outputs"}

    with pytest.raises(ValueError, match="safe run identifier"):
        run_dir(config, unsafe_run_id)


@pytest.mark.parametrize("safe_run_id", ["fleet", "site-a_2026.06.24", "RUN-01"])
def test_run_dir_accepts_safe_run_id(tmp_path, safe_run_id):
    config = {"_root": tmp_path, "outputs_dir": "outputs"}

    assert run_dir(config, safe_run_id) == (
        tmp_path / "outputs" / "runs" / safe_run_id
    ).resolve()
