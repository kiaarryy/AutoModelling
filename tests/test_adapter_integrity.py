from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from autofmu.adapters.mapping import adapt_csv
from autofmu.pipeline.attribute import _normalized_timestamps


def _case(tmp_path: Path, timestamps, policy=None):
    pd.DataFrame({"DateTime": timestamps, "Power": range(len(timestamps))}).to_csv(
        tmp_path / "raw.csv", index=False
    )
    config = {
        "source_csv": "raw.csv",
        "timestamp": "DateTime",
        "columns": {"power_W": {"source": "Power"}},
    }
    if policy:
        config["timestamp_policy"] = policy
    config_path = tmp_path / "adapter.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "out" / "canonical.csv"
    return config_path, output


@pytest.mark.parametrize(
    "timestamps,message",
    [
        (["2024-01-01", "not-a-time"], "invalid timestamp"),
        (["2024-01-01", "2024-01-01"], "duplicate timestamp"),
        (["2024-01-02", "2024-01-01"], "not sorted"),
    ],
)
def test_adapter_rejects_invalid_duplicate_or_unsorted_timestamps(tmp_path, timestamps, message):
    config, output = _case(tmp_path, timestamps)
    with pytest.raises(ValueError, match=message):
        adapt_csv(config, tmp_path, output)


def test_adapter_explicit_policy_sorts_deduplicates_and_normalizes_timezone(tmp_path):
    config, output = _case(
        tmp_path,
        ["2024-01-01T09:00:00+08:00", "2024-01-01T08:00:00+08:00", "2024-01-01T08:00:00+08:00"],
        policy="sort_deduplicate",
    )
    adapt_csv(config, tmp_path, output)
    canonical = pd.read_csv(output)
    assert canonical["timestamp"].tolist() == [
        "2024-01-01T00:00:00Z",
        "2024-01-01T01:00:00Z",
    ]


def test_fleet_generators_keep_portable_data_and_fmu_roots():
    repo = Path(__file__).resolve().parents[1]
    for name in ("generate_sitea_fleet.py", "generate_tencent_fleet.py"):
        source = (repo / "scripts" / name).read_text(encoding="utf-8")
        assert '"${AUTOFMU_DATA_ROOT}"' in source
        assert '"fmu_config_dir": "../fmu"' in source


def test_attribute_join_rejects_duplicate_or_invalid_keys():
    with pytest.raises(ValueError, match="duplicate timestamp"):
        _normalized_timestamps(pd.Series(["2024-01-01", "2024-01-01"]), "weather join")
    with pytest.raises(ValueError, match="invalid timestamp"):
        _normalized_timestamps(pd.Series(["not-a-time"]), "weather join")
