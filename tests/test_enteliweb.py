from datetime import datetime

import pandas as pd
import pytest

from autofmu.adapters.enteliweb import parse_trend_log
from scripts.fetch_site_d_enteliweb import _align_device_parts


def test_parse_full_circular_trend_buffer_in_chronological_order():
    xml = b'''<?xml version="1.0" encoding="UTF-8"?>
    <Object xmlns="http://bacnet.org/csml/1.2">
      <Enumerated name="object-type" value="trend-log"/>
      <String name="object-name" value="TEST TL"/>
      <Unsigned name="log-interval" value="90000"/>
      <Array name="timestamp-buffer">
        <Unsigned name="1" value="180000"/>
        <Unsigned name="2" value="270000"/>
        <Unsigned name="3" value="0"/>
      </Array>
      <Array name="flags-buffer">
        <BitString name="1" value=""/>
        <BitString name="2" value="fault"/>
        <BitString name="3" value=""/>
      </Array>
      <Choice name="data-buffer"><List name="real">
        <Real name="1" value="2"/>
        <Real name="2" value="3"/>
        <Real name="3" value="1"/>
      </List></Choice>
      <DateTime name="time-reference" value="2026-01-01T00:00:00+00:00"/>
      <Unsigned name="time-reference-count" value="0"/>
      <Unsigned name="buffer-size" value="3"/>
      <Unsigned name="next-index" value="3"/>
      <Unsigned name="record-count" value="3"/>
      <Unsigned name="total-record-count" value="30"/>
    </Object>'''
    trend = parse_trend_log(xml)
    assert trend.interval_seconds == 900.0
    assert [sample.value for sample in trend.samples] == ["1", "2", "3"]
    assert [sample.timestamp for sample in trend.samples] == [
        datetime.fromisoformat("2026-01-01T00:00:00+00:00"),
        datetime.fromisoformat("2026-01-01T00:30:00+00:00"),
        datetime.fromisoformat("2026-01-01T00:45:00+00:00"),
    ]
    assert trend.samples[-1].flags == "fault"


def test_reject_non_trend_object():
    with pytest.raises(ValueError, match="not a trend-log"):
        parse_trend_log(
            b'<Object xmlns="http://bacnet.org/csml/1.2">'
            b'<Enumerated name="object-type" value="analog-input"/></Object>'
        )


def test_align_periodic_channels_and_forward_fill_past_events_only():
    periodic = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01T00:00:02Z", "2026-01-01T00:15:01Z", "2026-01-01T00:30:00Z"]
            ),
            "power_W": [100.0, 200.0, 300.0],
        }
    )
    status = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-12-31T23:50:00Z", "2026-01-01T00:20:00Z"]
            ),
            "run_signal": [0.0, 1.0],
        }
    )
    merged, interval = _align_device_parts([(periodic, 900.0), (status, 0.0)])
    assert interval == 900.0
    assert merged["power_W"].tolist() == [100.0, 200.0, 300.0]
    assert merged["run_signal"].tolist() == [0.0, 0.0, 1.0]
