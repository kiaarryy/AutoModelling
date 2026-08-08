"""Read Delta Controls enteliWEB BACnet Trend Log CSML responses."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List
import xml.etree.ElementTree as ET


TIMER_WRAP = 2 ** 32


@dataclass(frozen=True)
class TrendSample:
    timestamp: datetime
    value: str
    flags: str


@dataclass(frozen=True)
class TrendLog:
    object_name: str
    interval_seconds: float
    record_count: int
    total_record_count: int
    samples: List[TrendSample]


def _local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _named_child(root: ET.Element, name: str) -> ET.Element:
    for child in root:
        if child.attrib.get("name") == name:
            return child
    raise ValueError("enteliWEB Trend Log is missing property: %s" % name)


def _scalar(root: ET.Element, name: str) -> str:
    return _named_child(root, name).attrib.get("value", "")


def _indexed_values(container: ET.Element) -> Dict[int, str]:
    values: Dict[int, str] = {}
    for child in container:
        raw_index = child.attrib.get("name", "")
        if raw_index.isdigit():
            values[int(raw_index)] = child.attrib.get("value", "")
    return values


def _data_values(root: ET.Element) -> Dict[int, str]:
    choice = _named_child(root, "data-buffer")
    if not list(choice):
        return {}
    return _indexed_values(list(choice)[0])


def parse_trend_log(payload: bytes) -> TrendLog:
    """Parse one BACnet CSML Trend Log, including circular-buffer ordering.

    Delta's timer values and ``log-interval`` use BACnet hundredths of a
    second. ``next-index`` points at the oldest record when the buffer is full.
    """
    root = ET.fromstring(payload)
    if _local_name(root) != "Object":
        raise ValueError("enteliWEB response is not a BACnet Object")
    if _scalar(root, "object-type") != "trend-log":
        raise ValueError("enteliWEB response is not a trend-log object")

    reference = datetime.fromisoformat(_scalar(root, "time-reference"))
    interval_seconds = int(_scalar(root, "log-interval")) / 100.0
    record_count = int(_scalar(root, "record-count"))
    total_record_count = int(_scalar(root, "total-record-count"))
    buffer_size = int(_scalar(root, "buffer-size"))
    next_index = int(_scalar(root, "next-index"))
    reference_count = int(_scalar(root, "time-reference-count") or 0)

    timers = {k: int(v) for k, v in _indexed_values(
        _named_child(root, "timestamp-buffer")
    ).items()}
    flags = _indexed_values(_named_child(root, "flags-buffer"))
    values = _data_values(root)
    available = sorted(set(timers) & set(values))
    if record_count >= buffer_size and next_index in available:
        order = [i for i in range(next_index, buffer_size + 1) if i in available]
        order += [i for i in range(1, next_index) if i in available]
    else:
        order = available[:record_count]

    samples: List[TrendSample] = []
    wraps = reference_count
    previous_timer = None
    for index in order:
        timer = timers[index]
        if previous_timer is not None and timer < previous_timer:
            wraps += 1
        ticks = timer + wraps * TIMER_WRAP
        samples.append(
            TrendSample(
                timestamp=reference + timedelta(seconds=ticks / 100.0),
                value=values[index],
                flags=flags.get(index, ""),
            )
        )
        previous_timer = timer

    if len(samples) != min(record_count, len(available)):
        raise ValueError(
            "Trend Log record mismatch: declared=%d parsed=%d"
            % (record_count, len(samples))
        )
    if any(a.timestamp >= b.timestamp for a, b in zip(samples, samples[1:])):
        raise ValueError("Trend Log timestamps are not strictly increasing")
    return TrendLog(
        object_name=_scalar(root, "object-name"),
        interval_seconds=interval_seconds,
        record_count=record_count,
        total_record_count=total_record_count,
        samples=samples,
    )
