"""
Tests for resource-heavy test detection functionality in conftest.py
"""

import sys

import pytest


def test_resource_usage_current_returns_none_without_psutil(monkeypatch):
    from tests.conftest import ResourceUsage

    # Temporarily hide psutil
    original_psutil = sys.modules.get("psutil")
    if "psutil" in sys.modules:
        monkeypatch.setitem(sys.modules, "psutil", None)

    try:
        result = ResourceUsage.current()
        assert result is None
    finally:
        # Restore psutil
        if original_psutil is not None:
            sys.modules["psutil"] = original_psutil


def test_resource_usage_current_with_psutil():
    pytest.importorskip("psutil")

    from tests.conftest import ResourceUsage

    usage = ResourceUsage.current()
    assert usage is not None
    assert usage.mem_bytes > 0
    assert usage.disk_bytes > 0


def test_resource_delta_exceeds_threshold():
    from tests.conftest import ResourceDelta

    # Below threshold for both
    delta = ResourceDelta(mem_bytes=100 * 1024 * 1024, disk_bytes=100 * 1024 * 1024)
    assert not delta.exceeds_threshold()

    # Memory exceeds threshold
    delta = ResourceDelta(mem_bytes=600 * 1024 * 1024, disk_bytes=100 * 1024 * 1024)
    assert delta.exceeds_threshold()

    # Disk exceeds threshold
    delta = ResourceDelta(mem_bytes=100 * 1024 * 1024, disk_bytes=600 * 1024 * 1024)
    assert delta.exceeds_threshold()

    # Both exceed threshold
    delta = ResourceDelta(mem_bytes=600 * 1024 * 1024, disk_bytes=600 * 1024 * 1024)
    assert delta.exceeds_threshold()


def test_resource_delta_format_exceeded():
    from tests.conftest import ResourceDelta

    # Only memory exceeds
    delta = ResourceDelta(mem_bytes=600 * 1024 * 1024, disk_bytes=100 * 1024 * 1024)
    assert "mem: +0.6 GB" in delta.format_exceeded()
    assert "disk" not in delta.format_exceeded()

    # Only disk exceeds
    delta = ResourceDelta(mem_bytes=100 * 1024 * 1024, disk_bytes=800 * 1024 * 1024)
    assert "disk: +0.8 GB" in delta.format_exceeded()
    assert "mem" not in delta.format_exceeded()

    # Both exceed
    delta = ResourceDelta(mem_bytes=600 * 1024 * 1024, disk_bytes=800 * 1024 * 1024)
    formatted = delta.format_exceeded()
    assert "mem: +0.6 GB" in formatted
    assert "disk: +0.8 GB" in formatted
    assert ", " in formatted


def test_resource_usage_subtraction():
    from tests.conftest import ResourceUsage

    before = ResourceUsage(mem_bytes=1000 * 1024 * 1024, disk_bytes=2000 * 1024 * 1024)
    after = ResourceUsage(mem_bytes=1600 * 1024 * 1024, disk_bytes=2800 * 1024 * 1024)

    delta = after - before
    assert delta.mem_bytes == 600 * 1024 * 1024
    assert delta.disk_bytes == 800 * 1024 * 1024
