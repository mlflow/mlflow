import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dev"))

from classify_flaky_tests import _aggregate


def test_aggregate_collapses_events_per_test_and_counts():
    flakes = [
        {"shard": "python (1)", "test": "tests/a.py::t1", "error": "boom"},
        {"shard": "python (1)", "test": "tests/a.py::t1", "error": "boom again"},
        {"shard": "python (2)", "test": "tests/b.py::t2", "error": "kaboom"},
    ]
    result = _aggregate(flakes)
    # t1 flaked twice, t2 once; sorted by count descending.
    assert [(t["test"], t["count"]) for t in result] == [
        ("tests/a.py::t1", 2),
        ("tests/b.py::t2", 1),
    ]
    # The representative error is taken from the first event of each test.
    assert result[0]["error"] == "boom"


def test_aggregate_keys_shard_level_flakes_without_a_nodeid():
    # A job that flaked but yielded no test line (test is None) is keyed by shard so it
    # still appears in the report — and must not collide with real test-level entries.
    flakes = [
        {"shard": "windows (3)", "test": None, "error": None},
        {"shard": "windows (3)", "test": None, "error": None},
    ]
    result = _aggregate(flakes)
    assert len(result) == 1
    assert result[0]["test"] is None
    assert result[0]["shard"] == "windows (3)"
    assert result[0]["count"] == 2
