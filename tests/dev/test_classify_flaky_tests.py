import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dev"))

from classify_flaky_tests import CLASSIFIERS, _aggregate
from detect_flaky_tests import FRAMEWORKS


def test_classifiers_cover_exactly_the_detect_frameworks():
    # The two registries live in different modules; a mismatch would let `--framework X`
    # pass argparse in detect and then KeyError in classify. Keep them locked together.
    #
    # This intentionally duplicates the module-level assert in classify_flaky_tests.py:
    # the assert is the real guard (it fires at import, failing fast anywhere the module
    # loads), while this test exists to give a readable pytest diff of the offending keys.
    # If the assert ever fires on drift, importing classify here raises and this test
    # errors too — which is the intended signal, just surfaced through the test runner.
    assert CLASSIFIERS.keys() == FRAMEWORKS.keys()


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


def test_pytest_classifier_can_annotate_and_suggest_attempts():
    schema = CLASSIFIERS["pytest"].schema
    assert set(schema["properties"]["action"]["enum"]) == {"annotate", "fix", "investigate"}
    assert "attempts" in schema["properties"]


def test_jest_classifier_is_report_only():
    # Retries are banned in the JS suite, so the jest verdict schema must not offer
    # `annotate` (nor an `attempts` count) — every flake is fix/investigate for a human.
    schema = CLASSIFIERS["jest"].schema
    assert set(schema["properties"]["action"]["enum"]) == {"fix", "investigate"}
    assert "attempts" not in schema["properties"]
