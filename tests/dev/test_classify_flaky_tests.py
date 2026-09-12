import json
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dev"))

import classify_flaky_tests
from classify_flaky_tests import CLASSIFIERS, _aggregate, classify, render_summary
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


def _fake_response(verdict: dict[str, object]):
    """A stub urlopen context manager returning the Messages API envelope for `verdict`."""
    envelope = {"content": [{"text": json.dumps(verdict)}]}
    resp = mock.MagicMock()
    resp.read.return_value = json.dumps(envelope).encode()
    resp.__enter__.return_value = resp
    return resp


def test_classify_sends_jest_report_only_schema_and_formatted_prompt(monkeypatch):
    # End-to-end wiring for the JS path: the jest flake's test id is formatted into the
    # prompt, the request carries the report-only schema (so the model cannot return
    # `annotate`), and the parsed verdict flows back.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    verdict = {
        "category": "test-harness-race",
        "action": "investigate",
        "confidence": "medium",
        "rationale": "Likely a missing await on findBy*; wrap in waitFor.",
    }
    with mock.patch("urllib.request.urlopen", return_value=_fake_response(verdict)) as urlopen:
        result = classify(
            "src/foo/Bar.test.tsx › Bar › renders rows",
            2,
            "Unable to find element",
            CLASSIFIERS["jest"],
        )

    assert result == verdict
    urlopen.assert_called_once()
    sent = json.loads(urlopen.call_args[0][0].data)
    schema = sent["output_config"]["format"]["schema"]
    assert set(schema["properties"]["action"]["enum"]) == {"fix", "investigate"}
    assert "attempts" not in schema["properties"]
    assert "src/foo/Bar.test.tsx › Bar › renders rows" in sent["messages"][0]["content"]


def test_classify_sends_pytest_schema_allowing_annotate(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    verdict = {
        "category": "timeout",
        "action": "annotate",
        "attempts": 3,
        "confidence": "high",
        "rationale": "Load-induced timeout, safe to retry.",
    }
    with mock.patch("urllib.request.urlopen", return_value=_fake_response(verdict)) as urlopen:
        result = classify("tests/a.py::t1", 5, "TimeoutError", CLASSIFIERS["pytest"])

    assert result == verdict
    sent = json.loads(urlopen.call_args[0][0].data)
    schema = sent["output_config"]["format"]["schema"]
    assert "annotate" in schema["properties"]["action"]["enum"]
    assert "attempts" in schema["properties"]


def _run_main_on_shard_flake(tmp_path, monkeypatch, framework: str):
    """Run classify main() on a single shard-level (test=None) flake for `framework`,
    returning the one verdict it wrote. Shard flakes take the fallback path (no API call).
    """
    infile = tmp_path / "flakes.json"
    outfile = tmp_path / "classified.json"
    infile.write_text(json.dumps([{"shard": "js (rest)", "test": None, "error": None}]))
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--in", str(infile), "--out", str(outfile), "--framework", framework],
    )
    classify_flaky_tests.main()
    (entry,) = json.loads(outfile.read_text())
    return entry["verdict"]


def _assert_conforms(verdict: dict[str, object], schema: dict[str, object]):
    props = schema["properties"]
    # Every required field is present, and no field outside the schema leaked in.
    assert set(schema["required"]) <= set(verdict)
    assert set(verdict) <= set(props)
    assert verdict["action"] in props["action"]["enum"]


def test_shard_fallback_verdict_conforms_to_jest_schema(tmp_path, monkeypatch):
    # jest's schema omits `attempts`, so the fallback must not include it.
    verdict = _run_main_on_shard_flake(tmp_path, monkeypatch, "jest")
    assert "attempts" not in verdict
    _assert_conforms(verdict, CLASSIFIERS["jest"].schema)


def test_shard_fallback_verdict_conforms_to_pytest_schema(tmp_path, monkeypatch):
    # pytest's schema lists `attempts` as required, so the fallback must carry it (null).
    verdict = _run_main_on_shard_flake(tmp_path, monkeypatch, "pytest")
    assert verdict["attempts"] is None
    _assert_conforms(verdict, CLASSIFIERS["pytest"].schema)


def test_render_summary_surfaces_the_verdict_fields():
    # The weekly issue publishes this, so the LLM's action + rationale (its fix guidance)
    # must appear in the rendered markdown — not just the raw flake.
    results = [
        {
            "test": "src/foo/Bar.test.tsx › Bar › renders",
            "shard": "js (rest)",
            "count": 3,
            "error": "Unable to find element",
            "verdict": {
                "category": "test-harness-race",
                "action": "fix",
                "confidence": "high",
                "rationale": "Missing await on findBy*; wrap in waitFor.",
            },
        }
    ]
    md = render_summary(results, "jest")
    assert "src/foo/Bar.test.tsx › Bar › renders" in md
    assert "action: fix" in md
    assert "Missing await on findBy*; wrap in waitFor." in md
    assert "flaked 3×" in md
