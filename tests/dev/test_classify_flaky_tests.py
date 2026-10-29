import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dev"))

import classify_flaky_tests
from classify_flaky_tests import _SCHEMA, RETRY_MECHANISMS, _aggregate
from detect_flaky_tests import FRAMEWORKS


def test_retry_mechanisms_cover_exactly_the_detect_frameworks():
    # The two registries live in different modules; a mismatch would let `--framework X`
    # pass argparse in detect and then KeyError in classify. Keep them locked together.
    #
    # This intentionally duplicates the module-level assert in classify_flaky_tests.py:
    # the assert is the real guard (it fires at import, failing fast anywhere the module
    # loads), while this test exists to give a readable pytest diff of the offending keys.
    # If the assert ever fires on drift, importing classify here raises and this test
    # errors too — which is the intended signal, just surfaced through the test runner.
    assert RETRY_MECHANISMS.keys() == FRAMEWORKS.keys()


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


def test_classify_sends_gateway_request(monkeypatch):
    monkeypatch.setattr(
        classify_flaky_tests, "BASE_URL", "https://gateway.example/ai-gateway/anthropic"
    )
    monkeypatch.setattr(classify_flaky_tests, "DEFAULT_MODEL", "databricks-claude-sonnet-5")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "gateway-token")
    monkeypatch.setenv(
        "ANTHROPIC_CUSTOM_HEADERS",
        'Databricks-Ai-Gateway-Request-Tags: {"repository":"mlflow/mlflow",'
        '"task":"flaky-test-classifier:weekly"}',
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    verdict = {
        "category": "network",
        "action": "annotate",
        "attempts": 2,
        "confidence": "high",
        "rationale": "A transient network error.",
    }
    response = mock.MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps({
        "content": [{"type": "text", "text": json.dumps(verdict)}]
    }).encode()

    with mock.patch(
        "classify_flaky_tests.urllib.request.urlopen", return_value=response
    ) as urlopen:
        result = classify_flaky_tests.classify(
            "tests/example.py::test_network", 2, "connection reset", "@pytest.mark.flaky"
        )

    urlopen.assert_called_once()
    response.__enter__.assert_called_once()
    response.__enter__.return_value.read.assert_called_once()
    response.__exit__.assert_called_once()
    request = urlopen.call_args.args[0]
    assert request.full_url == "https://gateway.example/ai-gateway/anthropic/v1/messages"
    headers = {name.lower(): value for name, value in request.header_items()}
    scheme, separator, token = headers["authorization"].partition(" ")
    assert scheme == "Bearer"
    assert separator == " "
    assert token == os.environ["ANTHROPIC_AUTH_TOKEN"]
    assert headers["databricks-ai-gateway-request-tags"] == (
        '{"repository":"mlflow/mlflow","task":"flaky-test-classifier:weekly"}'
    )
    body = json.loads(request.data)
    assert body["model"] == "databricks-claude-sonnet-5"
    assert body["output_config"] == {"format": {"type": "json_schema", "schema": _SCHEMA}}
    assert result == verdict


@pytest.mark.parametrize(
    "flakes",
    [
        [],
        [{"shard": "python (1)", "test": None, "error": None}],
    ],
)
def test_classify_empty_or_shard_only_without_authentication(tmp_path, monkeypatch, flakes):
    input_path = tmp_path / "flakes.json"
    output_path = tmp_path / "classified.json"
    input_path.write_text(json.dumps(flakes))
    monkeypatch.setattr(
        sys,
        "argv",
        ["classify_flaky_tests.py", "--in", str(input_path), "--out", str(output_path)],
    )
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with mock.patch(
        "classify_flaky_tests.urllib.request.urlopen",
        side_effect=AssertionError("classification should not call the API"),
    ) as urlopen:
        classify_flaky_tests.main()

    urlopen.assert_not_called()
    results = json.loads(output_path.read_text())
    assert len(results) == len(flakes)
    if flakes:
        assert results[0]["verdict"]["action"] == "investigate"
