import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dev"))

from detect_flaky_tests import (
    FRAMEWORKS,
    Framework,
    failing_tests_from_log,
    gh_api_objects,
    parse_jest_failures,
    parse_pytest_failures,
)

# A captured pytest failure line as it appears in a raw GitHub Actions log: an ISO
# timestamp prefix, ANSI SGR color codes around FAILED/the nodeid, and the MLflow
# conftest's "| MEM ... DISK ..." annotation between the outcome and the nodeid.
_ANSI_FAILED_LINE = (
    "2026-07-20T10:00:00.1234567Z \x1b[31mFAILED\x1b[0m | MEM 3.2/7.0 GB | DISK 40/60 GB "
    "tests/tracing/test_x.py::test_span_flush - AssertionError: expected 1 span, got 0"
)


def test_parses_nodeid_and_error_from_ansi_timestamped_line():
    result = parse_pytest_failures(_ANSI_FAILED_LINE)
    assert result == {
        "tests/tracing/test_x.py::test_span_flush": "AssertionError: expected 1 span, got 0"
    }


def test_parses_error_outcome_and_parametrized_nodeid():
    log = (
        "2026-07-20T10:00:00Z \x1b[31mERROR\x1b[0m "
        "tests/store/test_y.py::test_z[case-1] - RuntimeError: boom"
    )
    assert parse_pytest_failures(log) == {
        "tests/store/test_y.py::test_z[case-1]": "RuntimeError: boom"
    }


def test_ignores_lines_without_a_failure_outcome():
    # A passing line and a bare traceback reference must not be mistaken for failures —
    # only lines carrying a FAILED/ERROR token are trusted.
    log = (
        "2026-07-20T10:00:00Z tests/test_a.py::test_ok PASSED\n"
        "2026-07-20T10:00:01Z   File 'tests/test_a.py', line 3, in test_ok\n"
        "2026-07-20T10:00:02Z collected 5 items"
    )
    assert parse_pytest_failures(log) == {}


def test_first_occurrence_wins_for_duplicate_nodeid():
    # pytest prints the nodeid in the progress line and again in the summary; keep the
    # first (concise) message rather than overwriting with a later, noisier one.
    log = (
        "2026-07-20T10:00:00Z FAILED tests/test_a.py::test_b - first message\n"
        "2026-07-20T10:00:05Z FAILED tests/test_a.py::test_b - second message"
    )
    assert parse_pytest_failures(log) == {"tests/test_a.py::test_b": "first message"}


def test_error_message_is_truncated():
    log = "2026-07-20T10:00:00Z FAILED tests/test_a.py::test_b - " + "x" * 500
    (msg,) = parse_pytest_failures(log).values()
    assert len(msg) == 300


def test_empty_log_yields_no_failures():
    assert parse_pytest_failures("") == {}


# A captured jest failure block as its default reporter renders it: a `FAIL <path>`
# header (with a trailing `(7.812 s)` timing), then one bullet block per failing test
# (`  <bullet> <suite chain> <test>`) whose first body line is the assertion message.
# ANSI SGR codes and the Actions ISO-timestamp prefix wrap the real lines and must be
# stripped first.
_JEST_TIMESTAMP = "2026-07-20T10:00:00.1234567Z "
_JEST_LOG = (
    f"{_JEST_TIMESTAMP}\x1b[1m\x1b[31m FAIL \x1b[0m src/\x1b[1m__flaky_probe__.test.tsx\x1b[0m "
    "(7.812 s)\n"
    f"{_JEST_TIMESTAMP}  \x1b[31m✕\x1b[0m top level failing test (1 ms)\n"
    f"{_JEST_TIMESTAMP}\n"
    f"{_JEST_TIMESTAMP}\x1b[1m\x1b[31m  ● \x1b[1mFlaky probe suite › renders the widget\x1b[0m\n"
    f"{_JEST_TIMESTAMP}\n"
    f"{_JEST_TIMESTAMP}    expect(received).toBe(expected) // Object.is equality\n"
    f"{_JEST_TIMESTAMP}    Expected: 3\n"
    f"{_JEST_TIMESTAMP}    Received: 2\n"
    f"{_JEST_TIMESTAMP}\x1b[1m\x1b[31m  ● \x1b[1mtop level failing test\x1b[0m\n"
    f"{_JEST_TIMESTAMP}\n"
    f"{_JEST_TIMESTAMP}    expect(received).toBe(expected) // Object.is equality\n"
)


def test_parses_jest_file_suite_and_test_into_id_with_error():
    assert parse_jest_failures(_JEST_LOG) == {
        "src/__flaky_probe__.test.tsx › Flaky probe suite › renders the widget": (
            "expect(received).toBe(expected) // Object.is equality"
        ),
        "src/__flaky_probe__.test.tsx › top level failing test": (
            "expect(received).toBe(expected) // Object.is equality"
        ),
    }


def test_jest_test_without_a_preceding_fail_file_is_ignored():
    # A bare ● with no `FAIL <path>` header before it has no file to attach to; the
    # detector must not emit a fileless id.
    assert parse_jest_failures("2026-07-20T10:00:00Z   ● Some suite › a test\n") == {}


def test_jest_empty_log_yields_no_failures():
    assert parse_jest_failures("") == {}


def test_jest_framework_is_registered_scanning_the_js_workflow():
    fw = FRAMEWORKS["jest"]
    assert fw.workflow_name == "JS"
    assert fw.parse_log is parse_jest_failures


def test_gh_api_objects_parses_concatenated_pages(monkeypatch):
    # `gh api --paginate` concatenates each page's JSON body back-to-back; the decoder
    # must recover every object, not just the first.
    import detect_flaky_tests

    monkeypatch.setattr(detect_flaky_tests, "gh_api", lambda *a, **k: '{"n": 1}\n{"n": 2}')
    assert gh_api_objects("any/path", paginate=True) == [{"n": 1}, {"n": 2}]


def test_gh_api_objects_returns_empty_on_no_output(monkeypatch):
    import detect_flaky_tests

    monkeypatch.setattr(detect_flaky_tests, "gh_api", lambda *a, **k: None)
    assert gh_api_objects("any/path") == []


def test_pytest_framework_is_registered_with_its_parser():
    fw = FRAMEWORKS["pytest"]
    assert fw.workflow_name == "MLflow tests"
    assert fw.parse_log is parse_pytest_failures


def test_failing_tests_from_log_delegates_to_the_frameworks_parser(monkeypatch):
    # A custom Framework's parse_log must be the function used to turn the fetched log
    # into failures, so a new framework only needs to register its own parser.
    import detect_flaky_tests

    monkeypatch.setattr(detect_flaky_tests, "gh_api_text_bytes", lambda path: "raw-log-text")
    custom = Framework(workflow_name="Custom", parse_log=lambda log: {f"parsed::{log}": "err"})
    assert failing_tests_from_log("owner/repo", 123, custom) == {"parsed::raw-log-text": "err"}


def test_failing_tests_from_log_returns_empty_when_no_log(monkeypatch):
    import detect_flaky_tests

    monkeypatch.setattr(detect_flaky_tests, "gh_api_text_bytes", lambda path: None)
    called = False

    def _parse(_log):
        nonlocal called
        called = True
        return {"x": "y"}

    assert failing_tests_from_log("owner/repo", 1, Framework("W", _parse)) == {}
    # The parser must not run when there is no log to parse.
    assert called is False


def test_workflow_override_replaces_only_the_workflow_name(monkeypatch):
    # `--workflow` swaps the scanned workflow while keeping the framework's parser, via
    # dataclasses.replace. Verify detect() resolves the overridden name.
    import detect_flaky_tests

    seen = {}

    def _get_workflow_id(repo, workflow_name):
        seen["workflow_name"] = workflow_name
        return None  # short-circuit detect() after the name is resolved

    monkeypatch.setattr(detect_flaky_tests, "get_workflow_id", _get_workflow_id)
    overridden = dataclasses.replace(FRAMEWORKS["pytest"], workflow_name="JS")
    assert detect_flaky_tests.detect("owner/repo", "2026-01-01", overridden) == []
    assert seen["workflow_name"] == "JS"
    # The parser is preserved through the override.
    assert overridden.parse_log is parse_pytest_failures
