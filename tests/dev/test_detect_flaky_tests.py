import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dev"))

from detect_flaky_tests import gh_api_objects, parse_failing_tests

# A captured pytest failure line as it appears in a raw GitHub Actions log: an ISO
# timestamp prefix, ANSI SGR color codes around FAILED/the nodeid, and the MLflow
# conftest's "| MEM ... DISK ..." annotation between the outcome and the nodeid.
_ANSI_FAILED_LINE = (
    "2026-07-20T10:00:00.1234567Z \x1b[31mFAILED\x1b[0m | MEM 3.2/7.0 GB | DISK 40/60 GB "
    "tests/tracing/test_x.py::test_span_flush - AssertionError: expected 1 span, got 0"
)


def test_parses_nodeid_and_error_from_ansi_timestamped_line():
    result = parse_failing_tests(_ANSI_FAILED_LINE)
    assert result == {
        "tests/tracing/test_x.py::test_span_flush": "AssertionError: expected 1 span, got 0"
    }


def test_parses_error_outcome_and_parametrized_nodeid():
    log = (
        "2026-07-20T10:00:00Z \x1b[31mERROR\x1b[0m "
        "tests/store/test_y.py::test_z[case-1] - RuntimeError: boom"
    )
    assert parse_failing_tests(log) == {
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
    assert parse_failing_tests(log) == {}


def test_first_occurrence_wins_for_duplicate_nodeid():
    # pytest prints the nodeid in the progress line and again in the summary; keep the
    # first (concise) message rather than overwriting with a later, noisier one.
    log = (
        "2026-07-20T10:00:00Z FAILED tests/test_a.py::test_b - first message\n"
        "2026-07-20T10:00:05Z FAILED tests/test_a.py::test_b - second message"
    )
    assert parse_failing_tests(log) == {"tests/test_a.py::test_b": "first message"}


def test_error_message_is_truncated():
    log = "2026-07-20T10:00:00Z FAILED tests/test_a.py::test_b - " + "x" * 500
    (msg,) = parse_failing_tests(log).values()
    assert len(msg) == 300


def test_empty_log_yields_no_failures():
    assert parse_failing_tests("") == {}


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
