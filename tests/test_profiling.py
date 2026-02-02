"""
Tests for test profiling functionality in conftest.py.
"""

import json

from tests.conftest import _should_profile_test, fetch_profile_tests


def test_should_profile_test_exact_match():
    from tests import conftest

    original_profile_tests = conftest._profile_tests.copy()
    try:
        conftest._profile_tests = {"tests/test_version.py::test_is_release_version"}
        assert _should_profile_test("tests/test_version.py::test_is_release_version")
        assert not _should_profile_test("tests/test_import.py::test_import_mlflow")
    finally:
        conftest._profile_tests = original_profile_tests


def test_should_profile_test_partial_match():
    from tests import conftest

    original_profile_tests = conftest._profile_tests.copy()
    try:
        conftest._profile_tests = {"tests/test_version.py"}
        assert _should_profile_test("tests/test_version.py::test_is_release_version")
        assert _should_profile_test("tests/test_version.py::another_test")
        assert not _should_profile_test("tests/test_import.py::test_import_mlflow")
    finally:
        conftest._profile_tests = original_profile_tests


def test_should_profile_test_empty():
    from tests import conftest

    original_profile_tests = conftest._profile_tests.copy()
    try:
        conftest._profile_tests = set()
        assert not _should_profile_test("tests/test_version.py::test_is_release_version")
    finally:
        conftest._profile_tests = original_profile_tests


def test_fetch_profile_tests_no_github(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    result = fetch_profile_tests()
    assert result == set()


def test_fetch_profile_tests_wrong_event(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    result = fetch_profile_tests()
    assert result == set()


def test_fetch_profile_tests_with_pr_body(tmp_path, monkeypatch):
    pr_body = """
Some PR description text

<!-- profile:
tests/test_version.py::test_is_release_version
tests/test_import.py
tests/foo/bar.py::test_baz
-->

More text here
"""

    event_path = tmp_path / "event.json"
    event_data = {
        "pull_request": {
            "body": pr_body,
            "labels": [],
        }
    }
    event_path.write_text(json.dumps(event_data))

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))

    result = fetch_profile_tests()
    assert "tests/test_version.py::test_is_release_version" in result
    assert "tests/test_import.py" in result
    assert "tests/foo/bar.py::test_baz" in result
    assert len(result) == 3


def test_fetch_profile_tests_multiple_blocks(tmp_path, monkeypatch):
    pr_body = """
<!-- profile:
tests/test_version.py
-->

Some text

<!-- profile:
tests/test_import.py
-->
"""

    event_path = tmp_path / "event.json"
    event_data = {
        "pull_request": {
            "body": pr_body,
            "labels": [],
        }
    }
    event_path.write_text(json.dumps(event_data))

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))

    result = fetch_profile_tests()
    assert "tests/test_version.py" in result
    assert "tests/test_import.py" in result
    assert len(result) == 2


def test_fetch_profile_tests_empty_pr_body(tmp_path, monkeypatch):
    event_path = tmp_path / "event.json"
    event_data = {
        "pull_request": {
            "body": "",
            "labels": [],
        }
    }
    event_path.write_text(json.dumps(event_data))

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))

    result = fetch_profile_tests()
    assert result == set()


def test_fetch_profile_tests_null_pr_body(tmp_path, monkeypatch):
    event_path = tmp_path / "event.json"
    event_data = {
        "pull_request": {
            "body": None,
            "labels": [],
        }
    }
    event_path.write_text(json.dumps(event_data))

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event_path))

    result = fetch_profile_tests()
    assert result == set()
