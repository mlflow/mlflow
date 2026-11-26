import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest import mock

import jsonschema  # type: ignore[import-untyped]
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]

# Rows, in order: L3/R3, L4, R4, R5, L5/R6, then a second hunk L20/R21, R22, L21/R23.
DIFF = """diff --git a/foo.py b/foo.py
index 1111111..2222222 100644
--- a/foo.py
+++ b/foo.py
@@ -3,3 +3,4 @@ def main():
     a = 1
-    b = 2
+    b = 3
+    c = 4
     return a
@@ -20,2 +21,3 @@ def other():
     x = 1
+    y = 2
     return x
"""


@pytest.fixture(scope="module")
def validate_review() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "validate_review", ROOT / ".claude/scripts/validate_review.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def review(**comment: Any) -> dict[str, Any]:
    return {"comments": [{"path": "foo.py", "body": "x", **comment}]}


@pytest.mark.parametrize(
    "comment",
    [
        {"side": "RIGHT", "line": 4},
        {"side": "LEFT", "line": 4},
        # An unchanged line is addressable on both sides, by its old and new numbers.
        {"side": "LEFT", "line": 5},
        {"side": "RIGHT", "line": 6},
        {"start_side": "RIGHT", "start_line": 4, "side": "RIGHT", "line": 6},
        {"start_side": "LEFT", "start_line": 4, "side": "RIGHT", "line": 5},
    ],
)
def test_positions_in_the_diff_are_accepted(
    validate_review: ModuleType, comment: dict[str, Any]
) -> None:
    assert validate_review.check_positions(review(**comment), DIFF) == []


@pytest.mark.parametrize(
    ("comment", "expected"),
    [
        ({"side": "RIGHT", "line": 7}, "comments[0]: R7 not in the diff of foo.py"),
        ({"side": "LEFT", "line": 6}, "comments[0]: L6 not in the diff of foo.py"),
        (
            {"path": "bar.py", "side": "RIGHT", "line": 3},
            "comments[0]: bar.py has no lines in the diff",
        ),
        (
            {"start_side": "RIGHT", "start_line": 5, "side": "RIGHT", "line": 4},
            "comments[0]: R5 does not come before R4 in foo.py",
        ),
        (
            # L3 and R3 are the same row.
            {"start_side": "LEFT", "start_line": 3, "side": "RIGHT", "line": 3},
            "comments[0]: L3 does not come before R3 in foo.py",
        ),
        (
            {"start_side": "RIGHT", "start_line": 6, "side": "RIGHT", "line": 21},
            "comments[0]: R6 and R21 are in different hunks of foo.py",
        ),
        (
            {"start_side": "RIGHT", "start_line": 1, "side": "RIGHT", "line": 99},
            "comments[0]: R1 and R99 not in the diff of foo.py",
        ),
    ],
)
def test_positions_outside_the_diff_are_rejected(
    validate_review: ModuleType, comment: dict[str, Any], expected: str
) -> None:
    assert validate_review.check_positions(review(**comment), DIFF) == [expected]


def test_every_bad_comment_is_reported(validate_review: ModuleType) -> None:
    payload = {
        "comments": [
            {"path": "foo.py", "body": "x", "side": "RIGHT", "line": 99},
            {"path": "foo.py", "body": "x", "side": "RIGHT", "line": 4},
            {"path": "foo.py", "body": "x", "side": "LEFT", "line": 99},
        ]
    }
    assert validate_review.check_positions(payload, DIFF) == [
        "comments[0]: R99 not in the diff of foo.py",
        "comments[2]: L99 not in the diff of foo.py",
    ]


def test_no_newline_marker_is_not_a_row(validate_review: ModuleType) -> None:
    diff = (
        "diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1,2 +1,3 @@\n a\n-b\n"
        "\\ No newline at end of file\n+c\n+d\n"
    )
    hunk = validate_review.read_hunks(diff)["f"][0]
    assert hunk == {
        ("LEFT", 1): 0,
        ("RIGHT", 1): 0,
        ("LEFT", 2): 1,
        ("RIGHT", 2): 2,
        ("RIGHT", 3): 3,
    }


def test_empty_line_is_a_blank_context_row(validate_review: ModuleType) -> None:
    # `diff.suppressBlankEmpty` drops the leading space of a blank context line.
    diff = "--- a/f\n+++ b/f\n@@ -1,3 +1,3 @@\n a\n\n-b\n+c\n"
    hunk = validate_review.read_hunks(diff)["f"][0]
    assert hunk == {
        ("LEFT", 1): 0,
        ("RIGHT", 1): 0,
        ("LEFT", 2): 1,
        ("RIGHT", 2): 1,
        ("LEFT", 3): 2,
        ("RIGHT", 3): 3,
    }


def test_deleted_file_is_addressed_by_its_old_path(validate_review: ModuleType) -> None:
    diff = "diff --git a/gone.py b/gone.py\n--- a/gone.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-a\n-b\n"
    assert (
        validate_review.check_positions(
            {"comments": [{"path": "gone.py", "body": "x", "side": "LEFT", "line": 2}]}, diff
        )
        == []
    )


def test_quoted_path_is_decoded(validate_review: ModuleType) -> None:
    path = r"\346\227\245 \346\234\254.txt"
    diff = (
        f'diff --git "a/{path}" "b/{path}"\n--- "a/{path}"\n+++ "b/{path}"\n@@ -1 +1,2 @@\n a\n+b\n'
    )
    assert list(validate_review.read_hunks(diff)) == ["日 本.txt"]


@pytest.mark.parametrize(
    "diff",
    [
        # Fewer lines than the header declares.
        "--- a/f\n+++ b/f\n@@ -1,3 +1,3 @@\n a\n",
        # One line fewer: the final newline must not count as a blank row.
        "--- a/f\n+++ b/f\n@@ -1,2 +1,2 @@\n a\n",
        # More old lines than the header declares.
        "--- a/f\n+++ b/f\n@@ -1,1 +1,2 @@\n-a\n-b\n+c\n",
    ],
)
def test_hunk_that_contradicts_its_header_is_rejected(
    validate_review: ModuleType, diff: str
) -> None:
    with pytest.raises(ValueError, match="(?i)hunk"):
        validate_review.read_hunks(diff)


@pytest.fixture(scope="module")
def schema() -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(
        yaml.safe_load((ROOT / ".claude/scripts/review.schema.yml").read_text())
    )


COMMENT = {"path": "foo.py", "body": "x", "side": "RIGHT", "line": 4}


@pytest.mark.parametrize(
    "payload",
    [
        {"event": "APPROVE", "body": "ok", "comments": []},
        {"event": "COMMENT", "body": "ok", "comments": [COMMENT]},
        {"event": "REQUEST_CHANGES", "body": "ok", "commit_id": "a" * 40, "comments": [COMMENT]},
        {
            "event": "COMMENT",
            "body": "ok",
            "comments": [COMMENT | {"start_side": "LEFT", "start_line": 3}],
        },
    ],
)
def test_schema_accepts_well_formed_payloads(
    schema: jsonschema.Draft202012Validator, payload: dict[str, Any]
) -> None:
    assert list(schema.iter_errors(payload)) == []


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"event": "MERGE", "body": "ok", "comments": []}, "'MERGE' is not one of"),
        ({"event": "COMMENT", "comments": []}, "'body' is a required property"),
        ({"event": "COMMENT", "body": " ", "comments": []}, "does not match"),
        ({"event": "COMMENT", "body": "ok", "commit_id": "abc", "comments": []}, "does not match"),
        (
            {"event": "COMMENT", "body": "ok", "comments": [COMMENT | {"line": 0}]},
            "0 is less than the minimum of 1",
        ),
        (
            {"event": "COMMENT", "body": "ok", "comments": [COMMENT | {"side": "BOTH"}]},
            "'BOTH' is not one of",
        ),
        (
            {"event": "COMMENT", "body": "ok", "comments": [COMMENT | {"position": 3}]},
            "Additional properties are not allowed",
        ),
        (
            {"event": "COMMENT", "body": "ok", "comments": [COMMENT | {"start_line": 3}]},
            "'start_side' is a dependency of 'start_line'",
        ),
        (
            {
                "event": "COMMENT",
                "body": "ok",
                "comments": [COMMENT | {"side": "LEFT", "start_side": "RIGHT", "start_line": 3}],
            },
            "'RIGHT' was expected",
        ),
    ],
)
def test_schema_rejects_malformed_payloads(
    schema: jsonschema.Draft202012Validator, payload: dict[str, Any], expected: str
) -> None:
    match [e.message for e in schema.iter_errors(payload)]:
        case [message]:
            assert expected in message
        case messages:
            pytest.fail(f"Expected one error, got {messages}")


def run_main(
    validate_review: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    comment: dict[str, Any],
    *,
    with_diff: bool,
    schema_exit: int = 0,
    diff_text: str = DIFF,
) -> mock.MagicMock:
    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps(review(**comment)))
    diff = tmp_path / "pr.diff"
    diff.write_text(diff_text)
    argv = ["validate_review.py", "--payload", str(payload)]
    if with_diff:
        argv += ["--diff", str(diff)]
    monkeypatch.setattr(sys, "argv", argv)
    result = subprocess.CompletedProcess([], schema_exit, stdout="schema says no\n")
    with mock.patch("subprocess.run", return_value=result) as mock_run:
        validate_review.main()
    return mock_run


def test_main_checks_the_schema_then_the_positions(
    validate_review: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match="R99 not in the diff of foo.py"):
        run_main(
            validate_review, monkeypatch, tmp_path, {"side": "RIGHT", "line": 99}, with_diff=True
        )

    mock_run = run_main(
        validate_review, monkeypatch, tmp_path, {"side": "RIGHT", "line": 4}, with_diff=True
    )
    mock_run.assert_called_once()
    assert mock_run.call_args.args[0][:4] == [
        "uvx",
        "check-jsonschema@0.37.4",
        "--schemafile",
        ROOT / ".claude/scripts/review.schema.yml",
    ]


def test_main_reports_a_malformed_diff(
    validate_review: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match="Invalid diff: Diff ends inside hunk"):
        run_main(
            validate_review,
            monkeypatch,
            tmp_path,
            {"side": "RIGHT", "line": 4},
            with_diff=True,
            diff_text="--- a/foo.py\n+++ b/foo.py\n@@ -1,3 +1,3 @@\n a\n",
        )


def test_main_skips_positions_without_a_diff(
    validate_review: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    mock_run = run_main(
        validate_review, monkeypatch, tmp_path, {"side": "RIGHT", "line": 99}, with_diff=False
    )
    mock_run.assert_called_once()


def test_main_stops_at_a_schema_failure(
    validate_review: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match="schema says no"):
        run_main(
            validate_review,
            monkeypatch,
            tmp_path,
            {"side": "RIGHT", "line": 99},
            with_diff=True,
            schema_exit=1,
        )
