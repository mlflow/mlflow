import json
from typing import Any

import pytest
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]
from skills.commands.validate_review import DEFAULT_SCHEMA

BODY = "Overall the change looks good."
CRITICAL = {"path": "a.py", "body": "🔴 **CRITICAL:** unhandled None", "line": 1, "side": "RIGHT"}
MODERATE = {"path": "a.py", "body": "🟡 **MODERATE:** unclear name", "line": 1, "side": "RIGHT"}
NIT = {"path": "a.py", "body": "🟢 **NIT:** stray blank line", "line": 1, "side": "RIGHT"}


@pytest.fixture(scope="module")
def validator() -> Draft202012Validator:
    return Draft202012Validator(json.loads(DEFAULT_SCHEMA.read_text()))


def test_schema_is_well_formed() -> None:
    # Catches malformed keyword values and invalid `pattern` regexes, which would otherwise
    # surface as a confusing failure on the first payload validated rather than here.
    Draft202012Validator.check_schema(json.loads(DEFAULT_SCHEMA.read_text()))


def payload(**overrides: Any) -> dict[str, Any]:
    return {"event": "APPROVE", "body": BODY, "comments": [], **overrides}


@pytest.mark.parametrize(
    ("event", "comments", "valid"),
    [
        ("APPROVE", [], True),
        ("APPROVE", [NIT, MODERATE], True),
        ("APPROVE", [CRITICAL], False),
        ("APPROVE", [NIT, CRITICAL], False),
        ("COMMENT", [CRITICAL], True),
        ("COMMENT", [NIT, CRITICAL], True),
        ("COMMENT", [], False),
        ("COMMENT", [NIT, MODERATE], False),
    ],
)
def test_event_is_derived_from_critical_comments(
    validator: Draft202012Validator,
    event: str,
    comments: list[dict[str, Any]],
    valid: bool,
) -> None:
    assert validator.is_valid(payload(event=event, comments=comments)) is valid


@pytest.mark.parametrize("event", ["approve", "REQUEST_CHANGES", "DISMISS", ""])
def test_event_rejects_unknown_actions(validator: Draft202012Validator, event: str) -> None:
    assert not validator.is_valid(payload(event=event))


@pytest.mark.parametrize(
    "body",
    [
        "unhandled None",
        "**CRITICAL:** unhandled None",
        "🔴 CRITICAL: unhandled None",
        "🔴 **critical:** unhandled None",
        "🟠 **MODERATE:** unclear name",
        " 🟢 **NIT:** stray blank line",
    ],
)
def test_comment_rejects_bodies_without_a_severity_prefix(
    validator: Draft202012Validator, body: str
) -> None:
    assert not validator.is_valid(payload(comments=[{**NIT, "body": body}]))


@pytest.mark.parametrize(
    ("body", "valid"),
    [
        (BODY, True),
        ("Looks good.", True),
        ("", False),
    ],
)
def test_body_must_be_non_empty(validator: Draft202012Validator, body: str, valid: bool) -> None:
    assert validator.is_valid(payload(body=body)) is valid


@pytest.mark.parametrize("field", ["event", "body", "comments"])
def test_top_level_fields_are_required(validator: Draft202012Validator, field: str) -> None:
    assert not validator.is_valid({k: v for k, v in payload().items() if k != field})


@pytest.mark.parametrize("field", ["path", "body", "line", "side"])
def test_comment_fields_are_required(validator: Draft202012Validator, field: str) -> None:
    comment = {k: v for k, v in NIT.items() if k != field}
    assert not validator.is_valid(payload(comments=[comment]))


def test_comment_path_must_not_be_empty(validator: Draft202012Validator) -> None:
    assert not validator.is_valid(payload(comments=[{**NIT, "path": ""}]))


@pytest.mark.parametrize(
    ("extra", "valid"),
    [
        ({"start_line": 1, "start_side": "RIGHT"}, True),
        ({"start_line": 1}, False),
        ({"start_side": "RIGHT"}, False),
    ],
)
def test_multi_line_anchor_requires_both_start_fields(
    validator: Draft202012Validator, extra: dict[str, Any], valid: bool
) -> None:
    assert validator.is_valid(payload(comments=[{**NIT, "line": 2, **extra}])) is valid


@pytest.mark.parametrize(
    ("commit_id", "valid"),
    [
        ("a" * 40, True),
        ("a" * 39, False),
        ("A" * 40, False),
        ("z" * 40, False),
    ],
)
def test_commit_id_must_be_a_full_sha(
    validator: Draft202012Validator, commit_id: str, valid: bool
) -> None:
    assert validator.is_valid(payload(commit_id=commit_id)) is valid


def test_unknown_fields_are_rejected(validator: Draft202012Validator) -> None:
    assert not validator.is_valid(payload(reviewer="claude"))
    assert not validator.is_valid(payload(comments=[{**NIT, "severity": "NIT"}]))


@pytest.mark.parametrize("side", ["LEFT", "RIGHT"])
def test_comment_accepts_both_sides(validator: Draft202012Validator, side: str) -> None:
    assert validator.is_valid(payload(comments=[{**NIT, "side": side}]))


@pytest.mark.parametrize("line", [0, -1])
def test_comment_line_must_be_positive(validator: Draft202012Validator, line: int) -> None:
    assert not validator.is_valid(payload(comments=[{**NIT, "line": line}]))


@pytest.mark.parametrize(
    "overrides",
    [
        {"line": "1"},
        {"path": ["a.py"]},
        {"start_line": "1", "start_side": "RIGHT"},
        {"start_line": 0, "start_side": "RIGHT"},
        {"start_line": 1, "start_side": "MIDDLE"},
    ],
)
def test_comment_rejects_wrongly_typed_values(
    validator: Draft202012Validator, overrides: dict[str, Any]
) -> None:
    assert not validator.is_valid(payload(comments=[{**NIT, "line": 2, **overrides}]))


@pytest.mark.parametrize(
    "value",
    [
        ["not", "an", "object"],
        payload(body=123),
        payload(commit_id=123),
        payload(comments=[123]),
    ],
)
def test_payload_rejects_wrongly_typed_values(validator: Draft202012Validator, value: Any) -> None:
    assert not validator.is_valid(value)


@pytest.mark.parametrize("comments", [None, "none", {"0": NIT}])
def test_comments_must_be_an_array(validator: Draft202012Validator, comments: Any) -> None:
    # A non-array makes `items` and `contains` no-op, disabling every comment-level
    # constraint and making the `if` branch always succeed, which pins `event` to COMMENT.
    assert not validator.is_valid(payload(event="COMMENT", comments=comments))
