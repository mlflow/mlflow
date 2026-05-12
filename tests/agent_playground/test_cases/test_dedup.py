import pytest

from mlflow.agent_playground.test_cases.dedup import (
    Duplicate,
    Unique,
    check_hard_match,
)


def test_no_existing_returns_unique():
    result = check_hard_match(
        new_source_assistant_message_id="msg-1",
        existing=[],
    )
    assert result == Unique()


def test_hard_match_on_anchored_message_id():
    result = check_hard_match(
        new_source_assistant_message_id="msg-789",
        existing=[("msg-789", "tc-001")],
    )
    assert result == Duplicate(test_case_id="tc-001", reason="anchored_message_id")


def test_no_msg_id_disables_hard_match():
    # New case has no anchor, so the hard match cannot fire even if an
    # existing row has a message id.
    result = check_hard_match(
        new_source_assistant_message_id=None,
        existing=[("msg-789", "tc-002")],
    )
    assert result == Unique()


def test_existing_with_no_msg_id_is_not_a_match():
    # None should not match None; a missing anchor on either side means
    # we don't know enough to call them duplicates.
    result = check_hard_match(
        new_source_assistant_message_id=None,
        existing=[(None, "tc-003")],
    )
    assert result == Unique()


def test_first_match_returns_immediately():
    result = check_hard_match(
        new_source_assistant_message_id="msg-x",
        existing=[
            ("msg-x", "tc-first"),
            ("msg-x", "tc-second"),
        ],
    )
    assert result == Duplicate(test_case_id="tc-first", reason="anchored_message_id")


def test_no_match_among_many_existing_returns_unique():
    result = check_hard_match(
        new_source_assistant_message_id="msg-new",
        existing=[
            ("msg-a", "tc-a"),
            ("msg-b", "tc-b"),
            ("msg-c", "tc-c"),
        ],
    )
    assert result == Unique()


@pytest.mark.parametrize(
    ("new_msg_id", "existing", "expected"),
    [
        # Match at the start of the list.
        ("msg-x", [("msg-x", "tc-1"), ("msg-y", "tc-2")], Duplicate(test_case_id="tc-1")),
        # Match in the middle.
        (
            "msg-y",
            [("msg-a", "tc-1"), ("msg-y", "tc-2"), ("msg-z", "tc-3")],
            Duplicate(test_case_id="tc-2"),
        ),
        # Match at the end.
        (
            "msg-z",
            [("msg-a", "tc-1"), ("msg-y", "tc-2"), ("msg-z", "tc-3")],
            Duplicate(test_case_id="tc-3"),
        ),
        # No match.
        ("msg-q", [("msg-a", "tc-1"), ("msg-b", "tc-2")], Unique()),
    ],
)
def test_hard_match_finds_first_in_iteration_order(new_msg_id, existing, expected):
    result = check_hard_match(
        new_source_assistant_message_id=new_msg_id,
        existing=existing,
    )
    assert result == expected
