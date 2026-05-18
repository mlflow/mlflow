import pytest

from mlflow.agent_playground.test_cases.dedup import (
    HardMatchDuplicate,
    HardMatchUnique,
    check_hard_match,
)


def test_no_existing_returns_unique():
    result = check_hard_match(
        new_source_assistant_message_id="msg-1",
        existing=[],
    )
    assert result == HardMatchUnique()


def test_hard_match_on_anchored_message_id():
    result = check_hard_match(
        new_source_assistant_message_id="msg-789",
        existing=[("msg-789", "tc-001")],
    )
    assert result == HardMatchDuplicate(
        existing_test_case_id="tc-001", reason="anchored_message_id"
    )


def test_no_msg_id_disables_hard_match():
    # New case has no anchor, so the hard match cannot fire even if an
    # existing row has a message id.
    result = check_hard_match(
        new_source_assistant_message_id=None,
        existing=[("msg-789", "tc-002")],
    )
    assert result == HardMatchUnique()


def test_existing_with_no_msg_id_is_not_a_match():
    # None should not match None; a missing anchor on either side means
    # we don't know enough to call them duplicates.
    result = check_hard_match(
        new_source_assistant_message_id=None,
        existing=[(None, "tc-003")],
    )
    assert result == HardMatchUnique()


def test_first_match_returns_immediately():
    result = check_hard_match(
        new_source_assistant_message_id="msg-x",
        existing=[
            ("msg-x", "tc-first"),
            ("msg-x", "tc-second"),
        ],
    )
    assert result == HardMatchDuplicate(
        existing_test_case_id="tc-first", reason="anchored_message_id"
    )


def test_no_match_among_many_existing_returns_unique():
    result = check_hard_match(
        new_source_assistant_message_id="msg-new",
        existing=[
            ("msg-a", "tc-a"),
            ("msg-b", "tc-b"),
            ("msg-c", "tc-c"),
        ],
    )
    assert result == HardMatchUnique()


def test_hard_match_duplicate_rejects_empty_existing_test_case_id():
    # Mirrors the ``min_length=1`` invariant on
    # ``DedupVerdict.existing_test_case_id`` so the hard-match counterpart
    # cannot emit a blank dedup pointer either.
    with pytest.raises(ValueError, match="must be non-empty"):
        HardMatchDuplicate(existing_test_case_id="")


def test_existing_accepts_a_generator():
    # The `existing` parameter is typed as Iterable, so a single-pass
    # generator must work without TypeError.
    def gen():
        yield ("msg-a", "tc-a")
        yield ("msg-x", "tc-x")
        yield ("msg-b", "tc-b")

    result = check_hard_match(
        new_source_assistant_message_id="msg-x",
        existing=gen(),
    )
    assert result == HardMatchDuplicate(existing_test_case_id="tc-x")


@pytest.mark.parametrize(
    ("new_msg_id", "existing", "expected"),
    [
        # Match at the start of the list.
        (
            "msg-x",
            [("msg-x", "tc-1"), ("msg-y", "tc-2")],
            HardMatchDuplicate(existing_test_case_id="tc-1"),
        ),
        # Match in the middle.
        (
            "msg-y",
            [("msg-a", "tc-1"), ("msg-y", "tc-2"), ("msg-z", "tc-3")],
            HardMatchDuplicate(existing_test_case_id="tc-2"),
        ),
        # Match at the end.
        (
            "msg-z",
            [("msg-a", "tc-1"), ("msg-y", "tc-2"), ("msg-z", "tc-3")],
            HardMatchDuplicate(existing_test_case_id="tc-3"),
        ),
        # No match.
        ("msg-q", [("msg-a", "tc-1"), ("msg-b", "tc-2")], HardMatchUnique()),
    ],
)
def test_hard_match_finds_first_in_iteration_order(new_msg_id, existing, expected):
    result = check_hard_match(
        new_source_assistant_message_id=new_msg_id,
        existing=existing,
    )
    assert result == expected
