import pytest

from mlflow.genai.judges.utils.multi_turn import (
    render_prior_conversation_block,
    truncate_prior_conversation,
)


def _msg(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


_FOUR_TURN_CONVERSATION = [
    _msg("user", "hello"),
    _msg("assistant", "hi"),
    _msg("user", "how are you?"),
    _msg("assistant", "good"),
]


def test_truncate_returns_empty_for_empty_input():
    assert truncate_prior_conversation([]) == []


def test_truncate_returns_empty_when_max_turns_is_zero():
    assert truncate_prior_conversation(_FOUR_TURN_CONVERSATION, max_turns=0) == []


def test_truncate_keeps_only_last_n_messages_when_max_turns_lt_total():
    result = truncate_prior_conversation(_FOUR_TURN_CONVERSATION, max_turns=2)
    assert result == _FOUR_TURN_CONVERSATION[-2:]


def test_truncate_returns_all_when_max_turns_ge_total():
    assert (
        truncate_prior_conversation(_FOUR_TURN_CONVERSATION, max_turns=10)
        == _FOUR_TURN_CONVERSATION
    )


def test_truncate_no_token_cap_when_token_budget_is_none():
    long_msg = _msg("user", "x" * 10_000)
    # Without a budget, the long message survives even with a generous max_turns.
    assert truncate_prior_conversation([long_msg], max_turns=5, token_budget=None) == [long_msg]


def test_truncate_drops_oldest_until_under_token_budget():
    msgs = [
        _msg("user", "a" * 800),  # ~200 tokens
        _msg("assistant", "b" * 800),  # ~200 tokens
        _msg("user", "c" * 800),  # ~200 tokens
    ]
    # Budget of ~250 tokens should keep just the last message.
    result = truncate_prior_conversation(msgs, max_turns=10, token_budget=250)
    assert result == msgs[-1:]


def test_truncate_keeps_at_least_one_message_when_budget_is_unrealistically_small():
    # A budget too small to fit even one message still keeps the last one
    # so the user notices the misconfiguration rather than getting silently
    # blank context.
    msg = _msg("user", "x" * 5000)
    result = truncate_prior_conversation([msg, msg], max_turns=10, token_budget=1)
    assert len(result) == 1


def test_truncate_max_turns_applied_before_token_budget():
    msgs = [
        _msg("user", "old"),
        _msg("user", "x" * 800),
        _msg("user", "y" * 800),
    ]
    # max_turns first chops to last two, token budget then trims one more.
    result = truncate_prior_conversation(msgs, max_turns=2, token_budget=250)
    assert result == msgs[-1:]


@pytest.mark.parametrize(
    ("env_max_turns", "expected_count"),
    [
        ("0", 0),
        ("1", 1),
        ("2", 2),
        ("10", 4),
    ],
)
def test_truncate_reads_max_turns_from_env_when_not_passed(
    monkeypatch, env_max_turns, expected_count
):
    monkeypatch.setenv("MLFLOW_JUDGE_MAX_PRIOR_TURNS", env_max_turns)
    result = truncate_prior_conversation(_FOUR_TURN_CONVERSATION)
    assert len(result) == expected_count


def test_truncate_reads_token_budget_from_env_when_not_passed(monkeypatch):
    monkeypatch.setenv("MLFLOW_JUDGE_PRIOR_TURNS_TOKEN_BUDGET", "200")
    # Each message is ~200 tokens; budget should drop the oldest.
    msgs = [
        _msg("user", "a" * 800),
        _msg("user", "b" * 800),
    ]
    result = truncate_prior_conversation(msgs, max_turns=10)
    assert len(result) == 1


def test_render_empty_messages_returns_empty_string():
    # Load-bearing: the empty case must produce "" (no header, no tag wrapper)
    # so the existing prompt template remains byte-for-byte unchanged when
    # there is no prior conversation.
    assert render_prior_conversation_block([]) == ""


def test_render_includes_header_and_role_prefixed_messages():
    rendered = render_prior_conversation_block(_FOUR_TURN_CONVERSATION)
    assert "<prior_conversation>" in rendered
    assert "</prior_conversation>" in rendered
    assert "[user]: hello" in rendered
    assert "[assistant]: hi" in rendered
    assert "[user]: how are you?" in rendered
    assert "[assistant]: good" in rendered


def test_rendered_block_ends_with_blank_line_separator():
    rendered = render_prior_conversation_block(_FOUR_TURN_CONVERSATION)
    assert rendered.endswith("\n\n")
