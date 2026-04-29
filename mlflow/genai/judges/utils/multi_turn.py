"""
Helpers for surfacing prior conversation context to single-turn LLM judges.

The prior conversation is supplied as an OpenAI-style message list
(``[{"role": ..., "content": ...}]``) — the same shape returned by
``mlflow.genai.utils.trace_utils.extract_prior_turns``. Truncation here is
purely cap enforcement: turn count first, then optional token budget.
Token estimation uses a 4-chars-per-token heuristic, which is sufficient for
budget-protection purposes and avoids pulling in a tokenizer dependency.
"""

from mlflow.environment_variables import (
    MLFLOW_JUDGE_MAX_PRIOR_TURNS,
    MLFLOW_JUDGE_PRIOR_TURNS_TOKEN_BUDGET,
)

_PRIOR_CONVERSATION_HEADER = (
    "Below is the prior conversation history that led up to the current turn. "
    "Use it as context when evaluating the answer below — for example, when the "
    'user\'s input is a continuation ("yes", "tell me more") that only makes '
    "sense given earlier turns."
)
_CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def truncate_prior_conversation(
    messages: list[dict[str, str]],
    *,
    max_turns: int | None = None,
    token_budget: int | None = None,
) -> list[dict[str, str]]:
    """
    Apply turn-count and token-budget caps to a prior conversation, dropping
    the oldest messages first.

    Args:
        messages: Conversation messages in chronological order.
        max_turns: Maximum number of messages to keep. ``None`` reads
            :data:`mlflow.environment_variables.MLFLOW_JUDGE_MAX_PRIOR_TURNS`.
            ``0`` returns an empty list.
        token_budget: Maximum estimated tokens for the kept messages. ``None``
            reads :data:`mlflow.environment_variables.MLFLOW_JUDGE_PRIOR_TURNS_TOKEN_BUDGET`,
            which is unset by default and means "no token cap".
    """
    if not messages:
        return []

    if max_turns is None:
        max_turns = MLFLOW_JUDGE_MAX_PRIOR_TURNS.get()
    if max_turns <= 0:
        return []
    truncated = messages[-max_turns:]

    if token_budget is None:
        token_budget = MLFLOW_JUDGE_PRIOR_TURNS_TOKEN_BUDGET.get()
    if token_budget is None:
        return truncated

    # Drop oldest turns until under budget. Always keep at least one message
    # if the budget cannot accommodate even that — a too-small budget is a
    # configuration issue and silently dropping all context would hide it.
    total = sum(_estimate_tokens(m.get("content", "")) for m in truncated)
    while len(truncated) > 1 and total > token_budget:
        dropped = truncated.pop(0)
        total -= _estimate_tokens(dropped.get("content", ""))
    return truncated


def render_prior_conversation_block(messages: list[dict[str, str]]) -> str:
    """
    Render the prior conversation as a prompt prefix, or an empty string if
    the list is empty. The empty-string return is load-bearing: callers
    concatenate this with their existing prompt template and rely on it
    producing byte-for-byte the original prompt when there is no prior
    context. Do not change the empty-case return value.
    """
    if not messages:
        return ""
    rendered_turns = "\n".join(
        f"[{message.get('role', 'unknown')}]: {message.get('content', '')}" for message in messages
    )
    return (
        f"{_PRIOR_CONVERSATION_HEADER}\n\n"
        f"<prior_conversation>\n{rendered_turns}\n</prior_conversation>\n\n"
    )
