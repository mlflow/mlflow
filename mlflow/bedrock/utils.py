import logging
from typing import Any, Callable, Optional

from mlflow.bedrock import FLAVOR_NAME
from mlflow.environment_variables import _MLFLOW_TESTING
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

_logger = logging.getLogger(__name__)


def capture_exception(logging_message: str):
    """
    A decorator to capture exceptions during a function execution.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                _logger.debug(logging_message)
                if _MLFLOW_TESTING:
                    raise

        return wrapper

    return decorator


def skip_if_trace_disabled(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    A decorator to apply the function only if trace autologging is enabled.
    This decorator is used to skip the test if the trace autologging is disabled.
    """

    def wrapper(original, self, *args, **kwargs):
        config = AutoLoggingConfig.init(flavor_name=FLAVOR_NAME)
        if not config.log_traces:
            return original(self, *args, **kwargs)

        return func(original, self, *args, **kwargs)

    return wrapper


def build_token_usage_dict(
    *,
    raw_usage: Optional[dict[str, int]] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
) -> Optional[dict[str, int]]:
    """Return a standardized token-usage dictionary.

    The returned dict uses :pydata:`mlflow.tracing.constant.TokenUsageKey` entries
    (``input_tokens``, ``output_tokens``, ``total_tokens``).

    Parameters
    ----------
    raw_usage
        Provider-specific usage dictionary. Keys that are automatically recognized:

        * ``input_tokens`` / ``inputTokens``
        * ``prompt_tokens`` / ``promptTokens`` (mapped to *input*)
        * ``output_tokens`` / ``outputTokens``
        * ``completion_tokens`` / ``completionTokens`` (mapped to *output*)
        * ``total_tokens`` / ``totalTokens``
    input_tokens, output_tokens, total_tokens
        Explicit counts. These override what is found in *raw_usage*.

    Returns
    -------
    dict or ``None``
        A dict with keys from :pydata:`TokenUsageKey` if both *input* and *output*
        can be determined, otherwise ``None``.
    """

    def _pick(d: dict, *names):
        for n in names:
            if n in d:
                return d[n]
        return None

    if raw_usage:
        # Use values from raw_usage when explicit args are not supplied
        input_tokens = (
            input_tokens
            if input_tokens is not None
            else _pick(
                raw_usage,
                "input_tokens",
                "inputTokens",
                "prompt_tokens",
                "promptTokens",
                "prompt_token_count",
            )
        )
        output_tokens = (
            output_tokens
            if output_tokens is not None
            else _pick(
                raw_usage,
                "output_tokens",
                "outputTokens",
                "completion_tokens",
                "completionTokens",
                "generation_token_count",
            )
        )
        total_tokens = (
            total_tokens
            if total_tokens is not None
            else _pick(
                raw_usage,
                "total_tokens",
                "totalTokens",
            )
        )

    # Need at least input & output to build a meaningful dict
    if input_tokens is None or output_tokens is None:
        return None

    if total_tokens is None:
        total_tokens = input_tokens + output_tokens

    return {
        TokenUsageKey.INPUT_TOKENS: input_tokens,
        TokenUsageKey.OUTPUT_TOKENS: output_tokens,
        TokenUsageKey.TOTAL_TOKENS: total_tokens,
    }


def set_chat_usage_if_valid(span, usage_dict: Optional[dict]):
    """Set ``SpanAttributeKey.CHAT_USAGE`` on *span* if *usage_dict* is truthy.

    Parameters
    ----------
    span : mlflow.entities.span.LiveSpan
        Span to annotate.
    usage_dict : dict | None
        Output of :func:`build_token_usage_dict`. The attribute is only written when this
        argument is a non-empty dictionary.
    """

    if usage_dict:
        span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage_dict)
