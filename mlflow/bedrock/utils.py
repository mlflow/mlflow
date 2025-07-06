import logging
from typing import Any, Callable, Optional

from mlflow.bedrock import FLAVOR_NAME
from mlflow.environment_variables import _MLFLOW_TESTING
from mlflow.tracing.constant import TokenUsageKey
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

_logger = logging.getLogger(__name__)

# Token key constants for different provider formats
INPUT_TOKEN_KEYS = [
    "input_tokens",
    "inputTokens",
    "prompt_tokens",
    "promptTokens",
    "prompt_token_count",
]

OUTPUT_TOKEN_KEYS = [
    "output_tokens",
    "outputTokens",
    "completion_tokens",
    "completionTokens",
    "generation_token_count",
]

TOTAL_TOKEN_KEYS = [
    "total_tokens",
    "totalTokens",
]


def _pick(d: dict[str, Any], *names: str) -> Optional[int]:
    """
    Pick the first available value from a dictionary using multiple possible key names.

    Args:
        d: Dictionary to search in
        *names: Variable number of key names to try in order

    Returns:
        The first integer value found for any of the provided keys, or None if none exist

    Example:
        >>> usage = {"input_tokens": 10, "output_tokens": 5}
        >>> _pick(usage, "input_tokens", "inputTokens", "prompt_tokens")
        10
        >>> _pick(usage, "total_tokens", "totalTokens")
        None
        >>> _pick(usage, "input_tokens", "inputTokens")  # where input_tokens = "10"
        None  # Returns None for string values
        >>> _pick(usage, "input_tokens", "inputTokens")  # where input_tokens = 10.5
        None  # Returns None for float values
    """
    return next((d[name] for name in names if name in d and isinstance(d[name], int)), None)


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

    The returned dict uses mlflow.tracing.constant.TokenUsageKey entries
    (input_tokens, output_tokens, total_tokens).

    Args:
        raw_usage: Provider-specific usage dictionary. Keys that are automatically recognized:
            - input_tokens / inputTokens
            - prompt_tokens / promptTokens (mapped to input)
            - output_tokens / outputTokens
            - completion_tokens / completionTokens (mapped to output)
            - total_tokens / totalTokens
        input_tokens: Explicit input token count. Overrides what is found in raw_usage.
        output_tokens: Explicit output token count. Overrides what is found in raw_usage.
        total_tokens: Explicit total token count. Overrides what is found in raw_usage.

    Returns:
        A dict with keys from TokenUsageKey if both input and output can be determined,
        otherwise None.
    """

    if raw_usage:
        input_tokens = (
            input_tokens if input_tokens is not None else _pick(raw_usage, *INPUT_TOKEN_KEYS)
        )
        output_tokens = (
            output_tokens if output_tokens is not None else _pick(raw_usage, *OUTPUT_TOKEN_KEYS)
        )
        total_tokens = (
            total_tokens if total_tokens is not None else _pick(raw_usage, *TOTAL_TOKEN_KEYS)
        )

    # Need at least input & output to build a meaningful dict
    return (
        {
            TokenUsageKey.INPUT_TOKENS: input_tokens,
            TokenUsageKey.OUTPUT_TOKENS: output_tokens,
            TokenUsageKey.TOTAL_TOKENS: total_tokens or (input_tokens + output_tokens),
        }
        if (input_tokens is not None and output_tokens is not None)
        else None
    )
