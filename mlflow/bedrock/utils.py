import logging
from typing import Any, Callable, Sequence

from mlflow.bedrock import FLAVOR_NAME
from mlflow.environment_variables import _MLFLOW_TESTING
from mlflow.tracing.constant import TokenUsageKey
from mlflow.utils.autologging_utils.config import AutoLoggingConfig

_logger = logging.getLogger(__name__)

# Token key constants for different provider formats
INPUT_TOKEN_KEYS: Sequence[str] = [
    "input_tokens",
    "inputTokens",
    "prompt_tokens",
    "promptTokens",
    "prompt_token_count",
]

OUTPUT_TOKEN_KEYS: Sequence[str] = [
    "output_tokens",
    "outputTokens",
    "completion_tokens",
    "completionTokens",
    "generation_token_count",
]

TOTAL_TOKEN_KEYS: Sequence[str] = [
    "total_tokens",
    "totalTokens",
]

# Common documentation for token key mappings used by parsing functions
_USAGE_DOCS = """The provider-specific usage dictionary. This function will attempt to
            extract token usage values using a variety of possible key names, including:
                - input_tokens / inputTokens: Input token count
                - prompt_tokens / promptTokens: Also mapped as input token count
                - output_tokens / outputTokens: Output token count
                - completion_tokens / completionTokens: Also mapped as output token count
                - total_tokens / totalTokens: Total token count (input + output)"""


def _validate_usage_input(usage_data: Any) -> bool:
    """Validate that usage_data is a dictionary suitable for token extraction."""
    return isinstance(usage_data, dict)


def _extract_token_value_by_keys(d: dict[str, Any], names: Sequence[str]) -> int | None:
    """Extract first integer value from dict using sequence of key names.

    Args:
        d: The dictionary to search for token values.
        names: A sequence of key names to try in order.

    Returns:
        The first integer value found for any of the provided keys, or None if none exist.
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


def parse_complete_token_usage_from_response(
    usage_data: dict[str, Any],
) -> dict[str, int] | None:
    """Parse token usage from response, requiring both input and output tokens.

    Args:
        usage_data: {_USAGE_DOCS}

    Returns:
        A dictionary with standardized token usage keys (from TokenUsageKey), or None if
        either input or output tokens are missing. The total_tokens will be calculated
        if not provided.
    """.format(_USAGE_DOCS=_USAGE_DOCS)
    # Input validation using shared validation function
    if not _validate_usage_input(usage_data):
        return None

    # Extract token values directly, only adding them if found
    token_usage_data = {}

    # Extract input tokens - required for complete usage
    if (input_tokens := _extract_token_value_by_keys(usage_data, INPUT_TOKEN_KEYS)) is not None:
        token_usage_data[TokenUsageKey.INPUT_TOKENS] = input_tokens
    else:
        return None  # Incomplete usage without input tokens

    # Extract output tokens - required for complete usage
    if (output_tokens := _extract_token_value_by_keys(usage_data, OUTPUT_TOKEN_KEYS)) is not None:
        token_usage_data[TokenUsageKey.OUTPUT_TOKENS] = output_tokens
    else:
        return None  # Incomplete usage without output tokens

    # Extract or calculate total tokens
    if (total_tokens := _extract_token_value_by_keys(usage_data, TOTAL_TOKEN_KEYS)) is not None:
        token_usage_data[TokenUsageKey.TOTAL_TOKENS] = total_tokens
    else:
        # Calculate total as input + output
        token_usage_data[TokenUsageKey.TOTAL_TOKENS] = input_tokens + output_tokens

    return token_usage_data


def parse_partial_token_usage_from_response(usage_data: dict[str, Any]) -> dict[str, int] | None:
    """Parse partial token usage from response, returning whatever is available.

    Args:
        usage_data: {_USAGE_DOCS}

    Returns:
        A dictionary with standardized token usage keys (from TokenUsageKey) containing
        whatever token data is available, or None if no token usage data is found.
    """.format(_USAGE_DOCS=_USAGE_DOCS)
    # Input validation using shared validation function
    if not _validate_usage_input(usage_data):
        return None

    token_usage_data = {}

    # Try to extract input token count (prompt tokens).
    if (input_tokens := _extract_token_value_by_keys(usage_data, INPUT_TOKEN_KEYS)) is not None:
        token_usage_data[TokenUsageKey.INPUT_TOKENS] = input_tokens

    # Try to extract output token count (completion tokens).
    if (output_tokens := _extract_token_value_by_keys(usage_data, OUTPUT_TOKEN_KEYS)) is not None:
        token_usage_data[TokenUsageKey.OUTPUT_TOKENS] = output_tokens

    # Try to extract total token count.
    if (total_tokens := _extract_token_value_by_keys(usage_data, TOTAL_TOKEN_KEYS)) is not None:
        token_usage_data[TokenUsageKey.TOTAL_TOKENS] = total_tokens

    # If no token usage data was found, return None. Otherwise, return the partial dictionary.
    return token_usage_data or None
