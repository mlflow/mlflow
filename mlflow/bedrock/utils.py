import logging
from typing import Any, Callable, Optional, Sequence

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


def _extract_token_value_by_keys(d: dict[str, Any], names: Sequence[str]) -> Optional[int]:
    """Extracts the first integer value from a dictionary using a sequence of possible key names.

    Args:
        d: The dictionary to search for token values.
        names: A sequence of key names to try in order.

    Returns:
        The first integer value found for any of the provided keys, or None if none exist.

    Example:
        >>> usage = {"input_tokens": 10, "output_tokens": 5}
        >>> _extract_token_value_by_keys(usage, ["input_tokens", "inputTokens", "prompt_tokens"])
        10
        >>> _extract_token_value_by_keys(usage, ["total_tokens", "totalTokens"])
        None
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


def parse_token_usage_from_response(
    raw_usage: dict[str, Any], require_full_usage: bool = True
) -> Optional[dict[str, int]]:
    """
    Parses token usage information from a provider-specific response dictionary and
    standardizes it into a dictionary with MLflow's canonical token usage keys.

    Args:
        raw_usage:
            The provider-specific usage dictionary. This function will attempt to extract
            token usage values using a variety of possible key names, including:
                - input_tokens / inputTokens: Input token count
                - prompt_tokens / promptTokens: Also mapped as input token count
                - output_tokens / outputTokens: Output token count
                - completion_tokens / completionTokens: Also mapped as output token count
                - total_tokens / totalTokens: Total token count (input + output)
        require_full_usage:
            If True, the function will only return a result if both input and output token
            counts are present (i.e., a "complete" usage record). If False, the function
            will return a dictionary with whatever token usage data is available, which is
            useful for streaming or partial responses.

    Returns:
        A dictionary with standardized token usage keys (from TokenUsageKey), or None if
        required data is missing. If require_full_usage=True, returns None unless both input
        and output tokens are present. If require_full_usage=False, returns a dictionary with
        any available token usage data, or None if none is found.

    Example:
        >>> parse_token_usage_from_response({"inputTokens": 10, "completionTokens": 15})
        {'input_tokens': 10, 'output_tokens': 15, 'total_tokens': 25}
    """
    if require_full_usage:
        # In "require_full_usage" mode, only return a result if both input and output
        # tokens are present.

        # First, extract all possible token values from the raw usage dictionary
        # using the known key sets.
        token_values = {
            TokenUsageKey.INPUT_TOKENS: _extract_token_value_by_keys(raw_usage, INPUT_TOKEN_KEYS),
            TokenUsageKey.OUTPUT_TOKENS: _extract_token_value_by_keys(raw_usage, OUTPUT_TOKEN_KEYS),
            TokenUsageKey.TOTAL_TOKENS: _extract_token_value_by_keys(raw_usage, TOTAL_TOKEN_KEYS),
        }

        # Remove any keys where the value is None (i.e., not found in the response).
        token_usage_dict = {
            token_key: token_value
            for token_key, token_value in token_values.items()
            if token_value is not None
        }

        # If either input or output tokens are missing, return None (incomplete usage).
        if (
            TokenUsageKey.INPUT_TOKENS not in token_usage_dict
            or TokenUsageKey.OUTPUT_TOKENS not in token_usage_dict
        ):
            return None

        # If total tokens are not provided, calculate as input + output.
        if TokenUsageKey.TOTAL_TOKENS not in token_usage_dict:
            token_usage_dict[TokenUsageKey.TOTAL_TOKENS] = (
                token_usage_dict[TokenUsageKey.INPUT_TOKENS]
                + token_usage_dict[TokenUsageKey.OUTPUT_TOKENS]
            )

        return token_usage_dict
    else:
        # In "partial" mode (for streaming or incomplete responses), extract whatever
        # token usage data is available.
        token_usage_dict = {}

        # Try to extract input token count (prompt tokens).
        input_tokens = _extract_token_value_by_keys(raw_usage, INPUT_TOKEN_KEYS)
        if input_tokens is not None:
            token_usage_dict[TokenUsageKey.INPUT_TOKENS] = input_tokens

        # Try to extract output token count (completion tokens).
        output_tokens = _extract_token_value_by_keys(raw_usage, OUTPUT_TOKEN_KEYS)
        if output_tokens is not None:
            token_usage_dict[TokenUsageKey.OUTPUT_TOKENS] = output_tokens

        # Try to extract total token count.
        total_tokens = _extract_token_value_by_keys(raw_usage, TOTAL_TOKEN_KEYS)
        if total_tokens is not None:
            token_usage_dict[TokenUsageKey.TOTAL_TOKENS] = total_tokens

        # If no token usage data was found, return None. Otherwise, return the partial dictionary.
        return token_usage_dict if token_usage_dict else None
