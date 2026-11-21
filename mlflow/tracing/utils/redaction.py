import logging
import re
from dataclasses import fields, is_dataclass
from typing import Any

import pydantic

_logger = logging.getLogger(__name__)

# Matches credential-like keys such as `api_key`, `_api_key`, `x-api-key`, `client_secret`,
# `Authorization`, `auth_token`, or `accessToken`. Credential keys end with `token`, so
# `max_tokens`, `tokenizer`, and `token_usage` (legitimate LLM configuration and usage fields)
# are kept.
SENSITIVE_KEY_PATTERN = re.compile(
    r"(?i:api[-_]?key|secret|password|authorization|credential)"  # anywhere in the key
    r"|(?<![A-Za-z])(?i:token)$"  # snake_case / standalone `token`
    r"|(?<=[a-z])Token$"  # camelCase `accessToken`
)


def strip_credentials(value: Any) -> Any:
    """
    Recursively drop credential-like keys from a value before it is attached to a span.

    Dicts, lists, tuples, and sets are traversed. Dataclasses and pydantic models are converted
    to dicts the same way ``TraceJSONEncoder`` serializes them, so credentials stored on nested
    objects (e.g. a model client's ``api_key`` field) are dropped too. Other objects are returned
    as-is.

    Fails closed: if sanitization raises, the value is replaced rather than logged unsanitized.
    Dicts become an empty dict so callers that merge the result into span attributes keep
    working; other values become a placeholder string.
    """
    try:
        return _strip_credentials(value, set())
    except Exception as e:
        _logger.debug("Failed to sanitize %s for span logging: %s", type(value).__name__, e)
        if isinstance(value, dict):
            return {}
        return f"<{type(value).__name__}: omitted, sanitization failed>"


def _strip_credentials(value: Any, ancestors: set[int]) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if id(value) in ancestors:
        return type(value).__name__
    ancestors.add(id(value))
    try:
        if isinstance(value, dict):
            return {
                k: _strip_credentials(v, ancestors)
                for k, v in value.items()
                if not (isinstance(k, str) and SENSITIVE_KEY_PATTERN.search(k))
            }
        if isinstance(value, (list, tuple, set)):
            sanitized = [_strip_credentials(item, ancestors) for item in value]
            # Sets become lists: sanitized items (e.g. expanded dataclasses) may be unhashable
            return tuple(sanitized) if isinstance(value, tuple) else sanitized
        if isinstance(value, pydantic.BaseModel):
            return _strip_credentials(value.model_dump(warnings=False), ancestors)
        if is_dataclass(value) and not isinstance(value, type):
            return _strip_credentials(
                {f.name: getattr(value, f.name) for f in fields(value)}, ancestors
            )
        return value
    finally:
        ancestors.discard(id(value))
