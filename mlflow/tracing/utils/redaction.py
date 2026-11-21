import functools
import logging
import re
import types
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

import pydantic

_logger = logging.getLogger(__name__)

# Encoding these with `str()` would embed the repr of a bound object or partial arguments, which
# can hold credentials (e.g. a method bound to an API client), so they are recorded by name.
_CALLABLE_TYPES = (
    types.FunctionType,
    types.MethodType,
    types.BuiltinFunctionType,
    types.BuiltinMethodType,
    functools.partial,
)

# Matches credential-like keys such as `api_key`, `_api_key`, `x-api-key`, `client_secret`,
# `private_key`, `Authorization`, `Cookie`, `Set-Cookie`, `auth_token`, or `accessToken`.
# Credential keys end with `token`, so `max_tokens`, `tokenizer`, and `token_usage` (legitimate
# LLM configuration and usage fields) are kept.
SENSITIVE_KEY_PATTERN = re.compile(
    r"(?i:api[-_]?key|private[-_]?key|secret|password|authorization|credential)"  # anywhere
    r"|(?<![A-Za-z])(?i:cookies?)$"  # HTTP `Cookie` / `Set-Cookie` headers
    r"|(?<![A-Za-z])(?i:token)$"  # snake_case / standalone `token`
    r"|(?<=[a-z])Token$"  # camelCase `accessToken`
)

# Keys whose value is a JSON schema, such as pydantic-ai's `ToolDefinition.parameters_json_schema`
# and `OutputObjectDefinition.json_schema` or an MCP tool's `inputSchema`. Property and definition
# names in a schema name fields rather than hold values, so they are kept there.
_SCHEMA_KEYS = frozenset({
    "parameters_json_schema",
    "json_schema",
    "input_schema",
    "inputSchema",
    "output_schema",
    "outputSchema",
})

# JSON schema keywords that hold example or default values rather than describe a field.
_SCHEMA_VALUE_KEYWORDS = frozenset({"default", "const", "enum", "examples", "example"})

# Keywords mapping names (properties or definitions) to subschemas.
_SCHEMA_NAMED_SUBSCHEMA_KEYWORDS = frozenset({
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
})

# Keywords holding a subschema or a list of subschemas.
_SCHEMA_SUBSCHEMA_KEYWORDS = frozenset({
    "items",
    "prefixItems",
    "additionalItems",
    "additionalProperties",
    "unevaluatedItems",
    "unevaluatedProperties",
    "propertyNames",
    "contains",
    "not",
    "if",
    "then",
    "else",
    "anyOf",
    "oneOf",
    "allOf",
})


def strip_credentials(value: Any) -> Any:
    """
    Recursively drop credential-like keys from a value before it is attached to a span.

    Mappings, lists, tuples, and sets are traversed, and functions and methods are replaced by
    their qualified name. Dataclasses and pydantic models are converted to dicts the same way
    ``TraceJSONEncoder`` serializes them, so credentials stored on nested objects (e.g. a model
    client's ``api_key`` field) are dropped too. Other objects are returned as-is.

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
        if isinstance(value, Mapping):
            return {
                k: (
                    _strip_schema(v, ancestors, sensitive=False)
                    if k in _SCHEMA_KEYS
                    else _strip_credentials(v, ancestors)
                )
                for k, v in value.items()
                if not _is_sensitive_key(k)
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
        if isinstance(value, _CALLABLE_TYPES):
            func = value.func if isinstance(value, functools.partial) else value
            return getattr(func, "__qualname__", type(func).__name__)
        return value
    finally:
        ancestors.discard(id(value))


def _is_sensitive_key(key: Any) -> bool:
    return isinstance(key, str) and bool(SENSITIVE_KEY_PATTERN.search(key))


def _strip_schema(schema: Any, ancestors: set[int], sensitive: bool) -> Any:
    # `sensitive` is set below a credential-like property or definition, where value-bearing
    # keywords (e.g. a secret `default`) are dropped from every nested subschema, including
    # `anyOf` / `oneOf` / `allOf` branches.
    if isinstance(schema, (list, tuple)):
        return [_strip_schema(item, ancestors, sensitive) for item in schema]
    if not isinstance(schema, Mapping):
        return _strip_credentials(schema, ancestors)

    if id(schema) in ancestors:
        return type(schema).__name__
    ancestors.add(id(schema))
    try:
        result = {}
        for key, value in schema.items():
            if sensitive and key in _SCHEMA_VALUE_KEYWORDS:
                continue
            if key in _SCHEMA_NAMED_SUBSCHEMA_KEYWORDS and isinstance(value, Mapping):
                result[key] = {
                    name: _strip_schema(sub, ancestors, sensitive or _is_sensitive_key(name))
                    for name, sub in value.items()
                    # A subschema is a mapping or a boolean; anything else may be a secret value
                    if isinstance(sub, (Mapping, bool)) or not _is_sensitive_key(name)
                }
            elif key in _SCHEMA_SUBSCHEMA_KEYWORDS:
                result[key] = _strip_schema(value, ancestors, sensitive)
            elif not _is_sensitive_key(key):
                result[key] = _strip_credentials(value, ancestors)
        return result
    finally:
        ancestors.discard(id(schema))
