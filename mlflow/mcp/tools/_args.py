"""
Normalization of tool arguments that also accept the shape the CLI-backed tools took.

The first generation of MCP tools forwarded arguments to CLI callbacks, so some lists and
objects were passed as comma-separated or JSON strings. The typed tools accept the structured
shape and keep accepting the old one.
"""

import json
from typing import Annotated, Any, Literal

from pydantic import Field

from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException

View = Annotated[
    Literal["active_only", "deleted_only", "all"],
    Field(description="Lifecycle view: 'active_only' (default), 'deleted_only' or 'all'."),
]
OrderBy = Annotated[
    list[str] | str | None,
    Field(
        description="Order-by clauses, e.g. ['name ASC', 'creation_time DESC']. A "
        "comma-separated string is also accepted."
    ),
]
PageToken = Annotated[
    str | None, Field(description="Token returned by a previous call to continue from.")
]
DeprecatedOutput = Annotated[
    Literal["table", "json"] | None,
    Field(description="Deprecated and ignored: results are always structured."),
]


def as_list(value: list[str] | str | None) -> list[str] | None:
    """A list of strings, or a comma-separated string of them."""
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def as_json_value(value: Any) -> Any:
    """A JSON value; strings are parsed as JSON when they are valid JSON, as the CLI did."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def as_json_object(value: dict[str, Any] | str | None, name: str) -> dict[str, Any] | None:
    """A JSON object, or a string holding one."""
    if value is None or isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as e:
        raise MlflowException.invalid_parameter_value(
            f"`{name}` must be a JSON object, got invalid JSON: {e}"
        ) from None
    if not isinstance(parsed, dict):
        raise MlflowException.invalid_parameter_value(
            f"`{name}` must be a JSON object, not {type(parsed).__name__}."
        )
    return parsed


def as_tag_dict(value: dict[str, str] | list[str] | None) -> dict[str, str]:
    """A tag mapping, or a list of ``key=value`` strings."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    tags: dict[str, str] = {}
    for tag in value:
        match tag.split("=", 1):
            case [key, tag_value]:
                if key in tags:
                    raise MlflowException.invalid_parameter_value(f"Duplicate tag key: '{key}'")
                tags[key] = tag_value
            case _:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid tag format: '{tag}'. Tags must be in key=value format."
                )
    return tags


def as_view_type(view: str) -> int:
    return ViewType.from_string(view) if view else ViewType.ACTIVE_ONLY


def check_non_negative(value: int | None, name: str) -> None:
    if value is not None and value < 0:
        raise MlflowException.invalid_parameter_value(f"`{name}` must be a non-negative integer.")
