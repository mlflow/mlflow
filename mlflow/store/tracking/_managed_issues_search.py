from typing import Any

from sqlparse.sql import Parenthesis

from mlflow.exceptions import MlflowException
from mlflow.utils.search_utils import SearchUtils


class _IssueSearchUtils(SearchUtils):
    VALID_SEARCH_ATTRIBUTE_KEYS = {"status", "severity", "category", "categories", "name", "title"}
    LIST_SUPPORTED_KEYS = None

    @classmethod
    def _get_value(cls, identifier_type, key, token):
        if identifier_type == cls._TAG_IDENTIFIER and isinstance(token, Parenthesis):
            return cls._parse_list_from_sql_token(token)
        return super()._get_value(identifier_type, key, token)


def _parse_issue_filter(filter_string: str | None) -> dict[str, Any]:
    """Translate MLflow filter syntax to the Issues REST API's structured filters.

    Supports AND, status/severity/category equality and IN, tag equality/IN/IS NOT NULL,
    and name ILIKE '%substring%'. Operators without an equivalent REST filter are rejected.
    """
    if filter_string is None:
        return {}
    if not isinstance(filter_string, str):
        raise MlflowException.invalid_parameter_value("filter_string must be a string")
    if not filter_string.strip():
        return {}

    result = {}
    seen = set()
    for comparison in _IssueSearchUtils.parse_search_filter(filter_string):
        kind = comparison["type"]
        key = comparison["key"]
        if kind == SearchUtils._ATTRIBUTE_IDENTIFIER and key == "categories":
            key = "category"
        if kind == SearchUtils._ATTRIBUTE_IDENTIFIER and key == "title":
            key = "name"
        operator = comparison["comparator"].upper()
        value = comparison["value"]
        if kind not in {SearchUtils._ATTRIBUTE_IDENTIFIER, SearchUtils._TAG_IDENTIFIER}:
            raise MlflowException.invalid_parameter_value(
                "Issue filters support status, severity, category, name, and tags.<key>"
            )
        if (kind, key) in seen:
            raise MlflowException.invalid_parameter_value(
                f"Only one filter clause is supported for '{key}'"
            )
        seen.add((kind, key))

        if kind == SearchUtils._TAG_IDENTIFIER and operator == "IS NOT NULL":
            result.setdefault("tag_filters", []).append({"key": key})
            continue
        if key == "name" and kind == SearchUtils._ATTRIBUTE_IDENTIFIER:
            if (
                operator != "ILIKE"
                or not isinstance(value, str)
                or not value.startswith("%")
                or not value.endswith("%")
                or len(value) < 3
                or any(character in value[1:-1] for character in "%_\\")
            ):
                raise MlflowException.invalid_parameter_value(
                    "Name filters require ILIKE '%substring%' without other wildcards"
                )
            result["query"] = value[1:-1]
            continue
        if operator not in {"=", "IN"} or isinstance(value, tuple) != (operator == "IN"):
            raise MlflowException.invalid_parameter_value(
                f"Issue filter '{key}' requires = with a string or IN with a list"
            )
        values = list(value) if isinstance(value, tuple) else [value]
        if kind == SearchUtils._TAG_IDENTIFIER:
            result.setdefault("tag_filters", []).append({"key": key, "values": values})
        else:
            result[
                {"status": "statuses", "severity": "severities", "category": "categories"}[key]
            ] = values
    return result


def _parse_issue_order_by(order_by: list[str] | None) -> dict[str, str]:
    if not order_by:
        return {}
    if not isinstance(order_by, list) or len(order_by) != 1:
        raise MlflowException.invalid_parameter_value(
            "Databricks issue search supports exactly one order_by field"
        )
    field, ascending = SearchUtils._parse_order_by_string(order_by[0])
    sort_keys = {
        "severity": "SEVERITY",
        "last_seen_timestamp": "LAST_SEEN_TIME",
        "last_updated_timestamp": "UPDATE_TIME",
        "created_timestamp": "CREATE_TIME",
    }
    if field not in sort_keys:
        raise MlflowException.invalid_parameter_value(
            f"Unsupported issue order_by field {field!r}. Expected one of {list(sort_keys)}"
        )
    return {"sort_key": sort_keys[field], "sort_order": "ASCENDING" if ascending else "DESCENDING"}
