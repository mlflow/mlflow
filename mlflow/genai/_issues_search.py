from typing import Any

import sqlparse
from sqlparse.sql import Comparison, Parenthesis
from sqlparse.tokens import Token as TokenType

from mlflow.exceptions import MlflowException
from mlflow.utils.search_utils import SearchUtils, _join_in_comparison_tokens


def _string_value(value: str) -> str:
    tokens = [
        (kind, text)
        for kind, text in sqlparse.lexer.tokenize(value)
        if kind not in TokenType.Whitespace
    ]
    if len(tokens) != 1 or tokens[0][0] not in {
        TokenType.Literal.String.Single,
        TokenType.Literal.String.Symbol,
    }:
        raise MlflowException.invalid_parameter_value("Issue filter values must be quoted strings")
    quote = value[0]
    return value[1:-1].replace(quote * 2, quote)


class _IssueSearchUtils(SearchUtils):
    VALID_SEARCH_ATTRIBUTE_KEYS = {"status", "severity", "category", "categories", "title"}

    @classmethod
    def _process_statement(cls, statement):
        tokens = [t for t in _join_in_comparison_tokens(statement.tokens) if not t.is_whitespace]
        if (
            not tokens
            or len(tokens) % 2 != 1
            or any(
                not isinstance(token, Comparison)
                if index % 2 == 0
                else not token.match(TokenType.Keyword, "AND")
                for index, token in enumerate(tokens)
            )
        ):
            raise MlflowException.invalid_parameter_value(
                "Issue filters must contain comparisons separated by AND"
            )
        return [cls._get_comparison(token) for token in tokens[::2]]

    @classmethod
    def _get_value(cls, identifier_type, key, token):
        if isinstance(token, Parenthesis):
            tokens = [t for t in token.flatten() if not t.is_whitespace][1:-1]
            if not tokens or len(tokens) % 2 != 1 or any(t.value != "," for t in tokens[1::2]):
                raise MlflowException.invalid_parameter_value(
                    "IN requires a nonempty parenthesized list of quoted strings"
                )
            return [_string_value(t.value) for t in tokens[::2]]
        return _string_value(token.value)


def _parse_issue_filter(filter_string: str | None) -> dict[str, Any]:
    """Translate MLflow filter syntax to the Issues REST API's structured filters.

    Supports AND, status/severity/category equality and IN, tag equality/IN/IS NOT NULL,
    and title ILIKE '%substring%'. Operators without an equivalent REST filter are rejected.
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
        operator = comparison["comparator"].upper()
        value = comparison["value"]
        if kind not in {SearchUtils._ATTRIBUTE_IDENTIFIER, SearchUtils._TAG_IDENTIFIER}:
            raise MlflowException.invalid_parameter_value(
                "Issue filters support status, severity, category, title, and tags.<key>"
            )
        if (kind, key) in seen:
            raise MlflowException.invalid_parameter_value(
                f"Only one filter clause is supported for '{key}'"
            )
        seen.add((kind, key))

        if kind == SearchUtils._TAG_IDENTIFIER and operator == "IS NOT NULL":
            result.setdefault("tag_filters", []).append({"key": key})
            continue
        if key == "title" and kind == SearchUtils._ATTRIBUTE_IDENTIFIER:
            if (
                operator != "ILIKE"
                or not isinstance(value, str)
                or not value.startswith("%")
                or not value.endswith("%")
                or len(value) < 3
                or any(character in value[1:-1] for character in "%_\\")
            ):
                raise MlflowException.invalid_parameter_value(
                    "Title filters require ILIKE '%substring%' without other wildcards"
                )
            result["query"] = value[1:-1]
            continue
        if operator not in {"=", "IN"} or isinstance(value, list) != (operator == "IN"):
            raise MlflowException.invalid_parameter_value(
                f"Issue filter '{key}' requires = with a string or IN with a list"
            )
        values = value if isinstance(value, list) else [value]
        if kind == SearchUtils._TAG_IDENTIFIER:
            result.setdefault("tag_filters", []).append({"key": key, "values": values})
        else:
            if key in {"status", "severity"}:
                values = [v.upper() for v in values]
                valid = (
                    {"PENDING", "RESOLVED", "REJECTED"}
                    if key == "status"
                    else {"LOW", "MEDIUM", "HIGH"}
                )
                if any(v not in valid for v in values):
                    raise MlflowException.invalid_parameter_value(
                        f"Invalid {key} filter. Expected values from {sorted(valid)}"
                    )
            result[
                {"status": "statuses", "severity": "severities", "category": "categories"}[key]
            ] = values
    return result
