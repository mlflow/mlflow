import ast
import base64
import json
import math
import operator
import re
import shlex
from dataclasses import asdict, dataclass
from typing import Any, Optional

import sqlparse
from packaging.version import Version
from sqlparse.sql import (
    Comparison,
    Identifier,
    Parenthesis,
    Statement,
    Token,
    TokenList,
)
from sqlparse.tokens import Token as TokenType

from mlflow.entities import LoggedModel, Metric, RunInfo
from mlflow.entities.model_registry.model_version_stages import STAGE_DELETED_INTERNAL
from mlflow.entities.model_registry.prompt_version import IS_PROMPT_TAG_KEY
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.db.db_types import MSSQL, MYSQL, POSTGRES, SQLITE
from mlflow.tracing.constant import TraceMetadataKey, TraceTagKey
from mlflow.utils.mlflow_tags import (
    MLFLOW_DATASET_CONTEXT,
)


def _convert_like_pattern_to_regex(pattern, flags=0):
    if not pattern.startswith("%"):
        pattern = "^" + pattern
    if not pattern.endswith("%"):
        pattern = pattern + "$"
    return re.compile(pattern.replace("_", ".").replace("%", ".*"), flags)


def _like(string, pattern):
    return _convert_like_pattern_to_regex(pattern).match(string) is not None


def _ilike(string, pattern):
    return _convert_like_pattern_to_regex(pattern, flags=re.IGNORECASE).match(string) is not None


def _join_in_comparison_tokens(tokens, search_traces=False):
    """
    Find a sequence of tokens that matches the pattern of an IN comparison or a NOT IN comparison,
    join the tokens into a single Comparison token. Otherwise, return the original list of tokens.
    """
    if Version(sqlparse.__version__) < Version("0.4.4"):
        # In sqlparse < 0.4.4, IN is treated as a comparison, we don't need to join tokens
        return tokens

    non_whitespace_tokens = [t for t in tokens if not t.is_whitespace]
    joined_tokens = []
    num_tokens = len(non_whitespace_tokens)
    iterator = enumerate(non_whitespace_tokens)
    while elem := next(iterator, None):
        index, first = elem
        # We need at least 3 tokens to form an IN comparison or a NOT IN comparison
        if num_tokens - index < 3:
            joined_tokens.extend(non_whitespace_tokens[index:])
            break

        if search_traces:
            # timestamp
            if first.match(ttype=TokenType.Name.Builtin, values=["timestamp", "timestamp_ms"]):
                (_, second) = next(iterator, (None, None))
                (_, third) = next(iterator, (None, None))
                if any(x is None for x in [second, third]):
                    raise MlflowException(
                        f"Invalid comparison clause with token `{first}, {second}, {third}`, "
                        "expected 3 tokens",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                if (
                    second.match(
                        ttype=TokenType.Operator.Comparison,
                        values=SearchTraceUtils.VALID_NUMERIC_ATTRIBUTE_COMPARATORS,
                    )
                    and third.ttype == TokenType.Literal.Number.Integer
                ):
                    joined_tokens.append(Comparison(TokenList([first, second, third])))
                    continue
                else:
                    joined_tokens.extend([first, second, third])

        # Wait until we encounter an identifier token
        if not isinstance(first, Identifier):
            joined_tokens.append(first)
            continue

        (_, second) = next(iterator)
        (_, third) = next(iterator)

        # IN
        if (
            isinstance(first, Identifier)
            and second.match(ttype=TokenType.Keyword, values=["IN"])
            and isinstance(third, Parenthesis)
        ):
            joined_tokens.append(Comparison(TokenList([first, second, third])))
            continue

        (_, fourth) = next(iterator, (None, None))
        if fourth is None:
            joined_tokens.extend([first, second, third])
            break

        # NOT IN
        if (
            isinstance(first, Identifier)
            and second.match(ttype=TokenType.Keyword, values=["NOT"])
            and third.match(ttype=TokenType.Keyword, values=["IN"])
            and isinstance(fourth, Parenthesis)
        ):
            joined_tokens.append(
                Comparison(TokenList([first, Token(TokenType.Keyword, "NOT IN"), fourth]))
            )
            continue

        joined_tokens.extend([first, second, third, fourth])

    return joined_tokens


class SearchUtils:
    LIKE_OPERATOR = "LIKE"
    ILIKE_OPERATOR = "ILIKE"
    ASC_OPERATOR = "asc"
    DESC_OPERATOR = "desc"
    VALID_ORDER_BY_TAGS = [ASC_OPERATOR, DESC_OPERATOR]
    VALID_METRIC_COMPARATORS = {">", ">=", "!=", "=", "<", "<="}
    VALID_PARAM_COMPARATORS = {"!=", "=", LIKE_OPERATOR, ILIKE_OPERATOR}
    VALID_TAG_COMPARATORS = {"!=", "=", LIKE_OPERATOR, ILIKE_OPERATOR}
    VALID_STRING_ATTRIBUTE_COMPARATORS = {"!=", "=", LIKE_OPERATOR, ILIKE_OPERATOR, "IN", "NOT IN"}
    VALID_NUMERIC_ATTRIBUTE_COMPARATORS = VALID_METRIC_COMPARATORS
    VALID_DATASET_COMPARATORS = {"!=", "=", LIKE_OPERATOR, ILIKE_OPERATOR, "IN", "NOT IN"}
    _BUILTIN_NUMERIC_ATTRIBUTES = {"start_time", "end_time"}
    _ALTERNATE_NUMERIC_ATTRIBUTES = {"created", "Created"}
    _ALTERNATE_STRING_ATTRIBUTES = {"run name", "Run name", "Run Name"}
    NUMERIC_ATTRIBUTES = set(
        list(_BUILTIN_NUMERIC_ATTRIBUTES) + list(_ALTERNATE_NUMERIC_ATTRIBUTES)
    )
    DATASET_ATTRIBUTES = {"name", "digest", "context"}
    VALID_SEARCH_ATTRIBUTE_KEYS = set(
        RunInfo.get_searchable_attributes()
        + list(_ALTERNATE_NUMERIC_ATTRIBUTES)
        + list(_ALTERNATE_STRING_ATTRIBUTES)
    )
    VALID_ORDER_BY_ATTRIBUTE_KEYS = set(
        RunInfo.get_orderable_attributes() + list(_ALTERNATE_NUMERIC_ATTRIBUTES)
    )
    _METRIC_IDENTIFIER = "metric"
    _ALTERNATE_METRIC_IDENTIFIERS = {"metrics"}
    _PARAM_IDENTIFIER = "parameter"
    _ALTERNATE_PARAM_IDENTIFIERS = {"parameters", "param", "params"}
    _TAG_IDENTIFIER = "tag"
    _ALTERNATE_TAG_IDENTIFIERS = {"tags"}
    _ATTRIBUTE_IDENTIFIER = "attribute"
    _ALTERNATE_ATTRIBUTE_IDENTIFIERS = {"attr", "attributes", "run"}
    _DATASET_IDENTIFIER = "dataset"
    _ALTERNATE_DATASET_IDENTIFIERS = {"datasets"}
    _IDENTIFIERS = [
        _METRIC_IDENTIFIER,
        _PARAM_IDENTIFIER,
        _TAG_IDENTIFIER,
        _ATTRIBUTE_IDENTIFIER,
        _DATASET_IDENTIFIER,
    ]
    _VALID_IDENTIFIERS = set(
        _IDENTIFIERS
        + list(_ALTERNATE_METRIC_IDENTIFIERS)
        + list(_ALTERNATE_PARAM_IDENTIFIERS)
        + list(_ALTERNATE_TAG_IDENTIFIERS)
        + list(_ALTERNATE_ATTRIBUTE_IDENTIFIERS)
        + list(_ALTERNATE_DATASET_IDENTIFIERS)
    )
    STRING_VALUE_TYPES = {TokenType.Literal.String.Single}
    DELIMITER_VALUE_TYPES = {TokenType.Punctuation}
    WHITESPACE_VALUE_TYPE = TokenType.Text.Whitespace
    NUMERIC_VALUE_TYPES = {TokenType.Literal.Number.Integer, TokenType.Literal.Number.Float}
    # Registered Models Constants
    ORDER_BY_KEY_TIMESTAMP = "timestamp"
    ORDER_BY_KEY_LAST_UPDATED_TIMESTAMP = "last_updated_timestamp"
    ORDER_BY_KEY_MODEL_NAME = "name"
    VALID_ORDER_BY_KEYS_REGISTERED_MODELS = {
        ORDER_BY_KEY_TIMESTAMP,
        ORDER_BY_KEY_LAST_UPDATED_TIMESTAMP,
        ORDER_BY_KEY_MODEL_NAME,
    }
    VALID_TIMESTAMP_ORDER_BY_KEYS = {ORDER_BY_KEY_TIMESTAMP, ORDER_BY_KEY_LAST_UPDATED_TIMESTAMP}
    # We encourage users to use timestamp for order-by
    RECOMMENDED_ORDER_BY_KEYS_REGISTERED_MODELS = {ORDER_BY_KEY_MODEL_NAME, ORDER_BY_KEY_TIMESTAMP}

    @staticmethod
    def get_comparison_func(comparator):
        return {
            ">": operator.gt,
            ">=": operator.ge,
            "=": operator.eq,
            "!=": operator.ne,
            "<=": operator.le,
            "<": operator.lt,
            "LIKE": _like,
            "ILIKE": _ilike,
            "IN": lambda x, y: x in y,
            "NOT IN": lambda x, y: x not in y,
        }[comparator]

    @staticmethod
    def get_sql_comparison_func(comparator, dialect):
        import sqlalchemy as sa

        def comparison_func(column, value):
            if comparator == "LIKE":
                return column.like(value)
            elif comparator == "ILIKE":
                return column.ilike(value)
            elif comparator == "IN":
                return column.in_(value)
            elif comparator == "NOT IN":
                return ~column.in_(value)
            return SearchUtils.get_comparison_func(comparator)(column, value)

        def mssql_comparison_func(column, value):
            if not isinstance(column.type, sa.types.String):
                return comparison_func(column, value)

            collated = column.collate("Japanese_Bushu_Kakusu_100_CS_AS_KS_WS")
            return comparison_func(collated, value)

        def mysql_comparison_func(column, value):
            if not isinstance(column.type, sa.types.String):
                return comparison_func(column, value)

            # MySQL is case insensitive by default, so we need to use the binary operator to
            # perform case sensitive comparisons.
            templates = {
                # Use non-binary ahead of binary comparison for runtime performance
                "=": "({column} = :value AND BINARY {column} = :value)",
                "!=": "({column} != :value OR BINARY {column} != :value)",
                "LIKE": "({column} LIKE :value AND BINARY {column} LIKE :value)",
            }
            if comparator in templates:
                column = f"{column.class_.__tablename__}.{column.key}"
                return sa.text(templates[comparator].format(column=column)).bindparams(
                    sa.bindparam("value", value=value, unique=True)
                )

            return comparison_func(column, value)

        return {
            POSTGRES: comparison_func,
            SQLITE: comparison_func,
            MSSQL: mssql_comparison_func,
            MYSQL: mysql_comparison_func,
        }[dialect]

    @staticmethod
    def translate_key_alias(key):
        if key in ["created", "Created"]:
            return "start_time"
        if key in ["run name", "Run name", "Run Name"]:
            return "run_name"
        return key

    @classmethod
    def _trim_ends(cls, string_value):
        return string_value[1:-1]

    @classmethod
    def _is_quoted(cls, value, pattern):
        return len(value) >= 2 and value.startswith(pattern) and value.endswith(pattern)

    @classmethod
    def _trim_backticks(cls, entity_type):
        """Remove backticks from identifier like `param`, if they exist."""
        if cls._is_quoted(entity_type, "`"):
            return cls._trim_ends(entity_type)
        return entity_type

    @classmethod
    def _strip_quotes(cls, value, expect_quoted_value=False):
        """
        Remove quotes for input string.
        Values of type strings are expected to have quotes.
        Keys containing special characters are also expected to be enclose in quotes.
        """
        if cls._is_quoted(value, "'") or cls._is_quoted(value, '"'):
            return cls._trim_ends(value)
        elif expect_quoted_value:
            raise MlflowException(
                "Parameter value is either not quoted or unidentified quote "
                f"types used for string value {value}. Use either single or double "
                "quotes.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        else:
            return value

    @classmethod
    def _valid_entity_type(cls, entity_type):
        entity_type = cls._trim_backticks(entity_type)
        if entity_type not in cls._VALID_IDENTIFIERS:
            raise MlflowException(
                f"Invalid entity type '{entity_type}'. Valid values are {cls._IDENTIFIERS}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if entity_type in cls._ALTERNATE_PARAM_IDENTIFIERS:
            return cls._PARAM_IDENTIFIER
        elif entity_type in cls._ALTERNATE_METRIC_IDENTIFIERS:
            return cls._METRIC_IDENTIFIER
        elif entity_type in cls._ALTERNATE_TAG_IDENTIFIERS:
            return cls._TAG_IDENTIFIER
        elif entity_type in cls._ALTERNATE_ATTRIBUTE_IDENTIFIERS:
            return cls._ATTRIBUTE_IDENTIFIER
        elif entity_type in cls._ALTERNATE_DATASET_IDENTIFIERS:
            return cls._DATASET_IDENTIFIER
        else:
            # one of ("metric", "parameter", "tag", or "attribute") since it a valid type
            return entity_type

    @classmethod
    def _get_identifier(cls, identifier, valid_attributes):
        try:
            tokens = identifier.split(".", 1)
            if len(tokens) == 1:
                key = tokens[0]
                entity_type = cls._ATTRIBUTE_IDENTIFIER
            else:
                entity_type, key = tokens
        except ValueError:
            raise MlflowException(
                f"Invalid identifier {identifier!r}. Columns should be specified as "
                "'attribute.<key>', 'metric.<key>', 'tag.<key>', 'dataset.<key>', or "
                "'param.'.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        identifier = cls._valid_entity_type(entity_type)
        key = cls._trim_backticks(cls._strip_quotes(key))
        if identifier == cls._ATTRIBUTE_IDENTIFIER and key not in valid_attributes:
            raise MlflowException.invalid_parameter_value(
                f"Invalid attribute key '{key}' specified. Valid keys are '{valid_attributes}'"
            )
        elif identifier == cls._DATASET_IDENTIFIER and key not in cls.DATASET_ATTRIBUTES:
            raise MlflowException.invalid_parameter_value(
                f"Invalid dataset key '{key}' specified. Valid keys are '{cls.DATASET_ATTRIBUTES}'"
            )
        return {"type": identifier, "key": key}

    @classmethod
    def validate_list_supported(cls, key: str) -> None:
        if key != "run_id":
            raise MlflowException(
                "Only the 'run_id' attribute supports comparison with a list of quoted "
                "string values.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    @classmethod
    def _get_value(cls, identifier_type, key, token):
        if identifier_type == cls._METRIC_IDENTIFIER:
            if token.ttype not in cls.NUMERIC_VALUE_TYPES:
                raise MlflowException(
                    f"Expected numeric value type for metric. Found {token.value}",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            return token.value
        elif identifier_type == cls._PARAM_IDENTIFIER or identifier_type == cls._TAG_IDENTIFIER:
            if token.ttype in cls.STRING_VALUE_TYPES or isinstance(token, Identifier):
                return cls._strip_quotes(token.value, expect_quoted_value=True)
            raise MlflowException(
                "Expected a quoted string value for "
                f"{identifier_type} (e.g. 'my-value'). Got value "
                f"{token.value}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        elif identifier_type == cls._ATTRIBUTE_IDENTIFIER:
            if key in cls.NUMERIC_ATTRIBUTES:
                if token.ttype not in cls.NUMERIC_VALUE_TYPES:
                    raise MlflowException(
                        f"Expected numeric value type for numeric attribute: {key}. "
                        f"Found {token.value}",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                return token.value
            elif token.ttype in cls.STRING_VALUE_TYPES or isinstance(token, Identifier):
                return cls._strip_quotes(token.value, expect_quoted_value=True)
            elif isinstance(token, Parenthesis):
                cls.validate_list_supported(key)
                return cls._parse_run_ids(token)
            else:
                raise MlflowException(
                    f"Expected a quoted string value for attributes. Got value {token.value}",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        elif identifier_type == cls._DATASET_IDENTIFIER:
            if key in cls.DATASET_ATTRIBUTES and (
                token.ttype in cls.STRING_VALUE_TYPES or isinstance(token, Identifier)
            ):
                return cls._strip_quotes(token.value, expect_quoted_value=True)
            elif isinstance(token, Parenthesis):
                if key not in ("name", "digest", "context"):
                    raise MlflowException(
                        "Only the dataset 'name' and 'digest' supports comparison with a list of "
                        "quoted string values.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                return cls._parse_run_ids(token)
            else:
                raise MlflowException(
                    "Expected a quoted string value for dataset attributes. "
                    f"Got value {token.value}",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        else:
            # Expected to be either "param" or "metric".
            raise MlflowException(
                "Invalid identifier type. Expected one of "
                f"{[cls._METRIC_IDENTIFIER, cls._PARAM_IDENTIFIER]}."
            )

    @classmethod
    def _validate_comparison(cls, tokens, search_traces=False):
        base_error_string = "Invalid comparison clause"
        if len(tokens) != 3:
            raise MlflowException(
                f"{base_error_string}. Expected 3 tokens found {len(tokens)}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if not isinstance(tokens[0], Identifier):
            if not search_traces:
                raise MlflowException(
                    f"{base_error_string}. Expected 'Identifier' found '{tokens[0]}'",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            if search_traces and not tokens[0].match(
                ttype=TokenType.Name.Builtin, values=["timestamp", "timestamp_ms"]
            ):
                raise MlflowException(
                    f"{base_error_string}. Expected 'TokenType.Name.Builtin' found '{tokens[0]}'",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        if not isinstance(tokens[1], Token) and tokens[1].ttype != TokenType.Operator.Comparison:
            raise MlflowException(
                f"{base_error_string}. Expected comparison found '{tokens[1]}'",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if not isinstance(tokens[2], Token) and (
            tokens[2].ttype not in cls.STRING_VALUE_TYPES.union(cls.NUMERIC_VALUE_TYPES)
            or isinstance(tokens[2], Identifier)
        ):
            raise MlflowException(
                f"{base_error_string}. Expected value token found '{tokens[2]}'",
                error_code=INVALID_PARAMETER_VALUE,
            )

    @classmethod
    def _get_comparison(cls, comparison):
        stripped_comparison = [token for token in comparison.tokens if not token.is_whitespace]
        cls._validate_comparison(stripped_comparison)
        comp = cls._get_identifier(stripped_comparison[0].value, cls.VALID_SEARCH_ATTRIBUTE_KEYS)
        comp["comparator"] = stripped_comparison[1].value
        comp["value"] = cls._get_value(comp.get("type"), comp.get("key"), stripped_comparison[2])
        return comp

    @classmethod
    def _invalid_statement_token_search_runs(cls, token):
        if (
            isinstance(token, Comparison)
            or token.is_whitespace
            or token.match(ttype=TokenType.Keyword, values=["AND"])
        ):
            return False
        return True

    @classmethod
    def _process_statement(cls, statement):
        # check validity
        tokens = _join_in_comparison_tokens(statement.tokens)
        invalids = list(filter(cls._invalid_statement_token_search_runs, tokens))
        if len(invalids) > 0:
            invalid_clauses = ", ".join(f"'{token}'" for token in invalids)
            raise MlflowException(
                f"Invalid clause(s) in filter string: {invalid_clauses}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return [cls._get_comparison(si) for si in tokens if isinstance(si, Comparison)]

    @classmethod
    def parse_search_filter(cls, filter_string):
        if not filter_string:
            return []
        try:
            parsed = sqlparse.parse(filter_string)
        except Exception:
            raise MlflowException(
                f"Error on parsing filter '{filter_string}'", error_code=INVALID_PARAMETER_VALUE
            )
        if len(parsed) == 0 or not isinstance(parsed[0], Statement):
            raise MlflowException(
                f"Invalid filter '{filter_string}'. Could not be parsed.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        elif len(parsed) > 1:
            raise MlflowException(
                f"Search filter contained multiple expression {filter_string!r}. "
                "Provide AND-ed expression list.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return cls._process_statement(parsed[0])

    @classmethod
    def is_metric(cls, key_type, comparator):
        if key_type == cls._METRIC_IDENTIFIER:
            if comparator not in cls.VALID_METRIC_COMPARATORS:
                raise MlflowException(
                    f"Invalid comparator '{comparator}' not one of '{cls.VALID_METRIC_COMPARATORS}",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            return True
        return False

    @classmethod
    def is_param(cls, key_type, comparator):
        if key_type == cls._PARAM_IDENTIFIER:
            if comparator not in cls.VALID_PARAM_COMPARATORS:
                raise MlflowException(
                    f"Invalid comparator '{comparator}' not one of '{cls.VALID_PARAM_COMPARATORS}'",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            return True
        return False

    @classmethod
    def is_tag(cls, key_type, comparator):
        if key_type == cls._TAG_IDENTIFIER:
            if comparator not in cls.VALID_TAG_COMPARATORS:
                raise MlflowException(
                    f"Invalid comparator '{comparator}' not one of '{cls.VALID_TAG_COMPARATORS}",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            return True
        return False

    @classmethod
    def is_attribute(cls, key_type, key_name, comparator):
        return cls.is_string_attribute(key_type, key_name, comparator) or cls.is_numeric_attribute(
            key_type, key_name, comparator
        )

    @classmethod
    def is_string_attribute(cls, key_type, key_name, comparator):
        if key_type == cls._ATTRIBUTE_IDENTIFIER and key_name not in cls.NUMERIC_ATTRIBUTES:
            if comparator not in cls.VALID_STRING_ATTRIBUTE_COMPARATORS:
                raise MlflowException(
                    f"Invalid comparator '{comparator}' not one of "
                    f"'{cls.VALID_STRING_ATTRIBUTE_COMPARATORS}'",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            return True
        return False

    @classmethod
    def is_numeric_attribute(cls, key_type, key_name, comparator):
        if key_type == cls._ATTRIBUTE_IDENTIFIER and key_name in cls.NUMERIC_ATTRIBUTES:
            if comparator not in cls.VALID_NUMERIC_ATTRIBUTE_COMPARATORS:
                raise MlflowException(
                    f"Invalid comparator '{comparator}' not one of "
                    f"'{cls.VALID_STRING_ATTRIBUTE_COMPARATORS}",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            return True
        return False

    @classmethod
    def is_dataset(cls, key_type, comparator):
        if key_type == cls._DATASET_IDENTIFIER:
            if comparator not in cls.VALID_DATASET_COMPARATORS:
                raise MlflowException(
                    f"Invalid comparator '{comparator}' "
                    f"not one of '{cls.VALID_DATASET_COMPARATORS}",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            return True
        return False

    @classmethod
    def _is_metric_on_dataset(cls, metric: Metric, dataset: dict[str, Any]) -> bool:
        return metric.dataset_name == dataset.get("dataset_name") and (
            dataset.get("dataset_digest") is None
            or dataset.get("dataset_digest") == metric.dataset_digest
        )

    @classmethod
    def _does_run_match_clause(cls, run, sed):
        key_type = sed.get("type")
        key = sed.get("key")
        value = sed.get("value")
        comparator = sed.get("comparator").upper()

        key = SearchUtils.translate_key_alias(key)

        if cls.is_metric(key_type, comparator):
            lhs = run.data.metrics.get(key, None)
            value = float(value)
        elif cls.is_param(key_type, comparator):
            lhs = run.data.params.get(key, None)
        elif cls.is_tag(key_type, comparator):
            lhs = run.data.tags.get(key, None)
        elif cls.is_string_attribute(key_type, key, comparator):
            lhs = getattr(run.info, key)
        elif cls.is_numeric_attribute(key_type, key, comparator):
            lhs = getattr(run.info, key)
            value = int(value)
        elif cls.is_dataset(key_type, comparator):
            if key == "context":
                return any(
                    SearchUtils.get_comparison_func(comparator)(tag.value if tag else None, value)
                    for dataset_input in run.inputs.dataset_inputs
                    for tag in dataset_input.tags
                    if tag.key == MLFLOW_DATASET_CONTEXT
                )
            else:
                return any(
                    SearchUtils.get_comparison_func(comparator)(
                        getattr(dataset_input.dataset, key), value
                    )
                    for dataset_input in run.inputs.dataset_inputs
                )
        else:
            raise MlflowException(
                f"Invalid search expression type '{key_type}'", error_code=INVALID_PARAMETER_VALUE
            )
        if lhs is None:
            return False

        return SearchUtils.get_comparison_func(comparator)(lhs, value)

    @classmethod
    def _does_model_match_clause(cls, model, sed):
        key_type = sed.get("type")
        key = sed.get("key")
        value = sed.get("value")
        comparator = sed.get("comparator").upper()

        key = SearchUtils.translate_key_alias(key)

        if cls.is_metric(key_type, comparator):
            matching_metrics = [metric for metric in model.metrics if metric.key == key]
            lhs = matching_metrics[0].value if matching_metrics else None
            value = float(value)
        elif cls.is_param(key_type, comparator):
            lhs = model.params.get(key, None)
        elif cls.is_tag(key_type, comparator):
            lhs = model.tags.get(key, None)
        elif cls.is_string_attribute(key_type, key, comparator):
            lhs = getattr(model.info, key)
        elif cls.is_numeric_attribute(key_type, key, comparator):
            lhs = getattr(model.info, key)
            value = int(value)
        else:
            raise MlflowException(
                f"Invalid model search expression type '{key_type}'",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if lhs is None:
            return False

        return SearchUtils.get_comparison_func(comparator)(lhs, value)

    @classmethod
    def filter(cls, runs, filter_string):
        """Filters a set of runs based on a search filter string."""
        if not filter_string:
            return runs
        parsed = cls.parse_search_filter(filter_string)

        def run_matches(run):
            return all(cls._does_run_match_clause(run, s) for s in parsed)

        return [run for run in runs if run_matches(run)]

    @classmethod
    def _validate_order_by_and_generate_token(cls, order_by):
        try:
            parsed = sqlparse.parse(order_by)
        except Exception:
            raise MlflowException(
                f"Error on parsing order_by clause '{order_by}'",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if len(parsed) != 1 or not isinstance(parsed[0], Statement):
            raise MlflowException(
                f"Invalid order_by clause '{order_by}'. Could not be parsed.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        statement = parsed[0]
        ttype_for_timestamp = (
            TokenType.Name.Builtin
            if Version(sqlparse.__version__) >= Version("0.4.3")
            else TokenType.Keyword
        )

        if len(statement.tokens) == 1 and isinstance(statement[0], Identifier):
            token_value = statement.tokens[0].value
        elif len(statement.tokens) == 1 and statement.tokens[0].match(
            ttype=ttype_for_timestamp, values=[cls.ORDER_BY_KEY_TIMESTAMP]
        ):
            token_value = cls.ORDER_BY_KEY_TIMESTAMP
        elif (
            statement.tokens[0].match(
                ttype=ttype_for_timestamp, values=[cls.ORDER_BY_KEY_TIMESTAMP]
            )
            and all(token.is_whitespace for token in statement.tokens[1:-1])
            and statement.tokens[-1].ttype == TokenType.Keyword.Order
        ):
            token_value = cls.ORDER_BY_KEY_TIMESTAMP + " " + statement.tokens[-1].value
        else:
            raise MlflowException(
                f"Invalid order_by clause '{order_by}'. Could not be parsed.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return token_value

    @classmethod
    def _parse_order_by_string(cls, order_by):
        token_value = cls._validate_order_by_and_generate_token(order_by)
        is_ascending = True
        tokens = shlex.split(token_value.replace("`", '"'))
        if len(tokens) > 2:
            raise MlflowException(
                f"Invalid order_by clause '{order_by}'. Could not be parsed.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        elif len(tokens) == 2:
            order_token = tokens[1].lower()
            if order_token not in cls.VALID_ORDER_BY_TAGS:
                raise MlflowException(
                    f"Invalid ordering key in order_by clause '{order_by}'.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            is_ascending = order_token == cls.ASC_OPERATOR
            token_value = tokens[0]
        return token_value, is_ascending

    @classmethod
    def parse_order_by_for_search_runs(cls, order_by):
        token_value, is_ascending = cls._parse_order_by_string(order_by)
        identifier = cls._get_identifier(token_value.strip(), cls.VALID_ORDER_BY_ATTRIBUTE_KEYS)
        return identifier["type"], identifier["key"], is_ascending

    @classmethod
    def parse_order_by_for_search_registered_models(cls, order_by):
        token_value, is_ascending = cls._parse_order_by_string(order_by)
        token_value = token_value.strip()
        if token_value not in cls.VALID_ORDER_BY_KEYS_REGISTERED_MODELS:
            raise MlflowException(
                f"Invalid order by key '{token_value}' specified. Valid keys "
                f"are '{cls.RECOMMENDED_ORDER_BY_KEYS_REGISTERED_MODELS}'",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return token_value, is_ascending

    @classmethod
    def _get_value_for_sort(cls, run, key_type, key, ascending):
        """Returns a tuple suitable to be used as a sort key for runs."""
        sort_value = None
        key = SearchUtils.translate_key_alias(key)
        if key_type == cls._METRIC_IDENTIFIER:
            sort_value = run.data.metrics.get(key)
        elif key_type == cls._PARAM_IDENTIFIER:
            sort_value = run.data.params.get(key)
        elif key_type == cls._TAG_IDENTIFIER:
            sort_value = run.data.tags.get(key)
        elif key_type == cls._ATTRIBUTE_IDENTIFIER:
            sort_value = getattr(run.info, key)
        else:
            raise MlflowException(
                f"Invalid order_by entity type '{key_type}'", error_code=INVALID_PARAMETER_VALUE
            )

        # Return a key such that None values are always at the end.
        is_none = sort_value is None
        is_nan = isinstance(sort_value, float) and math.isnan(sort_value)
        fill_value = (1 if ascending else -1) * math.inf

        if is_none:
            sort_value = fill_value
        elif is_nan:
            sort_value = -fill_value

        is_none_or_nan = is_none or is_nan

        return (is_none_or_nan, sort_value) if ascending else (not is_none_or_nan, sort_value)

    @classmethod
    def _get_model_value_for_sort(cls, model, key_type, key, ascending):
        """Returns a tuple suitable to be used as a sort key for models."""
        sort_value = None
        key = SearchUtils.translate_key_alias(key)
        if key_type == cls._METRIC_IDENTIFIER:
            matching_metrics = [metric for metric in model.metrics if metric.key == key]
            sort_value = float(matching_metrics[0].value) if matching_metrics else None
        elif key_type == cls._PARAM_IDENTIFIER:
            sort_value = model.params.get(key)
        elif key_type == cls._TAG_IDENTIFIER:
            sort_value = model.tags.get(key)
        elif key_type == cls._ATTRIBUTE_IDENTIFIER:
            sort_value = getattr(model, key)
        else:
            raise MlflowException(
                f"Invalid models order_by entity type '{key_type}'",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # Return a key such that None values are always at the end.
        is_none = sort_value is None
        is_nan = isinstance(sort_value, float) and math.isnan(sort_value)
        fill_value = (1 if ascending else -1) * math.inf

        if is_none:
            sort_value = fill_value
        elif is_nan:
            sort_value = -fill_value

        is_none_or_nan = is_none or is_nan

        return (is_none_or_nan, sort_value) if ascending else (not is_none_or_nan, sort_value)

    @classmethod
    def sort(cls, runs, order_by_list):
        """Sorts a set of runs based on their natural ordering and an overriding set of order_bys.
        Runs are naturally ordered first by start time descending, then by run id for tie-breaking.
        """
        runs = sorted(runs, key=lambda run: (-run.info.start_time, run.info.run_id))
        if not order_by_list:
            return runs
        # NB: We rely on the stability of Python's sort function, so that we can apply
        # the ordering conditions in reverse order.
        for order_by_clause in reversed(order_by_list):
            (key_type, key, ascending) = cls.parse_order_by_for_search_runs(order_by_clause)

            runs = sorted(
                runs,
                key=lambda run: cls._get_value_for_sort(run, key_type, key, ascending),
                reverse=not ascending,
            )
        return runs

    @classmethod
    def parse_start_offset_from_page_token(cls, page_token):
        # Note: the page_token is expected to be a base64-encoded JSON that looks like
        # { "offset": xxx }. However, this format is not stable, so it should not be
        # relied upon outside of this method.
        if not page_token:
            return 0

        try:
            decoded_token = base64.b64decode(page_token)
        except TypeError:
            raise MlflowException(
                "Invalid page token, could not base64-decode", error_code=INVALID_PARAMETER_VALUE
            )
        except base64.binascii.Error:
            raise MlflowException(
                "Invalid page token, could not base64-decode", error_code=INVALID_PARAMETER_VALUE
            )

        try:
            parsed_token = json.loads(decoded_token)
        except ValueError:
            raise MlflowException(
                f"Invalid page token, decoded value={decoded_token}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        offset_str = parsed_token.get("offset")
        if not offset_str:
            raise MlflowException(
                f"Invalid page token, parsed value={parsed_token}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        try:
            offset = int(offset_str)
        except ValueError:
            raise MlflowException(
                f"Invalid page token, not stringable {offset_str}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        return offset

    @classmethod
    def create_page_token(cls, offset):
        return base64.b64encode(json.dumps({"offset": offset}).encode("utf-8"))

    @classmethod
    def paginate(cls, runs, page_token, max_results):
        """Paginates a set of runs based on an offset encoded into the page_token and a max
        results limit. Returns a pair containing the set of paginated runs, followed by
        an optional next_page_token if there are further results that need to be returned.
        """
        start_offset = cls.parse_start_offset_from_page_token(page_token)
        final_offset = start_offset + max_results

        paginated_runs = runs[start_offset:final_offset]
        next_page_token = None
        if final_offset < len(runs):
            next_page_token = cls.create_page_token(final_offset)
        return (paginated_runs, next_page_token)

    # Model Registry specific parser
    # TODO: Tech debt. Refactor search code into common utils, tracking server, and model
    #       registry specific code.

    VALID_SEARCH_KEYS_FOR_MODEL_VERSIONS = {"name", "run_id", "source_path"}
    VALID_SEARCH_KEYS_FOR_REGISTERED_MODELS = {"name"}

    @classmethod
    def _check_valid_identifier_list(cls, tup: tuple[Any, ...]) -> None:
        """
        Validate that `tup` is a non-empty tuple of strings.
        """
        if len(tup) == 0:
            raise MlflowException(
                "While parsing a list in the query,"
                " expected a non-empty list of string values, but got empty list",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if not all(isinstance(x, str) for x in tup):
            raise MlflowException(
                "While parsing a list in the query, expected string value, punctuation, "
                f"or whitespace, but got different type in list: {tup}",
                error_code=INVALID_PARAMETER_VALUE,
            )

    @classmethod
    def _parse_list_from_sql_token(cls, token):
        try:
            parsed = ast.literal_eval(token.value)
        except SyntaxError as e:
            raise MlflowException(
                "While parsing a list in the query,"
                " expected a non-empty list of string values, but got ill-formed list.",
                error_code=INVALID_PARAMETER_VALUE,
            ) from e

        parsed = parsed if isinstance(parsed, tuple) else (parsed,)
        cls._check_valid_identifier_list(parsed)
        return parsed

    @classmethod
    def _parse_run_ids(cls, token):
        run_id_list = cls._parse_list_from_sql_token(token)
        # Because MySQL IN clause is case-insensitive, but all run_ids only contain lower
        # case letters, so that we filter out run_ids containing upper case letters here.
        return [run_id for run_id in run_id_list if run_id.islower()]


class SearchExperimentsUtils(SearchUtils):
    VALID_SEARCH_ATTRIBUTE_KEYS = {"name", "creation_time", "last_update_time"}
    VALID_ORDER_BY_ATTRIBUTE_KEYS = {"name", "experiment_id", "creation_time", "last_update_time"}
    NUMERIC_ATTRIBUTES = {"creation_time", "last_update_time"}

    @classmethod
    def _invalid_statement_token_search_experiments(cls, token):
        if (
            isinstance(token, Comparison)
            or token.is_whitespace
            or token.match(ttype=TokenType.Keyword, values=["AND"])
        ):
            return False
        return True

    @classmethod
    def _process_statement(cls, statement):
        tokens = _join_in_comparison_tokens(statement.tokens)
        invalids = list(filter(cls._invalid_statement_token_search_experiments, tokens))
        if len(invalids) > 0:
            invalid_clauses = ", ".join(map(str, invalids))
            raise MlflowException.invalid_parameter_value(
                f"Invalid clause(s) in filter string: {invalid_clauses}"
            )
        return [cls._get_comparison(t) for t in tokens if isinstance(t, Comparison)]

    @classmethod
    def _get_identifier(cls, identifier, valid_attributes):
        tokens = identifier.split(".", maxsplit=1)
        if len(tokens) == 1:
            key = tokens[0]
            identifier = cls._ATTRIBUTE_IDENTIFIER
        else:
            entity_type, key = tokens
            valid_entity_types = ("attribute", "tag", "tags")
            if entity_type not in valid_entity_types:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid entity type '{entity_type}'. "
                    f"Valid entity types are {valid_entity_types}"
                )
            identifier = cls._valid_entity_type(entity_type)

        key = cls._trim_backticks(cls._strip_quotes(key))
        if identifier == cls._ATTRIBUTE_IDENTIFIER and key not in valid_attributes:
            raise MlflowException.invalid_parameter_value(
                f"Invalid attribute key '{key}' specified. Valid keys are '{valid_attributes}'"
            )
        return {"type": identifier, "key": key}

    @classmethod
    def _get_comparison(cls, comparison):
        stripped_comparison = [token for token in comparison.tokens if not token.is_whitespace]
        cls._validate_comparison(stripped_comparison)
        left, comparator, right = stripped_comparison
        comp = cls._get_identifier(left.value, cls.VALID_SEARCH_ATTRIBUTE_KEYS)
        comp["comparator"] = comparator.value
        comp["value"] = cls._get_value(comp.get("type"), comp.get("key"), right)
        return comp

    @classmethod
    def parse_order_by_for_search_experiments(cls, order_by):
        token_value, is_ascending = cls._parse_order_by_string(order_by)
        identifier = cls._get_identifier(token_value.strip(), cls.VALID_ORDER_BY_ATTRIBUTE_KEYS)
        return identifier["type"], identifier["key"], is_ascending

    @classmethod
    def is_attribute(cls, key_type, comparator):
        if key_type == cls._ATTRIBUTE_IDENTIFIER:
            if comparator not in cls.VALID_STRING_ATTRIBUTE_COMPARATORS:
                raise MlflowException(
                    f"Invalid comparator '{comparator}' not one of "
                    f"'{cls.VALID_STRING_ATTRIBUTE_COMPARATORS}'"
                )
            return True
        return False

    @classmethod
    def _does_experiment_match_clause(cls, experiment, sed):
        key_type = sed.get("type")
        key = sed.get("key")
        value = sed.get("value")
        comparator = sed.get("comparator").upper()

        if cls.is_string_attribute(key_type, key, comparator):
            lhs = getattr(experiment, key)
        elif cls.is_numeric_attribute(key_type, key, comparator):
            lhs = getattr(experiment, key)
            value = float(value)
        elif cls.is_tag(key_type, comparator):
            if key not in experiment.tags:
                return False
            lhs = experiment.tags.get(key, None)
            if lhs is None:
                return experiment
        else:
            raise MlflowException(
                f"Invalid search expression type '{key_type}'", error_code=INVALID_PARAMETER_VALUE
            )

        return SearchUtils.get_comparison_func(comparator)(lhs, value)

    @classmethod
    def filter(cls, experiments, filter_string):
        if not filter_string:
            return experiments
        parsed = cls.parse_search_filter(filter_string)

        def experiment_matches(experiment):
            return all(cls._does_experiment_match_clause(experiment, s) for s in parsed)

        return list(filter(experiment_matches, experiments))

    @classmethod
    def _get_sort_key(cls, order_by_list):
        order_by = []
        parsed_order_by = map(cls.parse_order_by_for_search_experiments, order_by_list)
        for type_, key, ascending in parsed_order_by:
            if type_ == "attribute":
                order_by.append((key, ascending))
            else:
                raise MlflowException.invalid_parameter_value(f"Invalid order_by entity: {type_}")

        # Add a tie-breaker
        if not any(key == "experiment_id" for key, _ in order_by):
            order_by.append(("experiment_id", False))

        # https://stackoverflow.com/a/56842689
        class _Sorter:
            def __init__(self, obj, ascending):
                self.obj = obj
                self.ascending = ascending

            # Only need < and == are needed for use as a key parameter in the sorted function
            def __eq__(self, other):
                return other.obj == self.obj

            def __lt__(self, other):
                if self.obj is None:
                    return False
                elif other.obj is None:
                    return True
                elif self.ascending:
                    return self.obj < other.obj
                else:
                    return other.obj < self.obj

        def _apply_sorter(experiment, key, ascending):
            attr = getattr(experiment, key)
            return _Sorter(attr, ascending)

        return lambda experiment: tuple(_apply_sorter(experiment, k, asc) for (k, asc) in order_by)

    @classmethod
    def sort(cls, experiments, order_by_list):
        return sorted(experiments, key=cls._get_sort_key(order_by_list))


# https://stackoverflow.com/a/56842689
class _Reversor:
    def __init__(self, obj):
        self.obj = obj

    # Only need < and == are needed for use as a key parameter in the sorted function
    def __eq__(self, other):
        return other.obj == self.obj

    def __lt__(self, other):
        if self.obj is None:
            return False
        if other.obj is None:
            return True
        return other.obj < self.obj


def _apply_reversor(model, key, ascending):
    attr = getattr(model, key)
    return attr if ascending else _Reversor(attr)


class SearchModelUtils(SearchUtils):
    NUMERIC_ATTRIBUTES = {"creation_timestamp", "last_updated_timestamp"}
    VALID_SEARCH_ATTRIBUTE_KEYS = {"name"}
    VALID_ORDER_BY_KEYS_REGISTERED_MODELS = {"name", "creation_timestamp", "last_updated_timestamp"}

    @classmethod
    def _does_registered_model_match_clauses(cls, model, sed):
        key_type = sed.get("type")
        key = sed.get("key")
        value = sed.get("value")
        comparator = sed.get("comparator").upper()

        # what comparators do we support here?
        if cls.is_string_attribute(key_type, key, comparator):
            lhs = getattr(model, key)
        elif cls.is_numeric_attribute(key_type, key, comparator):
            lhs = getattr(model, key)
            value = int(value)
        elif cls.is_tag(key_type, comparator):
            # NB: We should use the private attribute `_tags` instead of the `tags` property
            # to consider all tags including reserved ones.
            lhs = model._tags.get(key, None)
        else:
            raise MlflowException(
                f"Invalid search expression type '{key_type}'", error_code=INVALID_PARAMETER_VALUE
            )

        # NB: Handling the special `mlflow.prompt.is_prompt` tag. This tag is used for
        #   distinguishing between prompt models and normal models. For example, we want to
        #   search for models only by the following filter string:
        #
        #     tags.`mlflow.prompt.is_prompt` != 'true'
        #     tags.`mlflow.prompt.is_prompt` = 'false'
        #
        #   However, models do not have this tag, so lhs is None in this case. Instead of returning
        #   False like normal tag filter, we need to return True here.
        if key == IS_PROMPT_TAG_KEY and lhs is None:
            return (comparator == "=" and value == "false") or (
                comparator == "!=" and value == "true"
            )

        if lhs is None:
            return False

        return SearchUtils.get_comparison_func(comparator)(lhs, value)

    @classmethod
    def filter(cls, registered_models, filter_string):
        """Filters a set of registered models based on a search filter string."""
        if not filter_string:
            return registered_models
        parsed = cls.parse_search_filter(filter_string)

        def registered_model_matches(model):
            return all(cls._does_registered_model_match_clauses(model, s) for s in parsed)

        return [
            registered_model
            for registered_model in registered_models
            if registered_model_matches(registered_model)
        ]

    @classmethod
    def parse_order_by_for_search_registered_models(cls, order_by):
        token_value, is_ascending = cls._parse_order_by_string(order_by)
        identifier = SearchExperimentsUtils._get_identifier(
            token_value.strip(), cls.VALID_ORDER_BY_KEYS_REGISTERED_MODELS
        )
        return identifier["type"], identifier["key"], is_ascending

    @classmethod
    def _get_sort_key(cls, order_by_list):
        order_by = []
        parsed_order_by = map(cls.parse_order_by_for_search_registered_models, order_by_list or [])
        for type_, key, ascending in parsed_order_by:
            if type_ == "attribute":
                order_by.append((key, ascending))
            else:
                raise MlflowException.invalid_parameter_value(f"Invalid order_by entity: {type_}")

        # Add a tie-breaker
        if not any(key == "name" for key, _ in order_by):
            order_by.append(("name", True))

        return lambda model: tuple(_apply_reversor(model, k, asc) for (k, asc) in order_by)

    @classmethod
    def sort(cls, models, order_by_list):
        return sorted(models, key=cls._get_sort_key(order_by_list))

    @classmethod
    def _process_statement(cls, statement):
        tokens = _join_in_comparison_tokens(statement.tokens)
        invalids = list(filter(cls._invalid_statement_token_search_model_registry, tokens))
        if len(invalids) > 0:
            invalid_clauses = ", ".join(map(str, invalids))
            raise MlflowException.invalid_parameter_value(
                f"Invalid clause(s) in filter string: {invalid_clauses}"
            )
        return [cls._get_comparison(t) for t in tokens if isinstance(t, Comparison)]

    @classmethod
    def _get_model_search_identifier(cls, identifier, valid_attributes):
        tokens = identifier.split(".", maxsplit=1)
        if len(tokens) == 1:
            key = tokens[0]
            identifier = cls._ATTRIBUTE_IDENTIFIER
        else:
            entity_type, key = tokens
            valid_entity_types = ("attribute", "tag", "tags")
            if entity_type not in valid_entity_types:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid entity type '{entity_type}'. "
                    f"Valid entity types are {valid_entity_types}"
                )
            identifier = (
                cls._TAG_IDENTIFIER if entity_type in ("tag", "tags") else cls._ATTRIBUTE_IDENTIFIER
            )

        if identifier == cls._ATTRIBUTE_IDENTIFIER and key not in valid_attributes:
            raise MlflowException.invalid_parameter_value(
                f"Invalid attribute key '{key}' specified. Valid keys are '{valid_attributes}'"
            )

        key = cls._trim_backticks(cls._strip_quotes(key))
        return {"type": identifier, "key": key}

    @classmethod
    def _get_comparison(cls, comparison):
        stripped_comparison = [token for token in comparison.tokens if not token.is_whitespace]
        cls._validate_comparison(stripped_comparison)
        left, comparator, right = stripped_comparison
        comp = cls._get_model_search_identifier(left.value, cls.VALID_SEARCH_ATTRIBUTE_KEYS)
        comp["comparator"] = comparator.value.upper()
        comp["value"] = cls._get_value(comp.get("type"), comp.get("key"), right)
        return comp

    @classmethod
    def _get_value(cls, identifier_type, key, token):
        if identifier_type == cls._TAG_IDENTIFIER:
            if token.ttype in cls.STRING_VALUE_TYPES or isinstance(token, Identifier):
                return cls._strip_quotes(token.value, expect_quoted_value=True)
            raise MlflowException(
                "Expected a quoted string value for "
                f"{identifier_type} (e.g. 'my-value'). Got value "
                f"{token.value}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        elif identifier_type == cls._ATTRIBUTE_IDENTIFIER:
            if token.ttype in cls.STRING_VALUE_TYPES or isinstance(token, Identifier):
                return cls._strip_quotes(token.value, expect_quoted_value=True)
            elif isinstance(token, Parenthesis):
                if key != "run_id":
                    raise MlflowException(
                        "Only the 'run_id' attribute supports comparison with a list of quoted "
                        "string values.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                return cls._parse_run_ids(token)
            else:
                raise MlflowException(
                    "Expected a quoted string value or a list of quoted string values for "
                    f"attributes. Got value {token.value}",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        else:
            # Expected to be either "param" or "metric".
            raise MlflowException(
                "Invalid identifier type. Expected one of "
                f"{[cls._ATTRIBUTE_IDENTIFIER, cls._TAG_IDENTIFIER]}.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    @classmethod
    def _invalid_statement_token_search_model_registry(cls, token):
        if (
            isinstance(token, Comparison)
            or token.is_whitespace
            or token.match(ttype=TokenType.Keyword, values=["AND"])
        ):
            return False
        return True


class SearchModelVersionUtils(SearchUtils):
    NUMERIC_ATTRIBUTES = {"version_number", "creation_timestamp", "last_updated_timestamp"}
    VALID_SEARCH_ATTRIBUTE_KEYS = {
        "name",
        "version_number",
        "run_id",
        "source_path",
    }
    VALID_ORDER_BY_ATTRIBUTE_KEYS = {
        "name",
        "version_number",
        "creation_timestamp",
        "last_updated_timestamp",
    }
    VALID_STRING_ATTRIBUTE_COMPARATORS = {"!=", "=", "LIKE", "ILIKE", "IN"}

    @classmethod
    def _does_model_version_match_clauses(cls, mv, sed):
        key_type = sed.get("type")
        key = sed.get("key")
        value = sed.get("value")
        comparator = sed.get("comparator").upper()

        if cls.is_string_attribute(key_type, key, comparator):
            lhs = getattr(mv, "source" if key == "source_path" else key)
        elif cls.is_numeric_attribute(key_type, key, comparator):
            if key == "version_number":
                key = "version"
            lhs = getattr(mv, key)
            value = int(value)
        elif cls.is_tag(key_type, comparator):
            lhs = mv.tags.get(key, None)
        else:
            raise MlflowException(
                f"Invalid search expression type '{key_type}'", error_code=INVALID_PARAMETER_VALUE
            )

        # NB: Handling the special `mlflow.prompt.is_prompt` tag. This tag is used for
        #   distinguishing between prompt models and normal models. For example, we want to
        #   search for models only by the following filter string:
        #
        #     tags.`mlflow.prompt.is_prompt` != 'true'
        #     tags.`mlflow.prompt.is_prompt` = 'false'
        #
        #   However, models do not have this tag, so lhs is None in this case. Instead of returning
        #   False like normal tag filter, we need to return True here.
        if key == IS_PROMPT_TAG_KEY and lhs is None:
            return (comparator == "=" and value == "false") or (
                comparator == "!=" and value == "true"
            )

        if lhs is None:
            return False

        if comparator == "IN" and isinstance(value, (set, list)):
            return lhs in set(value)

        return SearchUtils.get_comparison_func(comparator)(lhs, value)

    @classmethod
    def filter(cls, model_versions, filter_string):
        """Filters a set of model versions based on a search filter string."""
        model_versions = [mv for mv in model_versions if mv.current_stage != STAGE_DELETED_INTERNAL]
        if not filter_string:
            return model_versions
        parsed = cls.parse_search_filter(filter_string)

        def model_version_matches(mv):
            return all(cls._does_model_version_match_clauses(mv, s) for s in parsed)

        return [mv for mv in model_versions if model_version_matches(mv)]

    @classmethod
    def parse_order_by_for_search_model_versions(cls, order_by):
        token_value, is_ascending = cls._parse_order_by_string(order_by)
        identifier = SearchExperimentsUtils._get_identifier(
            token_value.strip(), cls.VALID_ORDER_BY_ATTRIBUTE_KEYS
        )
        return identifier["type"], identifier["key"], is_ascending

    @classmethod
    def _get_sort_key(cls, order_by_list):
        order_by = []
        parsed_order_by = map(cls.parse_order_by_for_search_model_versions, order_by_list or [])
        for type_, key, ascending in parsed_order_by:
            if type_ == "attribute":
                # Need to add this mapping because version is a keyword in sql
                if key == "version_number":
                    key = "version"
                order_by.append((key, ascending))
            else:
                raise MlflowException.invalid_parameter_value(f"Invalid order_by entity: {type_}")

        # Add a tie-breaker
        if not any(key == "name" for key, _ in order_by):
            order_by.append(("name", True))
        if not any(key == "version_number" for key, _ in order_by):
            order_by.append(("version", False))

        return lambda model_version: tuple(
            _apply_reversor(model_version, k, asc) for (k, asc) in order_by
        )

    @classmethod
    def sort(cls, model_versions, order_by_list):
        return sorted(model_versions, key=cls._get_sort_key(order_by_list))

    @classmethod
    def _get_model_version_search_identifier(cls, identifier, valid_attributes):
        tokens = identifier.split(".", maxsplit=1)
        if len(tokens) == 1:
            key = tokens[0]
            identifier = cls._ATTRIBUTE_IDENTIFIER
        else:
            entity_type, key = tokens
            valid_entity_types = ("attribute", "tag", "tags")
            if entity_type not in valid_entity_types:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid entity type '{entity_type}'. "
                    f"Valid entity types are {valid_entity_types}"
                )
            identifier = (
                cls._TAG_IDENTIFIER if entity_type in ("tag", "tags") else cls._ATTRIBUTE_IDENTIFIER
            )

        if identifier == cls._ATTRIBUTE_IDENTIFIER and key not in valid_attributes:
            raise MlflowException.invalid_parameter_value(
                f"Invalid attribute key '{key}' specified. Valid keys are '{valid_attributes}'"
            )

        key = cls._trim_backticks(cls._strip_quotes(key))
        return {"type": identifier, "key": key}

    @classmethod
    def _get_comparison(cls, comparison):
        stripped_comparison = [token for token in comparison.tokens if not token.is_whitespace]
        cls._validate_comparison(stripped_comparison)
        left, comparator, right = stripped_comparison
        comp = cls._get_model_version_search_identifier(left.value, cls.VALID_SEARCH_ATTRIBUTE_KEYS)
        comp["comparator"] = comparator.value.upper()
        comp["value"] = cls._get_value(comp.get("type"), comp.get("key"), right)
        return comp

    @classmethod
    def _get_value(cls, identifier_type, key, token):
        if identifier_type == cls._TAG_IDENTIFIER:
            if token.ttype in cls.STRING_VALUE_TYPES or isinstance(token, Identifier):
                return cls._strip_quotes(token.value, expect_quoted_value=True)
            raise MlflowException(
                "Expected a quoted string value for "
                f"{identifier_type} (e.g. 'my-value'). Got value "
                f"{token.value}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        elif identifier_type == cls._ATTRIBUTE_IDENTIFIER:
            if token.ttype in cls.STRING_VALUE_TYPES or isinstance(token, Identifier):
                return cls._strip_quotes(token.value, expect_quoted_value=True)
            elif isinstance(token, Parenthesis):
                if key != "run_id":
                    raise MlflowException(
                        "Only the 'run_id' attribute supports comparison with a list of quoted "
                        "string values.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                return cls._parse_run_ids(token)
            elif token.ttype in cls.NUMERIC_VALUE_TYPES:
                if key not in cls.NUMERIC_ATTRIBUTES:
                    raise MlflowException(
                        f"Only the '{cls.NUMERIC_ATTRIBUTES}' attributes support comparison with "
                        "numeric values.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                if token.ttype == TokenType.Literal.Number.Integer:
                    return int(token.value)
                elif token.ttype == TokenType.Literal.Number.Float:
                    return float(token.value)
            else:
                raise MlflowException(
                    "Expected a quoted string value or a list of quoted string values for "
                    f"attributes. Got value {token.value}",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        else:
            # Expected to be either "param" or "metric".
            raise MlflowException(
                "Invalid identifier type. Expected one of "
                f"{[cls._ATTRIBUTE_IDENTIFIER, cls._TAG_IDENTIFIER]}.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    @classmethod
    def _process_statement(cls, statement):
        tokens = _join_in_comparison_tokens(statement.tokens)
        invalids = list(filter(cls._invalid_statement_token_search_model_version, tokens))
        if len(invalids) > 0:
            invalid_clauses = ", ".join(map(str, invalids))
            raise MlflowException.invalid_parameter_value(
                f"Invalid clause(s) in filter string: {invalid_clauses}"
            )
        return [cls._get_comparison(t) for t in tokens if isinstance(t, Comparison)]

    @classmethod
    def _invalid_statement_token_search_model_version(cls, token):
        if (
            isinstance(token, Comparison)
            or token.is_whitespace
            or token.match(ttype=TokenType.Keyword, values=["AND"])
        ):
            return False
        return True

    @classmethod
    def parse_search_filter(cls, filter_string):
        if not filter_string:
            return []
        try:
            parsed = sqlparse.parse(filter_string)
        except Exception:
            raise MlflowException(
                f"Error on parsing filter '{filter_string}'", error_code=INVALID_PARAMETER_VALUE
            )
        if len(parsed) == 0 or not isinstance(parsed[0], Statement):
            raise MlflowException(
                f"Invalid filter '{filter_string}'. Could not be parsed.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        elif len(parsed) > 1:
            raise MlflowException(
                f"Search filter contained multiple expression {filter_string!r}. "
                "Provide AND-ed expression list.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return cls._process_statement(parsed[0])


class SearchTraceUtils(SearchUtils):
    """
    Utility class for searching traces.
    """

    VALID_SEARCH_ATTRIBUTE_KEYS = {
        "request_id",
        "timestamp",
        "timestamp_ms",
        "execution_time",
        "execution_time_ms",
        "status",
        # The following keys are mapped to tags or metadata
        "name",
        "run_id",
    }
    VALID_ORDER_BY_ATTRIBUTE_KEYS = {
        "experiment_id",
        "timestamp",
        "timestamp_ms",
        "execution_time",
        "execution_time_ms",
        "status",
        "request_id",
        # The following keys are mapped to tags or metadata
        "name",
        "run_id",
    }

    NUMERIC_ATTRIBUTES = {
        "timestamp_ms",
        "timestamp",
        "execution_time_ms",
        "execution_time",
    }

    # For now, don't support LIKE/ILIKE operators for trace search because it may
    # cause performance issues with large attributes and tags. We can revisit this
    # decision if we find a way to support them efficiently.
    VALID_TAG_COMPARATORS = {"!=", "="}
    VALID_STRING_ATTRIBUTE_COMPARATORS = {"!=", "=", "IN", "NOT IN"}

    _REQUEST_METADATA_IDENTIFIER = "request_metadata"
    _TAG_IDENTIFIER = "tag"
    _ATTRIBUTE_IDENTIFIER = "attribute"

    # These are aliases for the base identifiers
    # e.g. trace.status is equivalent to attribute.status
    _ALTERNATE_IDENTIFIERS = {
        "tags": _TAG_IDENTIFIER,
        "attributes": _ATTRIBUTE_IDENTIFIER,
        "trace": _ATTRIBUTE_IDENTIFIER,
        "metadata": _REQUEST_METADATA_IDENTIFIER,
    }
    _IDENTIFIERS = {_TAG_IDENTIFIER, _REQUEST_METADATA_IDENTIFIER, _ATTRIBUTE_IDENTIFIER}
    _VALID_IDENTIFIERS = _IDENTIFIERS | set(_ALTERNATE_IDENTIFIERS.keys())

    SUPPORT_IN_COMPARISON_ATTRIBUTE_KEYS = {"name", "status", "request_id", "run_id"}

    # Some search keys are defined differently in the DB models.
    # E.g. "name" is mapped to TraceTagKey.TRACE_NAME
    SEARCH_KEY_TO_TAG = {
        "name": TraceTagKey.TRACE_NAME,
    }
    SEARCH_KEY_TO_METADATA = {
        "run_id": TraceMetadataKey.SOURCE_RUN,
    }
    # Alias for attribute keys
    SEARCH_KEY_TO_ATTRIBUTE = {
        "timestamp": "timestamp_ms",
        "execution_time": "execution_time_ms",
    }

    @classmethod
    def filter(cls, traces, filter_string):
        """Filters a set of traces based on a search filter string."""
        if not filter_string:
            return traces
        parsed = cls.parse_search_filter_for_search_traces(filter_string)

        def trace_matches(trace):
            return all(cls._does_trace_match_clause(trace, s) for s in parsed)

        return list(filter(trace_matches, traces))

    @classmethod
    def _does_trace_match_clause(cls, trace, sed):
        type_ = sed.get("type")
        key = sed.get("key")
        value = sed.get("value")
        comparator = sed.get("comparator").upper()

        if cls.is_tag(type_, comparator):
            lhs = trace.tags.get(key)
        elif cls.is_request_metadata(type_, comparator):
            lhs = trace.request_metadata.get(key)
        elif cls.is_attribute(type_, key, comparator):
            lhs = getattr(trace, key)
        elif sed.get("type") == cls._TAG_IDENTIFIER:
            lhs = trace.tags.get(key)
        else:
            raise MlflowException(
                f"Invalid search key '{key}', supported are {cls.VALID_SEARCH_ATTRIBUTE_KEYS}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        if lhs is None:
            return False

        return SearchUtils.get_comparison_func(comparator)(lhs, value)

    @classmethod
    def sort(cls, traces, order_by_list):
        return sorted(traces, key=cls._get_sort_key(order_by_list))

    @classmethod
    def parse_order_by_for_search_traces(cls, order_by):
        token_value, is_ascending = cls._parse_order_by_string(order_by)
        identifier = cls._get_identifier(token_value.strip(), cls.VALID_ORDER_BY_ATTRIBUTE_KEYS)
        identifier = cls._replace_key_to_tag_or_metadata(identifier)
        return identifier["type"], identifier["key"], is_ascending

    @classmethod
    def parse_search_filter_for_search_traces(cls, filter_string):
        parsed = cls.parse_search_filter(filter_string)
        return [cls._replace_key_to_tag_or_metadata(p) for p in parsed]

    @classmethod
    def _replace_key_to_tag_or_metadata(cls, parsed: dict[str, Any]):
        """
        Replace search key to tag or metadata key if it is in the mapping.
        """
        key = parsed.get("key").lower()
        if key in cls.SEARCH_KEY_TO_TAG:
            parsed["type"] = cls._TAG_IDENTIFIER
            parsed["key"] = cls.SEARCH_KEY_TO_TAG[key]
        elif key in cls.SEARCH_KEY_TO_METADATA:
            parsed["type"] = cls._REQUEST_METADATA_IDENTIFIER
            parsed["key"] = cls.SEARCH_KEY_TO_METADATA[key]
        elif key in cls.SEARCH_KEY_TO_ATTRIBUTE:
            parsed["key"] = cls.SEARCH_KEY_TO_ATTRIBUTE[key]
        return parsed

    @classmethod
    def is_request_metadata(cls, key_type, comparator):
        if key_type == cls._REQUEST_METADATA_IDENTIFIER:
            # Request metadata accepts the same set of comparators as tags
            if comparator not in cls.VALID_TAG_COMPARATORS:
                raise MlflowException(
                    f"Invalid comparator '{comparator}' not one of '{cls.VALID_TAG_COMPARATORS}'",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            return True
        return False

    @classmethod
    def _valid_entity_type(cls, entity_type):
        entity_type = cls._trim_backticks(entity_type)
        if entity_type not in cls._VALID_IDENTIFIERS:
            raise MlflowException(
                f"Invalid entity type '{entity_type}'. Valid values are {cls._VALID_IDENTIFIERS}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        elif entity_type in cls._ALTERNATE_IDENTIFIERS:
            return cls._ALTERNATE_IDENTIFIERS[entity_type]
        else:
            return entity_type

    @classmethod
    def _get_sort_key(cls, order_by_list):
        order_by = []
        parsed_order_by = map(cls.parse_order_by_for_search_traces, order_by_list or [])
        for type_, key, ascending in parsed_order_by:
            if type_ == "attribute":
                order_by.append((key, ascending))
            else:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid order_by entity `{type_}` with key `{key}`"
                )

        # Add a tie-breaker
        if not any(key == "timestamp_ms" for key, _ in order_by):
            order_by.append(("timestamp_ms", False))
        if not any(key == "request_id" for key, _ in order_by):
            order_by.append(("request_id", True))

        return lambda trace: tuple(_apply_reversor(trace, k, asc) for (k, asc) in order_by)

    @classmethod
    def _get_value(cls, identifier_type, key, token):
        if identifier_type == cls._TAG_IDENTIFIER:
            if token.ttype in cls.STRING_VALUE_TYPES or isinstance(token, Identifier):
                return cls._strip_quotes(token.value, expect_quoted_value=True)
            elif isinstance(token, Parenthesis):
                return cls._parse_attribute_lists(token)
            raise MlflowException(
                "Expected a quoted string value for "
                f"{identifier_type} (e.g. 'my-value'). Got value "
                f"{token.value}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        elif identifier_type == cls._ATTRIBUTE_IDENTIFIER:
            if token.ttype in cls.STRING_VALUE_TYPES or isinstance(token, Identifier):
                return cls._strip_quotes(token.value, expect_quoted_value=True)
            elif isinstance(token, Parenthesis):
                if key not in cls.SUPPORT_IN_COMPARISON_ATTRIBUTE_KEYS:
                    raise MlflowException(
                        f"Only attributes in {cls.SUPPORT_IN_COMPARISON_ATTRIBUTE_KEYS} "
                        "supports comparison with a list of quoted string values.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                return cls._parse_attribute_lists(token)
            elif token.ttype in cls.NUMERIC_VALUE_TYPES:
                if key not in cls.NUMERIC_ATTRIBUTES:
                    raise MlflowException(
                        f"Only the '{cls.NUMERIC_ATTRIBUTES}' attributes support comparison with "
                        "numeric values.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                if token.ttype == TokenType.Literal.Number.Integer:
                    return int(token.value)
                elif token.ttype == TokenType.Literal.Number.Float:
                    return float(token.value)
            else:
                raise MlflowException(
                    "Expected a quoted string value or a list of quoted string values for "
                    f"attributes. Got value {token.value}",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        elif identifier_type == cls._REQUEST_METADATA_IDENTIFIER:
            if token.ttype in cls.STRING_VALUE_TYPES or isinstance(token, Identifier):
                return cls._strip_quotes(token.value, expect_quoted_value=True)
            else:
                raise MlflowException(
                    "Expected a quoted string value for "
                    f"{identifier_type} (e.g. 'my-value'). Got value "
                    f"{token.value}",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        else:
            # Expected to be either "param" or "metric".
            raise MlflowException(
                f"Invalid identifier type: {identifier_type}. "
                f"Expected one of {cls._VALID_IDENTIFIERS}.",
                error_code=INVALID_PARAMETER_VALUE,
            )

    @classmethod
    def _parse_attribute_lists(cls, token):
        return cls._parse_list_from_sql_token(token)

    @classmethod
    def _process_statement(cls, statement):
        # check validity
        tokens = _join_in_comparison_tokens(statement.tokens, search_traces=True)
        invalids = list(filter(cls._invalid_statement_token_search_traces, tokens))
        if len(invalids) > 0:
            invalid_clauses = ", ".join(f"'{token}'" for token in invalids)
            raise MlflowException(
                f"Invalid clause(s) in filter string: {invalid_clauses}",
                error_code=INVALID_PARAMETER_VALUE,
            )
        return [cls._get_comparison(si) for si in tokens if isinstance(si, Comparison)]

    @classmethod
    def _invalid_statement_token_search_traces(cls, token):
        if (
            isinstance(token, Comparison)
            or token.is_whitespace
            or token.match(ttype=TokenType.Keyword, values=["AND"])
        ):
            return False
        return True

    @classmethod
    def _get_comparison(cls, comparison):
        stripped_comparison = [token for token in comparison.tokens if not token.is_whitespace]
        cls._validate_comparison(stripped_comparison, search_traces=True)
        comp = cls._get_identifier(stripped_comparison[0].value, cls.VALID_SEARCH_ATTRIBUTE_KEYS)
        comp["comparator"] = stripped_comparison[1].value
        comp["value"] = cls._get_value(comp.get("type"), comp.get("key"), stripped_comparison[2])
        return comp


class SearchLoggedModelsUtils(SearchUtils):
    NUMERIC_ATTRIBUTES = {
        "creation_timestamp",
        "creation_time",
        "last_updated_timestamp",
        "last_updated_time",
    }
    VALID_SEARCH_ATTRIBUTE_KEYS = {
        "name",
        "model_id",
        "model_type",
        "status",
        "source_run_id",
    } | NUMERIC_ATTRIBUTES
    VALID_ORDER_BY_ATTRIBUTE_KEYS = VALID_SEARCH_ATTRIBUTE_KEYS

    @classmethod
    def _does_logged_model_match_clause(
        cls,
        model: LoggedModel,
        condition: dict[str, Any],
        datasets: Optional[list[dict[str, Any]]] = None,
    ):
        key_type = condition.get("type")
        key = condition.get("key")
        value = condition.get("value")
        comparator = condition.get("comparator").upper()

        key = SearchUtils.translate_key_alias(key)

        if cls.is_metric(key_type, comparator):
            matching_metrics = [metric for metric in model.metrics if metric.key == key]
            if datasets:
                matching_metrics = [
                    metric
                    for metric in matching_metrics
                    if any(cls._is_metric_on_dataset(metric, dataset) for dataset in datasets)
                ]
            lhs = matching_metrics[0].value if matching_metrics else None
            value = float(value)
        elif cls.is_param(key_type, comparator):
            lhs = model.params.get(key, None)
        elif cls.is_tag(key_type, comparator):
            lhs = model.tags.get(key, None)
        elif cls.is_numeric_attribute(key_type, key, comparator):
            lhs = getattr(model, key)
            value = int(value)
        elif hasattr(model, key):
            lhs = getattr(model, key)
        else:
            raise MlflowException.invalid_parameter_value(
                f"Invalid logged model search key '{key}'",
            )
        if lhs is None:
            return False

        return SearchUtils.get_comparison_func(comparator)(lhs, value)

    @classmethod
    def validate_list_supported(cls, key: str) -> None:
        """
        Override to allow logged model attributes to be used with IN/NOT IN.
        """

    @classmethod
    def filter_logged_models(
        cls,
        models: list[LoggedModel],
        filter_string: Optional[str] = None,
        datasets: Optional[list[dict[str, Any]]] = None,
    ):
        """Filters a set of runs based on a search filter string and list of dataset filters."""
        if not filter_string and not datasets:
            return models

        parsed = cls.parse_search_filter(filter_string)

        # If there are dataset filters but no metric filters in the filter string,
        # filter for models that have any metrics on the datasets
        if datasets and not any(
            cls.is_metric(s.get("type"), s.get("comparator").upper()) for s in parsed
        ):

            def model_has_metrics_on_datasets(model):
                return any(
                    any(cls._is_metric_on_dataset(metric, dataset) for dataset in datasets)
                    for metric in model.metrics
                )

            models = [model for model in models if model_has_metrics_on_datasets(model)]

        def model_matches(model):
            return all(cls._does_logged_model_match_clause(model, s, datasets) for s in parsed)

        return [model for model in models if model_matches(model)]

    @dataclass
    class OrderBy:
        field_name: str
        ascending: bool = True
        dataset_name: Optional[str] = None
        dataset_digest: Optional[str] = None

    @classmethod
    def parse_order_by_for_logged_models(cls, order_by: dict[str, Any]) -> OrderBy:
        if not isinstance(order_by, dict):
            raise MlflowException.invalid_parameter_value(
                "`order_by` must be a list of dictionaries."
            )
        field_name = order_by.get("field_name")
        if field_name is None:
            raise MlflowException.invalid_parameter_value(
                "`field_name` in the `order_by` clause must be specified."
            )
        if "." in field_name:
            entity = field_name.split(".", 1)[0]
            if entity != "metrics":
                raise MlflowException.invalid_parameter_value(
                    f"Invalid order by field name: {entity}, only `metrics.<name>` is allowed."
                )
        else:
            field_name = field_name.strip()
            if field_name not in cls.VALID_ORDER_BY_ATTRIBUTE_KEYS:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid order by field name: {field_name}."
                )
        ascending = order_by.get("ascending", True)
        if ascending not in [True, False]:
            raise MlflowException.invalid_parameter_value(
                "Value of `ascending` in the `order_by` clause must be a boolean, got "
                f"{type(ascending)} for field {field_name}."
            )
        dataset_name = order_by.get("dataset_name")
        dataset_digest = order_by.get("dataset_digest")
        if dataset_digest and not dataset_name:
            raise MlflowException.invalid_parameter_value(
                "`dataset_digest` can only be specified if `dataset_name` is also specified."
            )

        aliases = {
            "creation_time": "creation_timestamp",
        }
        return cls.OrderBy(
            aliases.get(field_name, field_name), ascending, dataset_name, dataset_digest
        )

    @classmethod
    def _apply_reversor_for_logged_model(
        cls,
        model: LoggedModel,
        order_by: OrderBy,
    ):
        if "." in order_by.field_name:
            metric_key = order_by.field_name.split(".", 1)[1]
            filtered_metrics = sorted(
                [
                    m
                    for m in model.metrics
                    if m.key == metric_key
                    and (not order_by.dataset_name or m.dataset_name == order_by.dataset_name)
                    and (not order_by.dataset_digest or m.dataset_digest == order_by.dataset_digest)
                ],
                key=lambda metric: metric.timestamp,
                reverse=True,
            )
            latest_metric_value = None if len(filtered_metrics) == 0 else filtered_metrics[0].value
            return (
                _LoggedModelMetricComp(latest_metric_value)
                if order_by.ascending
                else _Reversor(latest_metric_value)
            )
        else:
            value = getattr(model, order_by.field_name)
        return value if order_by.ascending else _Reversor(value)

    @classmethod
    def _get_sort_key(cls, order_by_list: Optional[list[dict[str, Any]]]):
        parsed_order_by = list(map(cls.parse_order_by_for_logged_models, order_by_list or []))

        # Add a tie-breaker
        if not any(order_by.field_name == "creation_timestamp" for order_by in parsed_order_by):
            parsed_order_by.append(cls.OrderBy("creation_timestamp", False))
        if not any(order_by.field_name == "model_id" for order_by in parsed_order_by):
            parsed_order_by.append(cls.OrderBy("model_id"))

        return lambda logged_model: tuple(
            cls._apply_reversor_for_logged_model(logged_model, order_by)
            for order_by in parsed_order_by
        )

    @classmethod
    def sort(cls, models, order_by_list):
        return sorted(models, key=cls._get_sort_key(order_by_list))


class _LoggedModelMetricComp:
    def __init__(self, obj):
        self.obj = obj

    def __eq__(self, other):
        return other.obj == self.obj

    def __lt__(self, other):
        if self.obj is None:
            return False
        if other.obj is None:
            return True
        return self.obj < other.obj


@dataclass
class SearchLoggedModelsPaginationToken:
    experiment_ids: list[str]
    filter_string: Optional[str] = None
    order_by: Optional[list[dict[str, Any]]] = None
    offset: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    def encode(self) -> str:
        return base64.b64encode(self.to_json().encode("utf-8")).decode("utf-8")

    @classmethod
    def decode(cls, token: str) -> "SearchLoggedModelsPaginationToken":
        try:
            token = json.loads(base64.b64decode(token.encode("utf-8")).decode("utf-8"))
        except json.JSONDecodeError as e:
            raise MlflowException.invalid_parameter_value(f"Invalid page token: {token}. {e}")

        return cls(
            experiment_ids=token.get("experiment_ids"),
            filter_string=token.get("filter_string") or None,
            order_by=token.get("order_by") or None,
            offset=token.get("offset") or 0,
        )

    def validate(
        self,
        experiment_ids: list[str],
        filter_string: Optional[str],
        order_by: Optional[list[dict[str, Any]]],
    ) -> None:
        if self.experiment_ids != experiment_ids:
            raise MlflowException.invalid_parameter_value(
                f"Experiment IDs in the page token do not match the requested experiment IDs. "
                f"Expected: {experiment_ids}. Found: {self.experiment_ids}"
            )

        if self.filter_string != filter_string:
            raise MlflowException.invalid_parameter_value(
                f"Filter string in the page token does not match the requested filter string. "
                f"Expected: {filter_string}. Found: {self.filter_string}"
            )

        if self.order_by != order_by:
            raise MlflowException.invalid_parameter_value(
                f"Order by in the page token does not match the requested order by. "
                f"Expected: {order_by}. Found: {self.order_by}"
            )
