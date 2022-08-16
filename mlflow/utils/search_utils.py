import base64
import json
import operator
import re
import ast
import shlex

import sqlparse
from sqlparse.sql import (
    Identifier,
    Token,
    Comparison,
    Statement,
    Parenthesis,
    IdentifierList,
)
from sqlparse.tokens import Token as TokenType

from mlflow.entities import RunInfo
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.db.db_types import MYSQL, MSSQL

import math


def _case_sensitive_match(string, pattern):
    return re.match(pattern, string) is not None


def _case_insensitive_match(string, pattern):
    return re.match(pattern, string, flags=re.IGNORECASE) is not None


class SearchUtils:
    LIKE_OPERATOR = "LIKE"
    ILIKE_OPERATOR = "ILIKE"
    ASC_OPERATOR = "asc"
    DESC_OPERATOR = "desc"
    VALID_ORDER_BY_TAGS = [ASC_OPERATOR, DESC_OPERATOR]
    VALID_METRIC_COMPARATORS = set([">", ">=", "!=", "=", "<", "<="])
    VALID_PARAM_COMPARATORS = set(["!=", "=", LIKE_OPERATOR, ILIKE_OPERATOR])
    VALID_TAG_COMPARATORS = set(["!=", "=", LIKE_OPERATOR, ILIKE_OPERATOR])
    VALID_STRING_ATTRIBUTE_COMPARATORS = set(["!=", "=", LIKE_OPERATOR, ILIKE_OPERATOR])
    VALID_NUMERIC_ATTRIBUTE_COMPARATORS = VALID_METRIC_COMPARATORS
    NUMERIC_ATTRIBUTES = set(["start_time"])
    CASE_INSENSITIVE_STRING_COMPARISON_OPERATORS = set([LIKE_OPERATOR, ILIKE_OPERATOR])
    VALID_REGISTERED_MODEL_SEARCH_COMPARATORS = CASE_INSENSITIVE_STRING_COMPARISON_OPERATORS.union(
        {"="}
    )
    VALID_MODEL_VERSIONS_SEARCH_COMPARATORS = set(["=", "IN"])
    VALID_SEARCH_ATTRIBUTE_KEYS = set(RunInfo.get_searchable_attributes())
    VALID_ORDER_BY_ATTRIBUTE_KEYS = set(RunInfo.get_orderable_attributes())
    _METRIC_IDENTIFIER = "metric"
    _ALTERNATE_METRIC_IDENTIFIERS = set(["metrics"])
    _PARAM_IDENTIFIER = "parameter"
    _ALTERNATE_PARAM_IDENTIFIERS = set(["parameters", "param", "params"])
    _TAG_IDENTIFIER = "tag"
    _ALTERNATE_TAG_IDENTIFIERS = set(["tags"])
    _ATTRIBUTE_IDENTIFIER = "attribute"
    _ALTERNATE_ATTRIBUTE_IDENTIFIERS = set(["attr", "attributes", "run"])
    _IDENTIFIERS = [_METRIC_IDENTIFIER, _PARAM_IDENTIFIER, _TAG_IDENTIFIER, _ATTRIBUTE_IDENTIFIER]
    _VALID_IDENTIFIERS = set(
        _IDENTIFIERS
        + list(_ALTERNATE_METRIC_IDENTIFIERS)
        + list(_ALTERNATE_PARAM_IDENTIFIERS)
        + list(_ALTERNATE_TAG_IDENTIFIERS)
        + list(_ALTERNATE_ATTRIBUTE_IDENTIFIERS)
    )
    STRING_VALUE_TYPES = set([TokenType.Literal.String.Single])
    DELIMITER_VALUE_TYPES = set([TokenType.Punctuation])
    WHITESPACE_VALUE_TYPE = TokenType.Text.Whitespace
    NUMERIC_VALUE_TYPES = set([TokenType.Literal.Number.Integer, TokenType.Literal.Number.Float])
    # Registered Models Constants
    ORDER_BY_KEY_TIMESTAMP = "timestamp"
    ORDER_BY_KEY_LAST_UPDATED_TIMESTAMP = "last_updated_timestamp"
    ORDER_BY_KEY_MODEL_NAME = "name"
    VALID_ORDER_BY_KEYS_REGISTERED_MODELS = set(
        [ORDER_BY_KEY_TIMESTAMP, ORDER_BY_KEY_LAST_UPDATED_TIMESTAMP, ORDER_BY_KEY_MODEL_NAME]
    )
    VALID_TIMESTAMP_ORDER_BY_KEYS = set(
        [ORDER_BY_KEY_TIMESTAMP, ORDER_BY_KEY_LAST_UPDATED_TIMESTAMP]
    )
    # We encourage users to use timestamp for order-by
    RECOMMENDED_ORDER_BY_KEYS_REGISTERED_MODELS = set(
        [ORDER_BY_KEY_MODEL_NAME, ORDER_BY_KEY_TIMESTAMP]
    )

    filter_ops = {
        ">": operator.gt,
        ">=": operator.ge,
        "=": operator.eq,
        "!=": operator.ne,
        "<=": operator.le,
        "<": operator.lt,
        "LIKE": _case_sensitive_match,
        "ILIKE": _case_insensitive_match,
    }

    @classmethod
    def get_sql_filter_ops(cls, column, operator, dialect):
        import sqlalchemy as sa

        col = f"{column.class_.__tablename__}.{column.key}"

        # Use case-sensitive collation for MSSQL
        if dialect == MSSQL:
            column = column.collate("Japanese_Bushu_Kakusu_100_CS_AS_KS_WS")

        # Use non-binary ahead of binary comparison for runtime performance
        # Use non-binary ahead of binary comparison for runtime performance
        def case_sensitive_mysql_eq(value):
            return sa.text(f"({col} = :value AND BINARY {col} = :value)").bindparams(
                sa.bindparam("value", value=value, unique=True)
            )

        def case_sensitive_mysql_ne(value):
            return sa.text(f"({col} != :value OR BINARY {col} != :value)").bindparams(
                sa.bindparam("value", value=value, unique=True)
            )

        def case_sensitive_mysql_like(value):
            return sa.text(f"({col} LIKE :value AND BINARY {col} LIKE :value)").bindparams(
                sa.bindparam("value", value=value, unique=True)
            )

        sql_filter_ops = {
            "=": case_sensitive_mysql_eq if dialect == MYSQL else column.__eq__,
            "!=": case_sensitive_mysql_ne if dialect == MYSQL else column.__ne__,
            "LIKE": case_sensitive_mysql_like if dialect == MYSQL else column.like,
            "ILIKE": column.ilike,
        }
        return sql_filter_ops[operator]

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
                "types used for string value %s. Use either single or double "
                "quotes." % value,
                error_code=INVALID_PARAMETER_VALUE,
            )
        else:
            return value

    @classmethod
    def _valid_entity_type(cls, entity_type):
        entity_type = cls._trim_backticks(entity_type)
        if entity_type not in cls._VALID_IDENTIFIERS:
            raise MlflowException(
                "Invalid entity type '%s'. "
                "Valid values are %s" % (entity_type, cls._IDENTIFIERS),
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
        else:
            # one of ("metric", "parameter", "tag", or "attribute") since it a valid type
            return entity_type

    @classmethod
    def _get_identifier(cls, identifier, valid_attributes):
        try:
            entity_type, key = identifier.split(".", 1)
        except ValueError:
            raise MlflowException(
                "Invalid identifier '%s'. Columns should be specified as "
                "'attribute.<key>', 'metric.<key>', 'tag.<key>', or "
                "'param.'." % identifier,
                error_code=INVALID_PARAMETER_VALUE,
            )
        identifier = cls._valid_entity_type(entity_type)
        key = cls._trim_backticks(cls._strip_quotes(key))
        if identifier == cls._ATTRIBUTE_IDENTIFIER and key not in valid_attributes:
            raise MlflowException.invalid_parameter_value(
                "Invalid attribute key '{}' specified. Valid keys "
                "are '{}'".format(key, valid_attributes)
            )
        return {"type": identifier, "key": key}

    @classmethod
    def _get_value(cls, identifier_type, key, token):
        if identifier_type == cls._METRIC_IDENTIFIER:
            if token.ttype not in cls.NUMERIC_VALUE_TYPES:
                raise MlflowException(
                    "Expected numeric value type for metric. Found {}".format(token.value),
                    error_code=INVALID_PARAMETER_VALUE,
                )
            return token.value
        elif identifier_type == cls._PARAM_IDENTIFIER or identifier_type == cls._TAG_IDENTIFIER:
            if token.ttype in cls.STRING_VALUE_TYPES or isinstance(token, Identifier):
                return cls._strip_quotes(token.value, expect_quoted_value=True)
            raise MlflowException(
                "Expected a quoted string value for "
                "{identifier_type} (e.g. 'my-value'). Got value "
                "{value}".format(identifier_type=identifier_type, value=token.value),
                error_code=INVALID_PARAMETER_VALUE,
            )
        elif identifier_type == cls._ATTRIBUTE_IDENTIFIER:
            if key in cls.NUMERIC_ATTRIBUTES:
                if token.ttype not in cls.NUMERIC_VALUE_TYPES:
                    raise MlflowException(
                        "Expected numeric value type for numeric attribute: {}. "
                        "Found {}".format(key, token.value),
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                return token.value
            elif token.ttype in cls.STRING_VALUE_TYPES or isinstance(token, Identifier):
                return cls._strip_quotes(token.value, expect_quoted_value=True)
            else:
                raise MlflowException(
                    "Expected a quoted string value for attributes. "
                    "Got value {value}".format(value=token.value),
                    error_code=INVALID_PARAMETER_VALUE,
                )
        else:
            # Expected to be either "param" or "metric".
            raise MlflowException(
                "Invalid identifier type. Expected one of "
                "{}.".format([cls._METRIC_IDENTIFIER, cls._PARAM_IDENTIFIER])
            )

    @classmethod
    def _validate_comparison(cls, tokens):
        base_error_string = "Invalid comparison clause"
        if len(tokens) != 3:
            raise MlflowException(
                "{}. Expected 3 tokens found {}".format(base_error_string, len(tokens)),
                error_code=INVALID_PARAMETER_VALUE,
            )
        if not isinstance(tokens[0], Identifier):
            raise MlflowException(
                "{}. Expected 'Identifier' found '{}'".format(base_error_string, str(tokens[0])),
                error_code=INVALID_PARAMETER_VALUE,
            )
        if not isinstance(tokens[1], Token) and tokens[1].ttype != TokenType.Operator.Comparison:
            raise MlflowException(
                "{}. Expected comparison found '{}'".format(base_error_string, str(tokens[1])),
                error_code=INVALID_PARAMETER_VALUE,
            )
        if not isinstance(tokens[2], Token) and (
            tokens[2].ttype not in cls.STRING_VALUE_TYPES.union(cls.NUMERIC_VALUE_TYPES)
            or isinstance(tokens[2], Identifier)
        ):
            raise MlflowException(
                "{}. Expected value token found '{}'".format(base_error_string, str(tokens[2])),
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
        if isinstance(token, Comparison):
            return False
        elif token.is_whitespace:
            return False
        elif token.match(ttype=TokenType.Keyword, values=["AND"]):
            return False
        else:
            return True

    @classmethod
    def _process_statement(cls, statement):
        # check validity
        invalids = list(filter(cls._invalid_statement_token_search_runs, statement.tokens))
        if len(invalids) > 0:
            invalid_clauses = ", ".join("'%s'" % token for token in invalids)
            raise MlflowException(
                "Invalid clause(s) in filter string: %s" % invalid_clauses,
                error_code=INVALID_PARAMETER_VALUE,
            )
        return [cls._get_comparison(si) for si in statement.tokens if isinstance(si, Comparison)]

    @classmethod
    def parse_search_filter(cls, filter_string):
        if not filter_string:
            return []
        try:
            parsed = sqlparse.parse(filter_string)
        except Exception:
            raise MlflowException(
                "Error on parsing filter '%s'" % filter_string, error_code=INVALID_PARAMETER_VALUE
            )
        if len(parsed) == 0 or not isinstance(parsed[0], Statement):
            raise MlflowException(
                "Invalid filter '%s'. Could not be parsed." % filter_string,
                error_code=INVALID_PARAMETER_VALUE,
            )
        elif len(parsed) > 1:
            raise MlflowException(
                "Search filter contained multiple expression '%s'. "
                "Provide AND-ed expression list." % filter_string,
                error_code=INVALID_PARAMETER_VALUE,
            )
        return cls._process_statement(parsed[0])

    @classmethod
    def is_metric(cls, key_type, comparator):
        if key_type == cls._METRIC_IDENTIFIER:
            if comparator not in cls.VALID_METRIC_COMPARATORS:
                raise MlflowException(
                    "Invalid comparator '%s' "
                    "not one of '%s" % (comparator, cls.VALID_METRIC_COMPARATORS),
                    error_code=INVALID_PARAMETER_VALUE,
                )
            return True
        return False

    @classmethod
    def is_param(cls, key_type, comparator):
        if key_type == cls._PARAM_IDENTIFIER:
            if comparator not in cls.VALID_PARAM_COMPARATORS:
                raise MlflowException(
                    "Invalid comparator '%s' "
                    "not one of '%s'" % (comparator, cls.VALID_PARAM_COMPARATORS),
                    error_code=INVALID_PARAMETER_VALUE,
                )
            return True
        return False

    @classmethod
    def is_tag(cls, key_type, comparator):
        if key_type == cls._TAG_IDENTIFIER:
            if comparator not in cls.VALID_TAG_COMPARATORS:
                raise MlflowException(
                    "Invalid comparator '%s' "
                    "not one of '%s" % (comparator, cls.VALID_TAG_COMPARATORS)
                )
            return True
        return False

    @classmethod
    def is_string_attribute(cls, key_type, key_name, comparator):
        if key_type == cls._ATTRIBUTE_IDENTIFIER and key_name not in cls.NUMERIC_ATTRIBUTES:
            if comparator not in cls.VALID_STRING_ATTRIBUTE_COMPARATORS:
                raise MlflowException(
                    "Invalid comparator '{}' not one of "
                    "'{}".format(comparator, cls.VALID_STRING_ATTRIBUTE_COMPARATORS)
                )
            return True
        return False

    @classmethod
    def is_numeric_attribute(cls, key_type, key_name, comparator):
        if key_type == cls._ATTRIBUTE_IDENTIFIER and key_name in cls.NUMERIC_ATTRIBUTES:
            if comparator not in cls.VALID_NUMERIC_ATTRIBUTE_COMPARATORS:
                raise MlflowException(
                    "Invalid comparator '{}' not one of "
                    "'{}".format(comparator, cls.VALID_STRING_ATTRIBUTE_COMPARATORS)
                )
            return True
        return False

    @classmethod
    def _convert_like_pattern_to_regex(cls, pattern):
        if not pattern.startswith("%"):
            pattern = "^" + pattern
        if not pattern.endswith("%"):
            pattern = pattern + "$"
        return pattern.replace("_", ".").replace("%", ".*")

    @classmethod
    def _does_run_match_clause(cls, run, sed):
        key_type = sed.get("type")
        key = sed.get("key")
        value = sed.get("value")
        comparator = sed.get("comparator").upper()

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
        else:
            raise MlflowException(
                "Invalid search expression type '%s'" % key_type, error_code=INVALID_PARAMETER_VALUE
            )
        if lhs is None:
            return False

        if comparator in cls.CASE_INSENSITIVE_STRING_COMPARISON_OPERATORS:
            value = cls._convert_like_pattern_to_regex(value)

        if comparator in cls.filter_ops.keys():
            return cls.filter_ops.get(comparator)(lhs, value)
        else:
            return False

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
                "Error on parsing order_by clause '{}'".format(order_by),
                error_code=INVALID_PARAMETER_VALUE,
            )
        if len(parsed) != 1 or not isinstance(parsed[0], Statement):
            raise MlflowException(
                "Invalid order_by clause '{}'. Could not be parsed.".format(order_by),
                error_code=INVALID_PARAMETER_VALUE,
            )
        statement = parsed[0]
        if len(statement.tokens) == 1 and isinstance(statement[0], Identifier):
            token_value = statement.tokens[0].value
        elif len(statement.tokens) == 1 and statement.tokens[0].match(
            ttype=TokenType.Keyword, values=[cls.ORDER_BY_KEY_TIMESTAMP]
        ):
            token_value = cls.ORDER_BY_KEY_TIMESTAMP
        elif (
            statement.tokens[0].match(ttype=TokenType.Keyword, values=[cls.ORDER_BY_KEY_TIMESTAMP])
            and all(token.is_whitespace for token in statement.tokens[1:-1])
            and statement.tokens[-1].ttype == TokenType.Keyword.Order
        ):
            token_value = cls.ORDER_BY_KEY_TIMESTAMP + " " + statement.tokens[-1].value
        else:
            raise MlflowException(
                "Invalid order_by clause '{}'. Could not be parsed.".format(order_by),
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
                "Invalid order_by clause '{}'. Could not be parsed.".format(order_by),
                error_code=INVALID_PARAMETER_VALUE,
            )
        elif len(tokens) == 2:
            order_token = tokens[1].lower()
            if order_token not in cls.VALID_ORDER_BY_TAGS:
                raise MlflowException(
                    "Invalid ordering key in order_by clause '{}'.".format(order_by),
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
                "Invalid order by key '{}' specified. Valid keys ".format(token_value)
                + "are '{}'".format(cls.RECOMMENDED_ORDER_BY_KEYS_REGISTERED_MODELS),
                error_code=INVALID_PARAMETER_VALUE,
            )
        return token_value, is_ascending

    @classmethod
    def _get_value_for_sort(cls, run, key_type, key, ascending):
        """Returns a tuple suitable to be used as a sort key for runs."""
        sort_value = None
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
                "Invalid order_by entity type '%s'" % key_type, error_code=INVALID_PARAMETER_VALUE
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
        runs = sorted(runs, key=lambda run: (-run.info.start_time, run.info.run_uuid))
        if not order_by_list:
            return runs
        # NB: We rely on the stability of Python's sort function, so that we can apply
        # the ordering conditions in reverse order.
        for order_by_clause in reversed(order_by_list):
            (key_type, key, ascending) = cls.parse_order_by_for_search_runs(order_by_clause)
            # pylint: disable=cell-var-from-loop
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
                "Invalid page token, decoded value=%s" % decoded_token,
                error_code=INVALID_PARAMETER_VALUE,
            )

        offset_str = parsed_token.get("offset")
        if not offset_str:
            raise MlflowException(
                "Invalid page token, parsed value=%s" % parsed_token,
                error_code=INVALID_PARAMETER_VALUE,
            )

        try:
            offset = int(offset_str)
        except ValueError:
            raise MlflowException(
                "Invalid page token, not stringable %s" % offset_str,
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

    VALID_SEARCH_KEYS_FOR_MODEL_VERSIONS = set(["name", "run_id", "source_path"])
    VALID_SEARCH_KEYS_FOR_REGISTERED_MODELS = set(["name"])

    @classmethod
    def _check_valid_identifier_list(cls, value_token):
        if len(value_token._groupable_tokens) == 0:
            raise MlflowException(
                "While parsing a list in the query,"
                " expected a non-empty list of string values, but got empty list",
                error_code=INVALID_PARAMETER_VALUE,
            )
        elif not isinstance(value_token._groupable_tokens[0], IdentifierList):
            raise MlflowException(
                "While parsing a list in the query,"
                " expected a non-empty list of string values, but got ill-formed list.",
                error_code=INVALID_PARAMETER_VALUE,
            )
        elif not all(
            map(
                lambda token: token.ttype
                in {*cls.STRING_VALUE_TYPES, *cls.DELIMITER_VALUE_TYPES, cls.WHITESPACE_VALUE_TYPE},
                value_token._groupable_tokens[0].tokens,
            )
        ):
            raise MlflowException(
                "While parsing a list in the query, expected string value, punctuation, "
                "or whitespace, but got different type in list: {value_token}".format(
                    value_token=value_token
                ),
                error_code=INVALID_PARAMETER_VALUE,
            )

    @classmethod
    def _parse_list_from_sql_token(cls, token):
        try:
            return ast.literal_eval(token.value)
        except SyntaxError:
            raise MlflowException(
                "While parsing a list in the query,"
                " expected a non-empty list of string values, but got ill-formed list.",
                error_code=INVALID_PARAMETER_VALUE,
            )


class SearchExperimentsUtils(SearchUtils):
    VALID_SEARCH_ATTRIBUTE_KEYS = ("name",)
    VALID_ORDER_BY_ATTRIBUTE_KEYS = ("name", "experiment_id")

    @classmethod
    def _invalid_statement_token_search_experiments(cls, token):
        if isinstance(token, (Comparison, Identifier, Parenthesis)):
            return False
        elif token.is_whitespace:
            return False
        elif token.match(ttype=TokenType.Keyword, values=["AND"]):
            return False
        else:
            return True

    @classmethod
    def _process_statement(cls, statement):
        invalids = list(filter(cls._invalid_statement_token_search_experiments, statement.tokens))
        if len(invalids) > 0:
            invalid_clauses = ", ".join(map(str, invalids))
            raise MlflowException.invalid_parameter_value(
                "Invalid clause(s) in filter string: %s" % invalid_clauses
            )
        return [cls._get_comparison(t) for t in statement.tokens if isinstance(t, Comparison)]

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
                "Invalid attribute key '{}' specified. Valid keys "
                "are '{}'".format(key, valid_attributes)
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
                    "Invalid comparator '{}' not one of "
                    "'{}".format(comparator, cls.VALID_STRING_ATTRIBUTE_COMPARATORS)
                )
            return True
        return False

    @classmethod
    def _does_experiment_match_clause(cls, experiment, sed):  # pylint: disable=arguments-renamed
        key_type = sed.get("type")
        key = sed.get("key")
        value = sed.get("value")
        comparator = sed.get("comparator").upper()

        if cls.is_attribute(key_type, comparator):
            lhs = getattr(experiment, key)
        elif cls.is_tag(key_type, comparator):
            if key not in experiment.tags:
                return False
            lhs = experiment.tags.get(key, None)
            if lhs is None:
                return experiment
        else:
            raise MlflowException(
                "Invalid search expression type '%s'" % key_type, error_code=INVALID_PARAMETER_VALUE
            )

        if comparator in cls.CASE_INSENSITIVE_STRING_COMPARISON_OPERATORS:
            value = cls._convert_like_pattern_to_regex(value)

        if comparator in cls.filter_ops.keys():
            return cls.filter_ops.get(comparator)(lhs, value)
        else:
            return False

    @classmethod
    def filter(cls, experiments, filter_string):  # pylint: disable=arguments-renamed
        if not filter_string:
            return experiments
        parsed = cls.parse_search_filter(filter_string)

        def experiment_matches(experiment):
            return all(cls._does_experiment_match_clause(experiment, s) for s in parsed)

        return list(filter(experiment_matches, experiments))

    @classmethod
    def _get_sort_key(cls, order_by_list):
        order_by = []
        parsed_order_by = map(cls.parse_order_by_for_search_experiments, order_by_list or [])
        for type_, key, ascending in parsed_order_by:
            if type_ == "attribute":
                order_by.append((key, ascending))
            else:
                raise MlflowException.invalid_parameter_value(f"Invalid order_by entity: {type_}")

        # Add a tie-breaker
        if not any(key == "experiment_id" for key, _ in order_by):
            order_by.append(("experiment_id", False))

        # https://stackoverflow.com/a/56842689
        class _Reversor:
            def __init__(self, obj):
                self.obj = obj

            # Only need < and == are needed for use as a key parameter in the sorted function
            def __eq__(self, other):
                return other.obj == self.obj

            def __lt__(self, other):
                return other.obj < self.obj

        def _apply_reversor(experiment, key, ascending):
            attr = getattr(experiment, key)
            return attr if ascending else _Reversor(attr)

        return lambda experiment: tuple(
            _apply_reversor(experiment, k, asc) for (k, asc) in order_by
        )

    @classmethod
    def sort(cls, experiments, order_by_list):  # pylint: disable=arguments-renamed
        return sorted(experiments, key=cls._get_sort_key(order_by_list))


class SearchModelUtils(SearchUtils):
    @classmethod
    def _process_statement(cls, statement):
        invalids = list(
            filter(cls._invalid_statement_token_search_model_registry, statement.tokens)
        )
        if len(invalids) > 0:
            invalid_clauses = ", ".join(map(str, invalids))
            raise MlflowException.invalid_parameter_value(
                "Invalid clause(s) in filter string: %s" % invalid_clauses
            )
        return [cls._get_comparison(t) for t in statement.tokens if isinstance(t, Comparison)]

    @classmethod
    def _get_model_search_identifier(cls, identifier):
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

        key = cls._trim_backticks(cls._strip_quotes(key))
        return {"type": identifier, "key": key}

    @classmethod
    def _get_comparison(cls, comparison):
        stripped_comparison = [token for token in comparison.tokens if not token.is_whitespace]
        cls._validate_comparison(stripped_comparison)
        left, comparator, right = stripped_comparison
        comp = cls._get_model_search_identifier(left.value)
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
                "{identifier_type} (e.g. 'my-value'). Got value "
                "{value}".format(identifier_type=identifier_type, value=token.value),
                error_code=INVALID_PARAMETER_VALUE,
            )
        elif identifier_type == cls._ATTRIBUTE_IDENTIFIER:
            if token.ttype in cls.STRING_VALUE_TYPES or isinstance(token, Identifier):
                return cls._strip_quotes(token.value, expect_quoted_value=True)
            elif isinstance(token, Parenthesis):
                cls._check_valid_identifier_list(token)
                if key != "run_id":
                    raise MlflowException(
                        "Only run_id attribute support compare with a list of quoted string "
                        "values.",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
                run_id_list = cls._parse_list_from_sql_token(token)
                # Because MySQL IN clause is case-insensitive, but all run_ids only contain lower
                # case letters, so that we filter out run_ids containing upper case letters here.
                run_id_list = [run_id for run_id in run_id_list if run_id.islower()]
                return run_id_list
            else:
                raise MlflowException(
                    "Expected a quoted string value or a list of quoted string values for "
                    "attributes. Got value {value}".format(value=token.value),
                    error_code=INVALID_PARAMETER_VALUE,
                )
        else:
            # Expected to be either "param" or "metric".
            raise MlflowException(
                "Invalid identifier type. Expected one of "
                "{}.".format([cls._ATTRIBUTE_IDENTIFIER, cls._TAG_IDENTIFIER]),
                error_code=INVALID_PARAMETER_VALUE,
            )

    @classmethod
    def _invalid_statement_token_search_model_registry(cls, token):
        if isinstance(token, (Comparison, Identifier, Parenthesis)):
            return False
        elif token.is_whitespace:
            return False
        elif token.match(ttype=TokenType.Keyword, values=["AND", "IN"]):
            return False
        else:
            return True
