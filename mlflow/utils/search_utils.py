import base64
import json
import operator
import re
import shlex

import sqlparse
import sqlalchemy.sql.expression as sql
import math
from sqlparse.sql import Identifier, Token, Comparison, Statement
from sqlparse.tokens import Token as TokenType

from mlflow.entities import RunInfo
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.tracking.dbmodels.models import (
    SqlRun,
    SqlParam,
    SqlTag,
    SqlLatestMetric,
)
from mlflow.store.model_registry.dbmodels.models import (
    SqlRegisteredModel,
    SqlRegisteredModelTag,
)


class SearchUtils(object):
    LIKE_OPERATOR = "LIKE"
    ILIKE_OPERATOR = "ILIKE"
    ASC_OPERATOR = "asc"
    DESC_OPERATOR = "desc"
    VALID_ORDER_BY_TAGS = [ASC_OPERATOR, DESC_OPERATOR]
    VALID_METRIC_COMPARATORS = set([">", ">=", "!=", "=", "<", "<="])
    VALID_PARAM_COMPARATORS = set(["!=", "=", LIKE_OPERATOR, ILIKE_OPERATOR])
    VALID_TAG_COMPARATORS = set(["!=", "=", LIKE_OPERATOR, ILIKE_OPERATOR])
    VALID_STRING_ATTRIBUTE_COMPARATORS = set(["!=", "=", LIKE_OPERATOR, ILIKE_OPERATOR])
    CASE_INSENSITIVE_STRING_COMPARISON_OPERATORS = set([LIKE_OPERATOR, ILIKE_OPERATOR])
    VALID_REGISTERED_MODEL_SEARCH_COMPARATORS = CASE_INSENSITIVE_STRING_COMPARISON_OPERATORS.union(
        {"="}
    )
    VALID_SEARCH_RUN_ATTRIBUTE_KEYS = set(RunInfo.get_searchable_attributes())
    VALID_ORDER_BY_RUN_ATTRIBUTE_KEYS = set(RunInfo.get_orderable_attributes())
    _METRIC_IDENTIFIER = "metric"
    _ALTERNATE_METRIC_IDENTIFIERS = set(["metrics"])
    _PARAM_IDENTIFIER = "parameter"
    _ALTERNATE_PARAM_IDENTIFIERS = set(["parameters", "param", "params"])
    _TAG_IDENTIFIER = "tag"
    _ALTERNATE_RUN_TAG_IDENTIFIERS = set(["tags"])
    _ATTRIBUTE_IDENTIFIER = "attribute"
    _ALTERNATE_RUN_ATTRIBUTE_IDENTIFIERS = set(["attr", "attributes", "run"])
    _ALTERNATE_MODEL_ATTRIBUTE_IDENTIFIERS = set(
        ["attr", "attributes", "model", "registered_model", "models", "registered_models"]
    )
    _ALTERNATE_MODEL_TAG_IDENTIFIERS = set(
        ["tags", "model_tag", "registered_model_tag", "model_tags", "registered_model_tags"]
    )
    _RUN_IDENTIFIERS = [
        _METRIC_IDENTIFIER,
        _PARAM_IDENTIFIER,
        _TAG_IDENTIFIER,
        _ATTRIBUTE_IDENTIFIER,
    ]
    _MODEL_IDENTIFIERS = [_TAG_IDENTIFIER, _ATTRIBUTE_IDENTIFIER]
    _VALID_RUN_IDENTIFIERS = set(
        _RUN_IDENTIFIERS
        + list(_ALTERNATE_METRIC_IDENTIFIERS)
        + list(_ALTERNATE_PARAM_IDENTIFIERS)
        + list(_ALTERNATE_RUN_TAG_IDENTIFIERS)
        + list(_ALTERNATE_RUN_ATTRIBUTE_IDENTIFIERS)
    )
    _VALID_MODEL_IDENTIFIERS = set(
        _MODEL_IDENTIFIERS
        + list(_ALTERNATE_MODEL_ATTRIBUTE_IDENTIFIERS)
        + list(_ALTERNATE_MODEL_TAG_IDENTIFIERS)
    )
    STRING_VALUE_TYPES = set([TokenType.Literal.String.Single])
    NUMERIC_VALUE_TYPES = set([TokenType.Literal.Number.Integer, TokenType.Literal.Number.Float])
    # Registered Models Constants
    ORDER_BY_KEY_TIMESTAMP = "timestamp"
    VALID_SEARCH_MODEL_VERSION_ATTRIBUTE_KEYS = set(["name", "run_id", "source_path"])
    VALID_SEARCH_REGISTERED_MODEL_ATTRIBUTE_KEYS = set(["name"])
    VALID_ORDER_BY_REGISTERED_MODEL_ATTRIBUTE_KEYS = set(
        [ORDER_BY_KEY_TIMESTAMP, "last_updated_timestamp", "name"]
    )
    VALID_TIMESTAMP_ORDER_BY_KEYS = set([ORDER_BY_KEY_TIMESTAMP, "last_updated_timestamp"])

    filter_ops = {
        ">": operator.gt,
        ">=": operator.ge,
        "=": operator.eq,
        "!=": operator.ne,
        "<=": operator.le,
        "<": operator.lt,
        "LIKE": re.match,
        "ILIKE": re.match,
    }

    @classmethod
    def get_sql_filter_ops(cls, column, operator):
        sql_filter_ops = {"LIKE": column.like, "ILIKE": column.ilike}
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
    def _valid_run_entity_type(cls, entity_type):
        entity_type = cls._trim_backticks(entity_type)
        if entity_type not in cls._VALID_RUN_IDENTIFIERS:
            raise MlflowException(
                "Invalid entity type '%s'. "
                "Valid values are %s" % (entity_type, cls._RUN_IDENTIFIERS),
                error_code=INVALID_PARAMETER_VALUE,
            )

        if entity_type in cls._ALTERNATE_PARAM_IDENTIFIERS:
            return cls._PARAM_IDENTIFIER
        elif entity_type in cls._ALTERNATE_METRIC_IDENTIFIERS:
            return cls._METRIC_IDENTIFIER
        elif entity_type in cls._ALTERNATE_RUN_TAG_IDENTIFIERS:
            return cls._TAG_IDENTIFIER
        elif entity_type in cls._ALTERNATE_RUN_ATTRIBUTE_IDENTIFIERS:
            return cls._ATTRIBUTE_IDENTIFIER
        else:
            # one of ("metric", "parameter", "tag", or "attribute") since it a valid type
            return entity_type

    @classmethod
    def _valid_model_entity_type(cls, entity_type):
        entity_type = cls._trim_backticks(entity_type)
        if entity_type not in cls._VALID_MODEL_IDENTIFIERS:
            raise MlflowException(
                "Invalid entity type '%s'. "
                "Valid values are %s" % (entity_type, cls._MODEL_IDENTIFIERS),
                error_code=INVALID_PARAMETER_VALUE,
            )
        if entity_type in cls._ALTERNATE_MODEL_ATTRIBUTE_IDENTIFIERS:
            return cls._ATTRIBUTE_IDENTIFIER
        elif entity_type in cls._ALTERNATE_MODEL_TAG_IDENTIFIERS:
            return cls._TAG_IDENTIFIER
        else:
            return entity_type

    @classmethod
    def _get_identifier(cls, identifier, valid_attributes, is_search_run):
        def split_identifier(full_column_name, missing_table_name):
            if missing_table_name:
                return "attribute", full_column_name
            else:
                try:
                    table_name, column_name = full_column_name.split(".", 1)
                    return table_name, column_name
                except ValueError:
                    raise MlflowException(
                        "Invalid identifier '%s'. Columns should be specified as "
                        "'attribute.<key>', 'metric.<key>', 'tag.<key>', or "
                        "'param.'." % full_column_name,
                        error_code=INVALID_PARAMETER_VALUE,
                    )

        # search models may not provide table name, in that case default to attribute table
        # (e.g name = 'CNN' is equivalent to attribute.model = 'CNN')
        column_only = identifier.find(".") == -1
        entity_type, key = split_identifier(identifier, column_only)
        identifier = (
            cls._valid_run_entity_type(entity_type)
            if is_search_run
            else cls._valid_model_entity_type(entity_type)
        )
        key = cls._trim_backticks(cls._strip_quotes(key))
        if identifier == cls._ATTRIBUTE_IDENTIFIER and key not in valid_attributes:
            raise MlflowException(
                "Invalid attribute key '{}' specified. Valid keys"
                " are {}".format(key, valid_attributes),
                error_code=INVALID_PARAMETER_VALUE,
            )
        return {"type": identifier, "key": key}

    @classmethod
    def _get_value(cls, identifier_type, token):
        if identifier_type == cls._METRIC_IDENTIFIER:
            if token.ttype not in cls.NUMERIC_VALUE_TYPES:
                raise MlflowException(
                    "Expected numeric value type for metric. " "Found {}".format(token.value),
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
            if token.ttype in cls.STRING_VALUE_TYPES or isinstance(token, Identifier):
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
    def _get_comparison(cls, comparison, valid_search_keys, is_search_run):
        stripped_comparison = [token for token in comparison.tokens if not token.is_whitespace]
        cls._validate_comparison(stripped_comparison)
        comp = cls._get_identifier(stripped_comparison[0].value, valid_search_keys, is_search_run)
        comp["comparator"] = stripped_comparison[1].value
        comp["value"] = cls._get_value(comp.get("type"), stripped_comparison[2])
        return comp

    @classmethod
    def _invalid_statement_token(cls, token):
        if isinstance(token, Comparison):
            return False
        elif token.is_whitespace:
            return False
        elif token.match(ttype=TokenType.Keyword, values=["AND"]):
            return False
        else:
            return True

    @classmethod
    def _process_statement(cls, statement, valid_search_keys, is_search_run):
        # check validity
        invalids = list(filter(cls._invalid_statement_token, statement.tokens))
        if len(invalids) > 0:
            invalid_clauses = ", ".join("'%s'" % token for token in invalids)
            raise MlflowException(
                "Invalid clause(s) in filter string: %s" % invalid_clauses,
                error_code=INVALID_PARAMETER_VALUE,
            )
        return [
            cls._get_comparison(si, valid_search_keys, is_search_run)
            for si in statement.tokens
            if isinstance(si, Comparison)
        ]

    @classmethod
    def _parse_search_filter(cls, filter_string, valid_search_keys, is_search_run):
        if not filter_string:
            return []
        try:
            parsed = sqlparse.parse(filter_string)
        except Exception:
            raise MlflowException(
                "Error on parsing filter '%s'" % filter_string, error_code=INVALID_PARAMETER_VALUE,
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
        return cls._process_statement(parsed[0], valid_search_keys, is_search_run)

    @classmethod
    def parse_filter_for_run(cls, filter_string):
        return cls._parse_search_filter(filter_string, cls.VALID_SEARCH_RUN_ATTRIBUTE_KEYS, True)

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
                    "not one of '%s" % (comparator, cls.VALID_TAG_COMPARATORS),
                    error_code=INVALID_PARAMETER_VALUE,
                )
            return True
        return False

    @classmethod
    def is_attribute(cls, key_type, comparator):
        if key_type == cls._ATTRIBUTE_IDENTIFIER:
            if comparator not in cls.VALID_STRING_ATTRIBUTE_COMPARATORS:
                raise MlflowException(
                    "Invalid comparator '{}' not one of "
                    "'{}".format(comparator, cls.VALID_STRING_ATTRIBUTE_COMPARATORS),
                    error_code=INVALID_PARAMETER_VALUE,
                )
            return True
        return False

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
        elif cls.is_attribute(key_type, comparator):
            lhs = getattr(run.info, key)
        else:
            raise MlflowException(
                "Invalid search expression type '%s'" % key_type,
                error_code=INVALID_PARAMETER_VALUE,
            )
        if lhs is None:
            return False

        if comparator in cls.CASE_INSENSITIVE_STRING_COMPARISON_OPERATORS:
            # Change value from sql syntax to regex syntax
            if comparator == "ILIKE":
                value = value.lower()
                lhs = lhs.lower()
            if not value.startswith("%"):
                value = "^" + value
            if not value.endswith("%"):
                value = value + "$"
            value = value.replace("_", ".").replace("%", ".*")
            return cls.filter_ops.get(comparator)(value, lhs)

        elif comparator in cls.filter_ops.keys():
            return cls.filter_ops.get(comparator)(lhs, value)
        else:
            return False

    @classmethod
    def filter_runs(cls, runs, filter_string):
        """Filters a set of runs based on a search filter string."""
        if not filter_string:
            return runs
        parsed = cls.parse_filter_for_run(filter_string)

        def run_matches(run):
            return all([cls._does_run_match_clause(run, s) for s in parsed])

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
            and all([token.is_whitespace for token in statement.tokens[1:-1]])
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
        identifier = cls._get_identifier(
            token_value.strip(), cls.VALID_ORDER_BY_RUN_ATTRIBUTE_KEYS, True
        )
        return identifier["type"], identifier["key"], is_ascending

    @classmethod
    def parse_order_by_for_search_registered_models(cls, order_by):
        token_value, is_ascending = cls._parse_order_by_string(order_by)
        identifier = cls._get_identifier(
            token_value.strip(), cls.VALID_ORDER_BY_REGISTERED_MODEL_ATTRIBUTE_KEYS, False,
        )
        # last updated timestamp field in search registered model has alias,
        # we want to map those alias to the correct db key
        if identifier["key"] in SearchUtils.VALID_TIMESTAMP_ORDER_BY_KEYS:
            identifier["key"] = SqlRegisteredModel.last_updated_time.key
        return identifier["type"], identifier["key"], is_ascending

    @classmethod
    def _get_subquery_entity(cls, key_type, comparator, is_search_run):
        if is_search_run:
            if cls.is_metric(key_type, comparator):  # any valid comparator
                return SqlLatestMetric
            elif cls.is_tag(key_type, comparator):
                return SqlTag
            elif cls.is_param(key_type, comparator):
                return SqlParam
            elif cls.is_attribute(key_type, comparator):
                return None
            else:
                raise MlflowException(
                    "Invalid identifier type '%s'" % key_type, error_code=INVALID_PARAMETER_VALUE,
                )
        else:
            if cls.is_tag(key_type, comparator):
                return SqlRegisteredModelTag
            elif cls.is_attribute(key_type, comparator):
                return None
            else:
                raise MlflowException(
                    "Invalid identifier type '%s'" % key_type, error_code=INVALID_PARAMETER_VALUE,
                )

    @classmethod
    def _get_order_by_clauses(cls, order_by_list, session, is_search_run):
        """Sorts a set of runs based on their natural ordering and an overriding set of order_bys.
        Runs are naturally ordered first by start time descending, then by run id for tie-breaking.
        """
        clauses = []
        ordering_joins = []
        clause_id = 0
        observed_order_by_clauses = set()
        # contrary to filters, it is not easily feasible to separately handle sorting
        # on attributes and on joined tables as we must keep all clauses in the same order
        if order_by_list:
            for order_by_clause in order_by_list:
                clause_id += 1
                (key_type, key, ascending) = (
                    cls.parse_order_by_for_search_runs(order_by_clause)
                    if is_search_run
                    else cls.parse_order_by_for_search_registered_models(order_by_clause)
                )
                if cls.is_attribute(key_type, "="):
                    order_value = (
                        getattr(SqlRun, SqlRun.get_attribute_name(key))
                        if is_search_run
                        else getattr(SqlRegisteredModel, key)
                    )
                    clauses.append(
                        sql.case([(order_value.is_(None), 1)], else_=0).label(
                            "clause_%s" % clause_id
                        )
                    )
                else:
                    entity = cls._get_subquery_entity(key_type, "=", is_search_run)
                    # build a subquery first because we will join it in the main request so that the
                    # metric we want to sort on is available when we apply the sorting clause
                    subquery = session.query(entity).filter(entity.key == key).subquery()
                    ordering_joins.append(subquery)
                    order_value = subquery.c.value
                    # sqlite does not support NULLS LAST expression, so we sort first by
                    # presence of the field (and is_nan for metrics), then by actual value
                    # As the subqueries are created independently and used later in the
                    # same main query, the CASE WHEN columns need to have unique names to
                    # avoid ambiguity
                    if is_search_run and cls.is_metric(key_type, "="):
                        clauses.append(
                            sql.case(
                                [(subquery.c.is_nan.is_(True), 1), (order_value.is_(None), 1)],
                                else_=0,
                            ).label("clause_%s" % clause_id)
                        )
                    else:  # other entities do not have an 'is_nan' field
                        clauses.append(
                            sql.case([(order_value.is_(None), 1)], else_=0).label(
                                "clause_%s" % clause_id
                            )
                        )

                if (key_type, key) in observed_order_by_clauses:
                    raise MlflowException(
                        "`order_by` contains duplicate fields: {}".format(order_by_list)
                    )
                observed_order_by_clauses.add((key_type, key))

                if ascending:
                    clauses.append(order_value)
                else:
                    clauses.append(order_value.desc())
        if is_search_run:
            if (
                SearchUtils._ATTRIBUTE_IDENTIFIER,
                SqlRun.start_time.key,
            ) not in observed_order_by_clauses:
                clauses.append(SqlRun.start_time.desc())
            clauses.append(SqlRun.run_uuid)
        else:
            if (
                SearchUtils._ATTRIBUTE_IDENTIFIER,
                SqlRegisteredModel.name.key,
            ) not in observed_order_by_clauses:
                clauses.append(SqlRegisteredModel.name.asc())
        return clauses, ordering_joins

    @classmethod
    def get_order_by_clauses_for_run(cls, order_by_list, session):
        return cls._get_order_by_clauses(order_by_list, session, True)

    @classmethod
    def get_order_by_clause_for_registered_model(cls, order_by_list, session):
        return cls._get_order_by_clauses(order_by_list, session, False)

    @classmethod
    def _to_sqlalchemy_filtering_statement(cls, sql_statement, session, is_search_run):
        key_type = sql_statement.get("type")
        key_name = sql_statement.get("key")
        value = sql_statement.get("value")
        comparator = sql_statement.get("comparator").upper()
        if SearchUtils.is_attribute(key_type, comparator):
            return None
        entity = cls._get_subquery_entity(key_type, comparator, is_search_run)
        if is_search_run and cls.is_metric(key_type, comparator):
            value = float(value)
        if comparator in cls.CASE_INSENSITIVE_STRING_COMPARISON_OPERATORS:
            op = cls.get_sql_filter_ops(entity.value, comparator)
            return session.query(entity).filter(entity.key == key_name, op(value)).subquery()
        elif comparator in cls.filter_ops:
            op = cls.filter_ops.get(comparator)
            return (
                session.query(entity)
                .filter(entity.key == key_name, op(entity.value, value))
                .subquery()
            )
        else:
            return None

    @classmethod
    def _get_sqlalchemy_filter_clauses(cls, parsed, session, is_search_run):
        """creates SqlAlchemy sub-queries
        that will be inner-joined to attribute table to act as multi-clause filters."""
        filters = []
        for sql_statement in parsed:
            filter_query = cls._to_sqlalchemy_filtering_statement(
                sql_statement, session, is_search_run
            )
            if filter_query is not None:
                filters.append(filter_query)
        return filters

    @classmethod
    def get_sqlalchemy_filter_clause_for_run(cls, parsed, session):
        return cls._get_sqlalchemy_filter_clauses(parsed, session, True)

    @classmethod
    def get_sqlalchemy_filter_clause_for_registered_model(cls, parsed, session):
        return cls._get_sqlalchemy_filter_clauses(parsed, session, False)

    @classmethod
    def _get_attributes_filtering_clauses(cls, parsed, is_search_run):
        clauses = []
        for sql_statement in parsed:
            key_type = sql_statement.get("type")
            key_name = sql_statement.get("key")
            value = sql_statement.get("value")
            comparator = sql_statement.get("comparator").upper()
            if cls.is_attribute(key_type, comparator):
                # key_name is guaranteed to be a valid searchable attribute of entities.RunInfo
                # by the call to parse_search_filter
                if is_search_run:
                    attribute = getattr(SqlRun, SqlRun.get_attribute_name(key_name))
                else:
                    attribute = getattr(SqlRegisteredModel, key_name)
                if comparator in cls.CASE_INSENSITIVE_STRING_COMPARISON_OPERATORS:
                    op = cls.get_sql_filter_ops(attribute, comparator)
                    clauses.append(op(value))
                elif comparator in cls.filter_ops:
                    op = cls.filter_ops.get(comparator)
                    clauses.append(op(attribute, value))
        return clauses

    @classmethod
    def get_attributes_filtering_clauses_for_run(cls, parsed):
        return cls._get_attributes_filtering_clauses(parsed, True)

    @classmethod
    def get_attributes_filtering_clauses_for_registered_model(cls, parsed):
        return cls._get_attributes_filtering_clauses(parsed, False)

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
                "Invalid order_by entity type '%s'" % key_type, error_code=INVALID_PARAMETER_VALUE,
            )

        # Return a key such that None values are always at the end.
        is_null_or_nan = sort_value is None or (
            isinstance(sort_value, float) and math.isnan(sort_value)
        )
        if ascending:
            return (is_null_or_nan, sort_value)
        return (not is_null_or_nan, sort_value)

    @classmethod
    def sort_runs(cls, runs, order_by_list):
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
                "Invalid page token, could not base64-decode", error_code=INVALID_PARAMETER_VALUE,
            )
        except base64.binascii.Error:
            raise MlflowException(
                "Invalid page token, could not base64-decode", error_code=INVALID_PARAMETER_VALUE,
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
    def paginate_runs(cls, runs, page_token, max_results):
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
        return paginated_runs, next_page_token

    @classmethod
    def _get_comparison_for_model_registry(cls, comparison, valid_search_keys):
        stripped_comparison = [token for token in comparison.tokens if not token.is_whitespace]
        cls._validate_comparison(stripped_comparison)
        key = stripped_comparison[0].value
        if key not in valid_search_keys:
            raise MlflowException(
                "Invalid attribute key '{}' specified. Valid keys "
                " are '{}'".format(key, valid_search_keys),
                error_code=INVALID_PARAMETER_VALUE,
            )
        value_token = stripped_comparison[2]
        if value_token.ttype not in cls.STRING_VALUE_TYPES:
            raise MlflowException(
                "Expected a quoted string value for attributes. "
                "Got value {value}".format(value=value_token.value),
                error_code=INVALID_PARAMETER_VALUE,
            )
        comp = {
            "key": key,
            "comparator": stripped_comparison[1].value,
            "value": cls._strip_quotes(value_token.value, expect_quoted_value=True),
        }
        return comp

    @classmethod
    def _parse_filter_for_model_registry(cls, filter_string, valid_search_keys):
        if not filter_string or filter_string == "":
            return []
        expected = "Expected search filter with single comparison operator. e.g. name='myModelName'"
        try:
            parsed = sqlparse.parse(filter_string)
        except Exception:
            raise MlflowException(
                "Error while parsing filter '%s'. %s" % (filter_string, expected),
                error_code=INVALID_PARAMETER_VALUE,
            )
        if len(parsed) == 0 or not isinstance(parsed[0], Statement):
            raise MlflowException(
                "Invalid filter '%s'. Could not be parsed. %s" % (filter_string, expected),
                error_code=INVALID_PARAMETER_VALUE,
            )
        elif len(parsed) > 1:
            raise MlflowException(
                "Search filter '%s' contains multiple expressions. "
                "%s " % (filter_string, expected),
                error_code=INVALID_PARAMETER_VALUE,
            )
        statement = parsed[0]
        invalids = list(filter(cls._invalid_statement_token, statement.tokens))
        if len(invalids) > 0:
            invalid_clauses = ", ".join("'%s'" % token for token in invalids)
            raise MlflowException(
                "Invalid clause(s) in filter string: %s. " "%s" % (invalid_clauses, expected),
                error_code=INVALID_PARAMETER_VALUE,
            )
        return [
            cls._get_comparison_for_model_registry(si, valid_search_keys)
            for si in statement.tokens
            if isinstance(si, Comparison)
        ]

    @classmethod
    def parse_filter_for_model_versions(cls, filter_string):
        return cls._parse_filter_for_model_registry(
            filter_string, cls.VALID_SEARCH_MODEL_VERSION_ATTRIBUTE_KEYS
        )

    @classmethod
    def parse_filter_for_registered_models(cls, filter_string):
        return cls._parse_search_filter(
            filter_string, cls.VALID_SEARCH_REGISTERED_MODEL_ATTRIBUTE_KEYS, False
        )
