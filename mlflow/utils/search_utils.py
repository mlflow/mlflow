from enum import Enum

import sqlparse
from sqlparse.sql import Identifier as SqlIdentifier, Token as SqlToken, \
    Comparison as SqlComparison, Statement as SqlStatement
from sqlparse.tokens import Token as SqlTokenType

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


class KeyType(Enum):
    METRIC = "metric"
    PARAM = "param"
    TAG = "tag"


KEY_TYPE_FROM_IDENTIFIER = {
    "metric": KeyType.METRIC,
    "metrics": KeyType.METRIC,
    "param": KeyType.PARAM,
    "params": KeyType.PARAM,
    "parameter": KeyType.PARAM,
    "tag": KeyType.TAG,
    "tags": KeyType.TAG
}


class ComparisonOperator(Enum):
    EQUAL = "="
    NOT_EQUAL = "!="
    GREATER_THAN = ">"
    GREATER_THAN_EQUAL = ">="
    LESS_THAN = "<"
    LESS_THAN_EQUAL = "<="


VALID_OPERATORS_FOR_KEY_TYPE = {
    KeyType.METRIC: set(ComparisonOperator),
    KeyType.PARAM: {ComparisonOperator.EQUAL, ComparisonOperator.NOT_EQUAL},
    KeyType.TAG: {ComparisonOperator.EQUAL, ComparisonOperator.NOT_EQUAL},
}


class Comparison(object):
    def __init__(self, key_type, key, operator, value):
        self.key_type = key_type
        self.key = key
        self.operator = operator
        self.value = value

    def __eq__(self, other):
        if not isinstance(other, Comparison):
            return False
        return (self.key_type == other.key_type and self.key == other.key and
                self.operator == other.operator and self.value == other.value)

    def __repr__(self):
        return "{}({}, {}, {}, {})".format(self.__class__.__name__, self.key_type, self.key,
                                           self.operator, self.value)

    def filter(self, run):
        valid_operators = VALID_OPERATORS_FOR_KEY_TYPE[self.key_type]
        if self.operator not in valid_operators:
            message = "Invalid comparator '{}' not one of '{}".format(self.operator.value,
                                                                      valid_operators)
            raise MlflowException(message, error_code=INVALID_PARAMETER_VALUE)
        value = float(self.value) if self.key_type == KeyType.METRIC else self.value
        lhs = _get_run_value(run, self.key_type, self.key)
        if lhs is None:
            return False
        elif self.operator == ComparisonOperator.GREATER_THAN:
            return lhs > value
        elif self.operator == ComparisonOperator.GREATER_THAN_EQUAL:
            return lhs >= value
        elif self.operator == ComparisonOperator.EQUAL:
            return lhs == value
        elif self.operator == ComparisonOperator.NOT_EQUAL:
            return lhs != value
        elif self.operator == ComparisonOperator.LESS_THAN_EQUAL:
            return lhs <= value
        elif self.operator == ComparisonOperator.LESS_THAN:
            return lhs < value
        else:
            return False


def _key_type_from_string(string):
    try:
        return KEY_TYPE_FROM_IDENTIFIER[string]
    except KeyError:
        message = "Invalid search expression type '{}'. Valid values are {}".format(
            string, set(KEY_TYPE_FROM_IDENTIFIER.keys()))
        raise MlflowException(message, error_code=INVALID_PARAMETER_VALUE)


def _comparison_operator_from_string(string):
    try:
        return ComparisonOperator(string)
    except ValueError:
        raise MlflowException("Invalid comparator '{}'".format(string),
                              error_code=INVALID_PARAMETER_VALUE)


def _get_run_value(run, key_type, key):
    if key_type == KeyType.METRIC:
        entities_to_search = run.data.metrics
    elif key_type == KeyType.PARAM:
        entities_to_search = run.data.params
    elif key_type == KeyType.TAG:
        entities_to_search = run.data.tags
    else:
        raise ValueError("Invalid key type: {}".format(key_type))

    matching_entity = next((e for e in entities_to_search if e.key == key), None)
    return matching_entity.value if matching_entity else None


INVALID_IDENTIFIER_TPL = (
    "Invalid comparison clause '{clause}'. "
    "Expected param, metric or tag identifier of format 'metric.<key> <comparator> <value>', "
    "'tag.<key> <comparator> <value>', or 'params.<key> <comparator> <value>' but found {token}."
)


def _comparison_from_sql_comparison(comparison):
    """
    Interpret a SQL comparison from  a filter string.

    :param sql_comparison: A sqlparse.sql.Comparison object.

    :return: A Comparison object.
    """

    stripped_comparison = [token for token in comparison.tokens if not token.is_whitespace]

    try:
        [identifier, operator, value] = stripped_comparison
    except ValueError:
        message = "Invalid comparison clause '{}'. Expected 3 tokens but found {}".format(
            comparison.value, len(stripped_comparison))
        raise MlflowException(message, error_code=INVALID_PARAMETER_VALUE)

    if not isinstance(identifier, SqlIdentifier):
        raise MlflowException(
            INVALID_IDENTIFIER_TPL.format(clause=comparison.value, token=identifier.value),
            error_code=INVALID_PARAMETER_VALUE
        )

    try:
        key_type, key = identifier.value.split(".", 1)
    except ValueError:
        raise MlflowException(
            INVALID_IDENTIFIER_TPL.format(clause=comparison.value, token=identifier.value),
            error_code=INVALID_PARAMETER_VALUE
        )

    if not isinstance(operator, SqlToken) and operator.ttype != SqlTokenType.Operator.Comparison:
        message = "Invalid comparison clause '{}'. Expected operator but found '{}'".format(
            comparison.value, operator.value)
        raise MlflowException(message, error_code=INVALID_PARAMETER_VALUE)

    if not isinstance(value, SqlToken) and \
            (value.ttype not in SearchFilter.STRING_VALUE_TYPES.union(SearchFilter.NUMERIC_VALUE_TYPES) or
             isinstance(value, SqlIdentifier)):
        message = "Invalid comparison clause '{}'. Expected value but found '{}'".format(
            comparison.value, value.value)
        raise MlflowException(message, error_code=INVALID_PARAMETER_VALUE)

    key_type = SearchFilter._valid_entity_type(key_type)
    key = SearchFilter._strip_quotes(key)
    operator = _comparison_operator_from_string(operator.value)
    value = SearchFilter._get_value(key_type, value)
    return Comparison(key_type, key, operator, value)


def parse_filter_string(string):
    try:
        parsed = sqlparse.parse(string)
    except Exception:
        raise MlflowException("Error on parsing filter '{}'".format(string),
                              error_code=INVALID_PARAMETER_VALUE)

    try:
        [statement] = parsed
    except ValueError:
        raise MlflowException("Invalid filter '{}'. Must be a single statement.".format(string),
                              error_code=INVALID_PARAMETER_VALUE)

    if not isinstance(statement, SqlStatement):
        raise MlflowException("Invalid filter '{}'. Must be a single statement.".format(string),
                              error_code=INVALID_PARAMETER_VALUE)

    return SearchFilter._process_statement(statement)


class SearchFilter(object):
    STRING_VALUE_TYPES = set([SqlTokenType.Literal.String.Single])
    NUMERIC_VALUE_TYPES = set([SqlTokenType.Literal.Number.Integer,
                               SqlTokenType.Literal.Number.Float])

    def __init__(self, filter_string=None, anded_expressions=None):
        self._filter_string = filter_string
        self._search_expressions = anded_expressions
        if self._filter_string and self._search_expressions:
            raise MlflowException("Can specify only one of 'filter' or 'search_expression'",
                                  error_code=INVALID_PARAMETER_VALUE)

        # lazy parsing
        self.parsed = None

    @property
    def filter_string(self):
        return self._filter_string

    @property
    def search_expressions(self):
        return self._search_expressions

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
            raise MlflowException("Parameter value is either not quoted or unidentified quote "
                                  "types used for string value %s. Use either single or double "
                                  "quotes." % value, error_code=INVALID_PARAMETER_VALUE)
        else:
            return value

    @classmethod
    def _valid_entity_type(cls, entity_type):
        entity_type = cls._trim_backticks(entity_type)
        return _key_type_from_string(entity_type)

    @classmethod
    def _get_value(cls, identifier_type, token):
        if identifier_type == KeyType.METRIC:
            if token.ttype not in cls.NUMERIC_VALUE_TYPES:
                raise MlflowException("Expected numeric value type for metric. "
                                      "Found {}".format(token.value),
                                      error_code=INVALID_PARAMETER_VALUE)
            return token.value
        elif identifier_type == KeyType.PARAM or identifier_type == KeyType.TAG:
            if token.ttype in cls.STRING_VALUE_TYPES or isinstance(token, SqlIdentifier):
                return cls._strip_quotes(token.value, expect_quoted_value=True)
            raise MlflowException("Expected a quoted string value for "
                                  "{identifier_type} (e.g. 'my-value'). Got value "
                                  "{value}".format(identifier_type=identifier_type.value,
                                                   value=token.value),
                                  error_code=INVALID_PARAMETER_VALUE)
        else:
            raise MlflowException("Invalid identifier type. Expected one of "
                                  "{}.".format({t.value for t in KeyType}))

    @classmethod
    def _invalid_statement_token(cls, token):
        if isinstance(token, SqlComparison):
            return False
        elif token.is_whitespace:
            return False
        elif token.match(ttype=SqlTokenType.Keyword, values=["AND"]):
            return False
        else:
            return True

    @classmethod
    def _process_statement(cls, statement):
        # check validity
        invalids = list(filter(cls._invalid_statement_token, statement.tokens))
        if len(invalids) > 0:
            invalid_clauses = ", ".join("'%s'" % token for token in invalids)
            raise MlflowException("Invalid clause(s) in filter string: %s" % invalid_clauses,
                                  error_code=INVALID_PARAMETER_VALUE)
        return [_comparison_from_sql_comparison(token)
                for token in statement.tokens if isinstance(token, SqlComparison)]

    @classmethod
    def search_expression_to_comparison(cls, search_expression):
        key_type = _key_type_from_string(search_expression.WhichOneof('expression'))
        if key_type == KeyType.METRIC:
            key = search_expression.metric.key
            metric_type = search_expression.metric.WhichOneof('clause')
            if metric_type == 'float':
                comparator = search_expression.metric.float.comparator
                value = search_expression.metric.float.value
            elif metric_type == 'double':
                comparator = search_expression.metric.double.comparator
                value = search_expression.metric.double.value
            else:
                raise MlflowException("Invalid metric type: '%s', expected float or double",
                                      error_code=INVALID_PARAMETER_VALUE)
            return Comparison(KeyType.METRIC, key, _comparison_operator_from_string(comparator),
                              value)
        elif key_type == KeyType.PARAM:
            key = search_expression.parameter.key
            comparator = search_expression.parameter.string.comparator
            value = search_expression.parameter.string.value
            return Comparison(KeyType.PARAM, key, _comparison_operator_from_string(comparator),
                              value)
        else:
            raise MlflowException("Invalid search expression type '%s'" % key_type,
                                  error_code=INVALID_PARAMETER_VALUE)

    def _parse(self):
        if self._filter_string:
            return parse_filter_string(self._filter_string)
        elif self._search_expressions:
            return [self.search_expression_to_comparison(se) for se in self._search_expressions]
        else:
            return []

    def filter(self, run):
        if not self.parsed:
            self.parsed = self._parse()
        return all([comparison.filter(run) for comparison in self.parsed])
