import sqlparse
from sqlparse.sql import Identifier, Token, Comparison, Statement
from sqlparse.tokens import Token as TokenType

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


class SearchFilter(object):
    VALID_METRIC_COMPARATORS = set(['>', '>=', '!=', '=', '<', '<='])
    VALID_PARAM_COMPARATORS = set(['!=', '='])
    VALID_TAG_COMPARATORS = set(['!=', '='])
    _METRIC_IDENTIFIER = "metric"
    _ALTERNATE_METRIC_IDENTIFIERS = set(["metrics"])
    _PARAM_IDENTIFIER = "parameter"
    _ALTERNATE_PARAM_IDENTIFIERS = set(["param", "params"])
    _TAG_IDENTIFIER = "tag"
    _ALTERNATE_TAG_IDENTIFIERS = set(["tags"])
    VALID_KEY_TYPE = set([_METRIC_IDENTIFIER] + list(_ALTERNATE_METRIC_IDENTIFIERS)
                         + [_PARAM_IDENTIFIER] + list(_ALTERNATE_PARAM_IDENTIFIERS)
                         + [_TAG_IDENTIFIER] + list(_ALTERNATE_TAG_IDENTIFIERS))
    STRING_VALUE_TYPES = set([TokenType.Literal.String.Single])
    NUMERIC_VALUE_TYPES = set([TokenType.Literal.Number.Integer, TokenType.Literal.Number.Float])

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
        if entity_type not in cls.VALID_KEY_TYPE:
            raise MlflowException("Invalid search expression type '%s'. "
                                  "Valid values are '%s" % (entity_type, cls.VALID_KEY_TYPE),
                                  error_code=INVALID_PARAMETER_VALUE)

        if entity_type in cls._ALTERNATE_PARAM_IDENTIFIERS:
            return cls._PARAM_IDENTIFIER
        elif entity_type in cls._ALTERNATE_METRIC_IDENTIFIERS:
            return cls._METRIC_IDENTIFIER
        elif entity_type in cls._ALTERNATE_TAG_IDENTIFIERS:
            return cls._TAG_IDENTIFIER
        else:
            # either "metric" or "parameter", since valid type
            return entity_type

    @classmethod
    def _get_identifier(cls, identifier):
        try:
            entity_type, key = identifier.split(".", 1)
        except ValueError:
            raise MlflowException("Invalid filter string '%s'. Filter comparison is expected as "
                                  "'metric.<key> <comparator> <value>', "
                                  "'tag.<key> <comparator> <value>', or"
                                  "'params.<key> <comparator> <value>'." % identifier,
                                  error_code=INVALID_PARAMETER_VALUE)
        return {"type": cls._valid_entity_type(entity_type), "key": cls._strip_quotes(key)}

    @classmethod
    def _get_value(cls, identifier_type, token):
        if identifier_type == cls._METRIC_IDENTIFIER:
            if token.ttype not in cls.NUMERIC_VALUE_TYPES:
                raise MlflowException("Expected numeric value type for metric. "
                                      "Found {}".format(token.value),
                                      error_code=INVALID_PARAMETER_VALUE)
            return token.value
        elif identifier_type == cls._PARAM_IDENTIFIER or identifier_type == cls._TAG_IDENTIFIER:
            if token.ttype in cls.STRING_VALUE_TYPES or isinstance(token, Identifier):
                return cls._strip_quotes(token.value, expect_quoted_value=True)
            raise MlflowException("Expected a quoted string value for "
                                  "{identifier_type} (e.g. 'my-value'). Got value "
                                  "{value}".format(identifier_type=identifier_type,
                                                   value=token.value),
                                  error_code=INVALID_PARAMETER_VALUE)
        else:
            # Expected to be either "param" or "metric".
            raise MlflowException("Invalid identifier type. Expected one of "
                                  "{}.".format([cls._METRIC_IDENTIFIER, cls._PARAM_IDENTIFIER]))

    @classmethod
    def _validate_comparison(cls, tokens):
        base_error_string = "Invalid comparison clause"
        if len(tokens) != 3:
            raise MlflowException("{}. Expected 3 tokens found {}".format(base_error_string,
                                                                          len(tokens)),
                                  error_code=INVALID_PARAMETER_VALUE)
        if not isinstance(tokens[0], Identifier):
            raise MlflowException("{}. Expected 'Identifier' found '{}'".format(base_error_string,
                                                                                str(tokens[0])),
                                  error_code=INVALID_PARAMETER_VALUE)
        if not isinstance(tokens[1], Token) and tokens[1].ttype != TokenType.Operator.Comparison:
            raise MlflowException("{}. Expected comparison found '{}'".format(base_error_string,
                                                                              str(tokens[1])),
                                  error_code=INVALID_PARAMETER_VALUE)
        if not isinstance(tokens[2], Token) and \
                (tokens[2].ttype not in cls.STRING_VALUE_TYPES.union(cls.NUMERIC_VALUE_TYPES) or
                 isinstance(tokens[2], Identifier)):
            raise MlflowException("{}. Expected value token found '{}'".format(base_error_string,
                                                                               str(tokens[2])),
                                  error_code=INVALID_PARAMETER_VALUE)

    @classmethod
    def _get_comparison(cls, comparison):
        stripped_comparison = [token for token in comparison.tokens if not token.is_whitespace]
        cls._validate_comparison(stripped_comparison)
        comp = cls._get_identifier(stripped_comparison[0].value)
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
    def _process_statement(cls, statement):
        # check validity
        invalids = list(filter(cls._invalid_statement_token, statement.tokens))
        if len(invalids) > 0:
            invalid_clauses = ", ".join("'%s'" % token for token in invalids)
            raise MlflowException("Invalid clause(s) in filter string: %s" % invalid_clauses,
                                  error_code=INVALID_PARAMETER_VALUE)
        return [cls._get_comparison(si) for si in statement.tokens if isinstance(si, Comparison)]

    @classmethod
    def search_expression_to_dict(cls, search_expression):
        key_type = search_expression.WhichOneof('expression')
        if key_type == cls._METRIC_IDENTIFIER:
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
            return {
                "type": cls._METRIC_IDENTIFIER,
                "key": key,
                "comparator": comparator,
                "value": value
            }
        elif key_type == cls._PARAM_IDENTIFIER:
            key = search_expression.parameter.key
            comparator = search_expression.parameter.string.comparator
            value = search_expression.parameter.string.value
            return {
                "type": cls._PARAM_IDENTIFIER,
                "key": key,
                "comparator": comparator,
                "value": value
            }
        else:
            raise MlflowException("Invalid search expression type '%s'" % key_type,
                                  error_code=INVALID_PARAMETER_VALUE)

    def _parse(self):
        if self._filter_string:
            try:
                parsed = sqlparse.parse(self._filter_string)
            except Exception:
                raise MlflowException("Error on parsing filter '%s'" % self._filter_string,
                                      error_code=INVALID_PARAMETER_VALUE)
            if len(parsed) == 0 or not isinstance(parsed[0], Statement):
                raise MlflowException("Invalid filter '%s'. Could not be parsed." %
                                      self._filter_string, error_code=INVALID_PARAMETER_VALUE)
            elif len(parsed) > 1:
                raise MlflowException("Search filter contained multiple expression '%s'. "
                                      "Provide AND-ed expression list." % self._filter_string,
                                      error_code=INVALID_PARAMETER_VALUE)
            return self._process_statement(parsed[0])
        elif self._search_expressions:
            return [self.search_expression_to_dict(se) for se in self._search_expressions]
        else:
            return []

    @classmethod
    def does_run_match_clause(cls, run, sed):
        key_type = sed.get('type')
        key = sed.get('key')
        value = sed.get('value')
        comparator = sed.get('comparator')
        if key_type == cls._METRIC_IDENTIFIER:
            if comparator not in cls.VALID_METRIC_COMPARATORS:
                raise MlflowException("Invalid comparator '%s' "
                                      "not one of '%s" % (comparator,
                                                          cls.VALID_METRIC_COMPARATORS),
                                      error_code=INVALID_PARAMETER_VALUE)
            metric = next((m for m in run.data.metrics if m.key == key), None)
            lhs = metric.value if metric else None
            value = float(value)
        elif key_type == cls._PARAM_IDENTIFIER:
            if comparator not in cls.VALID_PARAM_COMPARATORS:
                raise MlflowException("Invalid comparator '%s' "
                                      "not one of '%s" % (comparator, cls.VALID_PARAM_COMPARATORS),
                                      error_code=INVALID_PARAMETER_VALUE)
            param = next((p for p in run.data.params if p.key == key), None)
            lhs = param.value if param else None
        elif key_type == cls._TAG_IDENTIFIER:
            if comparator not in cls.VALID_TAG_COMPARATORS:
                raise MlflowException("Invalid comparator '%s' "
                                      "not one of '%s" % (comparator, cls.VALID_TAG_COMPARATORS))
            tag = next((t for t in run.data.tags if t.key == key), None)
            lhs = tag.value if tag else None
        else:
            raise MlflowException("Invalid search expression type '%s'" % key_type,
                                  error_code=INVALID_PARAMETER_VALUE)
        if lhs is None:
            return False
        elif comparator == '>':
            return lhs > value
        elif comparator == '>=':
            return lhs >= value
        elif comparator == '=':
            return lhs == value
        elif comparator == '!=':
            return lhs != value
        elif comparator == '<=':
            return lhs <= value
        elif comparator == '<':
            return lhs < value
        else:
            return False

    def filter(self, run):
        if not self.parsed:
            self.parsed = self._parse()
        return all([self.does_run_match_clause(run, s) for s in self.parsed])
