import sqlparse
from sqlparse.sql import Identifier, Token, Comparison, Statement
from sqlparse.tokens import Token as TokenType

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE


class SearchFilter(object):
    VALID_METRIC_COMPARATORS = set(['>', '>=', '!=', '=', '<', '<='])
    VALID_PARAM_COMPARATORS = set(['!=', '='])
    _METRIC_IDENTIFIER = "metric"
    _ALTERNATE_METRIC_IDENTIFIERS = set(["metrics"])
    _PARAM_IDENTIFIER = "parameter"
    _ALTERNATE_PARAM_IDENTIFIERS = set(["param", "params"])
    VALID_KEY_TYPE = set([_METRIC_IDENTIFIER] + list(_ALTERNATE_METRIC_IDENTIFIERS)
                         + [_PARAM_IDENTIFIER] + list(_ALTERNATE_PARAM_IDENTIFIERS))
    VALUE_TYPES = set([TokenType.Literal.String.Single,
                       TokenType.Literal.Number.Integer,
                       TokenType.Literal.Number.Float])

    def __init__(self, search_runs=None):
        self._filter_string = search_runs.filter if search_runs else None
        self._search_expressions = search_runs.anded_expressions if search_runs else None
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
    def _trim_backticks(cls, entity_type):
        if entity_type.startswith("`"):
            assert entity_type.endswith("`")
            return entity_type[1:-1]
        return entity_type

    @classmethod
    def _valid_entity_type(cls, entity_type):
        entity_type = cls._trim_backticks(entity_type)
        if entity_type not in cls.VALID_KEY_TYPE:
            raise MlflowException("Invalid search expression type '%s'. "
                                  "Valid values are '%s" % (entity_type, cls.VALID_KEY_TYPE))

        if entity_type in cls._ALTERNATE_PARAM_IDENTIFIERS:
            return cls._PARAM_IDENTIFIER
        elif entity_type in cls._ALTERNATE_METRIC_IDENTIFIERS:
            return cls._METRIC_IDENTIFIER
        else:
            # either "metric" or "parameter", since valid type
            return entity_type

    @classmethod
    def _get_identifier(cls, identifier):
        try:
            entity_type, key = identifier.split(".")
        except ValueError:
            raise MlflowException("Invalid filter string '%s'. Filter comparison is expected as "
                                  "'metric.<key> <comparator> <value>' or"
                                  "'params.<key> <comparator> <value>'." % identifier,
                                  error_code=INVALID_PARAMETER_VALUE)
        return {"type": cls._valid_entity_type(entity_type), "key": key}

    @classmethod
    def _process_token(cls, token):
        if token.ttype == TokenType.Operator.Comparison:
            return {"comparator": token.value}
        elif token.ttype in cls.VALUE_TYPES:
            return {"value": token.value}
        else:
            return {}

    @classmethod
    def _get_comparison(cls, comparison):
        comp = {}
        for t in comparison.tokens:
            if isinstance(t, Identifier):
                comp.update(cls._get_identifier(t.value))
            elif isinstance(t, Token):
                comp.update(cls._process_token(t))
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
            raise MlflowException("Invalid clause(s) in filter string: %s" % invalid_clauses)
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
                raise MlflowException("Invalid metric type: '%s', expected float or double")
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
            raise MlflowException("Invalid search expression type '%s'" % key_type)

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
                                      "not one of '%s" % (comparator, cls.VALID_METRIC_COMPARATORS))
            metric = next((m for m in run.data.metrics if m.key == key), None)
            lhs = metric.value if metric else None
            value = float(value)
        elif key_type == cls._PARAM_IDENTIFIER:
            if comparator not in cls.VALID_PARAM_COMPARATORS:
                raise MlflowException("Invalid comparator '%s' "
                                      "not one of '%s" % (comparator, cls.VALID_PARAM_COMPARATORS))
            param = next((p for p in run.data.params if p.key == key), None)
            lhs = param.value if param else None
        else:
            raise MlflowException("Invalid search expression type '%s'" % key_type)

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
