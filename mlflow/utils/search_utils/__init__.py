from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.search_utils.models import Comparison, KeyType
from mlflow.utils.search_utils.parse import parse_filter_string, _key_type_from_string, \
    _comparison_operator_from_string


class SearchFilter(object):

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
