from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.search_utils.parse import parse_filter_string, search_expression_to_comparison


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

    def comparisons(self):
        if self.parsed is None:
            if self._filter_string:
                self.parsed = parse_filter_string(self._filter_string)
            elif self._search_expressions:
                self.parsed = [search_expression_to_comparison(search_expression)
                               for search_expression in self._search_expressions]
            else:
                self.parsed = []
        return self.parsed

    def filter(self, run):
        return all([comparison.filter(run) for comparison in self.comparisons()])
