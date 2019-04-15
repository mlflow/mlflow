import pytest
import mock

from mlflow.utils.search_utils import SearchFilter
from mlflow.exceptions import MlflowException


def test_search_filter_basics():
    search_filter = "This is a filter string"
    anded_expressions = [mock.Mock(), mock.Mock()]

    # only anded_expressions
    SearchFilter(anded_expressions=anded_expressions)

    # only search filter
    SearchFilter(filter_string=search_filter)

    # both
    with pytest.raises(MlflowException,
                       match="Can specify only one of 'filter' or 'search_expression'"):
        SearchFilter(anded_expressions=anded_expressions, filter_string=search_filter)
