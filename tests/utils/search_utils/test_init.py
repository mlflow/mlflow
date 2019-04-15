import pytest
import mock

import mlflow.utils.search_utils
from mlflow.utils.search_utils import SearchFilter
from mlflow.exceptions import MlflowException


COMPARISON_TEST_CASES = [
    ([], True),
    ([True, True], True),
    ([True, False, True], False),
    ([False, False], False)
]


@pytest.mark.parametrize("comparison_results, expected_result", COMPARISON_TEST_CASES)
def test_search_filter_filter_string(comparison_results, expected_result):

    filter_string = mock.Mock()
    comparisons = [mock.Mock(filter=mock.Mock(return_value=r)) for r in comparison_results]
    run = mock.Mock()

    search_filter = SearchFilter(filter_string)

    with mock.patch("mlflow.utils.search_utils.parse_filter_string", return_value=comparisons):
        assert search_filter.filter(run) is expected_result
        mlflow.utils.search_utils.parse_filter_string.assert_called_once_with(filter_string)

    for comparison in comparisons:
        comparison.filter.assert_called_once_with(run)


@pytest.mark.parametrize("comparison_results, expected_result", COMPARISON_TEST_CASES)
def test_search_filter_anded_expressions(comparison_results, expected_result):

    anded_expressions = [mock.Mock() for _ in comparison_results]
    comparisons = [mock.Mock(filter=mock.Mock(return_value=r)) for r in comparison_results]
    run = mock.Mock()

    search_filter = SearchFilter(anded_expressions=anded_expressions)

    with mock.patch("mlflow.utils.search_utils.search_expression_to_comparison",
                    side_effect=comparisons):
        assert search_filter.filter(run) is expected_result
        assert mlflow.utils.search_utils.search_expression_to_comparison.call_args_list == [
            mock.call(expression) for expression in anded_expressions
        ]

    for comparison in comparisons:
        comparison.filter.assert_called_once_with(run)


def test_search_filter_both():
    with pytest.raises(MlflowException,
                       match="Can specify only one of 'filter' or 'search_expression'"):
        SearchFilter(filter_string=mock.Mock(), anded_expressions=mock.Mock())
