import pytest

from mlflow.exceptions import MlflowException
from mlflow.utils.search_logged_model_utils import EntityType, parse_filter_string


@pytest.mark.parametrize(
    "filter_string",
    [
        "metrics.loss > abc",
        "params.foo IN ('a', xyz)",
    ],
)
def test_parse_filter_string_rejects_malformed_values_as_invalid_parameter(filter_string):
    with pytest.raises(MlflowException, match="Invalid filter string") as exc:
        parse_filter_string(filter_string)

    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    # The original conversion error must be chained, not swallowed.
    assert isinstance(exc.value.__cause__, (ValueError, TypeError, SyntaxError))


def test_parse_filter_string_accepts_valid_values():
    assert parse_filter_string("") == []

    [numeric] = parse_filter_string("metrics.loss > 0.5")
    assert numeric.entity.type == EntityType.METRIC
    assert numeric.value == 0.5

    [string_value] = parse_filter_string("params.model = 'LR'")
    assert string_value.value == "LR"

    [in_list] = parse_filter_string("params.foo IN ('a', 'b')")
    assert in_list.value == ("a", "b")
