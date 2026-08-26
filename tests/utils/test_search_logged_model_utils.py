import pytest

from mlflow.exceptions import MlflowException
from mlflow.utils.search_logged_model_utils import parse_filter_string


def test_invalid_operator_message_lists_the_applicable_operators():
    # `LIKE` is rejected for a numeric entity, so it must not appear in the
    # list of operators the message offers as alternatives.
    with pytest.raises(MlflowException, match="Invalid comparison operator") as exc:
        parse_filter_string("metrics.loss LIKE 'x'")

    message = str(exc.value)
    assert "'<'" in message
    assert "LIKE" not in message.split("Expected one of")[1]


def test_quoted_value_that_looks_like_a_tuple_stays_a_string():
    (comparison,) = parse_filter_string("params.shape = '(1, 2)'")
    assert comparison.value == "(1, 2)"


def test_in_list_is_still_parsed_as_a_tuple():
    (comparison,) = parse_filter_string("params.shape IN ('a', 'b')")
    assert comparison.value == ("a", "b")
