import pytest

from mlflow.utils.search_logged_model_utils import parse_filter_string


@pytest.mark.parametrize(
    ("filter_string", "expected_op"),
    [
        ("attributes.name like 'model-%'", "LIKE"),
        ("attributes.name LIKE 'model-%'", "LIKE"),
        ("attributes.name Like 'model-%'", "LIKE"),
        ("attributes.name ilike 'MODEL-%'", "ILIKE"),
        ("attributes.name ILIKE 'MODEL-%'", "ILIKE"),
        ("attributes.name in ('model-a')", "IN"),
        ("attributes.name IN ('model-a')", "IN"),
        ("attributes.name In ('model-a')", "IN"),
        ("attributes.name not in ('other')", "NOT IN"),
        ("attributes.name NOT IN ('other')", "NOT IN"),
        ("params.lr like '0.%'", "LIKE"),
        ("tags.team in ('ml')", "IN"),
        ("metrics.loss > 0.1", ">"),
        ("metrics.loss = 0.1", "="),
    ],
)
def test_parse_filter_string_operators_are_case_insensitive(filter_string, expected_op):
    # https://github.com/mlflow/mlflow/issues/26216: lowercase operators must
    # behave like their uppercase forms on the SQL store.
    (comparison,) = parse_filter_string(filter_string)
    assert comparison.op == expected_op


def test_parse_filter_string_invalid_operator_still_rejected():
    # LIKE is invalid for numeric entities; must still be rejected.
    with pytest.raises(Exception, match="Invalid comparison operator"):
        parse_filter_string("metrics.loss like '0.1'")
