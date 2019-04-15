import pytest

from mlflow.exceptions import MlflowException
from mlflow.protos.service_pb2 import SearchExpression, DoubleClause, \
    MetricSearchExpression, FloatClause, ParameterSearchExpression, StringClause
from mlflow.utils.search_utils.models import KeyType, ComparisonOperator, Comparison
from mlflow.utils.search_utils.parse import parse_filter_string, search_expression_to_comparison


@pytest.mark.parametrize("filter_string, comparisons", [
    (
        "metric.acc >= 0.94",
        [Comparison(KeyType.METRIC, "acc", ComparisonOperator.GREATER_THAN_EQUAL, '0.94')]
    ),
    (
        "metric.acc>=100",
        [Comparison(KeyType.METRIC, "acc", ComparisonOperator.GREATER_THAN_EQUAL, '100')]
    ),
    (
        "params.m!='tf'",
        [Comparison(KeyType.PARAM, "m", ComparisonOperator.NOT_EQUAL, 'tf')]
    ),
    (
        'params."m"!="tf"',
        [Comparison(KeyType.PARAM, "m", ComparisonOperator.NOT_EQUAL, 'tf')]
    ),
    (
        'metric."legit name" >= 0.243',
        [Comparison(KeyType.METRIC, "legit name", ComparisonOperator.GREATER_THAN_EQUAL, '0.243')]
    ),
    (
        "metrics.XYZ = 3",
        [Comparison(KeyType.METRIC, "XYZ", ComparisonOperator.EQUAL, '3')]
    ),
    (
        'params."cat dog" = "pets"',
        [Comparison(KeyType.PARAM, "cat dog", ComparisonOperator.EQUAL, 'pets')]
    ),
    (
        'metrics."X-Y-Z" = 3',
        [Comparison(KeyType.METRIC, "X-Y-Z", ComparisonOperator.EQUAL, '3')]
    ),
    (
        'metrics."X//Y#$$@&Z" = 3',
        [Comparison(KeyType.METRIC, "X//Y#$$@&Z", ComparisonOperator.EQUAL, '3')]
    ),
    (
        "params.model = 'LinearRegression'",
        [Comparison(KeyType.PARAM, "model", ComparisonOperator.EQUAL, "LinearRegression")]
    ),
    (
        "metrics.rmse < 1 and params.model_class = 'LR'",
        [Comparison(KeyType.METRIC, "rmse", ComparisonOperator.LESS_THAN, '1'),
         Comparison(KeyType.PARAM, "model_class", ComparisonOperator.EQUAL, "LR")]
    ),
    (
        "`metric`.a >= 0.1",
        [Comparison(KeyType.METRIC, "a", ComparisonOperator.GREATER_THAN_EQUAL, '0.1')]
    ),
    (
        "`params`.model = 'LR'",
        [Comparison(KeyType.PARAM, "model", ComparisonOperator.EQUAL, "LR")]
    ),
    (
        "tags.version = 'commit-hash'",
        [Comparison(KeyType.TAG, "version", ComparisonOperator.EQUAL, "commit-hash")]
    ),
    (
        "`tags`.source_name = 'a notebook'",
        [Comparison(KeyType.TAG, "source_name", ComparisonOperator.EQUAL, "a notebook")]
    ),
    (
        'metrics."accuracy.2.0" > 5',
        [Comparison(KeyType.METRIC, "accuracy.2.0", ComparisonOperator.GREATER_THAN, '5')]
    ),
    (
        'params."p.a.r.a.m" != "a"',
        [Comparison(KeyType.PARAM, "p.a.r.a.m", ComparisonOperator.NOT_EQUAL, 'a')]
    ),
    (
        'tags."t.a.g" = "a"',
        [Comparison(KeyType.TAG, "t.a.g", ComparisonOperator.EQUAL, 'a')]
    ),
])
def test_parse_filter_string(filter_string, comparisons):
    assert parse_filter_string(filter_string) == comparisons


@pytest.mark.parametrize("filter_string, comparisons", [
    ("params.m = 'LR'", [Comparison(KeyType.PARAM, "m", ComparisonOperator.EQUAL, 'LR')]),
    ("params.m = \"LR\"", [Comparison(KeyType.PARAM, "m", ComparisonOperator.EQUAL, 'LR')]),
    ('params.m = "LR"', [Comparison(KeyType.PARAM, "m", ComparisonOperator.EQUAL, 'LR')]),
    ('params.m = "L\'Hosp"', [Comparison(KeyType.PARAM, "m", ComparisonOperator.EQUAL, "L'Hosp")]),
])
def test_parse_filter_string_quote_trimming(filter_string, comparisons):
    assert parse_filter_string(filter_string) == comparisons


@pytest.mark.parametrize("filter_string, error_message", [
    ("metric.acc >= 0.94; metrics.rmse < 1", "Must be a single statement"),
    ("m.acc >= 0.94", "Invalid search expression type"),
    ("acc >= 0.94", "Expected param, metric or tag identifier"),
    ("p.model >= 'LR'", "Invalid search expression type"),
    ("model >= 'LR'", "Expected param, metric or tag identifier"),
    ("metrics.A > 0.1 OR params.B = 'LR'", r"Invalid clause\(s\) in filter string"),
    ("metrics.A > 0.1 NAND params.B = 'LR'", r"Invalid clause\(s\) in filter string"),
    ("metrics.A > 0.1 AND (params.B = 'LR')", r"Invalid clause\(s\) in filter string"),
    ("`metrics.A > 0.1", r"Invalid clause\(s\) in filter string"),
    ("param`.A > 0.1", r"Invalid clause\(s\) in filter string"),
    ("`dummy.A > 0.1", r"Invalid clause\(s\) in filter string"),
    ("dummy`.A > 0.1", r"Invalid clause\(s\) in filter string"),
    ("params.acc LR !=", r"Invalid clause\(s\) in filter string"),
    ("params.acc LR", r"Invalid clause\(s\) in filter string"),
    ("metric.acc !=", r"Invalid clause\(s\) in filter string"),
    ("acc != 1.0", "Expected param, metric or tag identifier"),
    ("foo is null", r"Invalid clause\(s\) in filter string"),
    ("1=1", "Expected param, metric or tag identifier"),
    ("1==2", "Expected param, metric or tag identifier"),
])
def test_parse_filter_string_error(filter_string, error_message):
    with pytest.raises(MlflowException, match=error_message):
        parse_filter_string(filter_string)


@pytest.mark.parametrize("filter_string, error_message", [
    ("metric.model = 'LR'", "Expected a numeric value for metric"),
    ("metric.model = '5'", "Expected a numeric value for metric"),
    ("params.acc = 5", "Expected a quoted string value for param"),
    ("tags.acc = 5", "Expected a quoted string value for tag"),
    ("metrics.acc != metrics.acc", "Expected a numeric value for metric"),
    ("1.0 > metrics.acc", "Expected param, metric or tag identifier"),
])
def test_parse_filter_string_bad_value_type(filter_string, error_message):
    with pytest.raises(MlflowException, match=error_message):
        parse_filter_string(filter_string)


@pytest.mark.parametrize("filter_string, error_message", [
    ("params.acc = LR", "value is either not quoted or unidentified quote types"),
    ("tags.acc = LR", "value is either not quoted or unidentified quote types"),
    ("params.'acc = LR", r"Invalid clause\(s\) in filter string"),
    ("params.acc = 'LR", r"Invalid clause\(s\) in filter string"),
    ("params.acc = LR'", r"Invalid clause\(s\) in filter string"),
    ("params.acc = \"LR'", r"Invalid clause\(s\) in filter string"),
    ("tags.acc = \"LR'", r"Invalid clause\(s\) in filter string"),
    ("tags.acc = = 'LR'", r"Invalid clause\(s\) in filter string"),
])
def test_parse_filter_string_bad_quotes(filter_string, error_message):
    with pytest.raises(MlflowException, match=error_message):
        parse_filter_string(filter_string)


@pytest.mark.parametrize("entity_type, entity_value", [
    ("metrics", 1.0),
    ("params", "'my-param-value'"),
    ("tags", "'my-tag-value'")
])
@pytest.mark.parametrize("operator", ["~", "~="])
def test_parse_filter_string_invalid_operator(entity_type, operator, entity_value):
    filter_string = "{}.abc {} {}".format(entity_type, operator, entity_value)
    with pytest.raises(MlflowException, match="not a valid operator"):
        parse_filter_string(filter_string)


@pytest.mark.parametrize("entity_type, entity_value", [
    ("params", "'my-param-value'"),
    ("tags", "'my-tag-value'")
])
@pytest.mark.parametrize("operator", [">", "<", ">=", "<="])
def test_parse_filter_string_unsupported_operator(entity_type, operator, entity_value):
    filter_string = "{}.abc {} {}".format(entity_type, operator, entity_value)
    expected_message = "The operator '{}' is not supported for {}".format(operator, entity_type)
    with pytest.raises(MlflowException, match=expected_message):
        parse_filter_string(filter_string)


@pytest.mark.parametrize("search_expression, comparison", [
    (
        SearchExpression(metric=MetricSearchExpression(
            key="accuracy", double=DoubleClause(comparator=">=", value=0.94))),
        Comparison(KeyType.METRIC, "accuracy", ComparisonOperator.GREATER_THAN_EQUAL, 0.94)
    ),
    (
        SearchExpression(metric=MetricSearchExpression(
            key="accuracy", double=DoubleClause(comparator=">=", value=.94))),
        Comparison(KeyType.METRIC, "accuracy", ComparisonOperator.GREATER_THAN_EQUAL, 0.94)
    ),
    (
        SearchExpression(metric=MetricSearchExpression(
            key="error", double=DoubleClause(comparator="<", value=.01))),
        Comparison(KeyType.METRIC, "error", ComparisonOperator.LESS_THAN, 0.01)
    ),
    (
        SearchExpression(metric=MetricSearchExpression(
            key="mse", float=FloatClause(comparator=">=", value=5))),
        Comparison(KeyType.METRIC, "mse", ComparisonOperator.GREATER_THAN_EQUAL, 5)
    ),
    (
        SearchExpression(parameter=ParameterSearchExpression(
            key="a", string=StringClause(comparator="=", value="0"))),
        Comparison(KeyType.PARAM, "a", ComparisonOperator.EQUAL, '0')
    ),
    (
        SearchExpression(parameter=ParameterSearchExpression(
            key="b", string=StringClause(comparator="!=", value="blah"))),
        Comparison(KeyType.PARAM, "b", ComparisonOperator.NOT_EQUAL, 'blah')
    )
])
def test_search_expression_to_comparison(search_expression, comparison):
    assert search_expression_to_comparison(search_expression) == comparison
