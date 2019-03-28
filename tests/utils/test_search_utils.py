import pytest

from mlflow.exceptions import MlflowException
from mlflow.protos.service_pb2 import SearchExpression, DoubleClause, \
    MetricSearchExpression, FloatClause, ParameterSearchExpression, StringClause
from mlflow.utils.search_utils import SearchFilter


def test_search_filter_basics():
    search_filter = "This is a filter string"
    anded_expressions = [SearchExpression(), SearchExpression()]

    # only anded_expressions
    SearchFilter(anded_expressions=anded_expressions)

    # only search filter
    SearchFilter(filter_string=search_filter)

    # both
    with pytest.raises(MlflowException) as e:
        SearchFilter(anded_expressions=anded_expressions, filter_string=search_filter)
        assert e.message.contains("Can specify only one of 'filter' or 'search_expression'")


def test_anded_expression():
    se = SearchExpression(metric=MetricSearchExpression(key="accuracy",
                                                        double=DoubleClause(comparator=">=",
                                                                            value=.94)))
    sf = SearchFilter(anded_expressions=[se])
    assert sf._parse() == [{"type": "metric", "key": "accuracy", "comparator": ">=", "value": 0.94}]


def test_anded_expression_2():
    m1 = MetricSearchExpression(key="accuracy", double=DoubleClause(comparator=">=", value=.94))
    m2 = MetricSearchExpression(key="error", double=DoubleClause(comparator="<", value=.01))
    m3 = MetricSearchExpression(key="mse", float=FloatClause(comparator=">=", value=5))
    p1 = ParameterSearchExpression(key="a", string=StringClause(comparator="=", value="0"))
    p2 = ParameterSearchExpression(key="b", string=StringClause(comparator="!=", value="blah"))
    sf = SearchFilter(anded_expressions=[SearchExpression(metric=m1),
                                         SearchExpression(metric=m2),
                                         SearchExpression(metric=m3),
                                         SearchExpression(parameter=p1),
                                         SearchExpression(parameter=p2)])

    assert sf._parse() == [
        {'comparator': '>=', 'key': 'accuracy', 'type': 'metric', 'value': 0.94},
        {'comparator': '<', 'key': 'error', 'type': 'metric', 'value': 0.01},
        {'comparator': '>=', 'key': 'mse', 'type': 'metric', 'value': 5},
        {'comparator': '=', 'key': 'a', 'type': 'parameter', 'value': '0'},
        {'comparator': '!=', 'key': 'b', 'type': 'parameter', 'value': 'blah'}
    ]


@pytest.mark.parametrize("filter_string, parsed_filter", [
    ("metric.acc >= 0.94", [{'comparator': '>=', 'key': 'acc', 'type': 'metric', 'value': '0.94'}]),
    ('metric."legit name" >= 0.243', [{'comparator': '>=',
                                       'key': '"legit name"',
                                       'type': 'metric',
                                       'value': '0.243'}]),
    ("metrics.XYZ = 3", [{'comparator': '=', 'key': 'XYZ', 'type': 'metric', 'value': '3'}]),
    ("params.model = 'LinearRegression'", [{'comparator': '=',
                                            'key': 'model',
                                            'type': 'parameter',
                                            'value': "'LinearRegression'"}]),
    ("metrics.rmse < 1 and params.model_class = 'LR'", [
        {'comparator': '<', 'key': 'rmse', 'type': 'metric', 'value': '1'},
        {'comparator': '=', 'key': 'model_class', 'type': 'parameter', 'value': "'LR'"}
    ]),
    ("`metric`.a >= 0.1", [{'comparator': '>=', 'key': 'a', 'type': 'metric', 'value': '0.1'}]),
    ("`params`.model >= 'LR'", [{'comparator': '>=',
                                 'key': 'model',
                                 'type': 'parameter',
                                 'value': "'LR'"}]),
])
def test_filter(filter_string, parsed_filter):
    assert SearchFilter(filter_string=filter_string)._parse() == parsed_filter


@pytest.mark.parametrize("filter_string, error_message", [
    ("metric.acc >= 0.94; metrics.rmse < 1", "Search filter contained multiple expression"),
    ("m.acc >= 0.94", "Invalid search expression type"),
    ("acc >= 0.94", "Invalid filter string"),
    ("p.model >= 'LR'", "Invalid search expression type"),
    ("model >= 'LR'", "Invalid filter string"),
    ("metrics.A > 0.1 OR params.B = 'LR'", "Invalid clause(s) in filter string"),
    ("metrics.A > 0.1 NAND params.B = 'LR'", "Invalid clause(s) in filter string"),
    ("metrics.A > 0.1 AND (params.B = 'LR')", "Invalid clause(s) in filter string"),
    ("`metrics.A > 0.1", "Invalid clause(s) in filter string"),
    ("param`.A > 0.1", "Invalid clause(s) in filter string"),
    ("`dummy.A > 0.1", "Invalid clause(s) in filter string"),
    ("dummy`.A > 0.1", "Invalid clause(s) in filter string"),
    ("metrics.crazy.name > 0.1", "Invalid filter string"),
])
def test_error_filter(filter_string, error_message):
    with pytest.raises(MlflowException) as e:
        SearchFilter(filter_string=filter_string)._parse()
    assert error_message in e.value.message
