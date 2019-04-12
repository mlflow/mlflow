import pytest

from mlflow.entities import RunInfo, RunData, Run, SourceType, LifecycleStage, RunStatus
from mlflow.exceptions import MlflowException
from mlflow.protos.service_pb2 import SearchExpression, DoubleClause, \
    MetricSearchExpression, FloatClause, ParameterSearchExpression, StringClause
from mlflow.utils.search_utils import SearchFilter, KeyType, ComparisonOperator, Comparison


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
    expected_comparison = Comparison(KeyType.METRIC, "accuracy",
                                     ComparisonOperator.GREATER_THAN_EQUAL, 0.94)
    assert sf._parse() == [expected_comparison]


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
        Comparison(KeyType.METRIC, "accuracy", ComparisonOperator.GREATER_THAN_EQUAL, 0.94),
        Comparison(KeyType.METRIC, "error", ComparisonOperator.LESS_THAN, 0.01),
        Comparison(KeyType.METRIC, "mse", ComparisonOperator.GREATER_THAN_EQUAL, 5),
        Comparison(KeyType.PARAM, "a", ComparisonOperator.EQUAL, '0'),
        Comparison(KeyType.PARAM, "b", ComparisonOperator.NOT_EQUAL, 'blah')
    ]


@pytest.mark.parametrize("filter_string, parsed_filter", [
    ("metric.acc >= 0.94", [
        Comparison(KeyType.METRIC, "acc", ComparisonOperator.GREATER_THAN_EQUAL, '0.94')]),
    ("metric.acc>=100", [
        Comparison(KeyType.METRIC, "acc", ComparisonOperator.GREATER_THAN_EQUAL, '100')]),
    ("params.m!='tf'", [
        Comparison(KeyType.PARAM, "m", ComparisonOperator.NOT_EQUAL, 'tf')]),
    ('params."m"!="tf"', [
        Comparison(KeyType.PARAM, "m", ComparisonOperator.NOT_EQUAL, 'tf')]),
    ('metric."legit name" >= 0.243', [
        Comparison(KeyType.METRIC, "legit name", ComparisonOperator.GREATER_THAN_EQUAL, '0.243')]),
    ("metrics.XYZ = 3", [
        Comparison(KeyType.METRIC, "XYZ", ComparisonOperator.EQUAL, '3')]),
    ('params."cat dog" = "pets"', [
        Comparison(KeyType.PARAM, "cat dog", ComparisonOperator.EQUAL, 'pets')]),
    ('metrics."X-Y-Z" = 3', [
        Comparison(KeyType.METRIC, "X-Y-Z", ComparisonOperator.EQUAL, '3')]),
    ('metrics."X//Y#$$@&Z" = 3', [
        Comparison(KeyType.METRIC, "X//Y#$$@&Z", ComparisonOperator.EQUAL, '3')]),
    ("params.model = 'LinearRegression'", [
        Comparison(KeyType.PARAM, "model", ComparisonOperator.EQUAL, "LinearRegression")]),
    ("metrics.rmse < 1 and params.model_class = 'LR'", [
        Comparison(KeyType.METRIC, "rmse", ComparisonOperator.LESS_THAN, '1'),
        Comparison(KeyType.PARAM, "model_class", ComparisonOperator.EQUAL, "LR")
    ]),
    ('', []),
    ("`metric`.a >= 0.1", [
        Comparison(KeyType.METRIC, "a", ComparisonOperator.GREATER_THAN_EQUAL, '0.1')]),
    ("`params`.model >= 'LR'", [
        Comparison(KeyType.PARAM, "model", ComparisonOperator.GREATER_THAN_EQUAL, "LR")]),
    ("tags.version = 'commit-hash'", [
        Comparison(KeyType.TAG, "version", ComparisonOperator.EQUAL, "commit-hash")]),
    ("`tags`.source_name = 'a notebook'", [
        Comparison(KeyType.TAG, "source_name", ComparisonOperator.EQUAL, "a notebook")]),
    ('metrics."accuracy.2.0" > 5', [
        Comparison(KeyType.METRIC, "accuracy.2.0", ComparisonOperator.GREATER_THAN, '5')]),
    ('params."p.a.r.a.m" != "a"', [
        Comparison(KeyType.PARAM, "p.a.r.a.m", ComparisonOperator.NOT_EQUAL, 'a')]),
    ('tags."t.a.g" = "a"', [Comparison(KeyType.TAG, "t.a.g", ComparisonOperator.EQUAL, 'a')]),
])
def test_filter(filter_string, parsed_filter):
    assert SearchFilter(filter_string=filter_string)._parse() == parsed_filter


@pytest.mark.parametrize("filter_string, parsed_filter", [
    ("params.m = 'LR'", [Comparison(KeyType.PARAM, "m", ComparisonOperator.EQUAL, 'LR')]),
    ("params.m = \"LR\"", [Comparison(KeyType.PARAM, "m", ComparisonOperator.EQUAL, 'LR')]),
    ('params.m = "LR"', [Comparison(KeyType.PARAM, "m", ComparisonOperator.EQUAL, 'LR')]),
    ('params.m = "L\'Hosp"', [Comparison(KeyType.PARAM, "m", ComparisonOperator.EQUAL, "L'Hosp")]),
])
def test_correct_quote_trimming(filter_string, parsed_filter):
    assert SearchFilter(filter_string=filter_string)._parse() == parsed_filter


@pytest.mark.parametrize("filter_string, error_message", [
    ("metric.acc >= 0.94; metrics.rmse < 1", "Must be a single statement"),
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
])
def test_error_filter(filter_string, error_message):
    with pytest.raises(MlflowException) as e:
        SearchFilter(filter_string=filter_string)._parse()
    assert error_message in e.value.message


@pytest.mark.parametrize("filter_string, error_message", [
    ("metric.model = 'LR'", "Expected numeric value type for metric"),
    ("metric.model = '5'", "Expected numeric value type for metric"),
    ("params.acc = 5", "Expected a quoted string value for param"),
    ("tags.acc = 5", "Expected a quoted string value for tag"),
    ("metrics.acc != metrics.acc", "Expected numeric value type for metric"),
    ("1.0 > metrics.acc", "Expected param, metric or tag identifier"),
])
def test_error_comparison_clauses(filter_string, error_message):
    with pytest.raises(MlflowException) as e:
        SearchFilter(filter_string=filter_string)._parse()
    assert error_message in e.value.message


@pytest.mark.parametrize("filter_string, error_message", [
    ("params.acc = LR", "value is either not quoted or unidentified quote types"),
    ("tags.acc = LR", "value is either not quoted or unidentified quote types"),
    ("params.'acc = LR", "Invalid clause(s) in filter string"),
    ("params.acc = 'LR", "Invalid clause(s) in filter string"),
    ("params.acc = LR'", "Invalid clause(s) in filter string"),
    ("params.acc = \"LR'", "Invalid clause(s) in filter string"),
    ("tags.acc = \"LR'", "Invalid clause(s) in filter string"),
    ("tags.acc = = 'LR'", "Invalid clause(s) in filter string"),
])
def test_bad_quotes(filter_string, error_message):
    with pytest.raises(MlflowException) as e:
        SearchFilter(filter_string=filter_string)._parse()
    assert error_message in e.value.message


@pytest.mark.parametrize("filter_string, error_message", [
    ("params.acc LR !=", "Invalid clause(s) in filter string"),
    ("params.acc LR", "Invalid clause(s) in filter string"),
    ("metric.acc !=", "Invalid clause(s) in filter string"),
    ("acc != 1.0", "Invalid filter string"),
    ("foo is null", "Invalid clause(s) in filter string"),
    ("1=1", "Expected param, metric or tag identifier"),
    ("1==2", "Expected param, metric or tag identifier"),
])
def test_invalid_clauses(filter_string, error_message):
    with pytest.raises(MlflowException) as e:
        SearchFilter(filter_string=filter_string)._parse()
    assert error_message in e.value.message


@pytest.mark.parametrize("entity_type, bad_comparators, entity_value", [
    ("metrics", ["~", "~="], 1.0),
    ("params", [">", "<", ">=", "<=", "~"], "'my-param-value'"),
    ("tags", [">", "<", ">=", "<=", "~"], "'my-tag-value'"),
])
def test_bad_comparators(entity_type, bad_comparators, entity_value):
    run = Run(run_info=RunInfo(
        run_uuid="hi", experiment_id=0, name="name", source_type=SourceType.PROJECT,
        source_name="source-name", entry_point_name="entry-point-name",
        user_id="user-id", status=RunStatus.FAILED, start_time=0, end_time=1,
        source_version="version", lifecycle_stage=LifecycleStage.ACTIVE),
        run_data=RunData(metrics=[], params=[], tags=[])
    )
    for bad_comparator in bad_comparators:
        bad_filter = "{entity_type}.abc {comparator} {value}".format(
            entity_type=entity_type, comparator=bad_comparator, value=entity_value)
        sf = SearchFilter(filter_string=bad_filter)
        with pytest.raises(MlflowException) as e:
            sf.filter(run)
        assert "Invalid comparator" in str(e.value.message)
