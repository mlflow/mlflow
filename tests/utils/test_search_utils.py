import base64
import json
import pytest
import re

from mlflow.entities import RunInfo, RunData, Run, LifecycleStage, RunStatus, Metric, Param, RunTag
from mlflow.exceptions import MlflowException
from mlflow.utils.search_utils import SearchUtils


@pytest.mark.parametrize(
    "filter_string, parsed_filter",
    [
        (
            "metric.acc >= 0.94",
            [{"comparator": ">=", "key": "acc", "type": "metric", "value": "0.94"}],
        ),
        ("metric.acc>=100", [{"comparator": ">=", "key": "acc", "type": "metric", "value": "100"}]),
        ("params.m!='tf'", [{"comparator": "!=", "key": "m", "type": "parameter", "value": "tf"}]),
        (
            'params."m"!="tf"',
            [{"comparator": "!=", "key": "m", "type": "parameter", "value": "tf"}],
        ),
        (
            'metric."legit name" >= 0.243',
            [{"comparator": ">=", "key": "legit name", "type": "metric", "value": "0.243"}],
        ),
        ("metrics.XYZ = 3", [{"comparator": "=", "key": "XYZ", "type": "metric", "value": "3"}]),
        (
            'params."cat dog" = "pets"',
            [{"comparator": "=", "key": "cat dog", "type": "parameter", "value": "pets"}],
        ),
        (
            'metrics."X-Y-Z" = 3',
            [{"comparator": "=", "key": "X-Y-Z", "type": "metric", "value": "3"}],
        ),
        (
            'metrics."X//Y#$$@&Z" = 3',
            [{"comparator": "=", "key": "X//Y#$$@&Z", "type": "metric", "value": "3"}],
        ),
        (
            "params.model = 'LinearRegression'",
            [{"comparator": "=", "key": "model", "type": "parameter", "value": "LinearRegression"}],
        ),
        (
            "metrics.rmse < 1 and params.model_class = 'LR'",
            [
                {"comparator": "<", "key": "rmse", "type": "metric", "value": "1"},
                {"comparator": "=", "key": "model_class", "type": "parameter", "value": "LR"},
            ],
        ),
        ("", []),
        ("`metric`.a >= 0.1", [{"comparator": ">=", "key": "a", "type": "metric", "value": "0.1"}]),
        (
            "`params`.model >= 'LR'",
            [{"comparator": ">=", "key": "model", "type": "parameter", "value": "LR"}],
        ),
        (
            "tags.version = 'commit-hash'",
            [{"comparator": "=", "key": "version", "type": "tag", "value": "commit-hash"}],
        ),
        (
            "`tags`.source_name = 'a notebook'",
            [{"comparator": "=", "key": "source_name", "type": "tag", "value": "a notebook"}],
        ),
        (
            'metrics."accuracy.2.0" > 5',
            [{"comparator": ">", "key": "accuracy.2.0", "type": "metric", "value": "5"}],
        ),
        (
            "metrics.`spacey name` > 5",
            [{"comparator": ">", "key": "spacey name", "type": "metric", "value": "5"}],
        ),
        (
            'params."p.a.r.a.m" != "a"',
            [{"comparator": "!=", "key": "p.a.r.a.m", "type": "parameter", "value": "a"}],
        ),
        ('tags."t.a.g" = "a"', [{"comparator": "=", "key": "t.a.g", "type": "tag", "value": "a"}]),
        (
            "attribute.artifact_uri = '1/23/4'",
            [{"type": "attribute", "comparator": "=", "key": "artifact_uri", "value": "1/23/4"}],
        ),
        (
            "attribute.start_time >= 1234",
            [{"type": "attribute", "comparator": ">=", "key": "start_time", "value": "1234"}],
        ),
        (
            "run.status = 'RUNNING'",
            [{"type": "attribute", "comparator": "=", "key": "status", "value": "RUNNING"}],
        ),
    ],
)
def test_filter(filter_string, parsed_filter):
    assert SearchUtils.parse_search_filter(filter_string) == parsed_filter


@pytest.mark.parametrize(
    "filter_string, parsed_filter",
    [
        ("params.m = 'LR'", [{"type": "parameter", "comparator": "=", "key": "m", "value": "LR"}]),
        ('params.m = "LR"', [{"type": "parameter", "comparator": "=", "key": "m", "value": "LR"}]),
        ('params.m = "LR"', [{"type": "parameter", "comparator": "=", "key": "m", "value": "LR"}]),
        (
            'params.m = "L\'Hosp"',
            [{"type": "parameter", "comparator": "=", "key": "m", "value": "L'Hosp"}],
        ),
    ],
)
def test_correct_quote_trimming(filter_string, parsed_filter):
    assert SearchUtils.parse_search_filter(filter_string) == parsed_filter


@pytest.mark.parametrize(
    "filter_string, error_message",
    [
        ("metric.acc >= 0.94; metrics.rmse < 1", "Search filter contained multiple expression"),
        ("m.acc >= 0.94", "Invalid entity type"),
        ("acc >= 0.94", "Invalid identifier"),
        ("p.model >= 'LR'", "Invalid entity type"),
        ("attri.x != 1", "Invalid entity type"),
        ("a.x != 1", "Invalid entity type"),
        ("model >= 'LR'", "Invalid identifier"),
        ("metrics.A > 0.1 OR params.B = 'LR'", "Invalid clause(s) in filter string"),
        ("metrics.A > 0.1 NAND params.B = 'LR'", "Invalid clause(s) in filter string"),
        ("metrics.A > 0.1 AND (params.B = 'LR')", "Invalid clause(s) in filter string"),
        ("`metrics.A > 0.1", "Invalid clause(s) in filter string"),
        ("param`.A > 0.1", "Invalid clause(s) in filter string"),
        ("`dummy.A > 0.1", "Invalid clause(s) in filter string"),
        ("dummy`.A > 0.1", "Invalid clause(s) in filter string"),
        ("attribute.start != 1", "Invalid attribute key"),
        ("attribute.end_time != 1", "Invalid attribute key"),
        ("attribute.run_id != 1", "Invalid attribute key"),
        ("attribute.run_uuid != 1", "Invalid attribute key"),
        ("attribute.experiment_id != 1", "Invalid attribute key"),
        ("attribute.lifecycle_stage = 'ACTIVE'", "Invalid attribute key"),
        ("attribute.name != 1", "Invalid attribute key"),
        ("attribute.time != 1", "Invalid attribute key"),
        ("attribute._status != 'RUNNING'", "Invalid attribute key"),
        ("attribute.status = true", "Invalid clause(s) in filter string"),
    ],
)
def test_error_filter(filter_string, error_message):
    with pytest.raises(MlflowException, match=re.escape(error_message)):
        SearchUtils.parse_search_filter(filter_string)


@pytest.mark.parametrize(
    "filter_string, error_message",
    [
        ("metric.model = 'LR'", "Expected numeric value type for metric"),
        ("metric.model = '5'", "Expected numeric value type for metric"),
        ("params.acc = 5", "Expected a quoted string value for param"),
        ("tags.acc = 5", "Expected a quoted string value for tag"),
        ("metrics.acc != metrics.acc", "Expected numeric value type for metric"),
        ("1.0 > metrics.acc", "Expected 'Identifier' found"),
        ("attribute.status = 1", "Expected a quoted string value for attributes"),
    ],
)
def test_error_comparison_clauses(filter_string, error_message):
    with pytest.raises(MlflowException, match=error_message):
        SearchUtils.parse_search_filter(filter_string)


@pytest.mark.parametrize(
    "filter_string, error_message",
    [
        ("params.acc = LR", "value is either not quoted or unidentified quote types"),
        ("tags.acc = LR", "value is either not quoted or unidentified quote types"),
        ("params.acc = `LR`", "value is either not quoted or unidentified quote types"),
        ("params.'acc = LR", "Invalid clause(s) in filter string"),
        ("params.acc = 'LR", "Invalid clause(s) in filter string"),
        ("params.acc = LR'", "Invalid clause(s) in filter string"),
        ("params.acc = \"LR'", "Invalid clause(s) in filter string"),
        ("tags.acc = \"LR'", "Invalid clause(s) in filter string"),
        ("tags.acc = = 'LR'", "Invalid clause(s) in filter string"),
        ("attribute.status IS 'RUNNING'", "Invalid clause(s) in filter string"),
    ],
)
def test_bad_quotes(filter_string, error_message):
    with pytest.raises(MlflowException, match=re.escape(error_message)):
        SearchUtils.parse_search_filter(filter_string)


@pytest.mark.parametrize(
    "filter_string, error_message",
    [
        ("params.acc LR !=", "Invalid clause(s) in filter string"),
        ("params.acc LR", "Invalid clause(s) in filter string"),
        ("metric.acc !=", "Invalid clause(s) in filter string"),
        ("acc != 1.0", "Invalid identifier"),
        ("foo is null", "Invalid clause(s) in filter string"),
        ("1=1", "Expected 'Identifier' found"),
        ("1==2", "Expected 'Identifier' found"),
    ],
)
def test_invalid_clauses(filter_string, error_message):
    with pytest.raises(MlflowException, match=re.escape(error_message)):
        SearchUtils.parse_search_filter(filter_string)


@pytest.mark.parametrize(
    "entity_type, bad_comparators, key, entity_value",
    [
        ("metrics", ["~", "~="], "abc", 1.0),
        ("params", [">", "<", ">=", "<=", "~"], "abc", "'my-param-value'"),
        ("tags", [">", "<", ">=", "<=", "~"], "abc", "'my-tag-value'"),
        ("attributes", [">", "<", ">=", "<=", "~"], "status", "'my-tag-value'"),
        ("attributes", ["LIKE", "ILIKE"], "start_time", 1234),
    ],
)
def test_bad_comparators(entity_type, bad_comparators, key, entity_value):
    run = Run(
        run_info=RunInfo(
            run_uuid="hi",
            run_id="hi",
            experiment_id=0,
            user_id="user-id",
            status=RunStatus.to_string(RunStatus.FAILED),
            start_time=0,
            end_time=1,
            lifecycle_stage=LifecycleStage.ACTIVE,
        ),
        run_data=RunData(metrics=[], params=[], tags=[]),
    )
    for bad_comparator in bad_comparators:
        bad_filter = "{entity_type}.{key} {comparator} {value}".format(
            entity_type=entity_type, key=key, comparator=bad_comparator, value=entity_value
        )
        with pytest.raises(MlflowException, match="Invalid comparator"):
            SearchUtils.filter([run], bad_filter)


@pytest.mark.parametrize(
    "filter_string, matching_runs",
    [
        (None, [0, 1, 2]),
        ("", [0, 1, 2]),
        ("attributes.status = 'FAILED'", [0, 2]),
        ("metrics.key1 = 123", [1]),
        ("metrics.key1 != 123", [0, 2]),
        ("metrics.key1 >= 123", [1, 2]),
        ("params.my_param = 'A'", [0, 1]),
        ("tags.tag1 = 'D'", [2]),
        ("tags.tag1 != 'D'", [1]),
        ("params.my_param = 'A' AND attributes.status = 'FAILED'", [0]),
    ],
)
def test_correct_filtering(filter_string, matching_runs):
    runs = [
        Run(
            run_info=RunInfo(
                run_uuid="hi",
                run_id="hi",
                experiment_id=0,
                user_id="user-id",
                status=RunStatus.to_string(RunStatus.FAILED),
                start_time=0,
                end_time=1,
                lifecycle_stage=LifecycleStage.ACTIVE,
            ),
            run_data=RunData(
                metrics=[Metric("key1", 121, 1, 0)], params=[Param("my_param", "A")], tags=[]
            ),
        ),
        Run(
            run_info=RunInfo(
                run_uuid="hi2",
                run_id="hi2",
                experiment_id=0,
                user_id="user-id",
                status=RunStatus.to_string(RunStatus.FINISHED),
                start_time=0,
                end_time=1,
                lifecycle_stage=LifecycleStage.ACTIVE,
            ),
            run_data=RunData(
                metrics=[Metric("key1", 123, 1, 0)],
                params=[Param("my_param", "A")],
                tags=[RunTag("tag1", "C")],
            ),
        ),
        Run(
            run_info=RunInfo(
                run_uuid="hi3",
                run_id="hi3",
                experiment_id=1,
                user_id="user-id",
                status=RunStatus.to_string(RunStatus.FAILED),
                start_time=0,
                end_time=1,
                lifecycle_stage=LifecycleStage.ACTIVE,
            ),
            run_data=RunData(
                metrics=[Metric("key1", 125, 1, 0)],
                params=[Param("my_param", "B")],
                tags=[RunTag("tag1", "D")],
            ),
        ),
    ]
    filtered_runs = SearchUtils.filter(runs, filter_string)
    assert set(filtered_runs) == set([runs[i] for i in matching_runs])


def test_filter_runs_by_start_time():
    runs = [
        Run(
            run_info=RunInfo(
                run_uuid=run_id,
                run_id=run_id,
                experiment_id=0,
                user_id="user-id",
                status=RunStatus.to_string(RunStatus.FINISHED),
                start_time=idx,
                end_time=1,
                lifecycle_stage=LifecycleStage.ACTIVE,
            ),
            run_data=RunData(),
        )
        for idx, run_id in enumerate(["a", "b", "c"])
    ]
    assert SearchUtils.filter(runs, "attribute.start_time >= 0") == runs
    assert SearchUtils.filter(runs, "attribute.start_time > 1") == runs[2:]
    assert SearchUtils.filter(runs, "attribute.start_time = 2") == runs[2:]


@pytest.mark.parametrize(
    "order_bys, matching_runs",
    [
        (None, [2, 1, 0]),
        ([], [2, 1, 0]),
        (["tags.noSuchTag"], [2, 1, 0]),
        (["attributes.status"], [2, 0, 1]),
        (["attributes.start_time"], [0, 2, 1]),
        (["metrics.key1 asc"], [0, 1, 2]),
        (['metrics."key1"  desc'], [2, 1, 0]),
        (["params.my_param"], [1, 0, 2]),
        (["params.my_param aSc", "attributes.status  ASC"], [0, 1, 2]),
        (["params.my_param", "attributes.status DESC"], [1, 0, 2]),
        (["params.my_param DESC", "attributes.status   DESC"], [2, 1, 0]),
        (["params.`my_param` DESC", "attributes.status DESC"], [2, 1, 0]),
        (["tags.tag1"], [1, 2, 0]),
        (["tags.tag1    DESC"], [2, 1, 0]),
    ],
)
def test_correct_sorting(order_bys, matching_runs):
    runs = [
        Run(
            run_info=RunInfo(
                run_uuid="9",
                run_id="9",
                experiment_id=0,
                user_id="user-id",
                status=RunStatus.to_string(RunStatus.FAILED),
                start_time=0,
                end_time=1,
                lifecycle_stage=LifecycleStage.ACTIVE,
            ),
            run_data=RunData(
                metrics=[Metric("key1", 121, 1, 0)], params=[Param("my_param", "A")], tags=[]
            ),
        ),
        Run(
            run_info=RunInfo(
                run_uuid="8",
                run_id="8",
                experiment_id=0,
                user_id="user-id",
                status=RunStatus.to_string(RunStatus.FINISHED),
                start_time=1,
                end_time=1,
                lifecycle_stage=LifecycleStage.ACTIVE,
            ),
            run_data=RunData(
                metrics=[Metric("key1", 123, 1, 0)],
                params=[Param("my_param", "A")],
                tags=[RunTag("tag1", "C")],
            ),
        ),
        Run(
            run_info=RunInfo(
                run_uuid="7",
                run_id="7",
                experiment_id=1,
                user_id="user-id",
                status=RunStatus.to_string(RunStatus.FAILED),
                start_time=1,
                end_time=1,
                lifecycle_stage=LifecycleStage.ACTIVE,
            ),
            run_data=RunData(
                metrics=[Metric("key1", 125, 1, 0)],
                params=[Param("my_param", "B")],
                tags=[RunTag("tag1", "D")],
            ),
        ),
    ]
    sorted_runs = SearchUtils.sort(runs, order_bys)
    sorted_run_indices = []
    for run in sorted_runs:
        for i, r in enumerate(runs):
            if r == run:
                sorted_run_indices.append(i)
                break
    assert sorted_run_indices == matching_runs


def test_order_by_metric_with_nans_infs_nones():
    metric_vals_str = ["nan", "inf", "-inf", "-1000", "0", "1000", "None"]
    runs = [
        Run(
            run_info=RunInfo(
                run_id=x,
                run_uuid=x,
                experiment_id=0,
                user_id="user",
                status=RunStatus.to_string(RunStatus.FINISHED),
                start_time=0,
                end_time=1,
                lifecycle_stage=LifecycleStage.ACTIVE,
            ),
            run_data=RunData(metrics=[Metric("x", None if x == "None" else float(x), 1, 0)]),
        )
        for x in metric_vals_str
    ]
    sorted_runs_asc = [x.info.run_id for x in SearchUtils.sort(runs, ["metrics.x asc"])]
    sorted_runs_desc = [x.info.run_id for x in SearchUtils.sort(runs, ["metrics.x desc"])]
    # asc
    assert ["-inf", "-1000", "0", "1000", "inf", "nan", "None"] == sorted_runs_asc
    # desc
    assert ["inf", "1000", "0", "-1000", "-inf", "nan", "None"] == sorted_runs_desc


@pytest.mark.parametrize(
    "order_by, error_message",
    [
        ("m.acc", "Invalid entity type"),
        ("acc", "Invalid identifier"),
        ("attri.x", "Invalid entity type"),
        ("`metrics.A", "Invalid order_by clause"),
        ("`metrics.A`", "Invalid entity type"),
        ("attribute.start", "Invalid attribute key"),
        ("attribute.run_id", "Invalid attribute key"),
        ("attribute.experiment_id", "Invalid attribute key"),
        ("metrics.A != 1", "Invalid order_by clause"),
        ("params.my_param ", "Invalid order_by clause"),
        ("attribute.run_id ACS", "Invalid ordering key"),
        ("attribute.run_id decs", "Invalid ordering key"),
    ],
)
def test_invalid_order_by_search_runs(order_by, error_message):
    with pytest.raises(MlflowException, match=error_message):
        SearchUtils.parse_order_by_for_search_runs(order_by)


@pytest.mark.parametrize(
    "order_by, ascending_expected",
    [
        ("metrics.`Mean Square Error`", True),
        ("metrics.`Mean Square Error` ASC", True),
        ("metrics.`Mean Square Error` DESC", False),
    ],
)
def test_space_order_by_search_runs(order_by, ascending_expected):
    identifier_type, identifier_name, ascending = SearchUtils.parse_order_by_for_search_runs(
        order_by
    )
    assert identifier_type == "metric"
    assert identifier_name == "Mean Square Error"
    assert ascending == ascending_expected


@pytest.mark.parametrize(
    "order_by, error_message",
    [
        ("creation_timestamp DESC", "Invalid order by key"),
        ("last_updated_timestamp DESC blah", "Invalid order_by clause"),
        ("", "Invalid order_by clause"),
        ("timestamp somerandomstuff ASC", "Invalid order_by clause"),
        ("timestamp somerandomstuff", "Invalid order_by clause"),
        ("timestamp decs", "Invalid order_by clause"),
        ("timestamp ACS", "Invalid order_by clause"),
        ("name aCs", "Invalid ordering key"),
    ],
)
def test_invalid_order_by_search_registered_models(order_by, error_message):
    with pytest.raises(MlflowException, match=re.escape(error_message)):
        SearchUtils.parse_order_by_for_search_registered_models(order_by)


@pytest.mark.parametrize(
    "page_token, max_results, matching_runs, expected_next_page_token",
    [
        (None, 1, [0], {"offset": 1}),
        (None, 2, [0, 1], {"offset": 2}),
        (None, 3, [0, 1, 2], None),
        (None, 5, [0, 1, 2], None),
        ({"offset": 1}, 1, [1], {"offset": 2}),
        ({"offset": 1}, 2, [1, 2], None),
        ({"offset": 1}, 3, [1, 2], None),
        ({"offset": 2}, 1, [2], None),
        ({"offset": 2}, 2, [2], None),
        ({"offset": 2}, 0, [], {"offset": 2}),
        ({"offset": 3}, 1, [], None),
    ],
)
def test_pagination(page_token, max_results, matching_runs, expected_next_page_token):
    runs = [
        Run(
            run_info=RunInfo(
                run_uuid="0",
                run_id="0",
                experiment_id=0,
                user_id="user-id",
                status=RunStatus.to_string(RunStatus.FAILED),
                start_time=0,
                end_time=1,
                lifecycle_stage=LifecycleStage.ACTIVE,
            ),
            run_data=RunData([], [], []),
        ),
        Run(
            run_info=RunInfo(
                run_uuid="1",
                run_id="1",
                experiment_id=0,
                user_id="user-id",
                status=RunStatus.to_string(RunStatus.FAILED),
                start_time=0,
                end_time=1,
                lifecycle_stage=LifecycleStage.ACTIVE,
            ),
            run_data=RunData([], [], []),
        ),
        Run(
            run_info=RunInfo(
                run_uuid="2",
                run_id="2",
                experiment_id=0,
                user_id="user-id",
                status=RunStatus.to_string(RunStatus.FAILED),
                start_time=0,
                end_time=1,
                lifecycle_stage=LifecycleStage.ACTIVE,
            ),
            run_data=RunData([], [], []),
        ),
    ]
    encoded_page_token = None
    if page_token:
        encoded_page_token = base64.b64encode(json.dumps(page_token).encode("utf-8"))
    paginated_runs, next_page_token = SearchUtils.paginate(runs, encoded_page_token, max_results)

    paginated_run_indices = []
    for run in paginated_runs:
        for i, r in enumerate(runs):
            if r == run:
                paginated_run_indices.append(i)
                break
    assert paginated_run_indices == matching_runs

    decoded_next_page_token = None
    if next_page_token:
        decoded_next_page_token = json.loads(base64.b64decode(next_page_token))
    assert decoded_next_page_token == expected_next_page_token


@pytest.mark.parametrize(
    "page_token, error_message",
    [
        (base64.b64encode(json.dumps({}).encode("utf-8")), "Invalid page token"),
        (base64.b64encode(json.dumps({"offset": "a"}).encode("utf-8")), "Invalid page token"),
        (base64.b64encode(json.dumps({"offsoot": 7}).encode("utf-8")), "Invalid page token"),
        (base64.b64encode("not json".encode("utf-8")), "Invalid page token"),
        ("not base64", "Invalid page token"),
    ],
)
def test_invalid_page_tokens(page_token, error_message):
    with pytest.raises(MlflowException, match=error_message):
        SearchUtils.paginate([], page_token, 1)
