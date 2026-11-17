import pytest

from mlflow.entities import TraceInfo, TraceState
from mlflow.entities.trace_location import TraceLocation
from mlflow.exceptions import MlflowException
from mlflow.utils.search_utils import SearchTraceMetricsUtils, SearchTraceUtils


@pytest.mark.parametrize(
    ("expression", "keys"),
    [
        ("", [[]]),
        ("(status = 'OK')", [["status"]]),
        ("span.name = 'a' OR span.name = 'b' AND span.type = 'LLM'", [["name"], ["name", "type"]]),
        (
            "status = 'OK' OR timestamp_ms > 1 AND tags.env = 'prod'",
            [["status"], ["timestamp_ms", "env"]],
        ),
        (
            "(status = 'OK' OR timestamp_ms > 1) AND tags.env = 'prod'",
            [["status", "env"], ["timestamp_ms", "env"]],
        ),
        (
            "(tags.a = '1' OR tags.b = '2') AND (metadata.c = '3' OR metadata.d = '4')",
            [["a", "c"], ["a", "d"], ["b", "c"], ["b", "d"]],
        ),
        (
            "status IN ('OK', 'ERROR') AND (tags.env IS NULL OR metadata.env IS NOT NULL)",
            [["status", "env"], ["status", "env"]],
        ),
        ("span.name NOT IN ('AND', 'OR') OR span.name = 'AND (OR)'", [["name"], ["name"]]),
    ],
)
def test_trace_filter_groups(expression, keys):
    groups = SearchTraceUtils.parse_search_filter_groups(expression)
    assert [[clause["key"] for clause in group] for group in groups] == keys


@pytest.mark.parametrize(
    "expression",
    [
        "()",
        "status = 'OK' OR",
        "status = 'OK' AND OR status = 'ERROR'",
        "status = 'OK' (status = 'ERROR')",
        "NOT (status = 'OK')",
        "status = 'OK'; status = 'ERROR'",
        "status = 'OK' -- ignored",
        "status = 'OK' UNION SELECT * FROM traces",
        "status = 'OK' OR 1 = 1",
        "status = 'OK' OR unknown.field = 'x'",
        "status = 'OK' OR (",
        "((status = 'OK')",
    ],
)
def test_trace_filter_groups_reject_invalid(expression):
    with pytest.raises(MlflowException, match="Invalid"):
        SearchTraceUtils.parse_search_filter_groups(expression)


def test_trace_filter_groups_bounds_expansion():
    def expression(count):
        return " AND ".join(f"(tags.a{i} = 'a' OR tags.b{i} = 'b')" for i in range(count))

    assert len(SearchTraceUtils.parse_search_filter_groups(expression(5))) == 32
    with pytest.raises(MlflowException, match="too complex"):
        SearchTraceUtils.parse_search_filter_groups(expression(6))
    with pytest.raises(MlflowException, match="too complex"):
        SearchTraceUtils.parse_search_filter_groups(
            expression(5) + " AND " + " AND ".join(["tags.c = 'c'"] * 4)
        )
    assert (
        len(SearchTraceUtils.parse_search_filter_groups(" AND ".join(["tags.c = 'c'"] * 260))[0])
        == 260
    )


def test_trace_filter_groups_preserve_legacy_parser_contract():
    with pytest.raises(MlflowException, match="Invalid"):
        SearchTraceUtils.parse_search_filter("status = 'OK' OR status = 'ERROR'")
    with pytest.raises(MlflowException, match="Invalid"):
        SearchTraceMetricsUtils.parse_search_filter("trace.status = 'OK' OR trace.status = 'ERROR'")


def test_trace_filter_groups_in_memory():
    def trace(trace_id, timestamp, env):
        return TraceInfo(
            trace_id=trace_id,
            trace_location=TraceLocation.from_experiment_id("1"),
            request_time=timestamp,
            execution_duration=1,
            state=TraceState.OK,
            tags={"env": env},
            trace_metadata={},
        )

    traces = [trace("tr-old", 1, "prod"), trace("tr-new", 20, "prod"), trace("tr-dev", 30, "dev")]
    expression = "timestamp_ms > 10 AND (tags.env = 'prod' OR tags.env = 'dev')"
    assert [t.trace_id for t in SearchTraceUtils.filter(traces, expression)] == ["tr-new", "tr-dev"]


def test_trace_filter_groups_fold_same_span_field_alternatives():
    expression = " OR ".join(f"span.name = 'name-{i}'" for i in range(200))
    groups = SearchTraceUtils.parse_search_filter_groups(expression)
    assert len(groups) == len(groups[0]) == 1
    assert groups[0][0]["comparator"] == "IN"
    assert groups[0][0]["value"] == tuple(f"name-{i}" for i in range(200))
    with pytest.raises(MlflowException, match="200 values"):
        SearchTraceUtils.parse_search_filter_groups(expression + " OR span.name = 'extra'")
    groups = SearchTraceUtils.parse_search_filter_groups(
        "span.attributes.foo = 'a' OR span.attributes.FOO = 'b'"
    )
    assert len(groups) == 2


@pytest.mark.parametrize(
    ("expressions", "expected"),
    [
        ((None, ""), ""),
        (("tags.env = 'prod'", None), "tags.env = 'prod'"),
        (("tags.env = 'a' OR tags.env = 'b'",), "tags.env = 'a' OR tags.env = 'b'"),
        (("tags.env = 'AND OR (x)'", "status = 'OK'"), "tags.env = 'AND OR (x)' AND status = 'OK'"),
        (
            ("(tags.env = 'a' OR tags.env = 'b')", "status = 'OK'"),
            "(tags.env = 'a' OR tags.env = 'b') AND status = 'OK'",
        ),
        (
            ("tags.env = 'a' OR tags.env = 'b'", "status = 'OK'"),
            "(tags.env = 'a' OR tags.env = 'b') AND status = 'OK'",
        ),
    ],
)
def test_combine_trace_filters(expressions, expected):
    assert SearchTraceUtils.combine_filters(*expressions) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "AND status = 'OK'",
        "status = 'OK' AND",
        "status = 'OK' tags.env = 'prod'",
    ],
)
def test_trace_filter_groups_preserve_legacy_flat_shapes(expression):
    assert SearchTraceUtils.parse_search_filter_groups(expression) == [
        SearchTraceUtils.parse_search_filter_for_search_traces(expression)
    ]


def test_trace_filter_groups_keep_custom_span_attributes_as_alternatives():
    groups = SearchTraceUtils.parse_search_filter_groups(
        "span.attributes.foo = 'a' OR span.attributes.foo = 'b'"
    )
    assert [[clause["comparator"] for clause in group] for group in groups] == [["="], ["="]]
