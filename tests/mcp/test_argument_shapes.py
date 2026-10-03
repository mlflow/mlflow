# Arguments whose typed shape differs from what the CLI-backed tools took. Each old shape is
# still accepted and gives the same result as the new one.

import pytest
import pytest_asyncio
from fastmcp import Client
from fastmcp.exceptions import ToolError

import mlflow
from mlflow import MlflowClient
from mlflow.mcp import server


@pytest_asyncio.fixture
async def client():
    async with Client(
        server.create_mcp(categories=["traces", "scorers", "experiments", "runs"])
    ) as c:
        yield c


@pytest.fixture
def experiment_id():
    experiment_id = MlflowClient().create_experiment("exp")
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def _trace() -> str:
    with mlflow.start_span("span") as span:
        span.set_inputs({"q": "hi"})
    return span.trace_id


async def _call(client, tool, **arguments):
    return (await client.call_tool(tool, arguments)).structured_content


@pytest.mark.parametrize(
    ("argument", "old", "new"),
    [
        ("order_by", "timestamp_ms DESC, status", ["timestamp_ms DESC", "status"]),
        ("extract_fields", "info.trace_id, info.state", ["info.trace_id", "info.state"]),
    ],
)
@pytest.mark.asyncio
async def test_search_traces_accepts_comma_separated_lists(
    client, experiment_id, argument, old, new
):
    _trace()
    _trace()
    old_result = await _call(
        client, "search_traces", experiment_id=experiment_id, **{argument: old}
    )
    new_result = await _call(
        client, "search_traces", experiment_id=experiment_id, **{argument: new}
    )
    assert old_result == new_result
    assert len(new_result["traces"]) == 2


@pytest.mark.asyncio
async def test_get_trace_accepts_comma_separated_extract_fields(client, experiment_id):
    trace_id = _trace()
    old = await _call(client, "get_trace", trace_id=trace_id, extract_fields="info.trace_id")
    new = await _call(client, "get_trace", trace_id=trace_id, extract_fields=["info.trace_id"])
    assert old == new == {"trace": {"info": {"trace_id": trace_id}}}


@pytest.mark.parametrize("shape", ["comma_separated", "list"])
@pytest.mark.asyncio
async def test_delete_traces_accepts_comma_separated_trace_ids(client, experiment_id, shape):
    trace_ids = [_trace(), _trace()]
    value = ",".join(trace_ids) if shape == "comma_separated" else trace_ids
    result = await _call(client, "delete_traces", experiment_id=experiment_id, trace_ids=value)
    assert result == {"experiment_id": experiment_id, "deleted_count": 2}


@pytest.mark.asyncio
async def test_search_experiments_order_by_accepts_a_comma_separated_string(client):
    for i in range(3):
        MlflowClient().create_experiment(f"exp-{i}")
    old = await _call(client, "search_experiments", order_by="name DESC, experiment_id")
    new = await _call(client, "search_experiments", order_by=["name DESC", "experiment_id"])
    assert old == new
    assert [e["name"] for e in new["experiments"]] == ["exp-2", "exp-1", "exp-0", "Default"]


@pytest.mark.asyncio
async def test_list_runs_order_by_accepts_a_comma_separated_string(client, experiment_id):
    for name in ("b", "a", "c"):
        MlflowClient().create_run(experiment_id, run_name=name)
    old = await _call(
        client, "list_runs", experiment_id=experiment_id, order_by="attributes.run_name, run_id"
    )
    new = await _call(
        client, "list_runs", experiment_id=experiment_id, order_by=["attributes.run_name", "run_id"]
    )
    assert old == new
    assert [r["run_name"] for r in new["runs"]] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_create_run_accepts_key_value_tag_list(client, experiment_id):
    old = await _call(client, "create_run", experiment_id=experiment_id, tags=["a=1", "b=x=y"])
    new = await _call(
        client, "create_run", experiment_id=experiment_id, tags={"a": "1", "b": "x=y"}
    )
    for result in (old, new):
        tags = MlflowClient().get_run(result["run_id"]).data.tags
        assert tags["a"] == "1"
        assert tags["b"] == "x=y"


@pytest.mark.asyncio
async def test_create_run_accepts_lowercase_status(client, experiment_id):
    result = await _call(client, "create_run", experiment_id=experiment_id, status="killed")
    assert result["status"] == "KILLED"


@pytest.mark.parametrize(
    ("old", "new"),
    [("0.9", 0.9), ("true", True), ('{"accuracy": 0.95}', {"accuracy": 0.95}), ("good", "good")],
)
@pytest.mark.asyncio
async def test_log_trace_feedback_accepts_json_string_values(client, experiment_id, old, new):
    trace_id = _trace()
    old_result = await _call(client, "log_trace_feedback", trace_id=trace_id, name="q", value=old)
    new_result = await _call(client, "log_trace_feedback", trace_id=trace_id, name="q", value=new)
    assert old_result["value"] == new_result["value"] == new


@pytest.mark.parametrize("tool", ["log_trace_feedback", "log_trace_expectation"])
@pytest.mark.asyncio
async def test_log_assessment_accepts_json_string_metadata(client, experiment_id, tool):
    trace_id = _trace()
    old = await _call(client, tool, trace_id=trace_id, name="q", value=1, metadata='{"k": "v"}')
    new = await _call(client, tool, trace_id=trace_id, name="q", value=1, metadata={"k": "v"})
    assert old["metadata"] == new["metadata"] == {"k": "v"}


@pytest.mark.asyncio
async def test_log_trace_expectation_accepts_json_string_value(client, experiment_id):
    trace_id = _trace()
    old = await _call(
        client, "log_trace_expectation", trace_id=trace_id, name="e", value='{"answer": "Paris"}'
    )
    new = await _call(
        client, "log_trace_expectation", trace_id=trace_id, name="e", value={"answer": "Paris"}
    )
    assert old["value"] == new["value"] == {"answer": "Paris"}


@pytest.mark.asyncio
async def test_update_trace_assessment_accepts_json_strings(client, experiment_id):
    trace_id = _trace()
    assessment_id = (
        await _call(client, "log_trace_feedback", trace_id=trace_id, name="q", value=1)
    )["assessment_id"]
    old = await _call(
        client,
        "update_trace_assessment",
        trace_id=trace_id,
        assessment_id=assessment_id,
        value='{"accuracy": 0.9}',
        metadata='{"k": "v"}',
    )
    assert old["value"] == {"accuracy": 0.9}
    assert old["metadata"] == {"k": "v"}
    new = await _call(
        client,
        "update_trace_assessment",
        trace_id=trace_id,
        assessment_id=assessment_id,
        value={"accuracy": 0.9},
        metadata={"k": "v"},
    )
    assert new["value"] == old["value"]
    assert new["metadata"] == old["metadata"]


@pytest.mark.parametrize("extra_headers", ['{"X-Team": "a"}', {"X-Team": "a"}])
@pytest.mark.asyncio
async def test_register_llm_judge_accepts_json_string_headers(client, experiment_id, extra_headers):
    result = await _call(
        client,
        "register_llm_judge_scorer",
        name="judge",
        instructions="Is {{ outputs }} good?",
        experiment_id=experiment_id,
        extra_headers=extra_headers,
    )
    assert result == {"name": "judge", "experiment_id": experiment_id}


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("get_experiment", {"experiment_id": "0"}),
        ("search_traces", {"experiment_id": "0"}),
        ("list_scorers", {"builtin": True}),
    ],
)
@pytest.mark.parametrize("output", ["table", "json"])
@pytest.mark.asyncio
async def test_output_format_is_accepted_and_ignored(client, tool, arguments, output):
    assert await _call(client, tool, **arguments, output=output) == await _call(
        client, tool, **arguments
    )


@pytest.mark.asyncio
async def test_search_experiments_without_max_results_returns_the_first_page(client):
    tool = next(t for t in await client.list_tools() if t.name == "search_experiments")
    assert tool.inputSchema["properties"]["max_results"]["default"] == 1000
    for i in range(3):
        MlflowClient().create_experiment(f"exp-{i}")
    result = await _call(client, "search_experiments")
    assert len(result["experiments"]) == 4
    assert result["next_page_token"] is None


@pytest.mark.asyncio
async def test_link_traces_to_run_rejects_more_than_100_trace_ids(client, experiment_id):
    tool = next(t for t in await client.list_tools() if t.name == "link_traces_to_run")
    assert tool.inputSchema["properties"]["trace_ids"]["maxItems"] == 100

    run_id = MlflowClient().create_run(experiment_id).info.run_id
    trace_ids = [f"tr-{i:032x}" for i in range(101)]
    with pytest.raises(ToolError, match="at most 100"):
        await _call(client, "link_traces_to_run", run_id=run_id, trace_ids=trace_ids)


@pytest.mark.parametrize("null", [None, "null"])
@pytest.mark.asyncio
async def test_update_trace_assessment_keeps_the_value_for_null(client, experiment_id, null):
    trace_id = _trace()
    logged = await _call(client, "log_trace_feedback", trace_id=trace_id, name="q", value=1)
    updated = await _call(
        client,
        "update_trace_assessment",
        trace_id=trace_id,
        assessment_id=logged["assessment_id"],
        value=null,
        rationale="kept",
    )
    assert updated["value"] == 1
    assert updated["rationale"] == "kept"


@pytest.mark.parametrize(
    ("tool", "arguments", "default"),
    [
        ("search_experiments", {}, 1000),
        ("list_runs", {"experiment_id": "0"}, 1000),
        ("search_traces", {"experiment_id": "0"}, 100),
    ],
)
@pytest.mark.asyncio
async def test_paged_tools_treat_a_null_max_results_as_omitted(client, tool, arguments, default):
    schema = next(t for t in await client.list_tools() if t.name == tool).inputSchema
    assert schema["properties"]["max_results"]["default"] == default
    omitted = await _call(client, tool, **arguments)
    assert await _call(client, tool, **arguments, max_results=None) == omitted
