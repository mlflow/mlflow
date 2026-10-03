import pytest

import mlflow
from mlflow import MlflowClient

from tests.mcp.helpers import stdio_client
from tests.tracking.integration_test_utils import _init_server


@pytest.fixture
def tracking_server(db_uri, tmp_path):
    with _init_server(
        backend_uri=db_uri, root_artifact_uri=tmp_path.joinpath("artifacts").as_uri()
    ) as url:
        mlflow.set_tracking_uri(url)
        yield url


@pytest.mark.asyncio
async def test_stdio_tools_work_against_a_remote_tracking_server(tracking_server):
    client = MlflowClient(tracking_server)

    async with stdio_client(tracking_server) as mcp:

        async def call(tool, **arguments):
            return (await mcp.call_tool(tool, arguments)).structured_content

        experiment = await call("create_experiment", experiment_name="remote-exp")
        experiment_id = experiment["experiment_id"]
        assert client.get_experiment(experiment_id).name == "remote-exp"

        run = await call("create_run", experiment_id=experiment_id, run_name="r", tags={"k": "v"})
        assert (await call("describe_run", run_id=run["run_id"]))["tags"]["k"] == "v"
        assert [
            r["run_id"] for r in (await call("list_runs", experiment_id=experiment_id))["runs"]
        ] == [run["run_id"]]
        assert client.get_run(run["run_id"]).info.status == "FINISHED"

        mlflow.set_experiment(experiment_id=experiment_id)
        with mlflow.start_span("span") as span:
            span.set_inputs({"q": "hi"})
        trace_id = span.trace_id

        page = await call("search_traces", experiment_id=experiment_id)
        assert [t["info"]["trace_id"] for t in page["traces"]] == [trace_id]

        await call("set_trace_tag", trace_id=trace_id, key="env", value="prod")
        feedback = await call(
            "log_trace_feedback",
            trace_id=trace_id,
            name="quality",
            value=1,
            rationale="ok",
            metadata={"confidence": 0.9, "round": 2},
        )
        trace = (await call("get_trace", trace_id=trace_id))["trace"]
        assert trace["info"]["tags"]["env"] == "prod"
        assert [a["assessment_id"] for a in trace["info"]["assessments"]] == [
            feedback["assessment_id"]
        ]
        # Metadata is a string map over REST; the tool stores and returns the same strings the
        # endpoint does on a direct store connection.
        stored = mlflow.get_assessment(trace_id, feedback["assessment_id"]).metadata
        assert feedback["metadata"] == stored == {"confidence": "0.9", "round": "2"}
        updated = await call(
            "update_trace_assessment",
            trace_id=trace_id,
            assessment_id=feedback["assessment_id"],
            metadata={"confidence": 0.5, "round": 3},
        )
        stored = mlflow.get_assessment(trace_id, feedback["assessment_id"]).metadata
        assert updated["metadata"] == stored == {"confidence": "0.5", "round": "3"}

        names = [e["name"] for e in (await call("search_experiments"))["experiments"]]
        assert "remote-exp" in names
