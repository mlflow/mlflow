import json

import pytest

import mlflow
from mlflow import MlflowClient
from mlflow.entities import LoggedModelInput
from mlflow.exceptions import MlflowException
from mlflow.mcp.request_context import MCP_HTTP_REQUEST, MCP_REQUEST_USERNAME
from mlflow.mcp.tools.runs import (
    create_run,
    delete_run,
    describe_run,
    link_traces_to_run,
    list_runs,
    restore_run,
)
from mlflow.utils.mlflow_tags import (
    MLFLOW_PARENT_RUN_ID,
    MLFLOW_RUN_NOTE,
    MLFLOW_SOURCE_NAME,
    MLFLOW_USER,
)


@pytest.fixture
def experiment_id():
    return MlflowClient().create_experiment("exp")


@pytest.fixture
def http_request():
    def enter(username):
        MCP_HTTP_REQUEST.set(True)
        MCP_REQUEST_USERNAME.set(username)

    http_token = MCP_HTTP_REQUEST.set(False)
    username_token = MCP_REQUEST_USERNAME.set(None)
    yield enter
    MCP_HTTP_REQUEST.reset(http_token)
    MCP_REQUEST_USERNAME.reset(username_token)


def test_create_run_uses_the_client_and_leaves_fluent_state_alone(experiment_id):
    created = create_run(
        experiment_id=experiment_id,
        run_name="baseline",
        description="first try",
        tags={"env": "prod"},
        status="failed",
    )
    assert created.status == "FAILED"
    assert created.run_name == "baseline"
    assert mlflow.active_run() is None
    assert mlflow.tracking.fluent._active_experiment_id is None

    run = MlflowClient().get_run(created.run_id)
    assert run.info.status == "FAILED"
    assert run.info.end_time is not None
    assert run.data.tags["env"] == "prod"
    assert run.data.tags[MLFLOW_RUN_NOTE] == "first try"
    # Created from the caller's machine: local context tags, like ``mlflow.start_run``.
    assert MLFLOW_SOURCE_NAME in run.data.tags
    assert MLFLOW_USER in run.data.tags


def test_create_run_by_name_creates_a_missing_experiment():
    created = create_run(experiment_name="new-exp")
    assert MlflowClient().get_experiment_by_name("new-exp").experiment_id == created.experiment_id
    assert create_run(experiment_name="new-exp").experiment_id == created.experiment_id


def test_create_run_over_http_is_attributed_to_the_authenticated_user(experiment_id, http_request):
    http_request("alice")
    run = MlflowClient().get_run(create_run(experiment_id=experiment_id).run_id)
    assert run.info.user_id == "alice"
    assert run.data.tags[MLFLOW_USER] == "alice"
    # The server process's own context says nothing about the caller.
    assert MLFLOW_SOURCE_NAME not in run.data.tags


def test_create_run_over_http_without_identity_does_not_use_the_server_user(
    experiment_id, http_request
):
    http_request(None)
    run = MlflowClient().get_run(create_run(experiment_id=experiment_id).run_id)
    assert run.info.user_id == "unknown"
    assert MLFLOW_USER not in run.data.tags


@pytest.mark.parametrize(
    "parent_arguments",
    [
        lambda parent: {"parent_run_id": parent},
        lambda parent: {"tags": {MLFLOW_PARENT_RUN_ID: parent}},
        lambda parent: {"tags": [f"{MLFLOW_PARENT_RUN_ID}={parent}"]},
    ],
    ids=["argument", "tag-object", "tag-list"],
)
def test_create_run_nests_under_parent(experiment_id, parent_arguments):
    parent = create_run(experiment_id=experiment_id)
    child = create_run(experiment_id=experiment_id, **parent_arguments(parent.run_id))
    tags = MlflowClient().get_run(child.run_id).data.tags
    assert tags[MLFLOW_PARENT_RUN_ID] == parent.run_id

    # Every way of naming the parent goes through the same validation.
    with pytest.raises(MlflowException, match="no-such-run"):
        create_run(experiment_id=experiment_id, **parent_arguments("no-such-run"))
    delete_run(parent.run_id)
    with pytest.raises(MlflowException, match="deleted state"):
        create_run(experiment_id=experiment_id, **parent_arguments(parent.run_id))


def test_create_run_rejects_a_parent_argument_and_tag_that_disagree(experiment_id):
    parent = create_run(experiment_id=experiment_id)
    other = create_run(experiment_id=experiment_id)
    with pytest.raises(MlflowException, match="name different runs"):
        create_run(
            experiment_id=experiment_id,
            parent_run_id=parent.run_id,
            tags={MLFLOW_PARENT_RUN_ID: other.run_id},
        )
    child = create_run(
        experiment_id=experiment_id,
        parent_run_id=parent.run_id,
        tags={MLFLOW_PARENT_RUN_ID: parent.run_id},
    )
    assert MlflowClient().get_run(child.run_id).data.tags[MLFLOW_PARENT_RUN_ID] == parent.run_id


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "exactly one of experiment_id or experiment_name"),
        ({"experiment_id": "0", "experiment_name": "Default"}, "exactly one of"),
        ({"experiment_id": "0", "status": "RUNNING"}, "Invalid status"),
        ({"experiment_id": "0", "tags": ["novalue"]}, "key=value format"),
        ({"experiment_id": "0", "tags": ["a=1", "a=2"]}, "Duplicate tag key"),
        (
            {"experiment_id": "0", "tags": {MLFLOW_RUN_NOTE: "x"}, "description": "y"},
            "Description is already set",
        ),
    ],
)
def test_create_run_validates_inputs(kwargs, match):
    with pytest.raises(MlflowException, match=match):
        create_run(**kwargs)


def test_list_runs_pages_and_filters_by_view(experiment_id):
    run_ids = [create_run(experiment_id=experiment_id).run_id for _ in range(3)]
    delete_run(run_ids[0])

    collected = []
    page_token = None
    while True:
        page = list_runs(experiment_id, max_results=1, page_token=page_token)
        collected.extend(run.run_id for run in page.runs)
        if not (page_token := page.next_page_token):
            break
    assert sorted(collected) == sorted(run_ids[1:])

    assert [r.run_id for r in list_runs(experiment_id, view="deleted_only").runs] == [run_ids[0]]
    restore_run(run_ids[0])
    assert len(list_runs(experiment_id, view="all").runs) == 3


def test_list_runs_filter_and_order(experiment_id):
    create_run(experiment_id=experiment_id, run_name="b", tags={"k": "1"})
    create_run(experiment_id=experiment_id, run_name="a", tags={"k": "1"})
    create_run(experiment_id=experiment_id, run_name="c", tags={"k": "2"})
    page = list_runs(experiment_id, filter_string="tags.k = '1'", order_by=["attributes.run_name"])
    assert [r.run_name for r in page.runs] == ["a", "b"]


def test_describe_run(experiment_id):
    client = MlflowClient()
    run_id = client.create_run(experiment_id, run_name="r").info.run_id
    client.log_metric(run_id, "rmse", 0.5)
    client.log_param(run_id, "alpha", "1")

    details = describe_run(run_id)
    assert details.run_id == run_id
    assert details.run_name == "r"
    assert details.metrics == {"rmse": 0.5}
    assert details.params == {"alpha": "1"}


def test_describe_run_serializes_logged_model_inputs(experiment_id):
    client = MlflowClient()
    run_id = client.create_run(experiment_id).info.run_id
    model_id = client.create_logged_model(experiment_id).model_id
    client.log_inputs(run_id, models=[LoggedModelInput(model_id)])

    details = describe_run(run_id)
    assert details.inputs == {"model_inputs": [{"model_id": model_id}], "dataset_inputs": []}
    assert json.loads(details.model_dump_json())["inputs"] == details.inputs


def test_link_traces_to_run(experiment_id):
    run_id = create_run(experiment_id=experiment_id).run_id
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_span("span") as span:
        pass

    result = link_traces_to_run(run_id, [span.trace_id])
    assert result.trace_ids == [span.trace_id]
    traces = mlflow.search_traces(locations=[experiment_id], run_id=run_id, return_type="list")
    assert [t.info.trace_id for t in traces] == [span.trace_id]
