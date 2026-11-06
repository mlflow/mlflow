import json
from contextlib import nullcontext
from unittest import mock

import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from mlflow.entities import LoggedModel, LoggedModelParameter, LoggedModelTag, Metric
from mlflow.entities.workspace import Workspace
from mlflow.environment_variables import MLFLOW_ENABLE_WORKSPACES
from mlflow.protos import service_pb2
from mlflow.server import app
from mlflow.server.handlers import _search_logged_models
from mlflow.store.entities import PagedList
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.tracking.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyStore
from mlflow.store.workspace.sqlalchemy_store import SqlAlchemyStore as SqlAlchemyWorkspaceStore
from mlflow.utils.proto_json_utils import parse_dict
from mlflow.utils.workspace_context import WorkspaceContext
from mlflow.utils.workspace_utils import DEFAULT_WORKSPACE_NAME


@pytest.fixture(params=[False, True], ids=["workspace-disabled", "workspace-enabled"])
def store(db_uri, tmp_path, request, monkeypatch, disable_workspace_mode_by_default):
    enabled = request.param
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, str(enabled).lower())
    if enabled:
        with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
            yield WorkspaceAwareSqlAlchemyStore(db_uri, str(tmp_path / "artifacts"))
    else:
        yield SqlAlchemyStore(db_uri, str(tmp_path / "artifacts"))


@pytest.mark.parametrize("include_metrics", [None, True, False])
@pytest.mark.parametrize("routed", [False, True], ids=["handler", "http-route"])
def test_search_logged_models_response(store, include_metrics, routed, mlflow_app_client):
    exp = store.create_experiment("synthetic-response")
    run = store.create_run(exp, "test", 0, [], "test")
    models = [
        store.create_logged_model(
            experiment_id=exp,
            name=f"model-{i}",
            tags=[LoggedModelTag("purpose", "test")],
            params=[LoggedModelParameter("width", "128")],
        )
        for i in range(3)
    ]
    for i, model in enumerate(models):
        store.log_metric(
            run.info.run_id,
            Metric(
                "accuracy",
                i / 10,
                123,
                0,
                model_id=model.model_id,
                dataset_name="validation",
                dataset_digest="d",
            ),
        )
    request = {
        "experiment_ids": [exp],
        "filter": "metrics.accuracy >= 0.1",
        "order_by": [{"field_name": "metrics.accuracy", "ascending": False}],
        "datasets": [{"dataset_name": "validation", "dataset_digest": "d"}],
        "max_results": 1,
    }
    if include_metrics is not None:
        request["include_metrics"] = include_metrics
    for expected in [models[2], models[1]]:
        with (
            nullcontext() if routed else app.test_request_context(method="POST", json=request),
            mock.patch(
                "mlflow.server.handlers._get_tracking_store", return_value=store
            ) as get_store,
            mock.patch.object(
                store, "search_logged_models", wraps=store.search_logged_models
            ) as search,
        ):
            response = (
                mlflow_app_client.post("/api/2.0/mlflow/logged-models/search", json=request)
                if routed
                else _search_logged_models()
            )
            get_store.assert_called_once()
            search.assert_called_once()
            assert "include_metrics" not in search.call_args.kwargs
        assert response.status_code == 200
        body = json.loads(response.data)
        assert len(body["models"]) == 1
        model = body["models"][0]
        assert model["info"]["model_id"] == expected.model_id
        assert model["info"]["tags"] == [{"key": "purpose", "value": "test"}]
        assert model["data"]["params"] == [{"key": "width", "value": "128"}]
        if include_metrics is False:
            assert not model["data"].get("metrics")
        else:
            assert model["data"]["metrics"][0]["key"] == "accuracy"
        request["page_token"] = body.get("next_page_token")
    assert not request["page_token"]
    assert store.search_logged_models([exp])[0].metrics


@pytest.mark.parametrize("include_metrics", [None, True, False])
def test_search_logged_models_empty_response(store, include_metrics, mlflow_app_client):
    exp = store.create_experiment("empty-models")
    request = {"experiment_ids": [exp]}
    if include_metrics is not None:
        request["include_metrics"] = include_metrics
    with mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store) as get_store:
        response = mlflow_app_client.post("/ajax-api/2.0/mlflow/logged-models/search", json=request)
        get_store.assert_called_once()
    assert response.status_code == 200
    assert not json.loads(response.data).get("models")


@pytest.mark.parametrize("include_metrics", [None, True, False])
def test_search_logged_models_cross_workspace(
    db_uri, tmp_path, monkeypatch, disable_workspace_mode_by_default, include_metrics
):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    SqlAlchemyWorkspaceStore(db_uri).create_workspace(Workspace(name="other"))
    store = WorkspaceAwareSqlAlchemyStore(db_uri, str(tmp_path / "artifacts"))
    with WorkspaceContext("other"):
        other_exp = store.create_experiment("other-exp")
        store.create_logged_model(other_exp, "other-model")
    with WorkspaceContext(DEFAULT_WORKSPACE_NAME):
        exp = store.create_experiment("default-exp")
        model = store.create_logged_model(exp, "default-model")
        request = {"experiment_ids": [exp, other_exp]}
        if include_metrics is not None:
            request["include_metrics"] = include_metrics
        with (
            app.test_request_context(method="POST", json=request),
            mock.patch(
                "mlflow.server.handlers._get_tracking_store", return_value=store
            ) as get_store,
        ):
            response = _search_logged_models()
            get_store.assert_called_once()
    assert response.status_code == 200
    assert [m["info"]["model_id"] for m in json.loads(response.data)["models"]] == [model.model_id]


@pytest.mark.parametrize("include_metrics", ["false", 0, [], {}])
def test_search_logged_models_invalid_include_metrics(include_metrics):
    with app.test_request_context(
        method="POST", json={"experiment_ids": ["1"], "include_metrics": include_metrics}
    ):
        response = _search_logged_models()
    assert response.status_code == 400
    assert "include_metrics" in json.loads(response.data)["message"]


@pytest.mark.parametrize(
    ("flag", "expected_metrics"),
    [({"include_metrics": None}, 1), ({"includeMetrics": True}, 1), ({"includeMetrics": False}, 0)],
)
def test_search_logged_models_json_names_and_nonmutation(flag, expected_metrics):
    metric = Metric("accuracy", 0.9, 123, 1)
    metrics = [metric]
    model = LoggedModel("1", "m-test", "test", "file:///synthetic", 1, 2, metrics=metrics)
    store = mock.Mock()
    store.search_logged_models.return_value = PagedList([model], token=None)
    with (
        app.test_request_context(method="POST", json={"experiment_ids": ["1"], **flag}),
        mock.patch("mlflow.server.handlers._get_tracking_store", return_value=store),
    ):
        response = _search_logged_models()
    assert response.status_code == 200
    body = json.loads(response.data)
    assert len(body["models"][0]["data"].get("metrics", [])) == expected_metrics
    assert model.metrics is metrics
    assert model.metrics == [metric]
    store.search_logged_models.assert_called_once()
    assert "include_metrics" not in store.search_logged_models.call_args.kwargs


@pytest.mark.parametrize("include_metrics", [True, False])
def test_search_logged_models_new_request_is_accepted_by_old_schema(include_metrics):
    # Reconstruct the previous schema in a separate pool, without field 7.
    schema = descriptor_pb2.FileDescriptorProto()
    service_pb2.DESCRIPTOR.CopyToProto(schema)
    search = next(m for m in schema.message_type if m.name == "SearchLoggedModels")
    search.field.remove(next(f for f in search.field if f.name == "include_metrics"))
    pool = descriptor_pool.DescriptorPool()
    added = set()

    def add_dependencies(descriptor):
        for dependency in descriptor.dependencies:
            if dependency.name not in added:
                add_dependencies(dependency)
                pool.AddSerializedFile(dependency.serialized_pb)
                added.add(dependency.name)

    add_dependencies(service_pb2.DESCRIPTOR)
    pool.Add(schema)
    descriptor = pool.FindMessageTypeByName(f"{schema.package}.SearchLoggedModels")
    old_class = (
        message_factory.GetMessageClass(descriptor)
        if hasattr(message_factory, "GetMessageClass")
        else message_factory.MessageFactory(pool).GetPrototype(descriptor)
    )
    old_message = old_class()
    parse_dict({"experiment_ids": ["1"], "include_metrics": include_metrics}, old_message)
    assert list(old_message.experiment_ids) == ["1"]
    assert "include_metrics" not in old_message.DESCRIPTOR.fields_by_name
    assert old_message.max_results == 50
