from types import SimpleNamespace
from unittest import mock

import pytest
from flask import Flask, g

from mlflow.demo.base import DEMO_EXPERIMENT_NAME
from mlflow.server import ARTIFACTS_ONLY_ENV_VAR, handlers


@pytest.fixture
def demo_request_context(monkeypatch):
    monkeypatch.delenv(ARTIFACTS_ONLY_ENV_VAR, raising=False)
    with Flask(__name__).test_request_context(
        "/ajax-api/3.0/mlflow/demo/generate",
        method="POST",
        json={"features": ["prompts"]},
    ):
        yield


@pytest.mark.parametrize("refresh_allowed", [None, False, True])
def test_generate_demo_returns_existing(demo_request_context, refresh_allowed):
    if refresh_allowed is not None:
        g.mlflow_demo_refresh_allowed = refresh_allowed
    experiment = SimpleNamespace(experiment_id="123", lifecycle_stage="active")
    store = SimpleNamespace(get_experiment_by_name=mock.Mock(return_value=experiment))

    with (
        mock.patch.object(handlers, "_get_tracking_store", return_value=store) as get_store,
        mock.patch("mlflow.demo.generate_all_demos", return_value=[]) as generate,
    ):
        response = handlers._generate_demo()

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "exists",
        "experiment_id": "123",
        "features_generated": [],
        "navigation_url": "/experiments/123",
    }
    get_store.assert_called_once_with()
    assert store.get_experiment_by_name.call_args_list == [
        mock.call(DEMO_EXPERIMENT_NAME),
        mock.call(DEMO_EXPERIMENT_NAME),
    ]
    generate.assert_called_once_with(features=["prompts"], can_refresh=mock.ANY)
    assert generate.call_args.kwargs["can_refresh"](object()) is (refresh_allowed is not False)


@pytest.mark.parametrize("experiment_exists", [False, True])
@pytest.mark.parametrize("refresh_allowed", [False, True])
def test_generate_demo_creates_missing_feature(
    demo_request_context, experiment_exists, refresh_allowed
):
    g.mlflow_demo_refresh_allowed = refresh_allowed
    experiment = SimpleNamespace(experiment_id="123", lifecycle_stage="active")
    store = SimpleNamespace(
        get_experiment_by_name=mock.Mock(
            side_effect=[experiment if experiment_exists else None, experiment]
        )
    )

    with (
        mock.patch.object(handlers, "_get_tracking_store", return_value=store) as get_store,
        mock.patch(
            "mlflow.demo.generate_all_demos", return_value=[SimpleNamespace(feature="prompts")]
        ) as generate,
    ):
        response = handlers._generate_demo()

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "created",
        "experiment_id": "123",
        "features_generated": ["prompts"],
        "navigation_url": "/experiments/123",
    }
    get_store.assert_called_once_with()
    assert store.get_experiment_by_name.call_args_list == [
        mock.call(DEMO_EXPERIMENT_NAME),
        mock.call(DEMO_EXPERIMENT_NAME),
    ]
    generate.assert_called_once_with(features=["prompts"], can_refresh=mock.ANY)
    assert generate.call_args.kwargs["can_refresh"](object()) is refresh_allowed


def test_generate_demo_rejects_unauthorized_restoration(demo_request_context):
    g.mlflow_demo_refresh_allowed = False
    experiment = SimpleNamespace(experiment_id="123", lifecycle_stage="deleted")
    store = SimpleNamespace(get_experiment_by_name=mock.Mock(return_value=experiment))

    with (
        mock.patch.object(handlers, "_get_tracking_store", return_value=store) as get_store,
        mock.patch("mlflow.demo.generate_all_demos") as generate,
    ):
        response = handlers._generate_demo()

    assert response.status_code == 403
    assert response.get_json()["error_code"] == "PERMISSION_DENIED"
    assert response.get_json()["message"] == (
        "You do not have permission to restore the demo experiment."
    )
    get_store.assert_called_once_with()
    store.get_experiment_by_name.assert_called_once_with(DEMO_EXPERIMENT_NAME)
    generate.assert_not_called()


@pytest.mark.parametrize("refresh_allowed", [None, True])
def test_generate_demo_allows_authorized_restoration(demo_request_context, refresh_allowed):
    if refresh_allowed is not None:
        g.mlflow_demo_refresh_allowed = refresh_allowed
    experiment = SimpleNamespace(experiment_id="123", lifecycle_stage="active")
    store = SimpleNamespace(
        get_experiment_by_name=mock.Mock(
            side_effect=[
                SimpleNamespace(experiment_id="123", lifecycle_stage="deleted"),
                experiment,
            ]
        )
    )

    with (
        mock.patch.object(handlers, "_get_tracking_store", return_value=store) as get_store,
        mock.patch(
            "mlflow.demo.generate_all_demos", return_value=[SimpleNamespace(feature="prompts")]
        ) as generate,
    ):
        response = handlers._generate_demo()

    assert response.status_code == 200
    assert response.get_json()["status"] == "created"
    get_store.assert_called_once_with()
    assert store.get_experiment_by_name.call_args_list == [
        mock.call(DEMO_EXPERIMENT_NAME),
        mock.call(DEMO_EXPERIMENT_NAME),
    ]
    generate.assert_called_once_with(features=["prompts"], can_refresh=mock.ANY)
    assert generate.call_args.kwargs["can_refresh"](object()) is True
