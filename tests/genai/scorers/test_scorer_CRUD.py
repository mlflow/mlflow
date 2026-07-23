import json
from unittest.mock import ANY, Mock, patch

import pytest

import mlflow
import mlflow.genai
from mlflow.entities import GatewayEndpointModelConfig, GatewayModelLinkageType
from mlflow.entities.gateway_endpoint import GatewayEndpoint
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import Guidelines, Scorer, list_scorer_versions, scorer
from mlflow.genai.scorers.base import ScorerSamplingConfig, ScorerStatus
from mlflow.genai.scorers.registry import (
    DatabricksStore,
    delete_scorer,
    get_scorer,
    list_scorers,
)
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_DOES_NOT_EXIST, ErrorCode
from mlflow.tracking._tracking_service.utils import _get_store


def test_scorer_registry_functions_accessible_from_mlflow_genai():
    assert mlflow.genai.get_scorer is get_scorer
    assert mlflow.genai.list_scorer_versions is list_scorer_versions
    assert mlflow.genai.list_scorers is list_scorers
    assert mlflow.genai.delete_scorer is delete_scorer


def test_mlflow_backend_scorer_operations():
    with (
        patch("mlflow.genai.scorers.base.is_databricks_uri", return_value=True),
    ):
        experiment_id = mlflow.create_experiment("test_scorer_mlflow_backend_experiment")
        mlflow.set_experiment(experiment_id=experiment_id)

        @scorer
        def test_mlflow_scorer_v1(outputs) -> bool:
            return len(outputs) > 0

        assert test_mlflow_scorer_v1.status == ScorerStatus.UNREGISTERED
        # Test register operation
        registered_scorer_v1 = test_mlflow_scorer_v1.register(
            experiment_id=experiment_id, name="test_mlflow_scorer"
        )

        assert registered_scorer_v1.status == ScorerStatus.STOPPED

        # Register a second version of the scorer
        @scorer
        def test_mlflow_scorer_v2(outputs) -> bool:
            return len(outputs) > 10  # Different logic for v2

        # Register the scorer in the active experiment.
        registered_scorer_v2 = test_mlflow_scorer_v2.register(name="test_mlflow_scorer")
        assert registered_scorer_v2.name == "test_mlflow_scorer"

        # Test list operation
        scorers = list_scorers(experiment_id=experiment_id)
        assert len(scorers) == 1
        assert scorers[0]._original_func.__name__ == "test_mlflow_scorer_v2"

        # Test list versions
        scorer_versions = list_scorer_versions(
            name="test_mlflow_scorer", experiment_id=experiment_id
        )
        assert len(scorer_versions) == 2

        # Test get_scorer with specific version
        retrieved_scorer_v1 = get_scorer(
            name="test_mlflow_scorer", experiment_id=experiment_id, version=1
        )
        assert retrieved_scorer_v1._original_func.__name__ == "test_mlflow_scorer_v1"

        retrieved_scorer_v2 = get_scorer(
            name="test_mlflow_scorer", experiment_id=experiment_id, version=2
        )
        assert retrieved_scorer_v2._original_func.__name__ == "test_mlflow_scorer_v2"

        retrieved_scorer_latest = get_scorer(name="test_mlflow_scorer", experiment_id=experiment_id)
        assert retrieved_scorer_latest._original_func.__name__ == "test_mlflow_scorer_v2"

        # Test delete_scorer with specific version
        delete_scorer(name="test_mlflow_scorer", experiment_id=experiment_id, version=2)
        scorers_after_delete = list_scorers(experiment_id=experiment_id)
        assert len(scorers_after_delete) == 1
        assert scorers_after_delete[0]._original_func.__name__ == "test_mlflow_scorer_v1"

        delete_scorer(name="test_mlflow_scorer", experiment_id=experiment_id, version=1)
        scorers_after_delete = list_scorers(experiment_id=experiment_id)
        assert len(scorers_after_delete) == 0

        # test delete all versions
        test_mlflow_scorer_v1.register(experiment_id=experiment_id, name="test_mlflow_scorer")
        test_mlflow_scorer_v2.register(experiment_id=experiment_id, name="test_mlflow_scorer")
        delete_scorer(name="test_mlflow_scorer", experiment_id=experiment_id, version="all")
        assert len(list_scorers(experiment_id=experiment_id)) == 0

        # Clean up
        mlflow.delete_experiment(experiment_id)


def test_databricks_backend_scorer_operations():
    # Mock the scheduled scorer responses
    mock_scheduled_scorer = Mock()
    mock_scheduled_scorer.scorer = Guidelines(
        name="test_databricks_scorer",
        guidelines=["Be concise"],
        model="databricks:/judge",
    )
    mock_scheduled_scorer.sample_rate = 0.5
    mock_scheduled_scorer.filter_string = "test_filter"

    with (
        patch("mlflow.tracking.get_tracking_uri", return_value="databricks"),
        patch("mlflow.genai.scorers.base.is_databricks_uri", return_value=True),
        patch("mlflow.genai.scorers.registry._get_scorer_store") as mock_get_store,
        patch("mlflow.genai.scorers.registry.DatabricksStore.add_registered_scorer") as mock_add,
        patch(
            "mlflow.genai.scorers.registry.DatabricksStore.list_scheduled_scorers",
            return_value=[mock_scheduled_scorer],
        ) as mock_list,
        patch(
            "mlflow.genai.scorers.registry.DatabricksStore.get_scheduled_scorer",
            return_value=mock_scheduled_scorer,
        ) as mock_get,
        patch(
            "mlflow.genai.scorers.registry.DatabricksStore.delete_scheduled_scorer",
            return_value=None,
        ) as mock_delete,
    ):
        # Set up the store mock
        mock_store = DatabricksStore()
        mock_get_store.return_value = mock_store

        # Test register operation
        @scorer
        def test_databricks_scorer(outputs) -> bool:
            return len(outputs) > 0

        assert test_databricks_scorer.status == ScorerStatus.UNREGISTERED
        registered_scorer = test_databricks_scorer.register(experiment_id="exp_123")
        assert registered_scorer.name == "test_databricks_scorer"
        assert registered_scorer.status == ScorerStatus.STOPPED

        # Verify add_registered_scorer was called during registration
        mock_add.assert_called_once_with(
            name="test_databricks_scorer",
            scorer=ANY,
            sample_rate=0.0,
            filter_string=None,
            experiment_id="exp_123",
        )

        # Test list operation
        scorers = list_scorers(experiment_id="exp_123")

        assert scorers[0].name == "test_databricks_scorer"
        assert scorers[0]._sampling_config == ScorerSamplingConfig(
            sample_rate=0.5, filter_string="test_filter"
        )

        assert len(scorers) == 1
        mock_list.assert_called_once_with("exp_123")

        # Test get operation
        retrieved_scorer = get_scorer(name="test_databricks_scorer", experiment_id="exp_123")
        assert retrieved_scorer.name == "test_databricks_scorer"
        mock_get.assert_called_once_with("test_databricks_scorer", "exp_123")

        # Test delete operation
        delete_scorer(name="test_databricks_scorer", experiment_id="exp_123")
        mock_delete.assert_called_once_with("exp_123", "test_databricks_scorer")


def _mock_response(payload):
    response = Mock(status_code=200, text=json.dumps(payload))
    response.json.return_value = payload
    return response


def _scheduled_scorers_response(configs):
    return _mock_response({"scheduled_scorers": {"scorers": configs}})


def _scorer_config(scorer, *, version=1, sample_rate=0.0, filter_string=None):
    config = {
        "name": scorer.name,
        "serialized_scorer": json.dumps(scorer.model_dump()),
        "builtin": {"name": scorer.name},
        "sample_rate": sample_rate,
        "scorer_version": version,
    }
    if filter_string is not None:
        config["filter_string"] = filter_string
    return config


def _scorer_version_config(scorer, *, experiment_id="exp_123", version=1):
    scorer_key = DatabricksStore._scorer_resource_key(scorer.name)
    return {
        "name": f"experiments/{experiment_id}/scorers/{scorer_key}/versions/{version}",
        "display_name": scorer.name,
        "scorer_version": version,
        "serialized_scorer": json.dumps(scorer.model_dump()),
        "create_time": "2026-07-20T00:00:00Z",
        "builtin": {"name": scorer.name},
    }


def test_databricks_backend_version_resource_names_use_unpadded_base64url():
    store = DatabricksStore(tracking_uri="databricks")

    assert DatabricksStore._scorer_resource_key("quality/foo") == "cXVhbGl0eS9mb28"
    assert DatabricksStore._scorer_resource_key("café/品質") == "Y2Fmw6kv5ZOB6LOq"
    assert store._scorer_versions_endpoint("123", "quality/foo") == (
        "/api/2.0/managed-evals/experiments/123/scorers/cXVhbGl0eS9mb28/versions"
    )
    assert store._scorer_version_endpoint("exp/123", "quality/foo", 7) == (
        "/api/2.0/managed-evals/experiments/exp%2F123/scorers/cXVhbGl0eS9mb28/versions/7"
    )


@pytest.mark.parametrize("version", [0, -1, 1.5, "1", True])
def test_databricks_backend_exact_version_operations_require_positive_integer(version):
    with patch("mlflow.genai.scorers.registry.http_request") as mock_http:
        store = DatabricksStore(tracking_uri="databricks")

        with pytest.raises(MlflowException, match="must be a positive integer"):
            store.get_scorer("exp_123", "test_scorer", version=version)
        with pytest.raises(MlflowException, match="must be a positive integer"):
            store.delete_scorer("exp_123", "test_scorer", version=version)

    mock_http.assert_not_called()


@pytest.mark.parametrize("version", [None, "all"])
def test_databricks_backend_delete_all_uses_scheduled_scorer_delete(version):
    with (
        patch(
            "mlflow.genai.scorers.registry.DatabricksStore.delete_scheduled_scorer"
        ) as mock_delete,
        patch("mlflow.genai.scorers.registry.http_request") as mock_http,
    ):
        DatabricksStore(tracking_uri="databricks").delete_scorer(
            "exp_123", "test_scorer", version=version
        )

    mock_delete.assert_called_once_with("exp_123", "test_scorer")
    mock_http.assert_not_called()


@pytest.mark.parametrize("next_page_token", [0, False, "", 123, "same-token"])
def test_databricks_backend_pagination_rejects_invalid_next_page_token(next_page_token):
    with (
        patch("mlflow.genai.scorers.registry.get_databricks_host_creds", return_value="creds"),
        patch("mlflow.genai.scorers.registry.http_request") as mock_http,
    ):
        mock_http.return_value = _mock_response({"next_page_token": next_page_token})
        store = DatabricksStore(tracking_uri="databricks")

        with pytest.raises(MlflowException, match="invalid `next_page_token`") as error:
            store._get_paginated_results(
                "/api/2.0/test",
                lambda response: [],
            )

    assert error.value.error_code == ErrorCode.Name(INTERNAL_ERROR)


def test_databricks_backend_config_response_errors_use_oss_error_codes():
    store = DatabricksStore(tracking_uri="databricks")

    with pytest.raises(MlflowException, match="Serialized scorer data is required") as malformed:
        Scorer._from_serialized_scorer(None)
    assert malformed.value.error_code == ErrorCode.Name(INTERNAL_ERROR)

    with (
        patch.object(store, "_list_current_scorer_configs", return_value=[]),
        pytest.raises(MlflowException, match="not found") as missing,
    ):
        store._find_current_scorer_config("exp_123", "test_scorer")
    assert missing.value.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST)


def test_databricks_backend_version_operations_use_managed_resource_endpoints():
    scorer_name = "folder/test_databricks_scorer"
    scorer_v1 = Guidelines(name=scorer_name, guidelines=["v1"], model="databricks:/judge")
    scorer_v2 = Guidelines(name=scorer_name, guidelines=["v2"], model="databricks:/judge")
    current_config = _scorer_config(
        scorer_v2,
        version=2,
        sample_rate=0.5,
        filter_string="trace.status = 'OK'",
    )
    v1_version_config = _scorer_version_config(scorer_v1, version=1)
    v2_version_config = _scorer_version_config(scorer_v2, version=2)
    v1_version_config.pop("display_name")
    v2_version_config.pop("display_name")

    with (
        patch("mlflow.genai.scorers.registry.get_databricks_host_creds", return_value="creds"),
        patch("mlflow.genai.scorers.registry.http_request") as mock_http,
    ):
        mock_http.side_effect = [
            _mock_response(v1_version_config),
            _scheduled_scorers_response([current_config]),
            _mock_response({"scorer_versions": [v1_version_config, v2_version_config]}),
            _scheduled_scorers_response([current_config]),
            _mock_response({}),
        ]
        store = DatabricksStore(tracking_uri="databricks://profile")

        exact = store.get_scorer("exp_123", scorer_name, version=1)
        versions = store.list_scorer_versions("exp_123", scorer_name)
        store.delete_scorer("exp_123", scorer_name, version=1)

    assert exact.name == scorer_name
    assert exact._sampling_config == ScorerSamplingConfig(
        sample_rate=0.5,
        filter_string="trace.status = 'OK'",
    )
    assert [version for _, version in versions] == [1, 2]
    assert [scorer.name for scorer, _ in versions] == [scorer_name, scorer_name]

    scorer_key = "Zm9sZGVyL3Rlc3RfZGF0YWJyaWNrc19zY29yZXI"
    assert mock_http.call_args_list[0].kwargs["endpoint"] == (
        f"/api/2.0/managed-evals/experiments/exp_123/scorers/{scorer_key}/versions/1"
    )
    assert mock_http.call_args_list[2].kwargs["endpoint"] == (
        f"/api/2.0/managed-evals/experiments/exp_123/scorers/{scorer_key}/versions"
    )
    assert mock_http.call_args_list[4].kwargs["method"] == "DELETE"
    assert mock_http.call_args_list[4].kwargs["endpoint"] == (
        f"/api/2.0/managed-evals/experiments/exp_123/scorers/{scorer_key}/versions/1"
    )


def test_databricks_backend_list_scorer_versions_paginates():
    scorer_name = "folder/test_databricks_scorer"
    scorer_v1 = Guidelines(name=scorer_name, guidelines=["v1"], model="databricks:/judge")
    scorer_v2 = Guidelines(name=scorer_name, guidelines=["v2"], model="databricks:/judge")
    v1_version_config = _scorer_version_config(scorer_v1, version=1)
    v2_version_config = _scorer_version_config(scorer_v2, version=2)

    with (
        patch("mlflow.genai.scorers.registry.get_databricks_host_creds", return_value="creds"),
        patch("mlflow.genai.scorers.registry.http_request") as mock_http,
    ):
        mock_http.side_effect = [
            _mock_response({
                "scorer_versions": [v1_version_config],
                "next_page_token": "page-2",
            }),
            _mock_response({"scorer_versions": [v2_version_config]}),
            _scheduled_scorers_response([_scorer_config(scorer_v2, version=2)]),
        ]

        versions = DatabricksStore(tracking_uri="databricks").list_scorer_versions(
            "exp_123", scorer_name
        )

    assert [version for _, version in versions] == [1, 2]
    assert mock_http.call_args_list[0].kwargs["params"] is None
    assert mock_http.call_args_list[1].kwargs["params"] == {"page_token": "page-2"}


def test_databricks_backend_current_scorer_configs_paginate():
    scorer_v1 = Guidelines(name="scorer_v1", guidelines=["v1"], model="databricks:/judge")
    scorer_v2 = Guidelines(name="scorer_v2", guidelines=["v2"], model="databricks:/judge")

    with (
        patch("mlflow.genai.scorers.registry.get_databricks_host_creds", return_value="creds"),
        patch("mlflow.genai.scorers.registry.http_request") as mock_http,
    ):
        mock_http.side_effect = [
            _mock_response({
                "scheduled_scorers": {"scorers": [_scorer_config(scorer_v1)]},
                "next_page_token": "page-2",
            }),
            _scheduled_scorers_response([_scorer_config(scorer_v2)]),
        ]

        configs = DatabricksStore(tracking_uri="databricks")._list_current_scorer_configs("exp_123")

    assert [config.name for config in configs] == ["scorer_v1", "scorer_v2"]
    assert mock_http.call_args_list[0].kwargs["params"] is None
    assert mock_http.call_args_list[1].kwargs["params"] == {"page_token": "page-2"}


def _mock_gateway_endpoint():
    """Returns a mock GatewayEndpoint for testing."""
    return GatewayEndpoint(
        endpoint_id="test-endpoint-id",
        name="test-endpoint",
        created_at=0,
        last_updated_at=0,
    )


def test_mlflow_backend_online_scoring_config_operations():
    experiment_id = mlflow.create_experiment("test_online_scoring_config_experiment")
    mlflow.set_experiment(experiment_id=experiment_id)

    test_scorer = Guidelines(
        name="test_online_config_scorer",
        guidelines=["Be helpful"],
        model="gateway:/test-endpoint",
    )

    with patch(
        "mlflow.store.tracking.sqlalchemy_store.SqlAlchemyStore.get_gateway_endpoint",
        return_value=_mock_gateway_endpoint(),
    ):
        registered_scorer = test_scorer.register(experiment_id=experiment_id)
        assert registered_scorer.sample_rate is None
        assert registered_scorer.filter_string is None
        assert registered_scorer.status == ScorerStatus.STOPPED

        started_scorer = registered_scorer.start(
            experiment_id=experiment_id,
            sampling_config=ScorerSamplingConfig(sample_rate=0.75, filter_string="status = 'OK'"),
        )
        assert started_scorer.sample_rate == 0.75
        assert started_scorer.filter_string == "status = 'OK'"
        assert started_scorer.status == ScorerStatus.STARTED

        retrieved_scorer = get_scorer(name="test_online_config_scorer", experiment_id=experiment_id)
        assert retrieved_scorer.sample_rate == 0.75
        assert retrieved_scorer.filter_string == "status = 'OK'"
        assert retrieved_scorer.status == ScorerStatus.STARTED

        scorers = list_scorers(experiment_id=experiment_id)
        assert len(scorers) == 1
        assert scorers[0].sample_rate == 0.75
        assert scorers[0].filter_string == "status = 'OK'"
        assert scorers[0].status == ScorerStatus.STARTED

        scorer_versions = list_scorer_versions(
            name="test_online_config_scorer", experiment_id=experiment_id
        )
        assert len(scorer_versions) == 1
        scorer_from_versions, version = scorer_versions[0]
        assert scorer_from_versions.sample_rate == 0.75
        assert scorer_from_versions.filter_string == "status = 'OK'"
        assert version == 1


def test_mlflow_backend_online_scoring_config_chained_update():
    with patch(
        "mlflow.store.tracking.sqlalchemy_store.SqlAlchemyStore.get_gateway_endpoint",
        return_value=_mock_gateway_endpoint(),
    ):
        experiment_id = mlflow.create_experiment("test_scorer_chained_update_experiment")
        mlflow.set_experiment(experiment_id=experiment_id)

        test_scorer = Guidelines(
            name="test_chained_scorer",
            guidelines=["Be helpful"],
            model="gateway:/test-endpoint",
        )

        registered_scorer = test_scorer.register(experiment_id=experiment_id)
        started_scorer = registered_scorer.start(
            experiment_id=experiment_id,
            sampling_config=ScorerSamplingConfig(sample_rate=0.5),
        )
        assert started_scorer.sample_rate == 0.5
        assert started_scorer.filter_string is None

        updated_scorer = get_scorer(name="test_chained_scorer", experiment_id=experiment_id).update(
            experiment_id=experiment_id,
            sampling_config=ScorerSamplingConfig(sample_rate=0.8, filter_string="status = 'OK'"),
        )
        assert updated_scorer.sample_rate == 0.8
        assert updated_scorer.filter_string == "status = 'OK'"
        assert updated_scorer.status == ScorerStatus.STARTED

        retrieved_scorer = get_scorer(name="test_chained_scorer", experiment_id=experiment_id)
        assert retrieved_scorer.sample_rate == 0.8
        assert retrieved_scorer.filter_string == "status = 'OK'"

        stopped_scorer = get_scorer(name="test_chained_scorer", experiment_id=experiment_id).stop(
            experiment_id=experiment_id
        )
        assert stopped_scorer.sample_rate == 0.0
        assert stopped_scorer.status == ScorerStatus.STOPPED

        retrieved_after_stop = get_scorer(name="test_chained_scorer", experiment_id=experiment_id)
        assert retrieved_after_stop.sample_rate == 0.0
        assert retrieved_after_stop.status == ScorerStatus.STOPPED

        restarted_scorer = get_scorer(
            name="test_chained_scorer", experiment_id=experiment_id
        ).start(
            experiment_id=experiment_id,
            sampling_config=ScorerSamplingConfig(sample_rate=0.3),
        )
        assert restarted_scorer.sample_rate == 0.3
        assert restarted_scorer.status == ScorerStatus.STARTED

        retrieved_after_restart = get_scorer(
            name="test_chained_scorer", experiment_id=experiment_id
        )
        assert retrieved_after_restart.sample_rate == 0.3
        assert retrieved_after_restart.status == ScorerStatus.STARTED


def _setup_gateway_endpoint(store):
    secret = store.create_gateway_secret(
        secret_name="test-binding-secret",
        secret_value={"api_key": "test-key"},
        provider="openai",
    )

    model_def = store.create_gateway_model_definition(
        name="test-binding-model",
        secret_id=secret.secret_id,
        provider="openai",
        model_name="gpt-4",
    )

    return store.create_gateway_endpoint(
        name="test-binding-endpoint",
        model_configs=[
            GatewayEndpointModelConfig(
                model_definition_id=model_def.model_definition_id,
                linkage_type=GatewayModelLinkageType.PRIMARY,
                weight=1.0,
            ),
        ],
    )


def test_register_scorer_creates_endpoint_binding(monkeypatch):
    monkeypatch.setenv("MLFLOW_CRYPTO_KEK_PASSPHRASE", "test-passphrase-for-binding-tests")

    experiment_id = mlflow.create_experiment("test_binding_creation_experiment")
    mlflow.set_experiment(experiment_id=experiment_id)

    store = _get_store()

    endpoint = _setup_gateway_endpoint(store)

    test_scorer = Guidelines(
        name="test_binding_scorer",
        guidelines=["Be helpful"],
        model=f"gateway:/{endpoint.name}",
    )

    # Binding should be created at registration time
    registered_scorer = test_scorer.register(experiment_id=experiment_id)
    assert registered_scorer.status == ScorerStatus.STOPPED

    bindings = store.list_endpoint_bindings(endpoint_id=endpoint.endpoint_id)
    assert len(bindings) == 1
    assert bindings[0].resource_type == "scorer"
    assert bindings[0].endpoint_id == endpoint.endpoint_id
    assert bindings[0].display_name == "test_binding_scorer"  # Scorer name

    # Binding should persist even after stopping the scorer
    # (stopping only changes sample_rate, not the endpoint reference)
    started_scorer = registered_scorer.start(
        experiment_id=experiment_id,
        sampling_config=ScorerSamplingConfig(sample_rate=0.5),
    )
    assert started_scorer.status == ScorerStatus.STARTED

    stopped_scorer = started_scorer.stop(experiment_id=experiment_id)
    assert stopped_scorer.status == ScorerStatus.STOPPED

    # Binding should still exist after stopping
    bindings_after_stop = store.list_endpoint_bindings(endpoint_id=endpoint.endpoint_id)
    assert len(bindings_after_stop) == 1

    mlflow.delete_experiment(experiment_id)


def test_delete_scorer_removes_endpoint_binding(monkeypatch):
    monkeypatch.setenv("MLFLOW_CRYPTO_KEK_PASSPHRASE", "test-passphrase-for-binding-tests")

    experiment_id = mlflow.create_experiment("test_binding_deletion_experiment")
    mlflow.set_experiment(experiment_id=experiment_id)

    store = _get_store()

    endpoint = _setup_gateway_endpoint(store)

    test_scorer = Guidelines(
        name="test_delete_binding_scorer",
        guidelines=["Be helpful"],
        model=f"gateway:/{endpoint.name}",
    )

    test_scorer.register(experiment_id=experiment_id)

    bindings = store.list_endpoint_bindings(endpoint_id=endpoint.endpoint_id)
    assert len(bindings) == 1

    delete_scorer(name="test_delete_binding_scorer", experiment_id=experiment_id, version="all")

    bindings_after_delete = store.list_endpoint_bindings(endpoint_id=endpoint.endpoint_id)
    assert len(bindings_after_delete) == 0

    mlflow.delete_experiment(experiment_id)
