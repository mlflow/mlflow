from unittest.mock import ANY, Mock, patch

import mlflow
import mlflow.genai
from mlflow.entities.gateway_endpoint import GatewayEndpoint
from mlflow.genai.scorers import Guidelines, Scorer, scorer
from mlflow.genai.scorers.base import ScorerSamplingConfig, ScorerStatus
from mlflow.genai.scorers.registry import (
    DatabricksStore,
    delete_scorer,
    get_scorer,
    list_scorer_versions,
    list_scorers,
)


def test_scorer_registry_functions_accessible_from_mlflow_genai():
    assert mlflow.genai.get_scorer is get_scorer
    assert mlflow.genai.list_scorers is list_scorers
    assert mlflow.genai.delete_scorer is delete_scorer


def test_mlflow_backend_scorer_operations():
    with (
        patch("mlflow.genai.scorers.base.is_in_databricks_runtime", return_value=True),
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
    mock_scheduled_scorer.scorer = Mock(spec=Scorer)
    mock_scheduled_scorer.scorer.name = "test_databricks_scorer"
    mock_scheduled_scorer.sample_rate = 0.5
    mock_scheduled_scorer.filter_string = "test_filter"

    with (
        patch("mlflow.tracking.get_tracking_uri", return_value="databricks"),
        patch("mlflow.genai.scorers.base.is_in_databricks_runtime", return_value=True),
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
