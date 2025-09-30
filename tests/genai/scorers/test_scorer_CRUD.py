from unittest.mock import ANY, Mock, patch

import mlflow
from mlflow.genai.scorers import Scorer, scorer
from mlflow.genai.scorers.base import Scorer, ScorerSamplingConfig, ScorerStatus
from mlflow.genai.scorers.registry import (
    delete_scorer,
    get_scorer,
    list_scorer_versions,
    list_scorers,
)


def test_mlflow_backend_scorer_operations():
    """Test all scorer operations with MLflow backend"""

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
    scorer_versions = list_scorer_versions(name="test_mlflow_scorer", experiment_id=experiment_id)
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


@patch("mlflow.tracking._tracking_service.utils.get_tracking_uri", return_value="databricks")
@patch("mlflow.genai.scorers.registry.DatabricksStore.add_registered_scorer")
@patch("mlflow.genai.scorers.registry.DatabricksStore.list_scheduled_scorers")
@patch("mlflow.genai.scorers.registry.DatabricksStore.get_scheduled_scorer")
@patch("mlflow.genai.scorers.registry.DatabricksStore.delete_scheduled_scorer")
def test_databricks_backend_scorer_operations(mock_delete, mock_get, mock_list, mock_add, _):
    """Test all scorer operations with Databricks backend"""

    # Mock the scheduled scorer responses
    mock_scheduled_scorer = Mock()
    mock_scheduled_scorer.scorer = Mock(spec=Scorer)
    mock_scheduled_scorer.scorer.name = "test_databricks_scorer"
    mock_scheduled_scorer.sample_rate = 0.5
    mock_scheduled_scorer.filter_string = "test_filter"

    mock_list.return_value = [mock_scheduled_scorer]
    mock_get.return_value = mock_scheduled_scorer
    mock_delete.return_value = None

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
