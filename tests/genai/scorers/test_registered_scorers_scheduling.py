from unittest.mock import patch

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import Guidelines, scorer
from mlflow.genai.scorers.base import Scorer, ScorerSamplingConfig


@pytest.fixture(autouse=True)
def mock_databricks_runtime():
    from mlflow.genai.scorers.registry import DatabricksStore

    with (
        patch("mlflow.tracking.get_tracking_uri", return_value="databricks"),
        patch(
            "mlflow.tracking._tracking_service.utils.get_tracking_uri", return_value="databricks"
        ),
        patch("mlflow.genai.scorers.base.is_in_databricks_runtime", return_value=True),
        patch("mlflow.genai.scorers.base.is_databricks_uri", return_value=True),
        patch("mlflow.genai.scorers.registry._get_scorer_store") as mock_get_store,
    ):
        mock_store = DatabricksStore()
        mock_get_store.return_value = mock_store
        yield


@scorer
def length_check(outputs):
    """Check if response is adequately detailed"""
    return len(str(outputs)) > 100


@scorer
def serialization_scorer(outputs) -> bool:
    """Scorer for serialization tests"""
    return len(outputs) > 5


def test_scorer_register():
    """Test registering a scorer."""
    my_scorer = length_check
    with patch("mlflow.genai.scorers.registry.DatabricksStore.add_registered_scorer") as mock_add:
        registered = my_scorer.register(name="my_length_check")

    # Check immutability - returns new instance
    assert registered is not my_scorer
    assert registered.name == "my_length_check"
    assert registered._sampling_config.sample_rate == 0.0
    assert registered._sampling_config.filter_string is None

    # Check that the original scorer is unchanged
    assert my_scorer.name == "length_check"
    assert my_scorer._sampling_config is None

    # Check the mock was called correctly
    mock_add.assert_called_once()
    call_args = mock_add.call_args.kwargs
    assert call_args["name"] == "my_length_check"
    assert call_args["sample_rate"] == 0.0
    assert call_args["filter_string"] is None


def test_scorer_register_default_name():
    """Test registering with default name."""
    my_scorer = length_check
    with patch("mlflow.genai.scorers.registry.DatabricksStore.add_registered_scorer") as mock_add:
        registered = my_scorer.register()

    assert registered.name == "length_check"  # Uses scorer's name
    mock_add.assert_called_once()
    assert mock_add.call_args.kwargs["name"] == "length_check"


def test_scorer_start():
    """Test starting a scorer."""
    my_scorer = length_check
    my_scorer = my_scorer._create_copy()
    my_scorer.name = "my_length_check"
    my_scorer._sampling_config = ScorerSamplingConfig(sample_rate=0.0)

    with patch(
        "mlflow.genai.scorers.registry.DatabricksStore.update_registered_scorer"
    ) as mock_update:
        # Mock the return value
        mock_update.return_value = my_scorer._create_copy()
        mock_update.return_value.name = "my_length_check"
        mock_update.return_value._sampling_config = ScorerSamplingConfig(
            sample_rate=0.5, filter_string="trace.status = 'OK'"
        )

        started = my_scorer.start(
            sampling_config=ScorerSamplingConfig(
                sample_rate=0.5, filter_string="trace.status = 'OK'"
            )
        )

    # Check immutability
    assert started is not my_scorer
    assert started.name == "my_length_check"
    assert started._sampling_config.sample_rate == 0.5
    assert started._sampling_config.filter_string == "trace.status = 'OK'"

    # Original unchanged
    assert my_scorer._sampling_config.sample_rate == 0.0

    # Check mock called correctly
    mock_update.assert_called_once()
    call_args = mock_update.call_args.kwargs
    assert call_args["name"] == "my_length_check"
    assert call_args["sample_rate"] == 0.5
    assert call_args["filter_string"] == "trace.status = 'OK'"


@pytest.mark.parametrize("sample_rate", [0, -0.1])
def test_scorer_start_with_zero_sample_rate_raises_error(sample_rate):
    with pytest.raises(MlflowException, match="sample rate must be greater than 0"):
        length_check.start(sampling_config=ScorerSamplingConfig(sample_rate=sample_rate))


def test_scorer_start_not_registered():
    """Test starting a scorer that isn't registered."""
    my_scorer = length_check

    # Should work fine - start doesn't require pre-registration
    with patch(
        "mlflow.genai.scorers.registry.DatabricksStore.update_registered_scorer"
    ) as mock_update:
        mock_update.return_value = my_scorer._create_copy()
        my_scorer.start(sampling_config=ScorerSamplingConfig(sample_rate=0.5))

    assert mock_update.called


def test_scorer_update():
    """Test updating a scorer."""
    my_scorer = length_check
    my_scorer = my_scorer._create_copy()
    my_scorer.name = "my_length_check"
    my_scorer._sampling_config = ScorerSamplingConfig(sample_rate=0.5, filter_string="old filter")

    with patch(
        "mlflow.genai.scorers.registry.DatabricksStore.update_registered_scorer"
    ) as mock_update:
        # Mock the return value
        mock_update.return_value = my_scorer._create_copy()
        mock_update.return_value._sampling_config = ScorerSamplingConfig(
            sample_rate=0.4, filter_string="old filter"
        )

        # Update with new config
        updated = my_scorer.update(
            sampling_config=ScorerSamplingConfig(sample_rate=0.4, filter_string="old filter")
        )

    assert updated._sampling_config.sample_rate == 0.4
    assert updated._sampling_config.filter_string == "old filter"

    # Check mock was called correctly
    mock_update.assert_called_once()
    call_args = mock_update.call_args.kwargs
    assert call_args["name"] == "my_length_check"
    assert call_args["sample_rate"] == 0.4
    assert call_args["filter_string"] == "old filter"


def test_scorer_stop():
    """Test stopping a scorer."""
    my_scorer = length_check
    my_scorer = my_scorer._create_copy()
    my_scorer.name = "my_length_check"
    my_scorer._sampling_config = ScorerSamplingConfig(sample_rate=0.5)

    with patch(
        "mlflow.genai.scorers.registry.DatabricksStore.update_registered_scorer"
    ) as mock_update:
        # Mock the return value
        mock_update.return_value = my_scorer._create_copy()
        mock_update.return_value._sampling_config = ScorerSamplingConfig(sample_rate=0.0)

        stopped = my_scorer.stop()

    assert stopped._sampling_config.sample_rate == 0.0
    mock_update.assert_called_once()
    assert mock_update.call_args.kwargs["sample_rate"] == 0.0


def test_scorer_register_with_experiment_id():
    """Test registering a scorer with experiment_id."""
    my_scorer = length_check
    with patch("mlflow.genai.scorers.registry.DatabricksStore.add_registered_scorer") as mock_add:
        my_scorer.register(name="test_scorer", experiment_id="exp123")

    mock_add.assert_called_once()
    call_args = mock_add.call_args.kwargs
    assert call_args["experiment_id"] == "exp123"
    assert call_args["name"] == "test_scorer"


def test_scorer_start_with_name_param():
    """Test starting a scorer using name parameter."""
    my_scorer = length_check
    # Scorer doesn't need name modification

    with patch(
        "mlflow.genai.scorers.registry.DatabricksStore.update_registered_scorer"
    ) as mock_update:
        mock_update.return_value = my_scorer._create_copy()
        mock_update.return_value.name = "different_name"
        mock_update.return_value._sampling_config = ScorerSamplingConfig(sample_rate=0.7)

        started = my_scorer.start(
            name="different_name", sampling_config=ScorerSamplingConfig(sample_rate=0.7)
        )

    assert started.name == "different_name"
    assert started._sampling_config.sample_rate == 0.7

    mock_update.assert_called_once()
    call_args = mock_update.call_args.kwargs
    assert call_args["name"] == "different_name"


def test_scorer_update_with_all_params():
    my_scorer = length_check
    my_scorer = my_scorer._create_copy()
    my_scorer.name = "original_name"

    with patch(
        "mlflow.genai.scorers.registry.DatabricksStore.update_registered_scorer"
    ) as mock_update:
        mock_update.return_value = my_scorer._create_copy()
        mock_update.return_value._sampling_config = ScorerSamplingConfig(
            sample_rate=0.9, filter_string="new_filter"
        )

        my_scorer.update(
            name="override_name",
            experiment_id="exp456",
            sampling_config=ScorerSamplingConfig(sample_rate=0.9, filter_string="new_filter"),
        )

    mock_update.assert_called_once()
    call_args = mock_update.call_args.kwargs
    assert call_args["name"] == "override_name"
    assert call_args["experiment_id"] == "exp456"
    assert call_args["sample_rate"] == 0.9
    assert call_args["filter_string"] == "new_filter"


def test_builtin_scorer_register():
    """Test registering a builtin scorer with custom name."""
    guidelines_scorer = Guidelines(guidelines="Be helpful")

    # Verify original serialization
    original_dump = guidelines_scorer.model_dump()
    assert original_dump["name"] == "guidelines"

    with patch("mlflow.genai.scorers.registry.DatabricksStore.add_registered_scorer"):
        # Register with custom name
        registered = guidelines_scorer.register(name="my_guidelines")

    assert registered is not guidelines_scorer
    assert registered.name == "my_guidelines"
    assert registered._sampling_config.sample_rate == 0.0

    # Check original fields preserved
    assert registered.guidelines == "Be helpful"

    # Verify serialization reflects the new name (key test for BuiltinScorers)
    registered_dump = registered.model_dump()
    assert registered_dump["name"] == "my_guidelines"

    # Verify original scorer is unchanged
    assert guidelines_scorer.name == "guidelines"
    original_dump_after = guidelines_scorer.model_dump()
    assert original_dump_after["name"] == "guidelines"


def test_builtin_scorer_update():
    guidelines_scorer = Guidelines(guidelines="Be helpful")
    guidelines_scorer = guidelines_scorer._create_copy()
    guidelines_scorer.name = "my_guidelines"
    guidelines_scorer._sampling_config = ScorerSamplingConfig(sample_rate=0.5)

    with patch(
        "mlflow.genai.scorers.registry.DatabricksStore.update_registered_scorer"
    ) as mock_update:
        # Mock the return value
        mock_update.return_value = guidelines_scorer._create_copy()
        mock_update.return_value._sampling_config = ScorerSamplingConfig(sample_rate=0.3)

        updated = guidelines_scorer.update(sampling_config=ScorerSamplingConfig(sample_rate=0.3))

    assert updated._sampling_config.sample_rate == 0.3
    assert updated.guidelines == "Be helpful"  # Original field preserved


def test_all_methods_are_immutable():
    """Test that all methods return new instances and don't modify the original."""
    original = length_check

    # Set up some state
    original = original._create_copy()
    original.name = "original_name"
    original._sampling_config = ScorerSamplingConfig(sample_rate=0.1, filter_string="original")

    # Test each method
    with patch("mlflow.genai.scorers.registry.DatabricksStore.add_registered_scorer"):
        registered = original.register(name="new_name")
        assert registered is not original
        assert original.name == "original_name"  # Unchanged

    with patch(
        "mlflow.genai.scorers.registry.DatabricksStore.update_registered_scorer"
    ) as mock_update:
        # Mock return values
        mock_update.return_value = original._create_copy()
        mock_update.return_value._sampling_config = ScorerSamplingConfig(sample_rate=0.9)

        started = original.start(sampling_config=ScorerSamplingConfig(sample_rate=0.9))
        assert started is not original
        assert original._sampling_config.sample_rate == 0.1  # Unchanged

        mock_update.return_value = original._create_copy()
        mock_update.return_value._sampling_config = ScorerSamplingConfig(
            sample_rate=0.1, filter_string="new filter"
        )

        updated = original.update(
            sampling_config=ScorerSamplingConfig(sample_rate=0.1, filter_string="new filter")
        )
        assert updated is not original
        assert original._sampling_config.filter_string == "original"  # Unchanged

        mock_update.return_value = original._create_copy()
        mock_update.return_value._sampling_config = ScorerSamplingConfig(sample_rate=0.0)

        stopped = original.stop()
        assert stopped is not original
        assert original._sampling_config.sample_rate == 0.1  # Unchanged


def test_class_scorer_cannot_be_registered():
    """Test that class-based scorers cannot be registered."""

    class CustomScorer(Scorer):
        name: str = "custom"

        def __call__(self, outputs):
            return True

    custom_scorer = CustomScorer()

    # Test all methods that require registrable scorers
    with pytest.raises(MlflowException, match="Scorer must be a builtin or decorator scorer"):
        custom_scorer.register()

    with pytest.raises(MlflowException, match="Scorer must be a builtin or decorator scorer"):
        custom_scorer.start(sampling_config=ScorerSamplingConfig(sample_rate=0.5))

    with pytest.raises(MlflowException, match="Scorer must be a builtin or decorator scorer"):
        custom_scorer.update(sampling_config=ScorerSamplingConfig(sample_rate=0.5))

    with pytest.raises(MlflowException, match="Scorer must be a builtin or decorator scorer"):
        custom_scorer.stop()


def test_register_with_custom_name_updates_serialization():
    """Test that registering with a custom name properly updates serialization data."""
    # Use the pre-defined scorer to avoid source extraction issues
    test_scorer = serialization_scorer

    # Serialize to populate cache
    original_dump = test_scorer.model_dump()
    assert original_dump["name"] == "serialization_scorer"

    with patch("mlflow.genai.scorers.registry.DatabricksStore.add_registered_scorer") as mock_add:
        # Register with custom name
        registered = test_scorer.register(name="custom_test_name")

    # Verify the registered scorer has the correct name
    assert registered.name == "custom_test_name"

    # Verify serialization reflects the new name
    registered_dump = registered.model_dump()
    assert registered_dump["name"] == "custom_test_name"

    # Verify original scorer is unchanged
    assert test_scorer.name == "serialization_scorer"

    # Verify the server was called with the correct name
    mock_add.assert_called_once()
    assert mock_add.call_args.kwargs["name"] == "custom_test_name"
