"""Tests for registered scorer functionality."""

from unittest.mock import patch

import pytest

from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import Guidelines, get_scorer, list_scorers, scorer
from mlflow.genai.scorers.base import Scorer, ScorerSamplingConfig


@scorer
def length_check(outputs):
    """Check if response is adequately detailed"""
    return len(str(outputs)) > 100


class TestScorerMethods:
    """Test the new methods on the Scorer class."""

    @patch("mlflow.genai.scorers.registry.add_registered_scorer")
    def test_scorer_register(self, mock_add):
        """Test registering a scorer."""
        # Test decorator scorer
        my_scorer = length_check
        registered = my_scorer.register(name="my_length_check")

        # Check immutability - returns new instance
        assert registered is not my_scorer
        assert registered._server_name == "my_length_check"
        assert registered._sampling_config.sample_rate == 0.0
        assert registered._sampling_config.filter_string is None

        # Check that the original scorer is unchanged
        assert my_scorer._server_name is None
        assert my_scorer._sampling_config is None

        # Check the mock was called correctly
        mock_add.assert_called_once()
        call_args = mock_add.call_args.kwargs
        assert call_args["name"] == "my_length_check"
        assert call_args["sample_rate"] == 0.0
        assert call_args["filter_string"] is None

    @patch("mlflow.genai.scorers.registry.add_registered_scorer")
    def test_scorer_register_default_name(self, mock_add):
        """Test registering with default name."""
        my_scorer = length_check
        registered = my_scorer.register()

        assert registered._server_name == "length_check"  # Uses scorer's name
        mock_add.assert_called_once()
        assert mock_add.call_args.kwargs["name"] == "length_check"

    @patch("mlflow.genai.scorers.registry.update_registered_scorer")
    def test_scorer_start(self, mock_update):
        """Test starting a scorer."""
        my_scorer = length_check
        my_scorer._server_name = "my_length_check"
        my_scorer._sampling_config = ScorerSamplingConfig(sample_rate=0.0)

        # Mock the return value
        mock_update.return_value = my_scorer._create_copy()
        mock_update.return_value._server_name = "my_length_check"
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
        assert started._server_name == "my_length_check"
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

    def test_scorer_start_not_registered(self):
        """Test starting a scorer that isn't registered."""
        my_scorer = length_check

        # Should work fine - start doesn't require pre-registration
        with patch("mlflow.genai.scorers.registry.update_registered_scorer") as mock_update:
            mock_update.return_value = my_scorer._create_copy()
            my_scorer.start(sampling_config=ScorerSamplingConfig(sample_rate=0.5))
            assert mock_update.called

    @patch("mlflow.genai.scorers.registry.update_registered_scorer")
    def test_scorer_update(self, mock_update):
        """Test updating a scorer."""
        my_scorer = length_check
        my_scorer._server_name = "my_length_check"
        my_scorer._sampling_config = ScorerSamplingConfig(
            sample_rate=0.5, filter_string="old filter"
        )

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

    @patch("mlflow.genai.scorers.registry.update_registered_scorer")
    def test_scorer_stop(self, mock_update):
        """Test stopping a scorer."""
        my_scorer = length_check
        my_scorer._server_name = "my_length_check"
        my_scorer._sampling_config = ScorerSamplingConfig(sample_rate=0.5)

        # Mock the return value
        mock_update.return_value = my_scorer._create_copy()
        mock_update.return_value._sampling_config = ScorerSamplingConfig(sample_rate=0.0)

        stopped = my_scorer.stop()

        assert stopped._sampling_config.sample_rate == 0.0
        mock_update.assert_called_once()
        assert mock_update.call_args.kwargs["sample_rate"] == 0.0

    @patch("mlflow.genai.scorers.registry.delete_registered_scorer")
    def test_scorer_delete(self, mock_delete):
        """Test deleting a scorer."""
        my_scorer = length_check
        my_scorer._server_name = "my_length_check"
        my_scorer._sampling_config = ScorerSamplingConfig(sample_rate=0.5)

        my_scorer.delete()

        # delete() returns None, not a new scorer
        # Original unchanged
        assert my_scorer._server_name == "my_length_check"

        mock_delete.assert_called_once()
        assert mock_delete.call_args.kwargs["name"] == "my_length_check"

    @patch("mlflow.genai.scorers.registry.add_registered_scorer")
    def test_scorer_register_with_experiment_id(self, mock_add):
        """Test registering a scorer with experiment_id."""
        my_scorer = length_check
        my_scorer.register(name="test_scorer", experiment_id="exp123")

        mock_add.assert_called_once()
        call_args = mock_add.call_args.kwargs
        assert call_args["experiment_id"] == "exp123"
        assert call_args["name"] == "test_scorer"

    @patch("mlflow.genai.scorers.registry.update_registered_scorer")
    def test_scorer_start_with_name_param(self, mock_update):
        """Test starting a scorer using name parameter."""
        my_scorer = length_check
        # Scorer doesn't need _server_name set

        mock_update.return_value = my_scorer._create_copy()
        mock_update.return_value._server_name = "different_name"
        mock_update.return_value._sampling_config = ScorerSamplingConfig(sample_rate=0.7)

        started = my_scorer.start(
            name="different_name", sampling_config=ScorerSamplingConfig(sample_rate=0.7)
        )

        assert started._server_name == "different_name"
        assert started._sampling_config.sample_rate == 0.7

        mock_update.assert_called_once()
        call_args = mock_update.call_args.kwargs
        assert call_args["name"] == "different_name"

    @patch("mlflow.genai.scorers.registry.update_registered_scorer")
    def test_scorer_update_with_all_params(self, mock_update):
        """Test updating with all parameters."""
        my_scorer = length_check
        my_scorer._server_name = "original_name"

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

    @patch("mlflow.genai.scorers.registry.delete_registered_scorer")
    def test_scorer_delete_with_experiment_id(self, mock_delete):
        """Test deleting with experiment_id."""
        my_scorer = length_check
        my_scorer._server_name = "test_scorer"

        my_scorer.delete(experiment_id="exp789")

        mock_delete.assert_called_once()
        call_args = mock_delete.call_args.kwargs
        assert call_args["experiment_id"] == "exp789"


class TestBuiltinScorerMethods:
    """Test that builtin scorers work with the new methods."""

    @patch("mlflow.genai.scorers.registry.add_registered_scorer")
    def test_builtin_scorer_register(self, mock_add):
        """Test registering a builtin scorer."""
        guidelines_scorer = Guidelines(guidelines="Be helpful")
        registered = guidelines_scorer.register(name="my_guidelines")

        assert registered is not guidelines_scorer
        assert registered._server_name == "my_guidelines"
        assert registered._sampling_config.sample_rate == 0.0

        # Check original fields preserved
        assert registered.guidelines == "Be helpful"

    @patch("mlflow.genai.scorers.registry.update_registered_scorer")
    def test_builtin_scorer_update(self, mock_update):
        """Test updating a builtin scorer."""
        guidelines_scorer = Guidelines(guidelines="Be helpful")
        guidelines_scorer._server_name = "my_guidelines"
        guidelines_scorer._sampling_config = ScorerSamplingConfig(sample_rate=0.5)

        # Mock the return value
        mock_update.return_value = guidelines_scorer._create_copy()
        mock_update.return_value._sampling_config = ScorerSamplingConfig(sample_rate=0.3)

        updated = guidelines_scorer.update(sampling_config=ScorerSamplingConfig(sample_rate=0.3))

        assert updated._sampling_config.sample_rate == 0.3
        assert updated.guidelines == "Be helpful"  # Original field preserved


class TestRegistryFunctions:
    """Test the list_scorers and get_scorer functions."""

    def test_list_scorers_import_error(self):
        """Test that list_scorers raises ImportError when databricks-agents is not installed."""
        with pytest.raises(ImportError, match="databricks-agents"):
            list_scorers(experiment_id="123")

    def test_get_scorer_import_error(self):
        """Test that get_scorer raises ImportError when databricks-agents is not installed."""
        with pytest.raises(ImportError, match="databricks-agents"):
            get_scorer(name="my_scorer", experiment_id="456")


class TestImmutability:
    """Test that all methods return new instances (immutability)."""

    def test_all_methods_are_immutable(self):
        """Test that all methods return new instances and don't modify the original."""
        original = length_check

        # Set up some state
        original._server_name = "original_name"
        original._sampling_config = ScorerSamplingConfig(sample_rate=0.1, filter_string="original")

        # Test each method
        with patch("mlflow.genai.scorers.registry.add_registered_scorer"):
            registered = original.register(name="new_name")
            assert registered is not original
            assert original._server_name == "original_name"  # Unchanged

        with patch("mlflow.genai.scorers.registry.update_registered_scorer") as mock_update:
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

        with patch("mlflow.genai.scorers.registry.delete_registered_scorer"):
            # delete() returns None, not a new instance
            result = original.delete()
            assert result is None
            assert original._server_name == "original_name"  # Unchanged


class TestValidation:
    """Test validation for scorer registration."""

    def test_class_scorer_cannot_be_registered(self):
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

        with pytest.raises(MlflowException, match="Scorer must be a builtin or decorator scorer"):
            custom_scorer.delete()

    def test_decorator_scorer_can_be_registered(self):
        """Test that decorator scorers can be registered."""
        my_scorer = length_check

        # Should not raise
        with patch("mlflow.genai.scorers.registry.add_registered_scorer"):
            my_scorer.register()

    def test_builtin_scorer_can_be_registered(self):
        """Test that builtin scorers can be registered."""
        guidelines_scorer = Guidelines(guidelines="Test")

        # Should not raise
        with patch("mlflow.genai.scorers.registry.add_registered_scorer"):
            guidelines_scorer.register()
