"""Tests for registered scorer functionality."""
from unittest.mock import Mock, patch

import pytest

from mlflow.genai.scorers import Guidelines, get_scorer, list_scorers, scorer
from mlflow.genai.scorers.base import Scorer, ScorerSamplingConfig
from mlflow.genai.scheduled_scorers import ScorerScheduleConfig


@scorer
def length_check(outputs):
    """Check if response is adequately detailed"""
    return len(str(outputs)) > 100


class TestScorerMethods:
    """Test the new methods on the Scorer class."""

    @patch("mlflow.genai.scorers.registry._add_registered_scorer")
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
        assert call_args["scheduled_scorer_name"] == "my_length_check"
        assert call_args["sample_rate"] == 0.0
        assert call_args["filter_string"] is None

    @patch("mlflow.genai.scorers.registry._add_registered_scorer")
    def test_scorer_register_default_name(self, mock_add):
        """Test registering with default name."""
        my_scorer = length_check
        registered = my_scorer.register()
        
        assert registered._server_name == "length_check"  # Uses scorer's name
        mock_add.assert_called_once()
        assert mock_add.call_args.kwargs["scheduled_scorer_name"] == "length_check"

    @patch("mlflow.genai.scorers.registry._update_registered_scorer")
    def test_scorer_start(self, mock_update):
        """Test starting a scorer."""
        my_scorer = length_check
        my_scorer._server_name = "my_length_check"
        my_scorer._sampling_config = ScorerSamplingConfig(sample_rate=0.0)
        
        started = my_scorer.start(sample_rate=0.5, filter_string="trace.status = 'OK'")
        
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
        assert call_args["scheduled_scorer_name"] == "my_length_check"
        assert call_args["sample_rate"] == 0.5
        assert call_args["filter_string"] == "trace.status = 'OK'"

    def test_scorer_start_not_registered(self):
        """Test starting a scorer that isn't registered."""
        my_scorer = length_check
        
        with pytest.raises(Exception, match="must be registered"):
            my_scorer.start(sample_rate=0.5)

    @patch("mlflow.genai.scorers.registry._update_registered_scorer")
    def test_scorer_update(self, mock_update):
        """Test updating a scorer."""
        my_scorer = length_check
        my_scorer._server_name = "my_length_check"
        my_scorer._sampling_config = ScorerSamplingConfig(sample_rate=0.5, filter_string="old filter")
        
        # Update only sample rate
        updated = my_scorer.update(sample_rate=0.4)
        
        assert updated._sampling_config.sample_rate == 0.4
        assert updated._sampling_config.filter_string == "old filter"  # Unchanged
        
        # Update only filter
        updated2 = my_scorer.update(filter_string="new filter")
        
        assert updated2._sampling_config.sample_rate == 0.5  # Original value
        assert updated2._sampling_config.filter_string == "new filter"

    @patch("mlflow.genai.scorers.registry._update_registered_scorer")
    def test_scorer_stop(self, mock_update):
        """Test stopping a scorer."""
        my_scorer = length_check
        my_scorer._server_name = "my_length_check"
        my_scorer._sampling_config = ScorerSamplingConfig(sample_rate=0.5)
        
        stopped = my_scorer.stop()
        
        assert stopped._sampling_config.sample_rate == 0.0
        mock_update.assert_called_once()
        assert mock_update.call_args.kwargs["sample_rate"] == 0.0

    @patch("mlflow.genai.scorers.registry._delete_registered_scorer")
    def test_scorer_delete(self, mock_delete):
        """Test deleting a scorer."""
        my_scorer = length_check
        my_scorer._server_name = "my_length_check"
        my_scorer._sampling_config = ScorerSamplingConfig(sample_rate=0.5)
        
        deleted = my_scorer.delete()
        
        # Check immutability
        assert deleted is not my_scorer
        assert deleted._server_name is None
        assert deleted._sampling_config is None
        
        # Original unchanged
        assert my_scorer._server_name == "my_length_check"
        
        mock_delete.assert_called_once()
        assert mock_delete.call_args.kwargs["scheduled_scorer_name"] == "my_length_check"


class TestBuiltinScorerMethods:
    """Test that builtin scorers work with the new methods."""

    @patch("mlflow.genai.scorers.registry._add_registered_scorer")
    def test_builtin_scorer_register(self, mock_add):
        """Test registering a builtin scorer."""
        guidelines_scorer = Guidelines(guidelines="Be helpful")
        registered = guidelines_scorer.register(name="my_guidelines")
        
        assert registered is not guidelines_scorer
        assert registered._server_name == "my_guidelines"
        assert registered._sampling_config.sample_rate == 0.0
        
        # Check original fields preserved
        assert registered.guidelines == "Be helpful"

    @patch("mlflow.genai.scorers.registry._update_registered_scorer")
    def test_builtin_scorer_update(self, mock_update):
        """Test updating a builtin scorer."""
        guidelines_scorer = Guidelines(guidelines="Be helpful")
        guidelines_scorer._server_name = "my_guidelines"
        guidelines_scorer._sampling_config = ScorerSamplingConfig(sample_rate=0.5)
        
        updated = guidelines_scorer.update(sample_rate=0.3)
        
        assert updated._sampling_config.sample_rate == 0.3
        assert updated.guidelines == "Be helpful"  # Original field preserved


class TestRegistryFunctions:
    """Test the list_scorers and get_scorer functions."""

    @patch("mlflow.genai.scorers.registry.list_scheduled_scorers")
    def test_list_scorers(self, mock_list):
        """Test listing scorers."""
        # Mock scheduled scorer configs
        mock_scorer1 = Mock(spec=Scorer)
        mock_scorer1.name = "scorer1"
        mock_scorer2 = Mock(spec=Scorer)
        mock_scorer2.name = "scorer2"
        
        mock_list.return_value = [
            ScorerScheduleConfig(
                scorer=mock_scorer1,
                scheduled_scorer_name="scorer1_scheduled",
                sample_rate=0.5,
                filter_string="filter1"
            ),
            ScorerScheduleConfig(
                scorer=mock_scorer2,
                scheduled_scorer_name="scorer2_scheduled",
                sample_rate=0.3,
                filter_string=None
            )
        ]
        
        scorers = list_scorers(experiment_id="123")
        
        assert len(scorers) == 2
        assert scorers[0]._server_name == "scorer1_scheduled"
        assert scorers[0]._sampling_config.sample_rate == 0.5
        assert scorers[0]._sampling_config.filter_string == "filter1"
        
        assert scorers[1]._server_name == "scorer2_scheduled"
        assert scorers[1]._sampling_config.sample_rate == 0.3
        assert scorers[1]._sampling_config.filter_string is None
        
        mock_list.assert_called_once_with("123")

    @patch("mlflow.genai.scorers.registry.get_scheduled_scorer")
    def test_get_scorer(self, mock_get):
        """Test getting a specific scorer."""
        mock_scorer = Mock(spec=Scorer)
        mock_scorer.name = "my_scorer"
        
        mock_get.return_value = ScorerScheduleConfig(
            scorer=mock_scorer,
            scheduled_scorer_name="my_scorer_scheduled",
            sample_rate=0.7,
            filter_string="status = 'OK'"
        )
        
        scorer = get_scorer(name="my_scorer_scheduled", experiment_id="456")
        
        assert scorer._server_name == "my_scorer_scheduled"
        assert scorer._sampling_config.sample_rate == 0.7
        assert scorer._sampling_config.filter_string == "status = 'OK'"
        
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args.kwargs
        assert call_kwargs["scheduled_scorer_name"] == "my_scorer_scheduled"
        assert call_kwargs["experiment_id"] == "456"


class TestScorerSerialization:
    """Test that new fields are properly serialized and deserialized."""

    def test_decorator_scorer_serialization_with_registered_fields(self):
        """Test serializing and deserializing a decorator scorer with registered fields."""
        my_scorer = length_check
        my_scorer._server_name = "test_server"
        my_scorer._sampling_config = ScorerSamplingConfig(sample_rate=0.8, filter_string="test filter")
        
        # Serialize
        dumped = my_scorer.model_dump()
        
        # Check that registered fields are included
        assert dumped["server_name"] == "test_server"
        assert dumped["sampling_config"]["sample_rate"] == 0.8
        assert dumped["sampling_config"]["filter_string"] == "test filter"
        
        # Deserialize
        restored = Scorer.model_validate(dumped)
        
        # Check that registered fields are restored
        assert restored._server_name == "test_server"
        assert restored._sampling_config.sample_rate == 0.8
        assert restored._sampling_config.filter_string == "test filter"
        
        # Check that the scorer still works
        assert restored(outputs="x" * 101) is True

    def test_builtin_scorer_serialization_with_registered_fields(self):
        """Test serializing and deserializing a builtin scorer with registered fields."""
        guidelines_scorer = Guidelines(guidelines="Be concise")
        guidelines_scorer._server_name = "guidelines_server"
        guidelines_scorer._sampling_config = ScorerSamplingConfig(sample_rate=0.2)
        
        # Serialize
        dumped = guidelines_scorer.model_dump()
        
        # Check that registered fields are included
        assert dumped["server_name"] == "guidelines_server"
        assert dumped["sampling_config"]["sample_rate"] == 0.2
        assert dumped["sampling_config"]["filter_string"] is None
        
        # Deserialize
        restored = Guidelines.model_validate(dumped)
        
        # Check that registered fields and original fields are restored
        assert restored._server_name == "guidelines_server"
        assert restored._sampling_config.sample_rate == 0.2
        assert restored.guidelines == "Be concise"


class TestImmutability:
    """Test that all methods return new instances (immutability)."""

    def test_all_methods_are_immutable(self):
        """Test that all methods return new instances and don't modify the original."""
        original = length_check
        
        # Set up some state
        original._server_name = "original_name"
        original._sampling_config = ScorerSamplingConfig(sample_rate=0.1, filter_string="original")
        
        # Test each method
        with patch("mlflow.genai.scorers.registry._add_registered_scorer"):
            registered = original.register(name="new_name")
            assert registered is not original
            assert original._server_name == "original_name"  # Unchanged
        
        with patch("mlflow.genai.scorers.registry._update_registered_scorer"):
            started = original.start(sample_rate=0.9)
            assert started is not original
            assert original._sampling_config.sample_rate == 0.1  # Unchanged
            
            updated = original.update(filter_string="new filter")
            assert updated is not original
            assert original._sampling_config.filter_string == "original"  # Unchanged
            
            stopped = original.stop()
            assert stopped is not original
            assert original._sampling_config.sample_rate == 0.1  # Unchanged
        
        with patch("mlflow.genai.scorers.registry._delete_registered_scorer"):
            deleted = original.delete()
            assert deleted is not original
            assert original._server_name == "original_name"  # Unchanged