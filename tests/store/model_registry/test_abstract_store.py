import json
from unittest import mock

import pytest

from mlflow.entities.logged_model import LoggedModel
from mlflow.entities.logged_model_tag import LoggedModelTag
from mlflow.entities.model_registry.prompt_version import PromptVersion
from mlflow.exceptions import MlflowException
from mlflow.prompt.constants import LINKED_PROMPTS_TAG_KEY
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode
from mlflow.store.model_registry.abstract_store import AbstractStore


class MockAbstractStore(AbstractStore):
    """Mock implementation of AbstractStore for testing."""

    def __init__(self):
        super().__init__()
        self.prompt_versions = {}

    def get_prompt_version(self, name: str, version: str) -> PromptVersion:
        key = f"{name}:{version}"
        if key not in self.prompt_versions:
            raise MlflowException(
                f"Prompt version '{name}' version '{version}' not found",
                error_code=ErrorCode.Name(RESOURCE_DOES_NOT_EXIST),
            )
        return self.prompt_versions[key]

    def add_prompt_version(self, name: str, version: str):
        """Helper method to add prompt versions for testing."""
        key = f"{name}:{version}"
        self.prompt_versions[key] = PromptVersion(
            name=name,
            version=int(version.replace("v", "")),  # Convert v1 -> 1
            template="Test template",
            creation_timestamp=1234567890,
        )


class TestAbstractStoreLinkPromptVersionToModel:
    """Test cases for the abstract store's link_prompt_version_to_model method."""

    @pytest.fixture
    def store(self):
        return MockAbstractStore()

    @pytest.fixture
    def mock_tracking_store(self):
        with mock.patch("mlflow.tracking._get_store") as mock_get_store:
            mock_store = mock.Mock()
            mock_get_store.return_value = mock_store
            yield mock_store

    def test_link_prompt_version_to_model_success(self, store, mock_tracking_store):
        """Test successful linking of prompt version to model."""
        # Setup
        store.add_prompt_version("test_prompt", "v1")
        model_id = "model_123"

        # Mock logged model with no existing linked prompts
        logged_model = LoggedModel(
            experiment_id="exp_123",
            model_id=model_id,
            name="test_model",
            artifact_location="/path/to/model",
            creation_timestamp=1234567890,
            last_updated_timestamp=1234567890,
            tags={},
        )
        mock_tracking_store.get_logged_model.return_value = logged_model

        # Execute
        store.link_prompt_version_to_model("test_prompt", "v1", model_id)

        # Verify
        mock_tracking_store.set_logged_model_tag.assert_called_once()
        call_args = mock_tracking_store.set_logged_model_tag.call_args
        assert call_args[0][0] == model_id

        logged_model_tag = call_args[0][1]
        assert isinstance(logged_model_tag, LoggedModelTag)
        assert logged_model_tag.key == LINKED_PROMPTS_TAG_KEY

        expected_value = [{"name": "test_prompt", "version": 1}]
        assert json.loads(logged_model_tag.value) == expected_value

    def test_link_prompt_version_to_model_append_to_existing(self, store, mock_tracking_store):
        """Test linking prompt version when other prompts are already linked."""
        # Setup
        store.add_prompt_version("test_prompt", "v1")
        model_id = "model_123"

        existing_prompts = [{"name": "existing_prompt", "version": "v1"}]
        logged_model = LoggedModel(
            experiment_id="exp_123",
            model_id=model_id,
            name="test_model",
            artifact_location="/path/to/model",
            creation_timestamp=1234567890,
            last_updated_timestamp=1234567890,
            tags={LINKED_PROMPTS_TAG_KEY: json.dumps(existing_prompts)},
        )
        mock_tracking_store.get_logged_model.return_value = logged_model

        # Execute
        store.link_prompt_version_to_model("test_prompt", "v1", model_id)

        # Verify
        call_args = mock_tracking_store.set_logged_model_tag.call_args
        logged_model_tag = call_args[0][1]

        expected_value = [
            {"name": "existing_prompt", "version": "v1"},
            {"name": "test_prompt", "version": 1},
        ]
        assert json.loads(logged_model_tag.value) == expected_value

    def test_link_prompt_version_to_model_no_model_found(self, store, mock_tracking_store):
        """Test error when model is not found."""
        # Setup
        store.add_prompt_version("test_prompt", "v1")
        mock_tracking_store.get_logged_model.return_value = None

        # Execute & Verify
        with pytest.raises(MlflowException) as exc_info:
            store.link_prompt_version_to_model("test_prompt", "v1", "nonexistent_model")

        assert "Could not find model with ID 'nonexistent_model'" in str(exc_info.value)
        assert exc_info.value.error_code == "INTERNAL_ERROR"

    def test_link_prompt_version_to_model_prompt_not_found(self, store, mock_tracking_store):
        """Test error when prompt version is not found."""
        # Setup
        model_id = "model_123"
        logged_model = LoggedModel(
            experiment_id="exp_123",
            model_id=model_id,
            name="test_model",
            artifact_location="/path/to/model",
            creation_timestamp=1234567890,
            last_updated_timestamp=1234567890,
            tags={},
        )
        mock_tracking_store.get_logged_model.return_value = logged_model

        # Execute & Verify
        with pytest.raises(MlflowException) as exc_info:
            store.link_prompt_version_to_model("nonexistent_prompt", "v1", model_id)

        assert "Prompt version 'nonexistent_prompt' version 'v1' not found" in str(exc_info.value)

    def test_link_prompt_version_to_model_invalid_json_tag(self, store, mock_tracking_store):
        """Test error when existing linked prompts tag has invalid JSON."""
        # Setup
        store.add_prompt_version("test_prompt", "v1")
        model_id = "model_123"

        logged_model = LoggedModel(
            experiment_id="exp_123",
            model_id=model_id,
            name="test_model",
            artifact_location="/path/to/model",
            creation_timestamp=1234567890,
            last_updated_timestamp=1234567890,
            tags={LINKED_PROMPTS_TAG_KEY: "invalid json"},
        )
        mock_tracking_store.get_logged_model.return_value = logged_model

        # Execute & Verify
        with pytest.raises(MlflowException) as exc_info:
            store.link_prompt_version_to_model("test_prompt", "v1", model_id)

        assert "Invalid JSON format for 'mlflow.linkedPrompts' tag" in str(exc_info.value)

    def test_link_prompt_version_to_model_invalid_format_tag(self, store, mock_tracking_store):
        """Test error when existing linked prompts tag has invalid format (not a list)."""
        # Setup
        store.add_prompt_version("test_prompt", "v1")
        model_id = "model_123"

        logged_model = LoggedModel(
            experiment_id="exp_123",
            model_id=model_id,
            name="test_model",
            artifact_location="/path/to/model",
            creation_timestamp=1234567890,
            last_updated_timestamp=1234567890,
            tags={LINKED_PROMPTS_TAG_KEY: json.dumps({"not": "a list"})},
        )
        mock_tracking_store.get_logged_model.return_value = logged_model

        # Execute & Verify
        with pytest.raises(MlflowException) as exc_info:
            store.link_prompt_version_to_model("test_prompt", "v1", model_id)

        assert "Invalid format for 'mlflow.linkedPrompts' tag" in str(exc_info.value)
