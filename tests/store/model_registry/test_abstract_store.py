import json
import threading
import time
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


@pytest.fixture
def store():
    return MockAbstractStore()


@pytest.fixture
def mock_tracking_store():
    with mock.patch("mlflow.tracking._get_store") as mock_get_store:
        mock_store = mock.Mock()
        mock_get_store.return_value = mock_store
        yield mock_store


def test_link_prompt_version_to_model_success(store, mock_tracking_store):
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
    mock_tracking_store.set_logged_model_tags.assert_called_once()
    call_args = mock_tracking_store.set_logged_model_tags.call_args
    assert call_args[0][0] == model_id

    logged_model_tags = call_args[0][1]
    assert len(logged_model_tags) == 1
    logged_model_tag = logged_model_tags[0]
    assert isinstance(logged_model_tag, LoggedModelTag)
    assert logged_model_tag.key == LINKED_PROMPTS_TAG_KEY

    expected_value = [{"name": "test_prompt", "version": 1}]
    assert json.loads(logged_model_tag.value) == expected_value


def test_link_prompt_version_to_model_append_to_existing(store, mock_tracking_store):
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
    call_args = mock_tracking_store.set_logged_model_tags.call_args
    logged_model_tags = call_args[0][1]
    assert len(logged_model_tags) == 1
    logged_model_tag = logged_model_tags[0]

    expected_value = [
        {"name": "existing_prompt", "version": "v1"},
        {"name": "test_prompt", "version": 1},
    ]
    assert json.loads(logged_model_tag.value) == expected_value


def test_link_prompt_version_to_model_no_model_found(store, mock_tracking_store):
    """Test error when model is not found."""
    # Setup
    store.add_prompt_version("test_prompt", "v1")
    mock_tracking_store.get_logged_model.return_value = None

    # Execute & Verify
    with pytest.raises(MlflowException, match="Could not find model with ID 'nonexistent_model'"):
        store.link_prompt_version_to_model("test_prompt", "v1", "nonexistent_model")


def test_link_prompt_version_to_model_prompt_not_found(store, mock_tracking_store):
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
    with pytest.raises(
        MlflowException, match="Prompt version 'nonexistent_prompt' version 'v1' not found"
    ):
        store.link_prompt_version_to_model("nonexistent_prompt", "v1", model_id)


def test_link_prompt_version_to_model_invalid_json_tag(store, mock_tracking_store):
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
    with pytest.raises(MlflowException, match="Invalid JSON format for 'mlflow.linkedPrompts' tag"):
        store.link_prompt_version_to_model("test_prompt", "v1", model_id)


def test_link_prompt_version_to_model_invalid_format_tag(store, mock_tracking_store):
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
    with pytest.raises(MlflowException, match="Invalid format for 'mlflow.linkedPrompts' tag"):
        store.link_prompt_version_to_model("test_prompt", "v1", model_id)


def test_link_prompt_version_to_model_duplicate_prevention(store, mock_tracking_store):
    """Test that linking the same prompt version twice doesn't create duplicates."""
    # Setup
    store.add_prompt_version("test_prompt", "v1")
    model_id = "model_123"

    # Create a logged model that will be updated by the mocked set_logged_model_tags
    logged_model = LoggedModel(
        experiment_id="exp_123",
        model_id=model_id,
        name="test_model",
        artifact_location="/path/to/model",
        creation_timestamp=1234567890,
        last_updated_timestamp=1234567890,
        tags={},
    )

    # Mock the behavior where set_logged_model_tags updates the model's tags
    def mock_set_tags(model_id, tags):
        for tag in tags:
            logged_model.tags[tag.key] = tag.value

    mock_tracking_store.get_logged_model.return_value = logged_model
    mock_tracking_store.set_logged_model_tags.side_effect = mock_set_tags

    # Execute - link the same prompt twice
    store.link_prompt_version_to_model("test_prompt", "v1", model_id)
    store.link_prompt_version_to_model("test_prompt", "v1", model_id)  # Should be idempotent

    # Verify set_logged_model_tags was called only once (second call should return early)
    assert mock_tracking_store.set_logged_model_tags.call_count == 1

    # Verify the tag contains only one entry
    tag_value = logged_model.tags[LINKED_PROMPTS_TAG_KEY]
    parsed_value = json.loads(tag_value)

    expected_value = [{"name": "test_prompt", "version": 1}]
    assert parsed_value == expected_value


def test_link_prompt_version_to_model_thread_safety(store, mock_tracking_store):
    """Test thread safety of linking prompt versions to models."""
    # Setup
    store.add_prompt_version("test_prompt_1", "v1")
    store.add_prompt_version("test_prompt_2", "v1")
    model_id = "model_123"

    # Create a shared logged model that will be updated
    logged_model = LoggedModel(
        experiment_id="exp_123",
        model_id=model_id,
        name="test_model",
        artifact_location="/path/to/model",
        creation_timestamp=1234567890,
        last_updated_timestamp=1234567890,
        tags={},
    )

    # Mock behavior to simulate updating the model's tags
    def mock_set_tags(model_id, tags):
        # Simulate concurrent access with small delay
        time.sleep(0.01)
        for tag in tags:
            logged_model.tags[tag.key] = tag.value

    mock_tracking_store.get_logged_model.return_value = logged_model
    mock_tracking_store.set_logged_model_tags.side_effect = mock_set_tags

    # Define thread worker function
    def link_prompt(prompt_name):
        try:
            store.link_prompt_version_to_model(prompt_name, "v1", model_id)
        except Exception as e:
            # Store any exceptions for later verification
            exceptions.append(e)

    # Track exceptions from threads
    exceptions = []

    # Create and start threads
    threads = []
    for i in range(2):
        thread = threading.Thread(target=link_prompt, args=[f"test_prompt_{i + 1}"])
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Verify no exceptions occurred
    assert len(exceptions) == 0, f"Thread exceptions: {exceptions}"

    # Verify final state contains both prompts (order may vary due to threading)
    final_tag_value = json.loads(logged_model.tags[LINKED_PROMPTS_TAG_KEY])

    expected_prompts = [
        {"name": "test_prompt_1", "version": 1},
        {"name": "test_prompt_2", "version": 1},
    ]
    assert len(final_tag_value) == 2
    for expected_prompt in expected_prompts:
        assert expected_prompt in final_tag_value


def test_link_prompts_to_trace_success(store, mock_tracking_store):
    """Test successful linking of prompt versions to a trace."""
    # Setup
    store.add_prompt_version("test_prompt", "v1")
    trace_id = "trace_123"

    # Mock trace info
    mock_trace_info = mock.Mock()
    mock_trace_info.tags = {}

    mock_tracking_store.get_trace_info.return_value = mock_trace_info

    # Execute - get the prompt version object
    prompt_version = store.get_prompt_version("test_prompt", "v1")
    store.link_prompts_to_trace(prompt_versions=[prompt_version], trace_id=trace_id)

    # Verify
    mock_tracking_store.set_trace_tag.assert_called_once()

    call_args = mock_tracking_store.set_trace_tag.call_args
    assert call_args[0][0] == "trace_123"
    assert call_args[0][1] == LINKED_PROMPTS_TAG_KEY
    expected_value = [{"name": "test_prompt", "version": "1"}]
    assert json.loads(call_args[0][2]) == expected_value


def test_link_prompts_to_trace_nonexistent_trace(store, mock_tracking_store):
    """Test error handling when trace is not found."""
    # Setup
    store.add_prompt_version("test_prompt", "v1")
    mock_tracking_store.get_trace_info.return_value = None

    # Execute - should log warning and continue
    prompt_version = store.get_prompt_version("test_prompt", "v1")
    store.link_prompts_to_trace(prompt_versions=[prompt_version], trace_id="nonexistent_trace")

    # Verify get_trace_info was called but set_trace_tag was not
    mock_tracking_store.get_trace_info.assert_called_once_with("nonexistent_trace")
    mock_tracking_store.set_trace_tag.assert_not_called()


def test_link_prompts_to_trace_unsupported_store(store, mock_tracking_store):
    """Test error handling when tracking store doesn't support get_trace_info."""
    # Setup
    store.add_prompt_version("test_prompt", "v1")
    # Mock tracking store that doesn't have get_trace_info method
    del mock_tracking_store.get_trace_info

    # Execute - should log warning and continue gracefully
    prompt_version = store.get_prompt_version("test_prompt", "v1")
    store.link_prompts_to_trace(prompt_versions=[prompt_version], trace_id="some_trace")

    # Verify set_trace_tag was not called since get_trace_info failed
    mock_tracking_store.set_trace_tag.assert_not_called()
