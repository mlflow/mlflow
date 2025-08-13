import json
import threading
import time
from unittest import mock

import pytest

from mlflow.entities.logged_model import LoggedModel
from mlflow.entities.logged_model_tag import LoggedModelTag
from mlflow.entities.model_registry.model_version import ModelVersion
from mlflow.entities.model_registry.model_version_tag import ModelVersionTag
from mlflow.entities.model_registry.prompt_version import PromptVersion
from mlflow.entities.run import Run
from mlflow.entities.run_data import RunData
from mlflow.entities.run_info import RunInfo
from mlflow.entities.run_tag import RunTag
from mlflow.exceptions import MlflowException
from mlflow.prompt.constants import LINKED_PROMPTS_TAG_KEY
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST, ErrorCode
from mlflow.store.model_registry.abstract_store import AbstractStore


class MockAbstractStore(AbstractStore):
    """Mock implementation of AbstractStore for testing."""

    def __init__(self):
        super().__init__()
        self.prompt_versions = {}
        self.model_versions = {}

    def get_prompt_version(self, name: str, version: str) -> PromptVersion:
        key = f"{name}:{version}"
        if key not in self.prompt_versions:
            raise MlflowException(
                f"Prompt version '{name}' version '{version}' not found",
                error_code=ErrorCode.Name(RESOURCE_DOES_NOT_EXIST),
            )
        return self.prompt_versions[key]

    def get_model_version(self, name: str, version: int) -> ModelVersion:
        key = f"{name}:{version}"
        if key not in self.model_versions:
            # Create a default model version for testing
            self.model_versions[key] = ModelVersion(
                name=name,
                version=str(version),
                creation_timestamp=1234567890,
                last_updated_timestamp=1234567890,
                description="Test model version",
                tags={},
            )
        return self.model_versions[key]

    def set_model_version_tag(self, name: str, version: int, tag: ModelVersionTag):
        """Mock implementation to set model version tags."""
        mv = self.get_model_version(name, version)
        if isinstance(mv.tags, dict):
            mv.tags[tag.key] = tag.value
        else:
            # Convert to dict if it's not already
            mv.tags = {tag.key: tag.value}

    def add_prompt_version(self, name: str, version: str):
        """Helper method to add prompt versions for testing."""
        # Convert version to integer for PromptVersion
        version_int = int(version[1:]) if version.startswith("v") else int(version)

        # Store using both formats to handle version lookups
        key_with_v = f"{name}:v{version_int}"
        key_without_v = f"{name}:{version_int}"

        prompt_version = PromptVersion(
            name=name,
            version=version_int,
            template="Test template",
            creation_timestamp=1234567890,
        )

        self.prompt_versions[key_with_v] = prompt_version
        self.prompt_versions[key_without_v] = prompt_version


@pytest.fixture
def store():
    return MockAbstractStore()


@pytest.fixture
def mock_tracking_store():
    with mock.patch("mlflow.tracking._get_store") as mock_get_store:
        mock_store = mock.Mock()
        mock_get_store.return_value = mock_store
        yield mock_store


@pytest.fixture
def mock_tracing_client():
    with mock.patch("mlflow.tracing.client.TracingClient") as mock_tracing_client:
        mock_client = mock.Mock()
        mock_tracing_client.return_value = mock_client
        yield mock_client


def test_link_prompt_version_to_model_success(store, mock_tracking_store):
    """Test successful linking of prompt version to model."""
    # Setup
    store.add_prompt_version("test_prompt", "1")
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

    expected_value = [{"name": "test_prompt", "version": "1"}]
    assert json.loads(logged_model_tag.value) == expected_value


def test_link_prompt_version_to_model_append_to_existing(store, mock_tracking_store):
    """Test linking prompt version when other prompts are already linked."""
    # Setup
    store.add_prompt_version("test_prompt", "1")
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
        {"name": "test_prompt", "version": "1"},
    ]
    assert json.loads(logged_model_tag.value) == expected_value


def test_link_prompt_version_to_model_no_model_found(store, mock_tracking_store):
    """Test error when model is not found."""
    # Setup
    store.add_prompt_version("test_prompt", "1")
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
    store.add_prompt_version("test_prompt", "1")
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
    store.add_prompt_version("test_prompt", "1")
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
    store.add_prompt_version("test_prompt", "1")
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

    expected_value = [{"name": "test_prompt", "version": "1"}]
    assert parsed_value == expected_value


def test_link_prompt_version_to_model_thread_safety(store, mock_tracking_store):
    """Test thread safety of linking prompt versions to models."""
    # Setup
    store.add_prompt_version("test_prompt_1", "1")
    store.add_prompt_version("test_prompt_2", "1")
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
        {"name": "test_prompt_1", "version": "1"},
        {"name": "test_prompt_2", "version": "1"},
    ]
    assert len(final_tag_value) == 2
    for expected_prompt in expected_prompts:
        assert expected_prompt in final_tag_value


def test_link_prompts_to_trace_success(store, mock_tracing_client):
    """Test successful linking of prompt versions to a trace."""
    # Setup
    store.add_prompt_version("test_prompt", "1")
    trace_id = "trace_123"

    # Mock trace info
    mock_trace_info = mock.Mock()
    mock_trace_info.tags = {}

    mock_tracing_client.get_trace_info.return_value = mock_trace_info

    # Execute - get the prompt version object
    prompt_version = store.get_prompt_version("test_prompt", "v1")
    store.link_prompts_to_trace(prompt_versions=[prompt_version], trace_id=trace_id)

    # Verify
    mock_tracing_client.set_trace_tag.assert_called_once()

    call_args = mock_tracing_client.set_trace_tag.call_args
    assert call_args[0][0] == "trace_123"
    assert call_args[0][1] == LINKED_PROMPTS_TAG_KEY
    expected_value = [{"name": "test_prompt", "version": "1"}]
    assert json.loads(call_args[0][2]) == expected_value


def test_link_prompts_to_trace_append_to_existing(store, mock_tracing_client):
    """Test linking prompt versions when other prompts are already linked to the trace."""
    # Setup
    store.add_prompt_version("test_prompt", "1")
    trace_id = "trace_123"

    existing_prompts = [{"name": "existing_prompt", "version": "v1"}]
    mock_trace_info = mock.Mock()
    mock_trace_info.tags = {LINKED_PROMPTS_TAG_KEY: json.dumps(existing_prompts)}

    mock_tracing_client.get_trace_info.return_value = mock_trace_info

    # Execute
    prompt_version = store.get_prompt_version("test_prompt", "v1")
    store.link_prompts_to_trace(prompt_versions=[prompt_version], trace_id=trace_id)

    # Verify
    call_args = mock_tracing_client.set_trace_tag.call_args
    assert call_args[0][0] == trace_id
    assert call_args[0][1] == LINKED_PROMPTS_TAG_KEY

    expected_value = [
        {"name": "existing_prompt", "version": "v1"},
        {"name": "test_prompt", "version": "1"},
    ]
    assert json.loads(call_args[0][2]) == expected_value


def test_link_prompts_to_trace_nonexistent_trace(store, mock_tracing_client):
    """Test error handling when trace is not found."""
    # Setup
    store.add_prompt_version("test_prompt", "1")
    mock_tracing_client.get_trace_info.return_value = None

    # Execute & Verify
    prompt_version = store.get_prompt_version("test_prompt", "v1")
    with pytest.raises(MlflowException, match="Could not find trace with ID 'nonexistent_trace'"):
        store.link_prompts_to_trace(prompt_versions=[prompt_version], trace_id="nonexistent_trace")

    # Verify get_trace_info was called but set_trace_tag was not
    mock_tracing_client.get_trace_info.assert_called_once_with("nonexistent_trace")
    mock_tracing_client.set_trace_tag.assert_not_called()


def test_link_prompts_to_trace_invalid_json_tag(store, mock_tracing_client):
    """Test error when existing linked prompts tag has invalid JSON."""
    # Setup
    store.add_prompt_version("test_prompt", "1")
    trace_id = "trace_123"

    mock_trace_info = mock.Mock()
    mock_trace_info.tags = {LINKED_PROMPTS_TAG_KEY: "invalid json"}

    mock_tracing_client.get_trace_info.return_value = mock_trace_info

    # Execute & Verify
    prompt_version = store.get_prompt_version("test_prompt", "v1")
    with pytest.raises(MlflowException, match="Invalid JSON format for 'mlflow.linkedPrompts' tag"):
        store.link_prompts_to_trace(prompt_versions=[prompt_version], trace_id=trace_id)


def test_link_prompts_to_trace_invalid_format_tag(store, mock_tracing_client):
    """Test error when existing linked prompts tag has invalid format (not a list)."""
    # Setup
    store.add_prompt_version("test_prompt", "1")
    trace_id = "trace_123"

    mock_trace_info = mock.Mock()
    mock_trace_info.tags = {LINKED_PROMPTS_TAG_KEY: json.dumps({"not": "a list"})}

    mock_tracing_client.get_trace_info.return_value = mock_trace_info

    # Execute & Verify
    prompt_version = store.get_prompt_version("test_prompt", "v1")
    with pytest.raises(MlflowException, match="Invalid format for 'mlflow.linkedPrompts' tag"):
        store.link_prompts_to_trace(prompt_versions=[prompt_version], trace_id=trace_id)


def test_link_prompts_to_trace_duplicate_prevention(store, mock_tracing_client):
    """Test that linking the same prompt version twice doesn't create duplicates."""
    # Setup
    store.add_prompt_version("test_prompt", "1")
    trace_id = "trace_123"

    # Create trace info that will be updated by the mocked set_trace_tag
    mock_trace_info = mock.Mock()
    mock_trace_info.tags = {}

    # Mock the behavior where set_trace_tag updates the trace's tags
    def mock_set_tag(trace_id, key, value):
        mock_trace_info.tags[key] = value

    mock_tracing_client.get_trace_info.return_value = mock_trace_info
    mock_tracing_client.set_trace_tag.side_effect = mock_set_tag

    # Execute - link the same prompt twice
    prompt_version = store.get_prompt_version("test_prompt", "v1")
    store.link_prompts_to_trace(prompt_versions=[prompt_version], trace_id=trace_id)
    store.link_prompts_to_trace(prompt_versions=[prompt_version], trace_id=trace_id)

    # Verify set_trace_tag was called only once (second call should return early)
    assert mock_tracing_client.set_trace_tag.call_count == 1

    # Verify the tag contains only one entry
    tag_value = mock_trace_info.tags[LINKED_PROMPTS_TAG_KEY]
    parsed_value = json.loads(tag_value)

    expected_value = [{"name": "test_prompt", "version": "1"}]
    assert parsed_value == expected_value


def test_link_prompts_to_trace_multiple_prompts(store, mock_tracing_client):
    """Test linking multiple prompt versions to a trace at once."""
    # Setup
    store.add_prompt_version("test_prompt_1", "v1")
    store.add_prompt_version("test_prompt_2", "v2")
    trace_id = "trace_123"

    mock_trace_info = mock.Mock()
    mock_trace_info.tags = {}

    mock_tracing_client.get_trace_info.return_value = mock_trace_info

    # Execute
    prompt_version_1 = store.get_prompt_version("test_prompt_1", "v1")
    prompt_version_2 = store.get_prompt_version("test_prompt_2", "v2")
    store.link_prompts_to_trace(
        prompt_versions=[prompt_version_1, prompt_version_2], trace_id=trace_id
    )

    # Verify
    call_args = mock_tracing_client.set_trace_tag.call_args
    assert call_args[0][0] == trace_id
    assert call_args[0][1] == LINKED_PROMPTS_TAG_KEY

    expected_value = [
        {"name": "test_prompt_1", "version": "1"},
        {"name": "test_prompt_2", "version": "2"},
    ]
    assert json.loads(call_args[0][2]) == expected_value


def test_link_prompts_to_trace_thread_safety(store, mock_tracing_client):
    """Test thread safety of linking prompt versions to traces."""
    # Setup
    store.add_prompt_version("test_prompt_1", "1")
    store.add_prompt_version("test_prompt_2", "1")
    trace_id = "trace_123"

    # Create shared trace info that will be updated
    mock_trace_info = mock.Mock()
    mock_trace_info.tags = {}

    # Mock behavior to simulate updating the trace's tags
    def mock_set_tag(trace_id, key, value):
        # Simulate concurrent access with small delay
        time.sleep(0.01)
        mock_trace_info.tags[key] = value

    mock_tracing_client.get_trace_info.return_value = mock_trace_info
    mock_tracing_client.set_trace_tag.side_effect = mock_set_tag

    # Define thread worker function
    def link_prompt(prompt_name):
        try:
            prompt_version = store.get_prompt_version(prompt_name, "v1")
            store.link_prompts_to_trace(prompt_versions=[prompt_version], trace_id=trace_id)
        except Exception as e:
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
    final_tag_value = json.loads(mock_trace_info.tags[LINKED_PROMPTS_TAG_KEY])

    expected_prompts = [
        {"name": "test_prompt_1", "version": "1"},
        {"name": "test_prompt_2", "version": "1"},
    ]
    assert len(final_tag_value) == 2
    for expected_prompt in expected_prompts:
        assert expected_prompt in final_tag_value


def test_link_prompts_to_trace_no_change_optimization(store, mock_tracing_client):
    """Test that tag is not updated when no change is needed."""
    # Setup
    store.add_prompt_version("test_prompt", "1")
    trace_id = "trace_123"

    existing_prompts = [{"name": "test_prompt", "version": "1"}]
    mock_trace_info = mock.Mock()
    mock_trace_info.tags = {LINKED_PROMPTS_TAG_KEY: json.dumps(existing_prompts)}

    mock_tracing_client.get_trace_info.return_value = mock_trace_info

    # Execute - try to link the same prompt that's already linked
    prompt_version = store.get_prompt_version("test_prompt", "v1")
    store.link_prompts_to_trace(prompt_versions=[prompt_version], trace_id=trace_id)

    # Verify set_trace_tag was not called since no change was needed
    mock_tracing_client.set_trace_tag.assert_not_called()


# Tests for link_prompt_version_to_run


def test_link_prompt_version_to_run_success(store, mock_tracking_store):
    """Test successful linking of prompt version to run."""
    # Setup
    store.add_prompt_version("test_prompt", "1")
    run_id = "run_123"

    # Mock run with no existing linked prompts
    run_data = RunData(metrics=[], params=[], tags={})
    run_info = RunInfo(
        run_id=run_id,
        experiment_id="exp_123",
        user_id="user_123",
        status="FINISHED",
        start_time=1234567890,
        end_time=1234567890,
        lifecycle_stage="active",
    )
    run = Run(run_info=run_info, run_data=run_data)
    mock_tracking_store.get_run.return_value = run

    # Execute
    store.link_prompt_version_to_run("test_prompt", "1", run_id)

    # Verify run tag was set
    mock_tracking_store.set_tag.assert_called_once()
    call_args = mock_tracking_store.set_tag.call_args
    assert call_args[0][0] == run_id

    run_tag = call_args[0][1]
    assert isinstance(run_tag, RunTag)
    assert run_tag.key == LINKED_PROMPTS_TAG_KEY

    expected_value = [{"name": "test_prompt", "version": "1"}]
    assert json.loads(run_tag.value) == expected_value


def test_link_prompt_version_to_run_append_to_existing(store, mock_tracking_store):
    """Test linking prompt version when other prompts are already linked to the run."""
    # Setup
    store.add_prompt_version("test_prompt_1", "1")
    store.add_prompt_version("test_prompt_2", "1")
    run_id = "run_123"

    # Mock run with existing linked prompts
    existing_prompts = [{"name": "existing_prompt", "version": "1"}]
    run_data = RunData(
        metrics=[], params=[], tags=[RunTag(LINKED_PROMPTS_TAG_KEY, json.dumps(existing_prompts))]
    )
    run_info = RunInfo(
        run_id=run_id,
        experiment_id="exp_123",
        user_id="user_123",
        status="FINISHED",
        start_time=1234567890,
        end_time=1234567890,
        lifecycle_stage="active",
    )
    run = Run(run_info=run_info, run_data=run_data)
    mock_tracking_store.get_run.return_value = run

    # Execute
    store.link_prompt_version_to_run("test_prompt_1", "1", run_id)

    # Verify run tag was updated with both prompts
    mock_tracking_store.set_tag.assert_called_once()
    call_args = mock_tracking_store.set_tag.call_args

    run_tag = call_args[0][1]
    linked_prompts = json.loads(run_tag.value)

    expected_prompts = [
        {"name": "existing_prompt", "version": "1"},
        {"name": "test_prompt_1", "version": "1"},
    ]
    assert len(linked_prompts) == 2
    for expected_prompt in expected_prompts:
        assert expected_prompt in linked_prompts


def test_link_prompt_version_to_run_no_run_found(store, mock_tracking_store):
    """Test error when run is not found."""
    # Setup
    store.add_prompt_version("test_prompt", "1")  # Use "1" instead of "v1"
    run_id = "nonexistent_run"

    mock_tracking_store.get_run.return_value = None

    # Execute and verify error
    with pytest.raises(MlflowException, match="Could not find run"):
        store.link_prompt_version_to_run("test_prompt", "1", run_id)


def test_link_prompt_version_to_run_prompt_not_found(store, mock_tracking_store):
    """Test error when prompt version is not found."""
    # Setup
    run_id = "run_123"

    run_data = RunData(metrics=[], params=[], tags={})
    run_info = RunInfo(
        run_id=run_id,
        experiment_id="exp_123",
        user_id="user_123",
        status="FINISHED",
        start_time=1234567890,
        end_time=1234567890,
        lifecycle_stage="active",
    )
    run = Run(run_info=run_info, run_data=run_data)
    mock_tracking_store.get_run.return_value = run

    # Execute and verify error
    with pytest.raises(MlflowException, match="not found"):
        store.link_prompt_version_to_run("nonexistent_prompt", "1", run_id)


def test_link_prompt_version_to_run_duplicate_prevention(store, mock_tracking_store):
    """Test that duplicate prompt linkings are prevented."""
    # Setup
    store.add_prompt_version("test_prompt", "1")
    run_id = "run_123"

    # Mock run with existing prompt already linked
    existing_prompts = [{"name": "test_prompt", "version": "1"}]
    run_data = RunData(
        metrics=[], params=[], tags=[RunTag(LINKED_PROMPTS_TAG_KEY, json.dumps(existing_prompts))]
    )
    run_info = RunInfo(
        run_id=run_id,
        experiment_id="exp_123",
        user_id="user_123",
        status="FINISHED",
        start_time=1234567890,
        end_time=1234567890,
        lifecycle_stage="active",
    )
    run = Run(run_info=run_info, run_data=run_data)
    mock_tracking_store.get_run.return_value = run

    # Execute - try to link the same prompt again
    store.link_prompt_version_to_run("test_prompt", "1", run_id)

    # Verify set_tag was not called since no change was needed
    mock_tracking_store.set_tag.assert_not_called()


def test_link_prompt_version_to_run_thread_safety(store, mock_tracking_store):
    """Test thread safety of linking prompt versions to runs."""
    # Setup
    store.add_prompt_version("test_prompt_1", "1")
    store.add_prompt_version("test_prompt_2", "1")
    run_id = "run_123"

    # Create a shared run that will be updated
    run_data = RunData(metrics=[], params=[], tags={})
    run_info = RunInfo(
        run_id=run_id,
        experiment_id="exp_123",
        user_id="user_123",
        status="FINISHED",
        start_time=1234567890,
        end_time=1234567890,
        lifecycle_stage="active",
    )
    run = Run(run_info=run_info, run_data=run_data)

    # Mock behavior to simulate updating the run's tags
    def mock_set_tag(run_id, tag):
        # Simulate concurrent access with small delay
        time.sleep(0.01)
        run.data.tags[tag.key] = tag.value

    mock_tracking_store.get_run.return_value = run
    mock_tracking_store.set_tag.side_effect = mock_set_tag

    # Define thread worker function
    def link_prompt(prompt_name):
        store.link_prompt_version_to_run(prompt_name, "1", run_id)

    # Execute concurrent linking
    threads = []
    for prompt_name in ["test_prompt_1", "test_prompt_2"]:
        thread = threading.Thread(target=link_prompt, args=(prompt_name,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Verify both prompts were linked
    final_tag_value = json.loads(run.data.tags[LINKED_PROMPTS_TAG_KEY])

    expected_prompts = [
        {"name": "test_prompt_1", "version": "1"},
        {"name": "test_prompt_2", "version": "1"},
    ]
    assert len(final_tag_value) == 2
    for expected_prompt in expected_prompts:
        assert expected_prompt in final_tag_value


def test_link_chat_prompt_to_model(store, mock_tracking_store):
    """Test linking chat prompts to models works correctly."""
    # Create chat prompt
    chat_template = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello {{name}}!"},
    ]

    prompt_version = PromptVersion("test_chat", 1, chat_template)
    store.prompt_versions["test_chat:1"] = prompt_version

    # Test linking
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

    store.link_prompt_version_to_model("test_chat", "1", model_id)

    # Verify linking worked
    mock_tracking_store.set_logged_model_tags.assert_called_once()
    call_args = mock_tracking_store.set_logged_model_tags.call_args
    logged_model_tags = call_args[0][1]
    assert len(logged_model_tags) == 1

    tag_value = json.loads(logged_model_tags[0].value)
    assert tag_value == [{"name": "test_chat", "version": "1"}]


def test_link_prompt_with_response_format_to_model(store, mock_tracking_store):
    """Test linking prompts with response format to models."""
    response_format = {"type": "string", "description": "A response"}
    prompt_version = PromptVersion(
        "test_response", 1, "Hello {{name}}!", response_format=response_format
    )

    store.prompt_versions["test_response:1"] = prompt_version

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

    store.link_prompt_version_to_model("test_response", "1", model_id)

    # Verify linking worked
    mock_tracking_store.set_logged_model_tags.assert_called_once()
    call_args = mock_tracking_store.set_logged_model_tags.call_args
    logged_model_tags = call_args[0][1]
    assert len(logged_model_tags) == 1

    tag_value = json.loads(logged_model_tags[0].value)
    assert tag_value == [{"name": "test_response", "version": "1"}]


def test_link_chat_prompt_to_run(store, mock_tracking_store):
    """Test linking chat prompts to runs works correctly."""
    chat_template = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello {{name}}!"},
    ]

    prompt_version = PromptVersion("test_chat", 1, chat_template)
    store.prompt_versions["test_chat:1"] = prompt_version

    run_id = "run_123"
    run_data = RunData(metrics=[], params=[], tags={})
    run_info = RunInfo(
        run_id=run_id,
        experiment_id="exp_123",
        user_id="user_123",
        status="FINISHED",
        start_time=1234567890,
        end_time=1234567890,
        lifecycle_stage="active",
    )
    run = Run(run_info=run_info, run_data=run_data)

    mock_tracking_store.get_run.return_value = run

    store.link_prompt_version_to_run("test_chat", "1", run_id)

    # Verify linking worked
    mock_tracking_store.set_tag.assert_called_once()
    call_args = mock_tracking_store.set_tag.call_args
    run_tag = call_args[0][1]
    assert run_tag.key == LINKED_PROMPTS_TAG_KEY

    tag_value = json.loads(run_tag.value)
    assert tag_value == [{"name": "test_chat", "version": "1"}]


def test_link_prompt_with_response_format_to_run(store, mock_tracking_store):
    """Test linking prompts with response format to runs."""
    response_format = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }
    prompt_version = PromptVersion(
        "test_response", 1, "What is {{question}}?", response_format=response_format
    )

    store.prompt_versions["test_response:1"] = prompt_version

    run_id = "run_123"
    run_data = RunData(metrics=[], params=[], tags={})
    run_info = RunInfo(
        run_id=run_id,
        experiment_id="exp_123",
        user_id="user_123",
        status="FINISHED",
        start_time=1234567890,
        end_time=1234567890,
        lifecycle_stage="active",
    )
    run = Run(run_info=run_info, run_data=run_data)

    mock_tracking_store.get_run.return_value = run

    store.link_prompt_version_to_run("test_response", "1", run_id)

    # Verify linking worked
    mock_tracking_store.set_tag.assert_called_once()
    call_args = mock_tracking_store.set_tag.call_args
    run_tag = call_args[0][1]
    assert run_tag.key == LINKED_PROMPTS_TAG_KEY

    tag_value = json.loads(run_tag.value)
    assert tag_value == [{"name": "test_response", "version": "1"}]


def test_link_multiple_prompt_types_to_model(store, mock_tracking_store):
    """Test linking both text and chat prompts to the same model."""
    # Create text prompt
    text_prompt = PromptVersion("test_text", 1, "Hello {{name}}!")

    # Create chat prompt
    chat_template = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "{{question}}"},
    ]
    chat_prompt = PromptVersion("test_chat", 1, chat_template)

    store.prompt_versions["test_text:1"] = text_prompt
    store.prompt_versions["test_chat:1"] = chat_prompt

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

    # Mock the behavior where set_logged_model_tags updates the model's tags
    def mock_set_tags(model_id, tags):
        for tag in tags:
            logged_model.tags[tag.key] = tag.value

    mock_tracking_store.get_logged_model.return_value = logged_model
    mock_tracking_store.set_logged_model_tags.side_effect = mock_set_tags

    # Link both prompts
    store.link_prompt_version_to_model("test_text", "1", model_id)
    store.link_prompt_version_to_model("test_chat", "1", model_id)

    # Verify both were linked
    assert mock_tracking_store.set_logged_model_tags.call_count == 2

    # Check final state
    final_call = mock_tracking_store.set_logged_model_tags.call_args_list[-1]
    logged_model_tags = final_call[0][1]
    tag_value = json.loads(logged_model_tags[0].value)

    expected_prompts = [
        {"name": "test_text", "version": "1"},
        {"name": "test_chat", "version": "1"},
    ]
    assert len(tag_value) == 2
    for expected_prompt in expected_prompts:
        assert expected_prompt in tag_value
