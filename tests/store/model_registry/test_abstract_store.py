import json
import threading
import time
from typing import Optional, Union
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
        self.registered_models = {}

    def create_model_version(
        self,
        name: str,
        source: str,
        run_id: Optional[str] = None,
        tags: Optional[list[ModelVersionTag]] = None,
        run_link: Optional[str] = None,
        description: Optional[str] = None,
        local_model_path: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> ModelVersion:
        """Mock implementation of create_model_version."""
        # Generate a new version number
        existing_versions = [
            int(mv.version) for k, mv in self.model_versions.items() if k.startswith(f"{name}:")
        ]
        version = str(max(existing_versions) + 1) if existing_versions else "1"

        # Create the model version
        mv = ModelVersion(
            name=name,
            version=version,
            creation_timestamp=1234567890,
            last_updated_timestamp=1234567890,
            description=description,
            tags=tags or [],
            run_id=run_id,
            run_link=run_link,
        )

        # Store it
        key = f"{name}:{version}"
        self.model_versions[key] = mv

        return mv

    def get_registered_model(self, name: str):
        """Mock implementation of get_registered_model."""
        if name not in self.registered_models:
            # Create a default registered model for testing
            from mlflow.entities.model_registry.registered_model import RegisteredModel
            from mlflow.entities.model_registry.registered_model_tag import (
                RegisteredModelTag,
            )

            self.registered_models[name] = RegisteredModel(
                name=name,
                creation_timestamp=1234567890,
                last_updated_timestamp=1234567890,
                description="Test registered model",
                tags=[RegisteredModelTag(key="test_tag", value="test_value")],
            )
        return self.registered_models[name]

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

    def create_prompt_version(
        self,
        name: str,
        template: Union[str, list[dict[str, str]]],
        prompt_type: str = "text",
        response_format: Optional[Union[dict, type]] = None,
        config: Optional[dict] = None,
        description: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> PromptVersion:
        """Mock implementation of create_prompt_version."""
        # Use the parent class implementation to create the prompt version
        prompt_version = super().create_prompt_version(
            name=name,
            template=template,
            prompt_type=prompt_type,
            response_format=response_format,
            config=config,
            description=description,
            tags=tags,
        )

        # Store the prompt version for later retrieval
        key = f"{name}:{prompt_version.version}"
        self.prompt_versions[key] = prompt_version

        return prompt_version


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
        MlflowException,
        match="Prompt version 'nonexistent_prompt' version 'v1' not found",
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
        metrics=[],
        params=[],
        tags=[RunTag(LINKED_PROMPTS_TAG_KEY, json.dumps(existing_prompts))],
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
        metrics=[],
        params=[],
        tags=[RunTag(LINKED_PROMPTS_TAG_KEY, json.dumps(existing_prompts))],
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


def test_create_prompt_version_text_basic(store):
    """Test basic text prompt creation."""
    prompt_version = store.create_prompt_version(
        name="text_prompt",
        template="Hello, {{name}}! How are you {{mood}}?",
        prompt_type="text",
        description="A friendly greeting",
        tags={"author": "Alice"},
    )

    assert prompt_version.name == "text_prompt"
    assert prompt_version.version == 1
    assert prompt_version.template == "Hello, {{name}}! How are you {{mood}}?"
    assert prompt_version.prompt_type == "text"
    assert prompt_version.response_format is None
    assert prompt_version.config is None
    assert prompt_version.variables == {"name", "mood"}
    assert prompt_version.commit_message == "A friendly greeting"
    assert prompt_version.tags == {"author": "Alice"}


def test_create_prompt_version_chat_basic(store):
    """Test basic chat prompt creation."""
    chat_template = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, {{name}}! How are you {{mood}}?"},
    ]

    prompt_version = store.create_prompt_version(
        name="chat_prompt",
        template=chat_template,
        prompt_type="chat",
        description="A chat assistant",
        tags={"model": "gpt-4"},
    )

    assert prompt_version.name == "chat_prompt"
    assert prompt_version.version == 1
    assert prompt_version.template == chat_template
    assert prompt_version.prompt_type == "chat"
    assert prompt_version.response_format is None
    assert prompt_version.config is None
    assert prompt_version.variables == {"name", "mood"}
    assert prompt_version.commit_message == "A chat assistant"
    assert prompt_version.tags == {"model": "gpt-4"}


def test_create_prompt_version_with_response_format(store):
    """Test prompt creation with response format."""
    response_format = {
        "type": "object",
        "properties": {
            "response": {"type": "string"},
            "confidence": {"type": "number"},
        },
    }

    prompt_version = store.create_prompt_version(
        name="structured_prompt",
        template="Generate a response for {{topic}}.",
        response_format=response_format,
        description="Structured response prompt",
    )

    assert prompt_version.name == "structured_prompt"
    assert prompt_version.version == 1
    assert prompt_version.template == "Generate a response for {{topic}}."
    assert prompt_version.prompt_type == "text"  # Default
    assert prompt_version.response_format == response_format
    assert prompt_version.config is None
    assert prompt_version.variables == {"topic"}


def test_create_prompt_version_with_config(store):
    """Test prompt creation with model configuration."""
    config = {
        "temperature": 0.7,
        "max_tokens": 100,
        "model": "gpt-4",
        "top_p": 0.9,
    }

    prompt_version = store.create_prompt_version(
        name="configured_prompt",
        template="Hello, {{name}}!",
        config=config,
        description="Prompt with config",
    )

    assert prompt_version.name == "configured_prompt"
    assert prompt_version.version == 1
    assert prompt_version.template == "Hello, {{name}}!"
    assert prompt_version.prompt_type == "text"  # Default
    assert prompt_version.response_format is None
    assert prompt_version.config == config
    assert prompt_version.variables == {"name"}


def test_create_prompt_version_with_all_features(store):
    """Test prompt creation with all new features."""
    chat_template = [
        {"role": "system", "content": "You are a {{assistant_type}} assistant."},
        {"role": "user", "content": "Help me with {{task}}."},
    ]
    response_format = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "explanation": {"type": "string"},
            "confidence": {"type": "number"},
        },
    }
    config = {
        "temperature": 0.5,
        "max_tokens": 200,
        "model": "gpt-4-turbo",
    }

    prompt_version = store.create_prompt_version(
        name="full_featured_prompt",
        template=chat_template,
        prompt_type="chat",
        response_format=response_format,
        config=config,
        description="Full featured prompt",
        tags={"project": "ai-assistant", "version": "1.0"},
    )

    assert prompt_version.name == "full_featured_prompt"
    assert prompt_version.version == 1
    assert prompt_version.template == chat_template
    assert prompt_version.prompt_type == "chat"
    assert prompt_version.response_format == response_format
    assert prompt_version.config == config
    assert prompt_version.variables == {"assistant_type", "task"}
    assert prompt_version.commit_message == "Full featured prompt"
    assert prompt_version.tags == {"project": "ai-assistant", "version": "1.0"}


def test_create_prompt_version_pydantic_response_format(store):
    """Test prompt creation with Pydantic response format."""
    try:
        from pydantic import BaseModel

        class ResponseModel(BaseModel):
            answer: str
            confidence: float
            metadata: dict

        prompt_version = store.create_prompt_version(
            name="pydantic_prompt",
            template="Answer: {{question}}",
            response_format=ResponseModel,
            description="Pydantic response format",
        )

        assert prompt_version.name == "pydantic_prompt"
        assert prompt_version.version == 1
        assert prompt_version.response_format is not None
        assert "properties" in prompt_version.response_format
        assert "answer" in prompt_version.response_format["properties"]
        assert "confidence" in prompt_version.response_format["properties"]
        assert "metadata" in prompt_version.response_format["properties"]

    except ImportError:
        pytest.skip("Pydantic not available")


def test_create_prompt_version_versioning(store):
    """Test that multiple versions of the same prompt work correctly."""
    # Create first version
    v1 = store.create_prompt_version(
        name="versioned_prompt",
        template="Hello, {{name}}!",
        prompt_type="text",
        description="Version 1",
    )

    assert v1.version == 1
    assert v1.template == "Hello, {{name}}!"
    assert v1.prompt_type == "text"

    # Create second version with different features
    chat_template = [
        {"role": "user", "content": "Hello, {{name}}!"},
    ]
    v2 = store.create_prompt_version(
        name="versioned_prompt",
        template=chat_template,
        prompt_type="chat",
        response_format={"type": "string"},
        config={"temperature": 0.7},
        description="Version 2 - chat",
    )

    assert v2.version == 2
    assert v2.template == chat_template
    assert v2.prompt_type == "chat"
    assert v2.response_format == {"type": "string"}
    assert v2.config == {"temperature": 0.7}

    # Verify both versions exist
    loaded_v1 = store.get_prompt_version("versioned_prompt", "1")
    loaded_v2 = store.get_prompt_version("versioned_prompt", "2")

    assert loaded_v1.template == "Hello, {{name}}!"
    assert loaded_v1.prompt_type == "text"
    assert loaded_v2.template == chat_template
    assert loaded_v2.prompt_type == "chat"


def test_create_prompt_version_backward_compatibility(store):
    """Test that create_prompt_version maintains backward compatibility."""
    # Test old-style creation (without new parameters)
    prompt_version = store.create_prompt_version(
        name="backward_compat",
        template="Hello, {{name}}!",
        description="Backward compatible",
    )

    assert prompt_version.name == "backward_compat"
    assert prompt_version.version == 1
    assert prompt_version.template == "Hello, {{name}}!"
    assert prompt_version.prompt_type == "text"  # Default
    assert prompt_version.response_format is None
    assert prompt_version.config is None
    assert prompt_version.variables == {"name"}


def test_create_prompt_version_invalid_chat_template(store):
    """Test that invalid chat templates are rejected."""
    # Invalid: not a list
    with pytest.raises(ValueError, match="Chat template must be a list of message dictionaries"):
        store.create_prompt_version(
            name="invalid_chat",
            template="Not a list",
            prompt_type="chat",
        )

    # Invalid: missing role
    with pytest.raises(ValueError, match="Each message must have 'role' and 'content' keys"):
        store.create_prompt_version(
            name="invalid_chat",
            template=[{"content": "Hello"}],
            prompt_type="chat",
        )

    # Invalid: missing content
    with pytest.raises(ValueError, match="Each message must have 'role' and 'content' keys"):
        store.create_prompt_version(
            name="invalid_chat",
            template=[{"role": "user"}],
            prompt_type="chat",
        )

    # Invalid: wrong role
    with pytest.raises(ValueError, match="Role must be one of: system, user, assistant"):
        store.create_prompt_version(
            name="invalid_chat",
            template=[{"role": "invalid", "content": "Hello"}],
            prompt_type="chat",
        )


def test_create_prompt_version_invalid_prompt_type(store):
    """Test that invalid prompt_type is rejected."""
    with pytest.raises(ValueError, match="prompt_type must be 'text' or 'chat'"):
        store.create_prompt_version(
            name="invalid_type",
            template="Hello",
            prompt_type="invalid",
        )


def test_create_prompt_version_invalid_response_format(store):
    """Test that invalid response_format is rejected."""
    with pytest.raises(ValueError, match="response_format must be a Pydantic class or dict"):
        store.create_prompt_version(
            name="invalid_format",
            template="Hello",
            response_format="not a dict or class",
        )


def test_create_prompt_version_complex_chat_scenario(store):
    """Test complex chat prompt scenario with multiple variables."""
    chat_template = [
        {
            "role": "system",
            "content": "You are a {{assistant_type}} assistant for {{company}}.",
        },
        {
            "role": "user",
            "content": "Hello! My name is {{name}} and I work at {{company}}.",
        },
        {
            "role": "assistant",
            "content": "Nice to meet you, {{name}}! I'm here to help with {{task}}.",
        },
        {"role": "user", "content": "Can you help me with {{specific_request}}?"},
    ]

    response_format = {
        "type": "object",
        "properties": {
            "solution": {"type": "string"},
            "steps": {"type": "array", "items": {"type": "string"}},
            "estimated_time": {"type": "string"},
        },
    }

    config = {
        "temperature": 0.3,
        "max_tokens": 500,
        "model": "gpt-4",
        "top_p": 0.8,
    }

    prompt_version = store.create_prompt_version(
        name="complex_assistant",
        template=chat_template,
        prompt_type="chat",
        response_format=response_format,
        config=config,
        description="Complex assistant prompt",
        tags={"category": "technical-support", "priority": "high"},
    )

    assert prompt_version.name == "complex_assistant"
    assert prompt_version.version == 1
    assert prompt_version.template == chat_template
    assert prompt_version.prompt_type == "chat"
    assert prompt_version.response_format == response_format
    assert prompt_version.config == config
    assert prompt_version.variables == {
        "assistant_type",
        "company",
        "name",
        "task",
        "specific_request",
    }
    assert prompt_version.tags == {"category": "technical-support", "priority": "high"}


def test_create_prompt_version_with_none_values(store):
    """Test that None values are handled correctly."""
    prompt_version = store.create_prompt_version(
        name="none_values",
        template="Hello, {{name}}!",
        prompt_type=None,  # Should default to "text"
        response_format=None,
        config=None,
        description=None,
        tags=None,
    )

    assert prompt_version.name == "none_values"
    assert prompt_version.template == "Hello, {{name}}!"
    assert prompt_version.prompt_type == "text"  # Default
    assert prompt_version.response_format is None
    assert prompt_version.config is None
    assert prompt_version.commit_message is None
    assert prompt_version.tags == {}


def test_create_prompt_version_format_functionality(store):
    """Test that created prompts can be formatted correctly."""
    # Create text prompt
    text_prompt = store.create_prompt_version(
        name="format_text",
        template="Hello, {{name}}! How are you {{mood}}?",
        prompt_type="text",
    )

    # Test formatting
    formatted = text_prompt.format(name="Alice", mood="happy")
    assert formatted == "Hello, Alice! How are you happy?"

    # Create chat prompt
    chat_template = [
        {"role": "system", "content": "You are a {{assistant_type}} assistant."},
        {"role": "user", "content": "Hello, {{name}}!"},
    ]
    chat_prompt = store.create_prompt_version(
        name="format_chat",
        template=chat_template,
        prompt_type="chat",
    )

    # Test formatting
    formatted = chat_prompt.format_chat(assistant_type="helpful", name="Bob")
    expected = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, Bob!"},
    ]
    assert formatted == expected
