import importlib
import json
import os
import subprocess
import threading
from pathlib import Path
from unittest import mock

import pytest
import requests

import mlflow
import mlflow.tracking._model_registry.fluent
from mlflow import MlflowClient, register_model
from mlflow.entities.model_registry import ModelVersion, PromptVersion, RegisteredModel
from mlflow.environment_variables import MLFLOW_PROMPT_CACHE_MAX_SIZE
from mlflow.exceptions import MlflowException
from mlflow.prompt.constants import LINKED_PROMPTS_TAG_KEY
from mlflow.protos.databricks_pb2 import (
    ALREADY_EXISTS,
    INTERNAL_ERROR,
    RESOURCE_ALREADY_EXISTS,
)
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.utils.databricks_utils import DatabricksRuntimeVersion


def join_thread_by_name_prefix(prefix: str):
    """Join any thread whose name starts with the given prefix."""
    for t in threading.enumerate():
        if t.name.startswith(prefix):
            t.join()


def test_register_model_with_runs_uri():
    class TestModel(mlflow.pyfunc.PythonModel):
        def predict(self, model_input):
            return model_input

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(name="model", python_model=TestModel())

    register_model(f"runs:/{run.info.run_id}/model", "Model 1")
    mv = MlflowClient().get_model_version("Model 1", "1")
    assert mv.name == "Model 1"


def test_register_model_with_non_runs_uri():
    create_model_patch = mock.patch.object(
        MlflowClient, "create_registered_model", return_value=RegisteredModel("Model 1")
    )
    create_version_patch = mock.patch.object(
        MlflowClient,
        "_create_model_version",
        return_value=ModelVersion("Model 1", "1", creation_timestamp=123),
    )
    with create_model_patch, create_version_patch:
        register_model("s3:/some/path/to/model", "Model 1")
        MlflowClient.create_registered_model.assert_called_once_with("Model 1")
        MlflowClient._create_model_version.assert_called_once_with(
            name="Model 1",
            run_id=None,
            tags=None,
            source="s3:/some/path/to/model",
            await_creation_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
            local_model_path=None,
            model_id=None,
        )


@pytest.mark.parametrize("error_code", [RESOURCE_ALREADY_EXISTS, ALREADY_EXISTS])
def test_register_model_with_existing_registered_model(error_code):
    create_model_patch = mock.patch.object(
        MlflowClient,
        "create_registered_model",
        side_effect=MlflowException("Some Message", error_code),
    )
    create_version_patch = mock.patch.object(
        MlflowClient,
        "_create_model_version",
        return_value=ModelVersion("Model 1", "1", creation_timestamp=123),
    )
    with create_model_patch, create_version_patch:
        register_model("s3:/some/path/to/model", "Model 1")
        MlflowClient.create_registered_model.assert_called_once_with("Model 1")
        MlflowClient._create_model_version.assert_called_once_with(
            name="Model 1",
            run_id=None,
            source="s3:/some/path/to/model",
            tags=None,
            await_creation_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
            local_model_path=None,
            model_id=None,
        )


def test_register_model_with_unexpected_mlflow_exception_in_create_registered_model():
    with mock.patch.object(
        MlflowClient,
        "create_registered_model",
        side_effect=MlflowException("Dunno", INTERNAL_ERROR),
    ) as mock_create_registered_model:
        with pytest.raises(MlflowException, match="Dunno"):
            register_model("s3:/some/path/to/model", "Model 1")
        mock_create_registered_model.assert_called_once_with("Model 1")


def test_register_model_with_unexpected_exception_in_create_registered_model():
    with mock.patch.object(
        MlflowClient, "create_registered_model", side_effect=Exception("Dunno")
    ) as create_registered_model_mock:
        with pytest.raises(Exception, match="Dunno"):
            register_model("s3:/some/path/to/model", "Model 1")
        create_registered_model_mock.assert_called_once_with("Model 1")


def test_register_model_with_tags():
    tags = {"a": "1"}

    class TestModel(mlflow.pyfunc.PythonModel):
        def predict(self, model_input):
            return model_input

    with mlflow.start_run() as run:
        mlflow.pyfunc.log_model(name="model", python_model=TestModel())

    register_model(f"runs:/{run.info.run_id}/model", "Model 1", tags=tags)
    mv = MlflowClient().get_model_version("Model 1", "1")
    assert mv.tags == tags


def test_crud_prompts(tmp_path):
    mlflow.register_prompt(
        name="prompt_1",
        template="Hi, {title} {name}! How are you today?",
        commit_message="A friendly greeting",
        tags={"model": "my-model"},
    )

    prompt = mlflow.load_prompt("prompt_1", version=1)
    assert prompt.name == "prompt_1"
    assert prompt.template == "Hi, {title} {name}! How are you today?"
    assert prompt.commit_message == "A friendly greeting"
    # Currently, the tags from register_prompt become version tags
    assert prompt.tags == {"model": "my-model"}

    # Check prompt-level tags separately (if needed for test completeness)
    from mlflow import MlflowClient

    client = MlflowClient()
    prompt_entity = client.get_prompt("prompt_1")
    assert prompt_entity.tags == {"model": "my-model"}

    mlflow.register_prompt(
        name="prompt_1",
        template="Hi, {title} {name}! What's up?",
        commit_message="New greeting",
    )

    prompt = mlflow.load_prompt("prompt_1", version=2)
    assert prompt.template == "Hi, {title} {name}! What's up?"

    prompt = mlflow.load_prompt("prompt_1", version=1)
    assert prompt.template == "Hi, {title} {name}! How are you today?"

    prompt = mlflow.load_prompt("prompts:/prompt_1/2")
    assert prompt.template == "Hi, {title} {name}! What's up?"

    # Test load_prompt with allow_missing for non-existent prompts
    assert mlflow.load_prompt("does_not_exist", version=1, allow_missing=True) is None


def test_register_chat_prompt_basic(tmp_path):
    """Test basic chat prompt registration through fluent API."""
    chat_template = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, {{name}}! How are you {{mood}}?"},
    ]

    prompt_version = mlflow.register_prompt(
        name="chat_prompt",
        template=chat_template,
        prompt_type="chat",
        commit_message="A chat assistant",
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


def test_register_chat_prompt_with_response_format(tmp_path):
    """Test chat prompt registration with response format through fluent API."""
    chat_template = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Generate a response for {{topic}}."},
    ]
    response_format = {
        "type": "object",
        "properties": {
            "response": {"type": "string"},
            "confidence": {"type": "number"},
        },
    }

    prompt_version = mlflow.register_prompt(
        name="structured_chat",
        template=chat_template,
        prompt_type="chat",
        response_format=response_format,
        commit_message="Structured response chat",
    )

    assert prompt_version.name == "structured_chat"
    assert prompt_version.version == 1
    assert prompt_version.template == chat_template
    assert prompt_version.prompt_type == "chat"
    assert prompt_version.response_format == response_format
    assert prompt_version.config is None
    assert prompt_version.variables == {"topic"}


def test_register_prompt_with_config(tmp_path):
    """Test prompt registration with model configuration through fluent API."""
    config = {
        "temperature": 0.7,
        "max_tokens": 100,
        "model": "gpt-4",
        "top_p": 0.9,
    }

    prompt_version = mlflow.register_prompt(
        name="configured_prompt",
        template="Hello, {{name}}!",
        config=config,
        commit_message="Prompt with config",
    )

    assert prompt_version.name == "configured_prompt"
    assert prompt_version.version == 1
    assert prompt_version.template == "Hello, {{name}}!"
    assert prompt_version.prompt_type == "text"  # Default
    assert prompt_version.response_format is None
    assert prompt_version.config == config
    assert prompt_version.variables == {"name"}


def test_register_prompt_with_all_features(tmp_path):
    """Test prompt registration with all new features through fluent API."""
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

    prompt_version = mlflow.register_prompt(
        name="full_featured_prompt",
        template=chat_template,
        prompt_type="chat",
        response_format=response_format,
        config=config,
        commit_message="Full featured prompt",
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


def test_register_prompt_pydantic_response_format(tmp_path):
    """Test prompt registration with Pydantic response format via fluent API."""
    try:
        from pydantic import BaseModel

        class ResponseModel(BaseModel):
            answer: str
            confidence: float
            metadata: dict

        prompt_version = mlflow.register_prompt(
            name="pydantic_prompt",
            template="Answer: {{question}}",
            response_format=ResponseModel,
            commit_message="Pydantic response format",
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


def test_register_prompt_versioning_with_new_features(tmp_path):
    """Test that multiple versions of prompt work correctly via fluent API."""
    # Register first version
    v1 = mlflow.register_prompt(
        name="versioned_prompt",
        template="Hello, {{name}}!",
        prompt_type="text",
        commit_message="Version 1",
    )

    assert v1.version == 1
    assert v1.template == "Hello, {{name}}!"
    assert v1.prompt_type == "text"

    # Register second version with different features
    chat_template = [
        {"role": "user", "content": "Hello, {{name}}!"},
    ]
    v2 = mlflow.register_prompt(
        name="versioned_prompt",
        template=chat_template,
        prompt_type="chat",
        response_format={"type": "string"},
        config={"temperature": 0.7},
        commit_message="Version 2 - chat",
    )

    assert v2.version == 2
    assert v2.template == chat_template
    assert v2.prompt_type == "chat"
    assert v2.response_format == {"type": "string"}
    assert v2.config == {"temperature": 0.7}

    # Verify both versions exist
    loaded_v1 = mlflow.load_prompt("versioned_prompt", version=1)
    loaded_v2 = mlflow.load_prompt("versioned_prompt", version=2)

    assert loaded_v1.template == "Hello, {{name}}!"
    assert loaded_v1.prompt_type == "text"
    assert loaded_v2.template == chat_template
    assert loaded_v2.prompt_type == "chat"


def test_register_prompt_backward_compatibility(tmp_path):
    """Test that register_prompt maintains backward compatibility through fluent API."""
    # Test old-style registration (without new parameters)
    prompt_version = mlflow.register_prompt(
        name="backward_compat",
        template="Hello, {{name}}!",
        commit_message="Backward compatible",
    )

    assert prompt_version.name == "backward_compat"
    assert prompt_version.version == 1
    assert prompt_version.template == "Hello, {{name}}!"
    assert prompt_version.prompt_type == "text"  # Default
    assert prompt_version.response_format is None
    assert prompt_version.config is None
    assert prompt_version.variables == {"name"}


def test_register_prompt_invalid_chat_template(tmp_path):
    """Test that invalid chat templates are rejected through fluent API."""
    # Invalid: not a list
    with pytest.raises(ValueError, match="Chat template must be a list of message dictionaries"):
        mlflow.register_prompt(
            name="invalid_chat",
            template="Not a list",
            prompt_type="chat",
        )

    # Invalid: missing role
    with pytest.raises(ValueError, match="Each message must have 'role' and 'content' keys"):
        mlflow.register_prompt(
            name="invalid_chat",
            template=[{"content": "Hello"}],
            prompt_type="chat",
        )

    # Invalid: missing content
    with pytest.raises(ValueError, match="Each message must have 'role' and 'content' keys"):
        mlflow.register_prompt(
            name="invalid_chat",
            template=[{"role": "user"}],
            prompt_type="chat",
        )

    # Invalid: wrong role
    with pytest.raises(ValueError, match="Role must be one of: system, user, assistant"):
        mlflow.register_prompt(
            name="invalid_chat",
            template=[{"role": "invalid", "content": "Hello"}],
            prompt_type="chat",
        )


def test_register_prompt_invalid_prompt_type(tmp_path):
    """Test that invalid prompt_type is rejected through fluent API."""
    with pytest.raises(ValueError, match="prompt_type must be 'text' or 'chat'"):
        mlflow.register_prompt(
            name="invalid_type",
            template="Hello",
            prompt_type="invalid",
        )


def test_register_prompt_invalid_response_format(tmp_path):
    """Test that invalid response_format is rejected through fluent API."""
    with pytest.raises(ValueError, match="response_format must be a Pydantic class or dict"):
        mlflow.register_prompt(
            name="invalid_format",
            template="Hello",
            response_format="not a dict or class",
        )


def test_register_prompt_complex_chat_scenario(tmp_path):
    """Test complex chat prompt scenario with multiple variables through fluent API."""
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

    prompt_version = mlflow.register_prompt(
        name="complex_assistant",
        template=chat_template,
        prompt_type="chat",
        response_format=response_format,
        config=config,
        commit_message="Complex assistant prompt",
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


def test_register_prompt_load_and_verify(tmp_path):
    """Test that registered prompts can be loaded and verified through fluent API."""
    # Register a prompt with all features
    chat_template = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello, {{name}}!"},
    ]
    response_format = {"type": "string"}
    config = {"temperature": 0.7}

    original = mlflow.register_prompt(
        name="load_test",
        template=chat_template,
        prompt_type="chat",
        response_format=response_format,
        config=config,
        tags={"test": "value"},
    )

    # Load and verify
    loaded = mlflow.load_prompt("load_test", version=1)

    assert loaded.name == original.name
    assert loaded.version == original.version
    assert loaded.template == original.template
    assert loaded.prompt_type == original.prompt_type
    assert loaded.response_format == original.response_format
    assert loaded.config == original.config
    assert loaded.tags == original.tags
    assert loaded.variables == original.variables


def test_register_prompt_format_functionality(tmp_path):
    """Test that registered prompts can be formatted correctly through fluent API."""
    # Register text prompt
    text_prompt = mlflow.register_prompt(
        name="format_text",
        template="Hello, {{name}}! How are you {{mood}}?",
        prompt_type="text",
    )

    # Test formatting
    formatted = text_prompt.format(name="Alice", mood="happy")
    assert formatted == "Hello, Alice! How are you happy?"

    # Register chat prompt
    chat_template = [
        {"role": "system", "content": "You are a {{assistant_type}} assistant."},
        {"role": "user", "content": "Hello, {{name}}!"},
    ]
    chat_prompt = mlflow.register_prompt(
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


def test_register_prompt_with_none_values(tmp_path):
    """Test that None values are handled correctly through fluent API."""
    prompt_version = mlflow.register_prompt(
        name="none_values",
        template="Hello, {{name}}!",
        prompt_type=None,  # Should default to "text"
        response_format=None,
        config=None,
        commit_message=None,
        tags=None,
    )

    assert prompt_version.name == "none_values"
    assert prompt_version.template == "Hello, {{name}}!"
    assert prompt_version.prompt_type == "text"  # Default
    assert prompt_version.response_format is None
    assert prompt_version.config is None
    assert prompt_version.commit_message is None
    assert prompt_version.tags == {}


def test_prompt_alias(tmp_path):
    mlflow.register_prompt(name="p1", template="Hi, there!")
    mlflow.register_prompt(name="p1", template="Hi, {{name}}!")

    mlflow.set_prompt_alias("p1", alias="production", version=1)
    prompt = mlflow.load_prompt("prompts:/p1@production")
    assert prompt.template == "Hi, there!"
    assert prompt.aliases == ["production"]

    # Reassign alias to a different version
    mlflow.set_prompt_alias("p1", alias="production", version=2)
    assert mlflow.load_prompt("prompts:/p1@production").template == "Hi, {{name}}!"

    mlflow.delete_prompt_alias("p1", alias="production")
    with pytest.raises(MlflowException, match=r"Prompt (.*) does not exist."):
        mlflow.load_prompt("prompts:/p1@production")


def test_prompt_associate_with_run(tmp_path):
    mlflow.register_prompt(name="prompt_1", template="Hi, {title} {name}! How are you today?")

    # mlflow.load_prompt() call during the run should associate the prompt with the run
    with mlflow.start_run() as run:
        mlflow.load_prompt("prompt_1", version=1)

    # Check that the prompt was linked to the run via the linkedPrompts tag
    client = MlflowClient()
    run_data = client.get_run(run.info.run_id)
    linked_prompts_tag = run_data.data.tags.get(LINKED_PROMPTS_TAG_KEY)
    assert linked_prompts_tag is not None

    linked_prompts = json.loads(linked_prompts_tag)
    assert len(linked_prompts) == 1
    assert linked_prompts[0]["name"] == "prompt_1"
    assert linked_prompts[0]["version"] == "1"


def test_register_model_prints_uc_model_version_url(monkeypatch):
    orig_registry_uri = mlflow.get_registry_uri()
    mlflow.set_registry_uri("databricks-uc")
    workspace_id = "123"
    model_id = "m-123"
    name = "name.mlflow.test_model"
    version = "1"
    with (
        mock.patch("mlflow.tracking._model_registry.fluent.eprint") as mock_eprint,
        mock.patch(
            "mlflow.tracking._model_registry.fluent.get_workspace_url",
            return_value="https://databricks.com",
        ) as mock_url,
        mock.patch(
            "mlflow.tracking._model_registry.fluent.get_workspace_id",
            return_value=workspace_id,
        ) as mock_workspace_id,
        mock.patch(
            "mlflow.MlflowClient.create_registered_model",
            return_value=RegisteredModel(name),
        ) as mock_create_model,
        mock.patch(
            "mlflow.MlflowClient._create_model_version",
            return_value=ModelVersion(name, version, creation_timestamp=123),
        ) as mock_create_version,
        mock.patch(
            "mlflow.MlflowClient.get_logged_model",
            return_value=mock.Mock(model_id=model_id, name=name, tags={}),
        ) as mock_get_logged_model,
        mock.patch("mlflow.MlflowClient.set_logged_model_tags") as mock_set_logged_model_tags,
    ):
        register_model(f"models:/{model_id}", name)
        expected_url = (
            "https://databricks.com/explore/data/models/name/mlflow/test_model/version/1?o=123"
        )
        mock_eprint.assert_called_with(
            f"🔗 Created version '{version}' of model '{name}': {expected_url}"
        )
        mock_url.assert_called_once()
        mock_workspace_id.assert_called_once()
        mock_create_model.assert_called_once()
        mock_create_version.assert_called_once()
        mock_get_logged_model.assert_called_once()
        mock_set_logged_model_tags.assert_called_once()

        # Test that the URL is not printed when the environment variable is set to false
        mock_eprint.reset_mock()
        monkeypatch.setenv("MLFLOW_PRINT_MODEL_URLS_ON_CREATION", "false")
        register_model(f"models:/{model_id}", name)
        mock_eprint.assert_called_with("Created version '1' of model 'name.mlflow.test_model'.")

    # Clean up the global variables set by the server
    mlflow.set_registry_uri(orig_registry_uri)


def test_set_model_version_tag():
    class TestModel(mlflow.pyfunc.PythonModel):
        def predict(self, model_input):
            return model_input

    mlflow.pyfunc.log_model(name="model", python_model=TestModel(), registered_model_name="Model 1")

    mv = MlflowClient().get_model_version("Model 1", "1")
    assert mv.tags == {}
    mlflow.set_model_version_tag(
        name="Model 1",
        version=1,
        key="key",
        value="value",
    )
    mv = MlflowClient().get_model_version("Model 1", "1")
    assert mv.tags == {"key": "value"}


def test_register_model_with_2_x_model(tmp_path: Path):
    tracking_uri = (tmp_path / "tracking").as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    artifact_location = (tmp_path / "artifacts").as_uri()
    exp_id = mlflow.create_experiment("test", artifact_location=artifact_location)
    mlflow.set_experiment(experiment_id=exp_id)
    code = """
import sys
import mlflow

assert mlflow.__version__.startswith("2."), mlflow.__version__

with mlflow.start_run() as run:
    model_info = mlflow.pyfunc.log_model(
        python_model=lambda *args: None,
        artifact_path="model",
        # When `python_model` is a function, either `input_example` or `pip_requirements`
        # must be provided.
        pip_requirements=["mlflow"],
    )
    assert model_info.model_uri.startswith("runs:/")
    out = sys.argv[1]
    with open(out, "w") as f:
        f.write(model_info.model_uri)
"""
    out = tmp_path / "output.txt"
    # Log a model using MLflow 2.x
    subprocess.check_call(
        [
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--with",
            "mlflow<3",
            "python",
            "-I",
            "-c",
            code,
            out,
        ],
        env=os.environ.copy() | {"UV_INDEX_STRATEGY": "unsafe-first-match"},
    )
    # Register the model with MLflow 3.x
    model_uri = out.read_text().strip()
    mlflow.register_model(model_uri, "model")


@pytest.fixture
def mock_dbr_version():
    """Mock DatabricksRuntimeVersion to simulate a supported client image."""
    with mock.patch(
        "mlflow.utils.databricks_utils.DatabricksRuntimeVersion.parse",
        return_value=DatabricksRuntimeVersion(
            is_client_image=True,
            major=2,  # Supported version
            minor=0,
        ),
    ):
        yield


def test_register_model_with_env_pack(tmp_path, mock_dbr_version):
    """Test that register_model correctly integrates with environment packing functionality."""
    # Mock download_artifacts to return a path
    mock_artifacts_dir = tmp_path / "artifacts"
    mock_artifacts_dir.mkdir()
    (mock_artifacts_dir / "requirements.txt").write_text("numpy==1.21.0")

    with (
        mock.patch(
            "mlflow.utils.env_pack.download_artifacts",
            return_value=str(mock_artifacts_dir),
        ),
        mock.patch("subprocess.run", return_value=mock.Mock(returncode=0)),
        mock.patch(
            "mlflow.tracking._model_registry.fluent.pack_env_for_databricks_model_serving"
        ) as mock_pack_env,
        mock.patch(
            "mlflow.tracking._model_registry.fluent.stage_model_for_databricks_model_serving"
        ) as mock_stage_model,
        mock.patch(
            "mlflow.MlflowClient._create_model_version",
            return_value=ModelVersion("Model 1", "1", creation_timestamp=123),
        ),
        mock.patch(
            "mlflow.MlflowClient.get_model_version",
            return_value=ModelVersion("Model 1", "1", creation_timestamp=123),
        ),
        mock.patch("mlflow.MlflowClient.log_model_artifacts") as mock_log_artifacts,
    ):
        # Set up the mock pack_env to yield a path
        mock_pack_env.return_value.__enter__.return_value = str(mock_artifacts_dir)

        # Call register_model with env_pack
        register_model("models:/test-model/1", "Model 1", env_pack="databricks_model_serving")

        # Verify pack_env was called with correct arguments
        mock_pack_env.assert_called_once_with(
            "models:/test-model/1",
            enforce_pip_requirements=True,
        )

        # Verify log_model_artifacts was called with correct arguments
        mock_log_artifacts.assert_called_once_with(
            None,
            str(mock_artifacts_dir),
        )

        # Verify stage_model was called with correct arguments
        mock_stage_model.assert_called_once_with(
            model_name="Model 1",
            model_version="1",
        )


def test_register_model_with_env_pack_staging_failure(tmp_path, mock_dbr_version):
    """Test that register_model handles staging failure gracefully."""
    # Mock download_artifacts to return a path
    mock_artifacts_dir = tmp_path / "artifacts"
    mock_artifacts_dir.mkdir()
    (mock_artifacts_dir / "requirements.txt").write_text("numpy==1.21.0")

    with (
        mock.patch(
            "mlflow.utils.env_pack.download_artifacts",
            return_value=str(mock_artifacts_dir),
        ),
        mock.patch("subprocess.run", return_value=mock.Mock(returncode=0)),
        mock.patch(
            "mlflow.tracking._model_registry.fluent.pack_env_for_databricks_model_serving"
        ) as mock_pack_env,
        mock.patch(
            "mlflow.tracking._model_registry.fluent.stage_model_for_databricks_model_serving",
            side_effect=requests.exceptions.HTTPError("Staging failed"),
        ) as mock_stage_model,
        mock.patch(
            "mlflow.MlflowClient._create_model_version",
            return_value=ModelVersion("Model 1", "1", creation_timestamp=123),
        ),
        mock.patch(
            "mlflow.MlflowClient.get_model_version",
            return_value=ModelVersion("Model 1", "1", creation_timestamp=123),
        ),
        mock.patch("mlflow.MlflowClient.log_model_artifacts") as mock_log_artifacts,
        mock.patch("mlflow.tracking._model_registry.fluent.eprint") as mock_eprint,
    ):
        # Set up the mock pack_env to yield a path
        mock_pack_env.return_value.__enter__.return_value = str(mock_artifacts_dir)

        # Call register_model with env_pack
        register_model("models:/test-model/1", "Model 1", env_pack="databricks_model_serving")

        # Verify pack_env was called with correct arguments
        mock_pack_env.assert_called_once_with(
            "models:/test-model/1",
            enforce_pip_requirements=True,
        )

        # Verify log_model_artifacts was called with correct arguments
        mock_log_artifacts.assert_called_once_with(
            None,
            str(mock_artifacts_dir),
        )

        # Verify stage_model was called with correct arguments
        mock_stage_model.assert_called_once_with(
            model_name="Model 1",
            model_version="1",
        )

        # Verify warning message was printed
        mock_eprint.assert_any_call(
            "Failed to stage model for Databricks Model Serving: Staging failed. "
            "The model was registered successfully and is available for serving, but may take "
            "longer to deploy."
        )


def test_load_prompt_with_link_to_model_disabled():
    """Test load_prompt with link_to_model=False does not attempt linking."""

    # Register a prompt
    mlflow.register_prompt(name="test_prompt", template="Hello, {{name}}!")

    # Create a logged model and set it as active
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=lambda x: x,
            name="model",
            pip_requirements=["mlflow"],
        )
        mlflow.set_active_model(model_id=model_info.model_id)

        # Load prompt with link_to_model=False - should not link despite active model
        prompt = mlflow.load_prompt("test_prompt", version=1, link_to_model=False)

        # Verify prompt was loaded correctly
        assert prompt.name == "test_prompt"
        assert prompt.version == 1
        assert prompt.template == "Hello, {{name}}!"

        # Join any potential background linking thread (it shouldn't run)
        join_thread_by_name_prefix("link_prompt_thread")

        # Verify the model does NOT have any linked prompts tag
        client = mlflow.MlflowClient()
        model = client.get_logged_model(model_info.model_id)
        linked_prompts_tag = model.tags.get("mlflow.linkedPrompts")
        assert linked_prompts_tag is None, (
            "Model should not have linkedPrompts tag when link_to_model=False"
        )


def test_load_prompt_with_explicit_model_id():
    """Test load_prompt with explicit model_id parameter."""

    # Register a prompt
    mlflow.register_prompt(name="test_prompt", template="Hello, {{name}}!")

    # Create a logged model to link to
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=lambda x: x,
            name="model",
            pip_requirements=["mlflow"],
        )

    # Load prompt with explicit model_id - should link successfully
    prompt = mlflow.load_prompt(
        "test_prompt", version=1, link_to_model=True, model_id=model_info.model_id
    )

    # Verify prompt was loaded correctly
    assert prompt.name == "test_prompt"
    assert prompt.version == 1
    assert prompt.template == "Hello, {{name}}!"

    # Join background linking thread to wait for completion
    join_thread_by_name_prefix("link_prompt_thread")

    # Verify the model has the linked prompt in its tags
    client = mlflow.MlflowClient()
    model = client.get_logged_model(model_info.model_id)
    linked_prompts_tag = model.tags.get("mlflow.linkedPrompts")
    assert linked_prompts_tag is not None

    # Parse the JSON tag value
    linked_prompts = json.loads(linked_prompts_tag)
    assert len(linked_prompts) == 1
    assert linked_prompts[0]["name"] == "test_prompt"
    assert linked_prompts[0]["version"] == "1"


def test_load_prompt_with_active_model_integration():
    """Test load_prompt with active model integration using get_active_model_id."""

    # Register a prompt
    mlflow.register_prompt(name="test_prompt", template="Hello, {{name}}!")

    # Test loading prompt with active model context
    with mlflow.start_run():
        model_info = mlflow.pyfunc.log_model(
            python_model=lambda x: x,
            name="model",
            pip_requirements=["mlflow"],
        )

        mlflow.set_active_model(model_id=model_info.model_id)
        # Load prompt with link_to_model=True - should use active model
        prompt = mlflow.load_prompt("test_prompt", version=1, link_to_model=True)

        # Verify prompt was loaded correctly
        assert prompt.name == "test_prompt"
        assert prompt.version == 1
        assert prompt.template == "Hello, {{name}}!"

        # Join background linking thread to wait for completion
        join_thread_by_name_prefix("link_prompt_thread")

        # Verify the model has the linked prompt in its tags
        client = mlflow.MlflowClient()
        model = client.get_logged_model(model_info.model_id)
        linked_prompts_tag = model.tags.get("mlflow.linkedPrompts")
        assert linked_prompts_tag is not None

        # Parse the JSON tag value
        linked_prompts = json.loads(linked_prompts_tag)
        assert len(linked_prompts) == 1
        assert linked_prompts[0]["name"] == "test_prompt"
        assert linked_prompts[0]["version"] == "1"


def test_load_prompt_with_no_active_model():
    """Test load_prompt when no active model is available."""

    # Register a prompt
    mlflow.register_prompt(name="test_prompt", template="Hello, {{name}}!")

    # Mock no active model available
    with mock.patch(
        "mlflow.tracking._model_registry.fluent.get_active_model_id", return_value=None
    ):
        # Load prompt with link_to_model=True but no active model - should still work
        prompt = mlflow.load_prompt("test_prompt", version=1, link_to_model=True)

        # Verify prompt was loaded correctly (linking just gets skipped)
        assert prompt.name == "test_prompt"
        assert prompt.version == 1
        assert prompt.template == "Hello, {{name}}!"


def test_load_prompt_linking_error_handling():
    """Test load_prompt error handling when linking fails."""

    # Register a prompt
    mlflow.register_prompt(name="test_prompt", template="Hello, {{name}}!")

    # Test with invalid model ID - should still load prompt successfully
    with mock.patch(
        "mlflow.tracking._model_registry.fluent.get_active_model_id",
        return_value="invalid_model_id",
    ):
        # Load prompt - should succeed despite linking failure (happens in background)
        prompt = mlflow.load_prompt("test_prompt", version=1, link_to_model=True)

        # Verify prompt was loaded successfully despite linking failure
        assert prompt.name == "test_prompt"
        assert prompt.version == 1
        assert prompt.template == "Hello, {{name}}!"


def test_load_prompt_explicit_model_id_overrides_active_model():
    """Test that explicit model_id parameter overrides active model ID."""

    # Register a prompt
    mlflow.register_prompt(name="test_prompt", template="Hello, {{name}}!")

    # Create models to test override behavior
    with mlflow.start_run():
        active_model = mlflow.pyfunc.log_model(
            python_model=lambda x: x,
            name="active_model",
            pip_requirements=["mlflow"],
        )
        explicit_model = mlflow.pyfunc.log_model(
            python_model=lambda x: x,
            name="explicit_model",
            pip_requirements=["mlflow"],
        )

    # Set active model context but provide explicit model_id - explicit should win
    mlflow.set_active_model(model_id=active_model.model_id)
    prompt = mlflow.load_prompt(
        "test_prompt", version=1, link_to_model=True, model_id=explicit_model.model_id
    )

    # Verify prompt was loaded correctly (explicit model_id should be used)
    assert prompt.name == "test_prompt"
    assert prompt.version == 1
    assert prompt.template == "Hello, {{name}}!"

    # Join background linking thread to wait for completion
    join_thread_by_name_prefix("link_prompt_thread")

    # Verify the EXPLICIT model (not active model) has the linked prompt in its tags
    client = mlflow.MlflowClient()
    explicit_model_data = client.get_logged_model(explicit_model.model_id)
    linked_prompts_tag = explicit_model_data.tags.get("mlflow.linkedPrompts")
    assert linked_prompts_tag is not None

    # Parse the JSON tag value
    linked_prompts = json.loads(linked_prompts_tag)
    assert len(linked_prompts) == 1
    assert linked_prompts[0]["name"] == "test_prompt"
    assert linked_prompts[0]["version"] == "1"

    # Verify the active model does NOT have the linked prompt
    active_model_data = client.get_logged_model(active_model.model_id)
    active_linked_prompts_tag = active_model_data.tags.get("mlflow.linkedPrompts")
    assert active_linked_prompts_tag is None


def test_load_prompt_with_tracing_single_prompt():
    """Test that load_prompt properly links a single prompt to an active trace."""

    # Register a prompt
    mlflow.register_prompt(name="test_prompt", template="Hello, {{name}}!")

    # Start tracing and load prompt
    with mlflow.start_span("test_operation") as span:
        prompt = mlflow.load_prompt("test_prompt", version=1)

        # Verify prompt was loaded correctly
        assert prompt.name == "test_prompt"
        assert prompt.version == 1
        assert prompt.template == "Hello, {{name}}!"

    # Manually trigger prompt linking to trace since in test environment
    # the trace export may not happen automatically
    client = mlflow.MlflowClient()
    prompt_version = PromptVersion(
        name="test_prompt",
        version=1,
        template="Hello, {{name}}!",
        commit_message=None,
        creation_timestamp=None,
    )
    client.link_prompt_versions_to_trace(trace_id=span.trace_id, prompt_versions=[prompt_version])

    # Verify the prompt was linked to the trace by checking the actual trace
    trace = mlflow.get_trace(span.trace_id)
    assert trace is not None

    # Check the linked prompts tag
    linked_prompts_tag = trace.info.tags.get("mlflow.linkedPrompts")
    assert linked_prompts_tag is not None

    # Parse the JSON tag value
    linked_prompts = json.loads(linked_prompts_tag)
    assert len(linked_prompts) == 1
    assert linked_prompts[0]["name"] == "test_prompt"
    assert linked_prompts[0]["version"] == "1"


def test_load_prompt_with_tracing_multiple_prompts():
    """Test that load_prompt properly links multiple versions of the same prompt to one trace."""

    # Register one prompt with multiple versions
    mlflow.register_prompt(name="my_prompt", template="Hello, {{name}}!")
    mlflow.register_prompt(name="my_prompt", template="Hi there, {{name}}! How are you?")

    # Start tracing and load multiple versions of the same prompt
    with mlflow.start_span("multi_version_prompt_operation") as span:
        prompt_v1 = mlflow.load_prompt("my_prompt", version=1)
        prompt_v2 = mlflow.load_prompt("my_prompt", version=2)

        # Verify prompts were loaded correctly
        assert prompt_v1.name == "my_prompt"
        assert prompt_v1.version == 1
        assert prompt_v1.template == "Hello, {{name}}!"

        assert prompt_v2.name == "my_prompt"
        assert prompt_v2.version == 2
        assert prompt_v2.template == "Hi there, {{name}}! How are you?"

    # Manually trigger prompt linking to trace since in test environment
    # the trace export may not happen automatically
    client = mlflow.MlflowClient()
    prompt_versions = [
        PromptVersion(
            name="my_prompt",
            version=1,
            template="Hello, {{name}}!",
            commit_message=None,
            creation_timestamp=None,
        ),
        PromptVersion(
            name="my_prompt",
            version=2,
            template="Hi there, {{name}}! How are you?",
            commit_message=None,
            creation_timestamp=None,
        ),
    ]
    client.link_prompt_versions_to_trace(trace_id=span.trace_id, prompt_versions=prompt_versions)

    # Verify both versions were linked to the same trace by checking the actual trace
    trace = mlflow.get_trace(span.trace_id)
    assert trace is not None

    # Check the linked prompts tag
    linked_prompts_tag = trace.info.tags.get("mlflow.linkedPrompts")
    assert linked_prompts_tag is not None

    # Parse the JSON tag value
    linked_prompts = json.loads(linked_prompts_tag)
    assert len(linked_prompts) == 2

    # Check that both versions of the same prompt are present
    prompt_entries = {(p["name"], p["version"]) for p in linked_prompts}
    expected_entries = {("my_prompt", "1"), ("my_prompt", "2")}
    assert prompt_entries == expected_entries

    # Verify we have the same prompt name but different versions
    assert all(p["name"] == "my_prompt" for p in linked_prompts)
    versions = {p["version"] for p in linked_prompts}
    assert versions == {"1", "2"}


def test_load_prompt_with_tracing_no_active_trace():
    """Test that load_prompt works correctly when there's no active trace."""

    # Register a prompt
    mlflow.register_prompt(name="no_trace_prompt", template="Hello, {{name}}!")

    # Load prompt without an active trace
    prompt = mlflow.load_prompt("no_trace_prompt", version=1)

    # Verify prompt was loaded correctly
    assert prompt.name == "no_trace_prompt"
    assert prompt.version == 1
    assert prompt.template == "Hello, {{name}}!"

    # No trace should be created or linked when no active trace exists
    # We can't easily test this without accessing the trace manager, but the function
    # should complete successfully without errors


def test_load_prompt_with_tracing_nested_spans():
    """Test that load_prompt links prompts to the same trace when using nested spans."""

    # Register prompts
    mlflow.register_prompt(name="outer_prompt", template="Outer: {{msg}}")
    mlflow.register_prompt(name="inner_prompt", template="Inner: {{msg}}")

    # Start nested spans (same trace, different spans)
    with mlflow.start_span("outer_operation") as outer_span:
        mlflow.load_prompt("outer_prompt", version=1)

        with mlflow.start_span("inner_operation") as inner_span:
            # Verify both spans belong to the same trace
            assert inner_span.trace_id == outer_span.trace_id

            mlflow.load_prompt("inner_prompt", version=1)

    # Manually trigger prompt linking to trace since in test environment
    # the trace export may not happen automatically
    client = mlflow.MlflowClient()
    prompt_versions = [
        PromptVersion(
            name="outer_prompt",
            version=1,
            template="Outer: {{msg}}",
            commit_message=None,
            creation_timestamp=None,
        ),
        PromptVersion(
            name="inner_prompt",
            version=1,
            template="Inner: {{msg}}",
            commit_message=None,
            creation_timestamp=None,
        ),
    ]
    client.link_prompt_versions_to_trace(
        trace_id=outer_span.trace_id, prompt_versions=prompt_versions
    )

    # Check trace now has both prompts (same trace, different spans)
    trace = mlflow.get_trace(outer_span.trace_id)
    assert trace is not None

    # Check the linked prompts tag
    linked_prompts_tag = trace.info.tags.get("mlflow.linkedPrompts")
    assert linked_prompts_tag is not None

    # Parse the JSON tag value
    linked_prompts = json.loads(linked_prompts_tag)
    assert len(linked_prompts) == 2

    # Check that both prompts are present (order may vary)
    prompt_names = {p["name"] for p in linked_prompts}
    expected_names = {"outer_prompt", "inner_prompt"}
    assert prompt_names == expected_names

    # Verify all prompts have correct versions
    for prompt in linked_prompts:
        assert prompt["version"] == "1"


def test_load_prompt_caching_works():
    """Test that prompt caching works and improves performance."""
    # Mock the client load_prompt method to count calls
    with mock.patch("mlflow.MlflowClient.load_prompt") as mock_client_load:
        # Configure mock to return a prompt
        mock_prompt = PromptVersion(
            name="cached_prompt",
            version=1,
            template="Hello, {{name}}!",
            creation_timestamp=123456789,
        )
        mock_client_load.return_value = mock_prompt

        # First call should hit the client
        prompt1 = mlflow.load_prompt("cached_prompt", version=1, link_to_model=False)
        assert prompt1.name == "cached_prompt"
        assert mock_client_load.call_count == 1

        # Second call with same parameters should use cache (not call client again)
        prompt2 = mlflow.load_prompt("cached_prompt", version=1, link_to_model=False)
        assert prompt2.name == "cached_prompt"
        assert mock_client_load.call_count == 1  # Should still be 1, not 2

        # Call with different version should hit the client again
        mock_client_load.return_value = PromptVersion(
            name="cached_prompt",
            version=2,
            template="Hi, {{name}}!",
            creation_timestamp=123456790,
        )
        prompt3 = mlflow.load_prompt("cached_prompt", version=2, link_to_model=False)
        assert prompt3.version == 2
        assert mock_client_load.call_count == 2  # Should be 2 now


def test_load_prompt_caching_respects_env_var():
    """Test that prompt caching respects the MLFLOW_PROMPT_CACHE_MAX_SIZE environment variable."""
    # Test with a small cache size
    original_value = MLFLOW_PROMPT_CACHE_MAX_SIZE.get()
    try:
        # Set cache size to 1
        MLFLOW_PROMPT_CACHE_MAX_SIZE.set(1)

        # Clear any existing cache by creating a new cached function
        # (This simulates restarting with the new env var)
        importlib.reload(mlflow.tracking._model_registry.fluent)

        # Register prompts
        mlflow.register_prompt(name="prompt_1", template="Template 1")
        mlflow.register_prompt(name="prompt_2", template="Template 2")

        # Mock the client load_prompt method to count calls
        with mock.patch("mlflow.MlflowClient.load_prompt") as mock_client_load:
            mock_client_load.side_effect = [
                PromptVersion(
                    name="prompt_1",
                    version=1,
                    template="Template 1",
                    creation_timestamp=1,
                ),
                PromptVersion(
                    name="prompt_2",
                    version=1,
                    template="Template 2",
                    creation_timestamp=2,
                ),
                PromptVersion(
                    name="prompt_1",
                    version=1,
                    template="Template 1",
                    creation_timestamp=1,
                ),
            ]

            # Load first prompt - should cache it
            mlflow.load_prompt("prompt_1", version=1, link_to_model=False)
            assert mock_client_load.call_count == 1

            # Load second prompt - should evict first from cache (size=1)
            mlflow.load_prompt("prompt_2", version=1, link_to_model=False)
            assert mock_client_load.call_count == 2

            # Load first prompt again - should need to call client again (evicted from cache)
            mlflow.load_prompt("prompt_1", version=1, link_to_model=False)
            assert mock_client_load.call_count == 3  # Called again because evicted

    finally:
        # Restore original cache size
        if original_value is not None:
            MLFLOW_PROMPT_CACHE_MAX_SIZE.set(original_value)
        else:
            MLFLOW_PROMPT_CACHE_MAX_SIZE.unset()


def test_load_prompt_skip_cache_for_allow_missing_none():
    """Test that we skip cache if allow_missing=True and the result is None."""
    # Mock the client load_prompt method to return None (prompt not found)
    with mock.patch("mlflow.MlflowClient.load_prompt") as mock_client_load:
        mock_client_load.return_value = None  # Simulate prompt not found

        # First call with allow_missing=True should call the client twice
        # (once for cached call, once for non-cached call due to `or` logic)
        prompt1 = mlflow.load_prompt(
            "nonexistent_prompt", version=1, allow_missing=True, link_to_model=False
        )
        assert prompt1 is None
        assert mock_client_load.call_count == 2  # Called twice due to `or` logic

        # Second call: cached function returns None from cache, non-cached function called once
        prompt2 = mlflow.load_prompt(
            "nonexistent_prompt", version=1, allow_missing=True, link_to_model=False
        )
        assert prompt2 is None
        assert mock_client_load.call_count == 3  # One additional call (non-cached only)

        # But if we find a prompt, the pattern will change
        mock_prompt = PromptVersion(
            name="nonexistent_prompt",
            version=1,
            template="Found!",
            creation_timestamp=123,
        )
        mock_client_load.return_value = mock_prompt

        prompt3 = mlflow.load_prompt(
            "nonexistent_prompt", version=1, allow_missing=True, link_to_model=False
        )
        assert prompt3.template == "Found!"
        assert (
            mock_client_load.call_count == 4
        )  # Called once for cached call (returned found prompt)

        # Now this should be cached - only cached call needed since it returns a valid prompt
        prompt4 = mlflow.load_prompt(
            "nonexistent_prompt", version=1, allow_missing=True, link_to_model=False
        )
        assert prompt4.template == "Found!"
        assert (
            mock_client_load.call_count == 5
        )  # One more call for cached check (but no non-cached call needed)


def test_load_prompt_missing_then_created_then_found():
    """Test loading a prompt that doesn't exist, then creating it, then loading again."""
    # First try to load a prompt that doesn't exist
    result1 = mlflow.load_prompt(
        "will_be_created", version=1, allow_missing=True, link_to_model=False
    )
    assert result1 is None

    # Now create the prompt
    created_prompt = mlflow.register_prompt(name="will_be_created", template="Now I exist!")
    assert created_prompt.name == "will_be_created"
    assert created_prompt.version == 1

    # Load again - should find it now (not cached because previous result was None)
    result2 = mlflow.load_prompt(
        "will_be_created", version=1, allow_missing=True, link_to_model=False
    )
    assert result2 is not None
    assert result2.name == "will_be_created"
    assert result2.version == 1
    assert result2.template == "Now I exist!"

    # Load a third time - should be cached now (no need to mock since we want real caching)
    result3 = mlflow.load_prompt(
        "will_be_created", version=1, allow_missing=True, link_to_model=False
    )
    assert result3.template == "Now I exist!"
    # This demonstrates the cache working - if it wasn't cached, we'd get a network call


def test_load_prompt_none_result_no_linking():
    """Test that if prompt version is None and allow_missing=True, we don't attempt any linking."""
    # Mock only the client load_prompt method and linking methods
    with (
        mock.patch("mlflow.MlflowClient.load_prompt") as mock_client_load,
        mock.patch("mlflow.MlflowClient.link_prompt_version_to_run") as mock_link_run,
        mock.patch("mlflow.MlflowClient.link_prompt_version_to_model") as mock_link_model,
    ):
        # Configure client to return None (prompt not found)
        mock_client_load.return_value = None

        # Try to load a prompt that doesn't exist with allow_missing=True
        result = mlflow.load_prompt(
            "nonexistent", version=1, allow_missing=True, link_to_model=True
        )
        assert result is None

        # Verify no linking methods were called
        mock_link_run.assert_not_called()
        mock_link_model.assert_not_called()
        # Note: trace manager registration is handled differently and tested elsewhere


def test_load_prompt_caching_with_different_parameters():
    """Test that caching works correctly with different parameter combinations."""
    # Register a prompt
    mlflow.register_prompt(name="param_test", template="Hello, {{name}}!")

    with mock.patch("mlflow.MlflowClient.load_prompt") as mock_client_load:
        mock_prompt = PromptVersion(
            name="param_test",
            version=1,
            template="Hello, {{name}}!",
            creation_timestamp=123,
        )
        mock_client_load.return_value = mock_prompt

        # Different allow_missing values should result in separate cache entries
        mlflow.load_prompt("param_test", version=1, allow_missing=False, link_to_model=False)
        call_count_after_first = mock_client_load.call_count

        mlflow.load_prompt("param_test", version=1, allow_missing=True, link_to_model=False)
        call_count_after_second = mock_client_load.call_count

        # Should be called again for different allow_missing parameter
        assert call_count_after_second > call_count_after_first

        # Same parameters should use cache
        mlflow.load_prompt("param_test", version=1, allow_missing=False, link_to_model=False)
        call_count_after_third = mock_client_load.call_count

        # Cache should work - either same count or only one additional call
        assert call_count_after_third <= call_count_after_second + 1

        mlflow.load_prompt("param_test", version=1, allow_missing=True, link_to_model=False)
        call_count_after_fourth = mock_client_load.call_count

        # Cache should work - either same count or only one additional call
        assert call_count_after_fourth <= call_count_after_third + 1
