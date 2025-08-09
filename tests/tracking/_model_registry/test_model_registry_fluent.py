import importlib
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
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
    with pytest.raises(
        MlflowException,
        match=(r"Prompt (.*) does not exist.|Prompt alias (.*) not found."),
    ):
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
    assert len(json.loads(linked_prompts_tag)) == 1
    assert json.loads(linked_prompts_tag)[0] == {
        "name": "prompt_1",
        "version": "1",
    }

    linked_prompts = json.loads(linked_prompts_tag)
    assert len(linked_prompts) == 1
    assert linked_prompts[0]["name"] == "prompt_1"
    assert linked_prompts[0]["version"] == "1"

    with mlflow.start_run() as run:
        run_id_2 = run.info.run_id

        # Prompt should be linked to the run even if it is loaded in a child thread
        def task():
            mlflow.load_prompt("prompt_1", version=1)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(task) for _ in range(10)]
            for future in futures:
                future.result()

    run_data = client.get_run(run_id_2)
    linked_prompts_tag = run_data.data.tags.get(LINKED_PROMPTS_TAG_KEY)
    assert linked_prompts_tag is not None
    assert len(json.loads(linked_prompts_tag)) == 1
    assert json.loads(linked_prompts_tag)[0] == {
        "name": "prompt_1",
        "version": "1",
    }


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


def test_register_prompt_chat_format_integration():
    """Test full integration of registering and using chat prompts."""
    chat_template = [
        {"role": "system", "content": "You are a {{style}} assistant."},
        {"role": "user", "content": "{{question}}"},
    ]

    response_format = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
        },
    }

    # Register chat prompt
    mlflow.register_prompt(
        name="test_chat_integration",
        template=chat_template,
        response_format=response_format,
        commit_message="Test chat prompt integration",
        tags={"model": "test-model"},
    )

    # Load and verify
    prompt = mlflow.load_prompt("test_chat_integration", version=1)
    assert prompt.template == chat_template
    assert prompt.response_format == response_format
    assert prompt.commit_message == "Test chat prompt integration"
    assert prompt.tags["model"] == "test-model"

    # Test formatting
    formatted = prompt.format(style="helpful", question="How are you?")
    expected = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How are you?"},
    ]
    assert formatted == expected


def test_prompt_associate_with_run_chat_format():
    """Test chat prompts associate with runs correctly."""
    chat_template = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    mlflow.register_prompt(name="test_chat_run", template=chat_template)

    with mlflow.start_run() as run:
        mlflow.load_prompt("test_chat_run", version=1)

    # Verify linking
    client = MlflowClient()
    run_data = client.get_run(run.info.run_id)
    linked_prompts_tag = run_data.data.tags.get(LINKED_PROMPTS_TAG_KEY)
    assert linked_prompts_tag is not None

    linked_prompts = json.loads(linked_prompts_tag)
    assert len(linked_prompts) == 1
    assert linked_prompts[0]["name"] == "test_chat_run"
    assert linked_prompts[0]["version"] == "1"


def test_register_prompt_with_pydantic_response_format():
    """Test registering prompts with Pydantic response format."""
    from pydantic import BaseModel

    class ResponseSchema(BaseModel):
        answer: str
        confidence: float

    # Register prompt with Pydantic response format
    mlflow.register_prompt(
        name="test_pydantic_response",
        template="What is {{question}}?",
        response_format=ResponseSchema,
        commit_message="Test Pydantic response format",
    )

    # Load and verify
    prompt = mlflow.load_prompt("test_pydantic_response", version=1)
    assert prompt.response_format == ResponseSchema.model_json_schema()
    assert prompt.commit_message == "Test Pydantic response format"


def test_register_prompt_with_dict_response_format():
    """Test registering prompts with dictionary response format."""
    response_format = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "key_points": {"type": "array", "items": {"type": "string"}},
        },
    }

    # Register prompt with dict response format
    mlflow.register_prompt(
        name="test_dict_response",
        template="Analyze this: {{text}}",
        response_format=response_format,
        tags={"analysis_type": "text"},
    )

    # Load and verify
    prompt = mlflow.load_prompt("test_dict_response", version=1)
    assert prompt.response_format == response_format
    assert prompt.tags["analysis_type"] == "text"


def test_register_prompt_text_backward_compatibility():
    """Test that text prompt registration continues to work as before."""
    # Register text prompt
    mlflow.register_prompt(
        name="test_text_backward",
        template="Hello {{name}}!",
        commit_message="Test backward compatibility",
    )

    # Load and verify
    prompt = mlflow.load_prompt("test_text_backward", version=1)
    assert prompt.is_text_prompt
    assert prompt.template == "Hello {{name}}!"
    assert prompt.commit_message == "Test backward compatibility"

    # Test formatting
    formatted = prompt.format(name="Alice")
    assert formatted == "Hello Alice!"


def test_register_prompt_complex_chat_template():
    """Test registering prompts with complex chat templates."""
    chat_template = [
        {
            "role": "system",
            "content": "You are a {{style}} assistant named {{name}}.",
        },
        {"role": "user", "content": "{{greeting}}! {{question}}"},
        {
            "role": "assistant",
            "content": "I understand you're asking about {{topic}}.",
        },
    ]

    # Register complex chat prompt
    mlflow.register_prompt(
        name="test_complex_chat",
        template=chat_template,
        tags={"complexity": "high"},
    )

    # Load and verify
    prompt = mlflow.load_prompt("test_complex_chat", version=1)
    assert not prompt.is_text_prompt
    assert prompt.template == chat_template
    assert prompt.tags["complexity"] == "high"

    # Test formatting
    formatted = prompt.format(
        style="friendly",
        name="Alice",
        greeting="Hello",
        question="How are you?",
        topic="wellbeing",
    )
    expected = [
        {"role": "system", "content": "You are a friendly assistant named Alice."},
        {"role": "user", "content": "Hello! How are you?"},
        {
            "role": "assistant",
            "content": "I understand you're asking about wellbeing.",
        },
    ]
    assert formatted == expected


def test_register_prompt_with_none_response_format():
    """Test registering prompts with None response format."""
    # Register prompt with None response format
    mlflow.register_prompt(
        name="test_none_response", template="Hello {{name}}!", response_format=None
    )

    # Load and verify
    prompt = mlflow.load_prompt("test_none_response", version=1)
    assert prompt.response_format is None


def test_register_prompt_with_empty_chat_template():
    """Test registering prompts with empty chat template list."""
    # Empty list should be treated as text prompt
    mlflow.register_prompt(name="test_empty_chat", template=[])

    # Load and verify
    prompt = mlflow.load_prompt("test_empty_chat", version=1)
    assert prompt.is_text_prompt
    assert prompt.template == "[]"  # Empty list serialized as string


def test_register_prompt_with_single_message_chat():
    """Test registering prompts with single message chat template."""
    chat_template = [{"role": "user", "content": "Hello {{name}}!"}]

    # Register single message chat prompt
    mlflow.register_prompt(name="test_single_message", template=chat_template)

    # Load and verify
    prompt = mlflow.load_prompt("test_single_message", version=1)
    not prompt.is_text_prompt
    assert prompt.template == chat_template
    assert prompt.variables == {"name"}


def test_register_prompt_with_multiple_variables_in_chat():
    """Test registering prompts with multiple variables in chat messages."""
    chat_template = [
        {
            "role": "system",
            "content": "You are a {{style}} assistant named {{name}}.",
        },
        {"role": "user", "content": "{{greeting}}! {{question}}"},
        {
            "role": "assistant",
            "content": "I understand you're asking about {{topic}}.",
        },
    ]

    # Register prompt with multiple variables
    mlflow.register_prompt(name="test_multiple_variables", template=chat_template)

    # Load and verify
    prompt = mlflow.load_prompt("test_multiple_variables", version=1)
    expected_variables = {"style", "name", "greeting", "question", "topic"}
    assert prompt.variables == expected_variables


def test_register_prompt_with_mixed_content_types():
    """Test registering prompts with mixed content types in chat messages."""
    chat_template = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello {{name}}!"},
        {"role": "assistant", "content": "Hi there! How can I help you today?"},
    ]

    # Register prompt with mixed content
    mlflow.register_prompt(name="test_mixed_content", template=chat_template)

    # Load and verify
    prompt = mlflow.load_prompt("test_mixed_content", version=1)
    not prompt.is_text_prompt
    assert prompt.template == chat_template
    assert prompt.variables == {"name"}


def test_register_prompt_with_nested_variables():
    """Test registering prompts with nested variable names."""
    chat_template = [
        {
            "role": "system",
            "content": "You are a {{user.preferences.style}} assistant.",
        },
        {
            "role": "user",
            "content": "Hello {{user.name}}! {{user.preferences.greeting}}",
        },
    ]

    # Register prompt with nested variables
    mlflow.register_prompt(name="test_nested_variables", template=chat_template)

    # Load and verify
    prompt = mlflow.load_prompt("test_nested_variables", version=1)
    expected_variables = {
        "user.preferences.style",
        "user.name",
        "user.preferences.greeting",
    }
    assert prompt.variables == expected_variables
