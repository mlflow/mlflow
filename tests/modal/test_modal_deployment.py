import os
from unittest import mock

import pytest

from mlflow.exceptions import MlflowException


@pytest.fixture
def mock_modal():
    with mock.patch.dict("sys.modules", {"modal": mock.MagicMock()}):
        yield


@pytest.fixture
def mock_modal_client(mock_modal):
    with mock.patch("mlflow.modal._import_modal"):
        from mlflow.modal import ModalDeploymentClient

        return ModalDeploymentClient("modal")


def test_import_modal_not_installed():
    from mlflow.modal import _import_modal

    with (
        mock.patch.dict("sys.modules", {"modal": None}),
        mock.patch("builtins.__import__", side_effect=ImportError("No module named 'modal'")),
        pytest.raises(MlflowException, match="modal.*package is required"),
    ):
        _import_modal()


def test_get_preferred_deployment_flavor_pyfunc():
    from mlflow.modal import _get_preferred_deployment_flavor

    mock_model = mock.MagicMock()
    mock_model.flavors = {"python_function": {}, "sklearn": {}}

    flavor = _get_preferred_deployment_flavor(mock_model)
    assert flavor == "python_function"


def test_get_preferred_deployment_flavor_no_pyfunc():
    from mlflow.modal import _get_preferred_deployment_flavor

    mock_model = mock.MagicMock()
    mock_model.flavors = {"sklearn": {}}

    with pytest.raises(MlflowException, match="python_function flavor"):
        _get_preferred_deployment_flavor(mock_model)


def test_validate_deployment_flavor_valid():
    from mlflow.modal import _validate_deployment_flavor

    mock_model = mock.MagicMock()
    mock_model.flavors = {"python_function": {}}

    # Should not raise
    _validate_deployment_flavor(mock_model, "python_function")


def test_validate_deployment_flavor_unsupported():
    from mlflow.modal import _validate_deployment_flavor

    mock_model = mock.MagicMock()
    mock_model.flavors = {"sklearn": {}}

    with pytest.raises(MlflowException, match="not supported for Modal deployment"):
        _validate_deployment_flavor(mock_model, "sklearn")


def test_validate_deployment_flavor_not_in_model():
    from mlflow.modal import _validate_deployment_flavor

    mock_model = mock.MagicMock()
    mock_model.flavors = {"sklearn": {}}

    with pytest.raises(MlflowException, match="does not contain"):
        _validate_deployment_flavor(mock_model, "python_function")


def test_generate_modal_app_code_basic():
    from mlflow.modal import _generate_modal_app_code

    config = {
        "gpu": None,
        "memory": 512,
        "cpu": 1.0,
        "timeout": 300,
        "container_idle_timeout": 60,
        "enable_batching": False,
        "python_version": "3.10",
    }

    code = _generate_modal_app_code("test-app", "/path/to/model", config)

    assert 'app = modal.App("test-app")' in code
    assert "memory=512" in code
    assert "cpu=1.0" in code
    assert "timeout=300" in code
    assert "@modal.web_endpoint" in code
    assert "def predict" in code


def test_generate_modal_app_code_with_gpu():
    from mlflow.modal import _generate_modal_app_code

    config = {
        "gpu": "T4",
        "memory": 2048,
        "cpu": 2.0,
        "timeout": 600,
        "container_idle_timeout": 120,
        "enable_batching": False,
        "python_version": "3.10",
    }

    code = _generate_modal_app_code("gpu-app", "/path/to/model", config)

    assert 'gpu="T4"' in code
    assert "memory=2048" in code


def test_generate_modal_app_code_with_batching():
    from mlflow.modal import _generate_modal_app_code

    config = {
        "gpu": None,
        "memory": 512,
        "cpu": 1.0,
        "timeout": 300,
        "container_idle_timeout": 60,
        "enable_batching": True,
        "max_batch_size": 16,
        "batch_wait_ms": 200,
        "python_version": "3.10",
    }

    code = _generate_modal_app_code("batch-app", "/path/to/model", config)

    assert "@modal.batched" in code
    assert "max_batch_size=16" in code
    assert "wait_ms=200" in code
    assert "predict_batch" in code


def test_client_init_default_uri(mock_modal_client):
    assert mock_modal_client.target_uri == "modal"
    assert mock_modal_client.workspace is None


def test_client_init_with_workspace(mock_modal):
    with mock.patch("mlflow.modal._import_modal"):
        from mlflow.modal import ModalDeploymentClient

        client = ModalDeploymentClient("modal:/my-workspace")
        assert client.workspace == "my-workspace"


def test_client_default_deployment_config(mock_modal_client):
    config = mock_modal_client._default_deployment_config()

    assert config["gpu"] is None
    assert config["memory"] == 512
    assert config["cpu"] == 1.0
    assert config["timeout"] == 300
    assert config["enable_batching"] is False


def test_client_apply_custom_config(mock_modal_client):
    default_config = mock_modal_client._default_deployment_config()
    custom_config = {
        "gpu": "A10G",
        "memory": 4096,
        "enable_batching": True,
        "max_batch_size": 32,
    }

    result = mock_modal_client._apply_custom_config(default_config, custom_config)

    assert result["gpu"] == "A10G"
    assert result["memory"] == 4096
    assert result["enable_batching"] is True
    assert result["max_batch_size"] == 32


def test_client_apply_custom_config_invalid_gpu(mock_modal_client):
    default_config = mock_modal_client._default_deployment_config()
    custom_config = {"gpu": "invalid-gpu"}

    with pytest.raises(MlflowException, match="Unsupported GPU type"):
        mock_modal_client._apply_custom_config(default_config, custom_config)


def test_client_apply_custom_config_type_conversion(mock_modal_client):
    default_config = mock_modal_client._default_deployment_config()
    custom_config = {
        "memory": "2048",
        "cpu": "2.5",
        "enable_batching": "true",
    }

    result = mock_modal_client._apply_custom_config(default_config, custom_config)

    assert result["memory"] == 2048
    assert isinstance(result["memory"], int)
    assert result["cpu"] == 2.5
    assert isinstance(result["cpu"], float)
    assert result["enable_batching"] is True
    assert isinstance(result["enable_batching"], bool)


def test_client_predict_no_deployment_or_endpoint(mock_modal_client):
    with pytest.raises(MlflowException, match="Either deployment_name or endpoint"):
        mock_modal_client.predict()


def test_client_predict_invalid_input_type(mock_modal_client):
    mock_deployment = {"endpoint_url": "http://test"}
    with (
        mock.patch.object(mock_modal_client, "get_deployment", return_value=mock_deployment),
        mock.patch("mlflow.modal.requests.post"),
        pytest.raises(MlflowException, match="must be a dictionary or pandas DataFrame"),
    ):
        mock_modal_client.predict(deployment_name="test", inputs="invalid")


def test_target_help_content():
    from mlflow.modal import target_help

    help_text = target_help()

    assert "Modal" in help_text
    assert "target_uri" in help_text.lower() or "Target URI" in help_text
    assert "gpu" in help_text.lower()
    assert "batching" in help_text.lower()
    assert "Example" in help_text


def test_run_local_calls_modal_serve(mock_modal):
    with (
        mock.patch("mlflow.modal._import_modal"),
        mock.patch("mlflow.modal._download_artifact_from_uri") as mock_download,
        mock.patch("mlflow.modal.TempDir") as mock_tempdir,
        mock.patch("mlflow.modal.subprocess.run") as mock_run,
        mock.patch("builtins.open", mock.mock_open()),
    ):
        mock_tempdir.return_value.__enter__ = mock.MagicMock(
            return_value=mock.MagicMock(path=lambda: "/tmp/test")
        )
        mock_tempdir.return_value.__exit__ = mock.MagicMock(return_value=False)
        mock_download.return_value = "/tmp/test/model"

        from mlflow.modal import run_local

        run_local("modal", "test-app", "runs:/abc123/model")

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "modal" in args
        assert "serve" in args


@pytest.mark.skipif(
    os.environ.get("TEST_MODAL_INTEGRATION") != "1",
    reason="Modal integration tests disabled. Set TEST_MODAL_INTEGRATION=1 to run.",
)
def test_create_deployment_integration():
    pass


@pytest.mark.skipif(
    os.environ.get("TEST_MODAL_INTEGRATION") != "1",
    reason="Modal integration tests disabled. Set TEST_MODAL_INTEGRATION=1 to run.",
)
def test_list_deployments_integration():
    pass
