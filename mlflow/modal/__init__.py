"""
The ``mlflow.modal`` module provides an API for deploying MLflow models to Modal.

Modal is a serverless platform for running Python code in the cloud.
See https://modal.com for more information.
"""

import json
import logging
import os
import re
import subprocess
import textwrap
import urllib.parse
from typing import Any

import requests

from mlflow.deployments import BaseDeploymentClient, PredictionsResponse
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST
from mlflow.pyfunc import FLAVOR_NAME as PYFUNC_FLAVOR_NAME
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.file_utils import TempDir

_logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_GPU = None  # No GPU by default
DEFAULT_MEMORY = 512  # MB
DEFAULT_CPU = 1.0
DEFAULT_TIMEOUT = 300  # seconds
DEFAULT_CONTAINER_IDLE_TIMEOUT = 60  # seconds
DEFAULT_ALLOW_CONCURRENT_INPUTS = 1
DEFAULT_MIN_CONTAINERS = 0
DEFAULT_MAX_CONTAINERS = None
DEFAULT_SCALEDOWN_WINDOW = None

# Supported GPU types
SUPPORTED_GPUS = ["T4", "L4", "A10G", "A100", "A100-80GB", "H100"]


def _get_model_requirements(model_path: str) -> list[str]:
    """Extract Python requirements from an MLflow model."""
    requirements = []
    req_file = os.path.join(model_path, "requirements.txt")
    if os.path.exists(req_file):
        with open(req_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and not line.lower().startswith("mlflow"):
                    requirements.append(line)
        return requirements
    conda_file = os.path.join(model_path, "conda.yaml")
    if os.path.exists(conda_file):
        try:
            import yaml

            with open(conda_file) as f:
                conda_env = yaml.safe_load(f)
            for dep in conda_env.get("dependencies", []):
                if isinstance(dep, dict) and "pip" in dep:
                    requirements.extend(
                        pip_dep
                        for pip_dep in dep["pip"]
                        if not pip_dep.lower().startswith("mlflow")
                    )
                elif isinstance(dep, str) and not dep.startswith("python"):
                    if not dep.lower().startswith("mlflow"):
                        requirements.append(dep)
        except Exception as e:
            _logger.warning(f"Failed to parse conda.yaml: {e}")
    return requirements


def _get_model_python_version(model_path: str) -> str | None:
    """Extract Python version from an MLflow model."""
    conda_file = os.path.join(model_path, "conda.yaml")
    if os.path.exists(conda_file):
        try:
            import yaml

            with open(conda_file) as f:
                conda_env = yaml.safe_load(f)
            for dep in conda_env.get("dependencies", []):
                if isinstance(dep, str) and dep.startswith("python"):
                    version = dep.split("=")[-1].split(">")[-1].split("<")[-1]
                    parts = version.split(".")
                    if len(parts) >= 2:
                        return f"{parts[0]}.{parts[1]}"
        except Exception as e:
            _logger.warning(f"Failed to parse Python version: {e}")
    return None


def _clear_volume(modal, volume_name: str) -> None:
    """Clear Modal volume to allow redeployment."""
    try:
        volume = modal.Volume.from_name(volume_name)
        for entry in volume.listdir("/"):
            try:
                volume.remove_file(f"/{entry.path}")
            except Exception:
                pass
        _logger.info(f"Cleared volume: {volume_name}")
    except Exception as e:
        _logger.debug(f"Could not clear volume {volume_name}: {e}")


def _get_preferred_deployment_flavor(model_config):
    """
    Obtains the flavor that MLflow would prefer to use when deploying the model.

    Args:
        model_config: An MLflow model object

    Returns:
        The name of the preferred deployment flavor for the specified model
    """
    if PYFUNC_FLAVOR_NAME in model_config.flavors:
        return PYFUNC_FLAVOR_NAME
    else:
        raise MlflowException(
            message=(
                "The specified model does not contain the python_function flavor "
                "which is required for Modal deployment. "
                f"The model contains the following flavors: {list(model_config.flavors.keys())}."
            ),
            error_code=RESOURCE_DOES_NOT_EXIST,
        )


def _validate_deployment_flavor(model_config, flavor):
    """
    Checks that the specified flavor is supported and is contained in the model.

    Args:
        model_config: An MLflow Model object
        flavor: The deployment flavor to validate
    """
    if flavor != PYFUNC_FLAVOR_NAME:
        raise MlflowException(
            message=(
                f"The specified flavor: `{flavor}` is not supported for Modal deployment. "
                f"Only `{PYFUNC_FLAVOR_NAME}` flavor is supported."
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )
    elif flavor not in model_config.flavors:
        raise MlflowException(
            message=(
                f"The specified model does not contain the `{flavor}` flavor. "
                f"The model contains the following flavors: {list(model_config.flavors.keys())}"
            ),
            error_code=RESOURCE_DOES_NOT_EXIST,
        )


def _import_modal():
    """Import modal and raise helpful error if not installed."""
    try:
        import modal

        return modal
    except ImportError as e:
        raise MlflowException(
            "The `modal` package is required for Modal deployments. "
            "Please install it with: pip install modal"
        ) from e


def _generate_modal_app_code(
    app_name: str,
    model_path: str,
    config: dict[str, Any],
    model_requirements: list[str] | None = None,
) -> str:
    """
    Generate the Modal app Python code for serving an MLflow model.

    Args:
        app_name: Name of the Modal app
        model_path: Path to the MLflow model directory
        config: Deployment configuration

    Returns:
        Python code string for the Modal app
    """
    gpu_config = config.get("gpu")
    memory = config.get("memory", DEFAULT_MEMORY)
    cpu = config.get("cpu", DEFAULT_CPU)
    timeout = config.get("timeout", DEFAULT_TIMEOUT)
    container_idle_timeout = config.get("container_idle_timeout", DEFAULT_CONTAINER_IDLE_TIMEOUT)
    enable_batching = config.get("enable_batching", False)
    max_batch_size = config.get("max_batch_size", 8)
    batch_wait_ms = config.get("batch_wait_ms", 100)
    python_version = config.get("python_version", "3.10")

    # Build GPU string for Modal
    gpu_str = f'"{gpu_config}"' if gpu_config else "None"

    # Build pip install string with model requirements
    pip_packages = ["mlflow"]
    if model_requirements:
        pip_packages.extend(model_requirements)
    pip_install_str = ", ".join(f'"{pkg}"' for pkg in pip_packages)

    # Build scaling configuration string
    scaling_parts = []
    min_containers = config.get("min_containers", DEFAULT_MIN_CONTAINERS)
    max_containers = config.get("max_containers")
    scaledown_window = config.get("scaledown_window")
    allow_concurrent_inputs = config.get("allow_concurrent_inputs", DEFAULT_ALLOW_CONCURRENT_INPUTS)

    if min_containers is not None and min_containers > 0:
        scaling_parts.append(f"min_containers={min_containers}")
    if max_containers is not None:
        scaling_parts.append(f"max_containers={max_containers}")
    if scaledown_window is not None:
        scaling_parts.append(f"scaledown_window={scaledown_window}")
    if allow_concurrent_inputs != DEFAULT_ALLOW_CONCURRENT_INPUTS:
        scaling_parts.append(f"allow_concurrent_inputs={allow_concurrent_inputs}")

    scaling_str = "\n            ".join(f"{part}," for part in scaling_parts)

    # Build the Modal app code
    code = textwrap.dedent(f'''
        """
        Modal app for serving MLflow model: {app_name}
        Auto-generated by mlflow.modal
        """
        import modal
        import os

        app = modal.App("{app_name}")

        # Create a volume to store the model
        model_volume = modal.Volume.from_name("{app_name}-model-volume", create_if_missing=True)
        MODEL_DIR = "/model"

        # Define the container image with MLflow dependencies
        image = (
            modal.Image.debian_slim(python_version="{python_version}")
            .pip_install({pip_install_str})
        )

        @app.cls(
            image=image,
            gpu={gpu_str},
            memory={memory},
            cpu={cpu},
            timeout={timeout},
            container_idle_timeout={container_idle_timeout},
            {scaling_str}
            volumes={{MODEL_DIR: model_volume}},
        )
        class MLflowModel:
            @modal.enter()
            def load_model(self):
                import mlflow.pyfunc
                model_volume.reload()
                self.model = mlflow.pyfunc.load_model(MODEL_DIR)

    ''')

    if enable_batching:
        code += textwrap.dedent(f"""
            @modal.batched(max_batch_size={max_batch_size}, wait_ms={batch_wait_ms})
            def predict_batch(self, inputs: list[dict]) -> list[dict]:
                import pandas as pd
                results = []
                for input_data in inputs:
                    df = pd.DataFrame(input_data)
                    prediction = self.model.predict(df)
                    results.append({{"predictions": prediction.tolist()}})
                return results

            @modal.web_endpoint(method="POST")
            def predict(self, input_data: dict) -> dict:
                return self.predict_batch.local([input_data])[0]
        """)
    else:
        code += textwrap.dedent("""
            @modal.web_endpoint(method="POST")
            def predict(self, input_data: dict) -> dict:
                import pandas as pd
                df = pd.DataFrame(input_data)
                prediction = self.model.predict(df)
                return {"predictions": prediction.tolist()}
        """)

    return code


class ModalDeploymentClient(BaseDeploymentClient):
    """
    Client for deploying MLflow models to Modal.

    Modal is a serverless platform for running Python code in the cloud.
    This client enables deploying MLflow models as Modal web endpoints.

    Args:
        target_uri: A URI that follows one of the following formats:

            - ``modal``: Uses default Modal workspace from environment
            - ``modal:/workspace-name``: Uses the specified workspace

    Example:
        .. code-block:: python

            from mlflow.deployments import get_deploy_client

            client = get_deploy_client("modal")
            client.create_deployment(
                name="my-model",
                model_uri="runs:/abc123/model",
                config={"gpu": "T4", "enable_batching": True},
            )
    """

    def __init__(self, target_uri: str):
        super().__init__(target_uri=target_uri)
        self.workspace = self._parse_workspace_from_uri(target_uri)
        self._validate_modal_auth()

    def _parse_workspace_from_uri(self, target_uri: str) -> str | None:
        """Parse workspace name from target URI.

        Expected formats:
        - "modal" -> workspace = None
        - "modal:/" -> workspace = None
        - "modal:/workspace-name" -> workspace = "workspace-name"
        """
        parsed = urllib.parse.urlparse(target_uri)
        # For "modal" without colon, the whole string becomes the path
        # For "modal:/workspace", scheme is "modal" and path is "/workspace"
        if parsed.scheme == "modal":
            path = parsed.path.strip("/")
            return path or None
        else:
            # No scheme (e.g., just "modal"), workspace is None
            return None

    def _validate_modal_auth(self):
        """Validate that Modal authentication is configured."""
        _import_modal()
        # Modal will raise an error if not authenticated when we try to use it
        # We don't validate here to allow offline configuration

    def _default_deployment_config(self) -> dict[str, Any]:
        """Return default deployment configuration."""
        return {
            "gpu": DEFAULT_GPU,
            "memory": DEFAULT_MEMORY,
            "cpu": DEFAULT_CPU,
            "timeout": DEFAULT_TIMEOUT,
            "container_idle_timeout": DEFAULT_CONTAINER_IDLE_TIMEOUT,
            "enable_batching": False,
            "max_batch_size": 8,
            "batch_wait_ms": 100,
            "allow_concurrent_inputs": DEFAULT_ALLOW_CONCURRENT_INPUTS,
            "min_containers": DEFAULT_MIN_CONTAINERS,
            "max_containers": DEFAULT_MAX_CONTAINERS,
            "scaledown_window": DEFAULT_SCALEDOWN_WINDOW,
            "python_version": None,  # Auto-detect from model
        }

    def _apply_custom_config(
        self, config: dict[str, Any], custom_config: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Apply custom configuration over defaults."""
        if not custom_config:
            return config

        int_fields = {
            "memory",
            "timeout",
            "container_idle_timeout",
            "max_batch_size",
            "batch_wait_ms",
            "min_containers",
            "max_containers",
            "scaledown_window",
            "allow_concurrent_inputs",
        }
        float_fields = {"cpu"}
        bool_fields = {"enable_batching"}

        for key, value in custom_config.items():
            if key not in config:
                # Allow passthrough of additional config options
                config[key] = value
                continue

            if value is None:
                config[key] = value
            elif key in int_fields and not isinstance(value, int):
                config[key] = int(value)
            elif key in float_fields and not isinstance(value, float):
                config[key] = float(value)
            elif key in bool_fields and not isinstance(value, bool):
                config[key] = str(value).lower() == "true"
            else:
                config[key] = value

        # Validate GPU if specified
        if config.get("gpu") and config["gpu"] not in SUPPORTED_GPUS:
            raise MlflowException(
                f"Unsupported GPU type: {config['gpu']}. Supported types: {SUPPORTED_GPUS}",
                error_code=INVALID_PARAMETER_VALUE,
            )

        return config

    def create_deployment(
        self,
        name: str,
        model_uri: str,
        flavor: str | None = None,
        config: dict[str, Any] | None = None,
        endpoint: str | None = None,
    ) -> dict[str, Any]:
        """
        Deploy an MLflow model to Modal.

        Args:
            name: Name of the deployment (will be used as Modal app name)
            model_uri: URI of the MLflow model to deploy. Examples:
                - ``runs:/<run_id>/model``
                - ``models:/<model_name>/<version>``
                - ``/path/to/local/model``
            flavor: Model flavor to deploy (only ``python_function`` is supported)
            config: Deployment configuration. Supported options:
                - ``gpu``: GPU type (T4, L4, A10G, A100, A100-80GB, H100)
                - ``memory``: Memory in MB (default: 512)
                - ``cpu``: CPU cores (default: 1.0)
                - ``timeout``: Request timeout in seconds (default: 300)
                - ``container_idle_timeout``: Idle timeout in seconds (default: 60)
                - ``enable_batching``: Enable dynamic batching (default: False)
                - ``max_batch_size``: Max batch size if batching enabled (default: 8)
                - ``batch_wait_ms``: Batch wait time in ms (default: 100)
                - ``python_version``: Python version (default: "3.10")
            endpoint: Not used for Modal deployments

        Returns:
            Dictionary containing deployment information including the endpoint URL
        """
        modal = _import_modal()

        # Download and validate model
        with TempDir() as tmp_dir:
            local_model_path = _download_artifact_from_uri(model_uri, output_path=tmp_dir.path())
            model_config = Model.load(local_model_path)

            # Validate flavor
            if flavor:
                _validate_deployment_flavor(model_config, flavor)
            else:
                flavor = _get_preferred_deployment_flavor(model_config)

            # Build configuration
            deployment_config = self._default_deployment_config()
            deployment_config = self._apply_custom_config(deployment_config, config)

            # Auto-detect Python version from model if not specified
            if deployment_config.get("python_version") is None:
                detected_version = _get_model_python_version(local_model_path)
                deployment_config["python_version"] = detected_version or "3.10"

            # Get model requirements
            model_requirements = _get_model_requirements(local_model_path)
            if model_requirements:
                _logger.info(f"Detected model requirements: {model_requirements}")

            # Generate Modal app code
            app_code = _generate_modal_app_code(
                name, local_model_path, deployment_config, model_requirements
            )

            # Write the app code to a temporary file
            app_file = os.path.join(tmp_dir.path(), "modal_app.py")
            with open(app_file, "w") as f:
                f.write(app_code)

            # Clear existing volume to allow redeployment (fixes FileExistsError)
            volume_name = f"{name}-model-volume"
            _clear_volume(modal, volume_name)

            # Upload model to Modal volume
            _logger.info(f"Uploading model to Modal volume: {volume_name}")
            volume = modal.Volume.from_name(volume_name, create_if_missing=True)

            # Upload model files to volume with force=True to overwrite
            with volume.batch_upload(force=True) as batch:
                for root, dirs, files in os.walk(local_model_path):
                    for file in files:
                        local_file = os.path.join(root, file)
                        relative_path = os.path.relpath(local_file, local_model_path)
                        batch.put_file(local_file, f"/{relative_path}")

            # Deploy the Modal app with workspace if specified
            _logger.info(f"Deploying Modal app: {name}")
            deploy_cmd = ["modal", "deploy", app_file]
            if self.workspace:
                deploy_cmd.extend(["--env", self.workspace])

            # Set environment for workspace if specified
            env = os.environ.copy()
            if self.workspace:
                env["MODAL_ENVIRONMENT"] = self.workspace

            result = subprocess.run(
                deploy_cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=tmp_dir.path(),
                env=env,
            )

            if result.returncode != 0:
                raise MlflowException(
                    f"Failed to deploy Modal app: {result.stderr}",
                    error_code=INVALID_PARAMETER_VALUE,
                )

            # Parse the endpoint URL from output
            endpoint_url = None
            for line in result.stdout.split("\n"):
                if "https://" in line and ".modal.run" in line:
                    # Extract URL from the line
                    if urls := re.findall(r"https://[^\s]+\.modal\.run[^\s]*", line):
                        endpoint_url = urls[0]
                        break

            _logger.info(f"Successfully deployed model to Modal: {endpoint_url}")

            return {
                "name": name,
                "flavor": flavor,
                "endpoint_url": endpoint_url,
                "config": deployment_config,
            }

    def update_deployment(
        self,
        name: str,
        model_uri: str | None = None,
        flavor: str | None = None,
        config: dict[str, Any] | None = None,
        endpoint: str | None = None,
    ) -> dict[str, Any]:
        """
        Update an existing Modal deployment with a new model or configuration.

        This is equivalent to deleting and recreating the deployment.

        Args:
            name: Name of the deployment to update
            model_uri: URI of the new MLflow model
            flavor: Model flavor to deploy
            config: Updated deployment configuration
            endpoint: Not used for Modal deployments

        Returns:
            Dictionary containing updated deployment information
        """
        if not model_uri:
            raise MlflowException(
                "model_uri is required for updating Modal deployments",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # Delete existing deployment
        self.delete_deployment(name)

        # Create new deployment
        return self.create_deployment(name, model_uri, flavor, config, endpoint)

    def delete_deployment(
        self,
        name: str,
        config: dict[str, Any] | None = None,
        endpoint: str | None = None,
    ) -> None:
        """
        Delete a Modal deployment.

        Args:
            name: Name of the deployment to delete
            config: Not used
            endpoint: Not used
        """
        modal = _import_modal()

        _logger.info(f"Stopping Modal app: {name}")

        # Build command with workspace if specified
        cmd = ["modal", "app", "stop", name]
        if self.workspace:
            cmd.extend(["--env", self.workspace])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        # Don't fail if app doesn't exist (idempotent delete)
        if result.returncode != 0 and "not found" not in result.stderr.lower():
            _logger.warning(f"Failed to stop Modal app {name}: {result.stderr}")

        # Clear the volume
        _clear_volume(modal, f"{name}-model-volume")

    def list_deployments(self, endpoint: str | None = None) -> list[dict[str, Any]]:
        """
        List all Modal deployments.

        Args:
            endpoint: Not used

        Returns:
            List of deployment dictionaries
        """
        cmd = ["modal", "app", "list", "--json"]
        if self.workspace:
            cmd.extend(["--env", self.workspace])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            raise MlflowException(f"Failed to list Modal apps: {result.stderr}")

        try:
            apps = json.loads(result.stdout)
            return [
                {
                    "name": app.get("Description", ""),
                    "app_id": app.get("App ID", ""),
                    "state": app.get("State", ""),
                    "created_at": app.get("Created at", ""),
                }
                for app in apps
            ]
        except json.JSONDecodeError:
            _logger.warning("Failed to parse Modal app list output as JSON")
            return []

    def get_deployment(self, name: str, endpoint: str | None = None) -> dict[str, Any]:
        """
        Get information about a specific Modal deployment.

        Args:
            name: Name of the deployment
            endpoint: Not used

        Returns:
            Dictionary containing deployment information
        """
        cmd = ["modal", "app", "list", "--json"]
        if self.workspace:
            cmd.extend(["--env", self.workspace])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            raise MlflowException(f"Failed to get Modal app info: {result.stderr}")

        try:
            apps = json.loads(result.stdout)
            for app in apps:
                if app.get("Description") == name:
                    workspace = self.workspace or os.environ.get("MODAL_WORKSPACE", "")
                    return {
                        "name": name,
                        "app_id": app.get("App ID", ""),
                        "state": app.get("State", ""),
                        "created_at": app.get("Created at", ""),
                        "endpoint_url": f"https://{workspace}--{name}-mlflowmodel-predict.modal.run"
                        if workspace
                        else None,
                    }
        except json.JSONDecodeError:
            pass

        raise MlflowException(
            f"Deployment '{name}' not found",
            error_code=RESOURCE_DOES_NOT_EXIST,
        )

    def predict(
        self,
        deployment_name: str | None = None,
        inputs: Any | None = None,
        endpoint: str | None = None,
    ) -> PredictionsResponse:
        """
        Make predictions using a deployed Modal model.

        Args:
            deployment_name: Name of the deployment
            inputs: Input data for prediction (dict or DataFrame-like)
            endpoint: Optional custom endpoint URL

        Returns:
            PredictionsResponse containing the model predictions
        """
        if not deployment_name and not endpoint:
            raise MlflowException(
                "Either deployment_name or endpoint must be provided",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # Get endpoint URL
        if endpoint:
            url = endpoint
        else:
            deployment = self.get_deployment(deployment_name)
            url = deployment.get("endpoint_url")
            if not url:
                raise MlflowException(
                    f"Could not determine endpoint URL for deployment '{deployment_name}'",
                    error_code=INVALID_PARAMETER_VALUE,
                )

        # Convert inputs to JSON-serializable format
        if hasattr(inputs, "to_dict"):
            # DataFrame
            input_data = inputs.to_dict(orient="list")
        elif isinstance(inputs, dict):
            input_data = inputs
        else:
            raise MlflowException(
                "inputs must be a dictionary or pandas DataFrame",
                error_code=INVALID_PARAMETER_VALUE,
            )

        # Make prediction request
        response = requests.post(url, json=input_data, timeout=300)

        if response.status_code != 200:
            raise MlflowException(
                f"Prediction request failed with status {response.status_code}: {response.text}"
            )

        return PredictionsResponse(predictions=response.json().get("predictions"))


def run_local(target, name, model_uri, flavor=None, config=None):
    """
    Deploy the specified model locally using Modal's local serving capability.

    This uses ``modal serve`` to run the model locally for testing.

    Args:
        target: Deployment target (should be "modal")
        name: Name for the local deployment
        model_uri: URI of the MLflow model to deploy
        flavor: Model flavor (only python_function supported)
        config: Deployment configuration

    Returns:
        None (runs until interrupted)
    """
    _import_modal()

    with TempDir() as tmp_dir:
        # Download model
        local_model_path = _download_artifact_from_uri(model_uri, output_path=tmp_dir.path())

        # Generate Modal app code
        deployment_config = ModalDeploymentClient("modal")._default_deployment_config()
        if config:
            deployment_config.update(config)

        # Auto-detect Python version from model if not specified
        if deployment_config.get("python_version") is None:
            detected_version = _get_model_python_version(local_model_path)
            deployment_config["python_version"] = detected_version or "3.10"

        # Get model requirements
        model_requirements = _get_model_requirements(local_model_path)
        if model_requirements:
            _logger.info(f"Detected model requirements: {model_requirements}")

        app_code = _generate_modal_app_code(
            name, local_model_path, deployment_config, model_requirements
        )

        # Write app code
        app_file = os.path.join(tmp_dir.path(), "modal_app.py")
        with open(app_file, "w") as f:
            f.write(app_code)

        # Run modal serve
        _logger.info(f"Starting local Modal server for {name}...")
        subprocess.run(["modal", "serve", app_file], cwd=tmp_dir.path())


def target_help():
    """
    Return help text for the Modal deployment target.
    """
    return """
    MLflow Modal Deployment Target
    ==============================

    The Modal deployment target enables deploying MLflow models to Modal's
    serverless platform (https://modal.com).

    Target URI Format
    -----------------
    - ``modal``: Use default workspace from Modal authentication
    - ``modal:/workspace-name``: Use a specific Modal environment/workspace

    Authentication
    --------------
    Modal authentication is handled via the Modal CLI or environment variables:
    - Run ``modal setup`` to configure authentication interactively
    - Or set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables

    Configuration Options
    ---------------------
    Pass these options via the ``config`` parameter in create_deployment():

    Resource Configuration:
    - ``gpu``: GPU type (T4, L4, A10G, A100, A100-80GB, H100)
    - ``memory``: Memory allocation in MB (default: 512)
    - ``cpu``: CPU cores (default: 1.0)
    - ``timeout``: Request timeout in seconds (default: 300)

    Scaling Configuration:
    - ``min_containers``: Minimum number of containers to keep warm (default: 0)
    - ``max_containers``: Maximum number of containers to scale to (default: None)
    - ``container_idle_timeout``: Container idle timeout in seconds (default: 60)
    - ``scaledown_window``: Time window for scale-down decisions in seconds
    - ``allow_concurrent_inputs``: Number of concurrent inputs per container (default: 1)

    Batching Configuration:
    - ``enable_batching``: Enable dynamic batching (default: False)
    - ``max_batch_size``: Maximum batch size when batching enabled (default: 8)
    - ``batch_wait_ms``: Batch wait time in milliseconds (default: 100)

    Environment Configuration:
    - ``python_version``: Python version for container (auto-detected from model, or "3.10")

    Note: Model dependencies are automatically detected from the MLflow model's
    requirements.txt or conda.yaml file and installed in the container.

    Example
    -------
    .. code-block:: python

        from mlflow.deployments import get_deploy_client

        client = get_deploy_client("modal")

        # Deploy a model with GPU and scaling
        deployment = client.create_deployment(
            name="my-classifier",
            model_uri="runs:/abc123/model",
            config={
                "gpu": "T4",
                "memory": 2048,
                "min_containers": 1,
                "max_containers": 10,
                "enable_batching": True,
                "max_batch_size": 16,
            }
        )

        # Make predictions
        predictions = client.predict(
            deployment_name="my-classifier",
            inputs={"feature1": [1, 2, 3], "feature2": [4, 5, 6]}
        )

    CLI Usage
    ---------
    .. code-block:: bash

        # Deploy a model
        mlflow deployments create -t modal -m runs:/abc123/model --name my-model

        # Deploy to a specific workspace/environment
        mlflow deployments create -t modal:/my-workspace -m runs:/abc123/model --name my-model

        # List deployments
        mlflow deployments list -t modal

        # Get deployment info
        mlflow deployments get -t modal --name my-model

        # Delete deployment
        mlflow deployments delete -t modal --name my-model
    """
