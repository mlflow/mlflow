import json
import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any, Callable, Literal, NamedTuple
from urllib.parse import urlparse

import yaml
from packaging.requirements import InvalidRequirement, Requirement

import mlflow
from mlflow.entities import LoggedModel, LoggedModelOutput, Metric
from mlflow.entities.model_registry.prompt_version import PromptVersion
from mlflow.environment_variables import (
    MLFLOW_PRINT_MODEL_URLS_ON_CREATION,
    MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING,
)
from mlflow.exceptions import MlflowException
from mlflow.models.auth_policy import AuthPolicy
from mlflow.models.resources import Resource, ResourceType, _ResourceBuilder
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking._tracking_service.utils import _resolve_tracking_uri
from mlflow.tracking.artifact_utils import _download_artifact_from_uri, _upload_artifact_to_uri
from mlflow.tracking.fluent import (
    _create_logged_model,
    _get_active_model_context,
    _last_logged_model_id,
    _set_active_model_id,
    _use_logged_model,
)
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import (
    _construct_databricks_logged_model_url,
    get_databricks_runtime_version,
    get_workspace_id,
    get_workspace_url,
    is_in_databricks_runtime,
)
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _add_or_overwrite_requirements,
    _get_requirements_from_file,
    _remove_requirements,
    _write_requirements_to_file,
)
from mlflow.utils.file_utils import TempDir
from mlflow.utils.logging_utils import eprint
from mlflow.utils.mlflow_tags import MLFLOW_MODEL_IS_EXTERNAL
from mlflow.utils.uri import (
    append_to_uri_path,
    get_uri_scheme,
    is_databricks_uri,
)

_logger = logging.getLogger(__name__)

# NOTE: The MLMODEL_FILE_NAME constant is considered @developer_stable
MLMODEL_FILE_NAME = "MLmodel"
_DATABRICKS_FS_LOADER_MODULE = "databricks.feature_store.mlflow_model"
_LOG_MODEL_METADATA_WARNING_TEMPLATE = (
    "Logging model metadata to the tracking server has failed. The model artifacts "
    "have been logged successfully under %s. Set logging level to DEBUG via "
    '`logging.getLogger("mlflow").setLevel(logging.DEBUG)` to see the full traceback.'
)
_LOG_MODEL_MISSING_SIGNATURE_WARNING = (
    "Model logged without a signature. Signatures are required for Databricks UC model registry "
    "as they validate model inputs and denote the expected schema of model outputs. "
    f"Please visit https://www.mlflow.org/docs/{mlflow.__version__.replace('.dev0', '')}/"
    "model/signatures.html#how-to-set-signatures-on-models for instructions on setting "
    "signature on models."
)
_LOG_MODEL_MISSING_INPUT_EXAMPLE_WARNING = (
    "Model logged without a signature and input example. Please set `input_example` parameter "
    "when logging the model to auto infer the model signature."
)
# NOTE: The _MLFLOW_VERSION_KEY constant is considered @developer_stable
_MLFLOW_VERSION_KEY = "mlflow_version"
METADATA_FILES = [
    MLMODEL_FILE_NAME,
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
]
MODEL_CONFIG = "config"
MODEL_CODE_PATH = "model_code_path"
SET_MODEL_ERROR = (
    "Model should either be an instance of PyFuncModel, Langchain type, or LlamaIndex index."
)
ENV_VAR_FILE_NAME = "environment_variables.txt"
ENV_VAR_FILE_HEADER = (
    "# This file records environment variable names that are used during model inference.\n"
    "# They might need to be set when creating a serving endpoint from this model.\n"
    "# Note: it is not guaranteed that all environment variables listed here are required\n"
)


class ModelInfo:
    """
    The metadata of a logged MLflow Model.
    """

    def __init__(
        self,
        artifact_path: str,
        flavors: dict[str, Any],
        model_uri: str,
        model_uuid: str,
        run_id: str,
        saved_input_example_info: dict[str, Any] | None,
        signature,  # Optional[ModelSignature]
        utc_time_created: str,
        mlflow_version: str,
        metadata: dict[str, Any] | None = None,
        registered_model_version: int | None = None,
        env_vars: list[str] | None = None,
        prompts: list[str] | None = None,
        logged_model: LoggedModel | None = None,
    ):
        self._artifact_path = artifact_path
        self._flavors = flavors
        self._model_uri = model_uri
        self._model_uuid = model_uuid
        self._run_id = run_id
        self._saved_input_example_info = saved_input_example_info
        self._signature = signature
        self._utc_time_created = utc_time_created
        self._mlflow_version = mlflow_version
        self._metadata = metadata
        self._prompts = prompts
        self._registered_model_version = registered_model_version
        self._env_vars = env_vars
        self._logged_model = logged_model

    @property
    def artifact_path(self) -> str:
        """
        Run relative path identifying the logged model.

        :getter: Retrieves the relative path of the logged model.
        :type: str
        """
        return self._artifact_path

    @property
    def flavors(self) -> dict[str, Any]:
        """
        A dictionary mapping the flavor name to how to serve
        the model as that flavor.

        :getter: Gets the mapping for the logged model's flavor that defines parameters used in
            serving of the model
        :type: Dict[str, str]

        .. code-block:: python
            :caption: Example flavor mapping for a scikit-learn logged model

            {
                "python_function": {
                    "model_path": "model.pkl",
                    "loader_module": "mlflow.sklearn",
                    "python_version": "3.8.10",
                    "env": "conda.yaml",
                },
                "sklearn": {
                    "pickled_model": "model.pkl",
                    "sklearn_version": "0.24.1",
                    "serialization_format": "cloudpickle",
                },
            }

        """
        return self._flavors

    @property
    def model_uri(self) -> str:
        """
        The ``model_uri`` of the logged model in the format ``'runs:/<run_id>/<artifact_path>'``.

        :getter: Gets the uri path of the logged model from the uri `runs:/<run_id>` path
            encapsulation
        :type: str
        """
        return self._model_uri

    @property
    def model_uuid(self) -> str:
        """
        The ``model_uuid`` of the logged model,
        e.g., ``'39ca11813cfc46b09ab83972740b80ca'``.

        :getter: [Legacy] Gets the model_uuid (run_id) of a logged model
        :type: str
        """
        return self._model_uuid

    @property
    def run_id(self) -> str:
        """
        The ``run_id`` associated with the logged model,
        e.g., ``'8ede7df408dd42ed9fc39019ef7df309'``

        :getter: Gets the run_id identifier for the logged model
        :type: str
        """
        return self._run_id

    @property
    def saved_input_example_info(self) -> dict[str, Any] | None:
        """
        A dictionary that contains the metadata of the saved input example, e.g.,
        ``{"artifact_path": "input_example.json", "type": "dataframe", "pandas_orient": "split"}``.

        :getter: Gets the input example if specified during model logging
        :type: Optional[Dict[str, str]]
        """
        return self._saved_input_example_info

    @property
    def signature(self):  # -> Optional[ModelSignature]
        """
        A :py:class:`ModelSignature <mlflow.models.ModelSignature>` that describes the
        model input and output.

        :getter: Gets the model signature if it is defined
        :type: Optional[ModelSignature]
        """
        return self._signature

    @property
    def utc_time_created(self) -> str:
        """
        The UTC time that the logged model is created, e.g., ``'2022-01-12 05:17:31.634689'``.

        :getter: Gets the UTC formatted timestamp for when the model was logged
        :type: str
        """
        return self._utc_time_created

    @property
    def mlflow_version(self) -> str:
        """
        Version of MLflow used to log the model

        :getter: Gets the version of MLflow that was installed when a model was logged
        :type: str
        """
        return self._mlflow_version

    @property
    def env_vars(self) -> list[str] | None:
        """
        Environment variables used during the model logging process.

        :getter: Gets the environment variables used during the model logging process.
        :type: Optional[List[str]]
        """
        return self._env_vars

    @env_vars.setter
    def env_vars(self, value: list[str] | None) -> None:
        if value and not (isinstance(value, list) and all(isinstance(x, str) for x in value)):
            raise TypeError(f"env_vars must be a list of strings. Got: {value}")
        self._env_vars = value

    @property
    def metadata(self) -> dict[str, Any] | None:
        """
        User defined metadata added to the model.

        :getter: Gets the user-defined metadata about a model
        :type: Optional[Dict[str, Any]]

        .. code-block:: python
            :caption: Example usage of Model Metadata

            # Create and log a model with metadata to the Model Registry

            from sklearn import datasets
            from sklearn.ensemble import RandomForestClassifier
            import mlflow
            from mlflow.models import infer_signature

            with mlflow.start_run():
                iris = datasets.load_iris()
                clf = RandomForestClassifier()
                clf.fit(iris.data, iris.target)
                signature = infer_signature(iris.data, iris.target)
                mlflow.sklearn.log_model(
                    clf,
                    name="iris_rf",
                    signature=signature,
                    registered_model_name="model-with-metadata",
                    metadata={"metadata_key": "metadata_value"},
                )

            # model uri for the above model
            model_uri = "models:/model-with-metadata/1"

            # Load the model and access the custom metadata from its ModelInfo object
            model = mlflow.pyfunc.load_model(model_uri=model_uri)
            assert model.metadata.get_model_info().metadata["metadata_key"] == "metadata_value"

            # Load the ModelInfo and access the custom metadata
            model_info = mlflow.models.get_model_info(model_uri=model_uri)
            assert model_info.metadata["metadata_key"] == "metadata_value"
        """
        return self._metadata

    @property
    def prompts(self) -> list[str] | None:
        """A list of prompt URIs associated with the model."""
        return self._prompts

    @property
    def registered_model_version(self) -> int | None:
        """
        The registered model version, if the model is registered.

        :getter: Gets the registered model version, if the model is registered in Model Registry.
        :setter: Sets the registered model version.
        :type: Optional[int]
        """
        return self._registered_model_version

    @registered_model_version.setter
    def registered_model_version(self, value) -> None:
        self._registered_model_version = value

    @property
    def model_id(self) -> str:
        """
        The model ID of the logged model.

        :getter: Gets the model ID of the logged model
        """
        return self._logged_model.model_id if self._logged_model else None

    @property
    def metrics(self) -> list[Metric] | None:
        """
        Returns the metrics of the logged model.

        :getter: Retrieves the metrics of the logged model
        """
        return self._logged_model.metrics if self._logged_model else None

    @property
    def params(self) -> dict[str, str]:
        """
        Returns the parameters of the logged model.

        :getter: Retrieves the parameters of the logged model
        """
        return self._logged_model.params if self._logged_model else None

    @property
    def tags(self) -> dict[str, str]:
        """
        Returns the tags of the logged model.

        :getter: Retrieves the tags of the logged model
        """
        return self._logged_model.tags if self._logged_model else None

    @property
    def creation_timestamp(self) -> int:
        """
        Returns the creation timestamp of the logged model.

        :getter:  the creation timestamp of the logged model
        """
        return self._logged_model.creation_timestamp if self._logged_model else None

    @property
    def name(self) -> str:
        """
        Returns the name of the logged model.
        """
        return self._logged_model.name if self._logged_model else None


class Model:
    """
    An MLflow Model that can support multiple model flavors. Provides APIs for implementing
    new Model flavors.
    """

    def __init__(
        self,
        artifact_path=None,
        run_id=None,
        utc_time_created=None,
        flavors=None,
        signature=None,  # ModelSignature
        saved_input_example_info: dict[str, Any] | None = None,
        model_uuid: str | Callable[[], str] | None = lambda: uuid.uuid4().hex,
        mlflow_version: str | None = mlflow.version.VERSION,
        metadata: dict[str, Any] | None = None,
        model_size_bytes: int | None = None,
        resources: str | list[Resource] | None = None,
        env_vars: list[str] | None = None,
        auth_policy: AuthPolicy | None = None,
        model_id: str | None = None,
        prompts: list[str] | None = None,
        **kwargs,
    ):
        # store model id instead of run_id and path to avoid confusion when model gets exported
        self.run_id = run_id
        self.artifact_path = artifact_path
        self.utc_time_created = str(utc_time_created or datetime.utcnow())
        self.flavors = flavors if flavors is not None else {}
        self.signature = signature
        self.saved_input_example_info = saved_input_example_info
        self.model_uuid = model_uuid() if callable(model_uuid) else model_uuid
        self.mlflow_version = mlflow_version
        self.metadata = metadata
        self.prompts = prompts
        self.model_size_bytes = model_size_bytes
        self.resources = resources
        self.env_vars = env_vars
        self.auth_policy = auth_policy
        self.model_id = model_id
        self.__dict__.update(kwargs)

    def __eq__(self, other):
        if not isinstance(other, Model):
            return False
        return self.__dict__ == other.__dict__

    def get_input_schema(self):
        """
        Retrieves the input schema of the Model iff the model was saved with a schema definition.
        """
        return self.signature.inputs if self.signature is not None else None

    def get_output_schema(self):
        """
        Retrieves the output schema of the Model iff the model was saved with a schema definition.
        """
        return self.signature.outputs if self.signature is not None else None

    def get_params_schema(self):
        """
        Retrieves the parameters schema of the Model iff the model was saved with a schema
        definition.
        """
        return getattr(self.signature, "params", None)

    def get_serving_input(self, path: str) -> str | None:
        """
        Load serving input example from a model directory. Returns None if there is no serving input
        example.

        Args:
            path: Path to the model directory.

        Returns:
            Serving input example or None if the model has no serving input example.
        """
        from mlflow.models.utils import _load_serving_input_example

        return _load_serving_input_example(self, path)

    def load_input_example(self, path: str | None = None) -> str | None:
        """
        Load the input example saved along a model. Returns None if there is no example metadata
        (i.e. the model was saved without example). Raises FileNotFoundError if there is model
        metadata but the example file is missing.

        Args:
            path: Model or run URI, or path to the `model` directory.
                e.g. models://<model_name>/<model_version>, runs:/<run_id>/<artifact_path>
                or /path/to/model

        Returns:
            Input example (NumPy ndarray, SciPy csc_matrix, SciPy csr_matrix,
            pandas DataFrame, dict) or None if the model has no example.
        """

        # Just-in-time import to only load example-parsing libraries (e.g. numpy, pandas, etc.) if
        # example is requested.
        from mlflow.models.utils import _read_example

        if path is None:
            path = (
                f"runs:/{self.run_id}/{self.artifact_path}"
                if self.model_id is None
                else self.artifact_path
            )

        return _read_example(self, str(path))

    def load_input_example_params(self, path: str):
        """
        Load the params of input example saved along a model. Returns None if there are no params in
        the input_example.

        Args:
            path: Path to the model directory.

        Returns:
            params (dict) or None if the model has no params.
        """
        from mlflow.models.utils import _read_example_params

        return _read_example_params(self, path)

    def add_flavor(self, name, **params) -> "Model":
        """Add an entry for how to serve the model in a given format."""
        self.flavors[name] = params
        return self

    @property
    def metadata(self) -> dict[str, Any] | None:
        """
        Custom metadata dictionary passed to the model and stored in the MLmodel file.

        :getter: Retrieves custom metadata that have been applied to a model instance.
        :setter: Sets a dictionary of custom keys and values to be included with the model instance
        :type: Optional[Dict[str, Any]]

        Returns:
            A Dictionary of user-defined metadata iff defined.

        .. code-block:: python
            :caption: Example

            # Create and log a model with metadata to the Model Registry
            from sklearn import datasets
            from sklearn.ensemble import RandomForestClassifier
            import mlflow
            from mlflow.models import infer_signature

            with mlflow.start_run():
                iris = datasets.load_iris()
                clf = RandomForestClassifier()
                clf.fit(iris.data, iris.target)
                signature = infer_signature(iris.data, iris.target)
                mlflow.sklearn.log_model(
                    clf,
                    name="iris_rf",
                    signature=signature,
                    registered_model_name="model-with-metadata",
                    metadata={"metadata_key": "metadata_value"},
                )

            # model uri for the above model
            model_uri = "models:/model-with-metadata/1"

            # Load the model and access the custom metadata
            model = mlflow.pyfunc.load_model(model_uri=model_uri)
            assert model.metadata.metadata["metadata_key"] == "metadata_value"
        """

        return self._metadata

    @metadata.setter
    def metadata(self, value: dict[str, Any] | None) -> None:
        self._metadata = value

    @property
    def signature(self):  # -> Optional[ModelSignature]
        """
        An optional definition of the expected inputs to and outputs from a model object, defined
        with both field names and data types. Signatures support both column-based and tensor-based
        inputs and outputs.

        :getter: Retrieves the signature of a model instance iff the model was saved with a
            signature definition.
        :setter: Sets a signature to a model instance.
        :type: Optional[ModelSignature]
        """
        return self._signature

    @signature.setter
    def signature(self, value) -> None:
        # signature cannot be set to `False`, which is used in `log_model` and `save_model` calls
        # to disable automatic signature inference
        if value is not False:
            self._signature = value

    @property
    def saved_input_example_info(self) -> dict[str, Any] | None:
        """
        A dictionary that contains the metadata of the saved input example, e.g.,
        ``{"artifact_path": "input_example.json", "type": "dataframe", "pandas_orient": "split"}``.
        """
        return self._saved_input_example_info

    @saved_input_example_info.setter
    def saved_input_example_info(self, value: dict[str, Any]) -> None:
        self._saved_input_example_info = value

    @property
    def model_size_bytes(self) -> int | None:
        """
        An optional integer that represents the model size in bytes

        :getter: Retrieves the model size if it's calculated when the model is saved
        :setter: Sets the model size to a model instance
        :type: Optional[int]
        """
        return self._model_size_bytes

    @model_size_bytes.setter
    def model_size_bytes(self, value: int | None) -> None:
        self._model_size_bytes = value

    @property
    def resources(self) -> dict[str, dict[ResourceType, list[dict[str, Any]]]]:
        """
        An optional dictionary that contains the resources required to serve the model.

        :getter: Retrieves the resources required to serve the model
        :setter: Sets the resources required to serve the model
        :type: Dict[str, Dict[ResourceType, List[Dict]]]
        """
        return self._resources

    @resources.setter
    def resources(self, value: str | list[Resource] | None) -> None:
        if isinstance(value, (Path, str)):
            serialized_resource = _ResourceBuilder.from_yaml_file(value)
        elif isinstance(value, list) and all(isinstance(resource, Resource) for resource in value):
            serialized_resource = _ResourceBuilder.from_resources(value)
        else:
            serialized_resource = value
        self._resources = serialized_resource

    @experimental(version="2.21.0")
    @property
    def auth_policy(self) -> dict[str, dict[str, Any]]:
        """
        An optional dictionary that contains the auth policy required to serve the model.

        :getter: Retrieves the auth_policy required to serve the model
        :setter: Sets the auth_policy required to serve the model
        :type: Dict[str, dict]
        """
        return self._auth_policy

    @experimental(version="2.21.0")
    @auth_policy.setter
    def auth_policy(self, value: dict[str, Any] | AuthPolicy | None) -> None:
        self._auth_policy = value.to_dict() if isinstance(value, AuthPolicy) else value

    @property
    def env_vars(self) -> list[str] | None:
        return self._env_vars

    @env_vars.setter
    def env_vars(self, value: list[str] | None) -> None:
        if value and not (isinstance(value, list) and all(isinstance(x, str) for x in value)):
            raise TypeError(f"env_vars must be a list of strings. Got: {value}")
        self._env_vars = value

    def _is_signature_from_type_hint(self):
        return self.signature._is_signature_from_type_hint if self.signature is not None else False

    def _is_type_hint_from_example(self):
        return self.signature._is_type_hint_from_example if self.signature is not None else False

    def get_model_info(self, logged_model: LoggedModel | None = None) -> ModelInfo:
        """
        Create a :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
        model metadata.
        """
        if logged_model is None and self.model_id is not None:
            logged_model = mlflow.get_logged_model(model_id=self.model_id)
        return ModelInfo(
            artifact_path=self.artifact_path,
            flavors=self.flavors,
            model_uri=(
                f"models:/{self.model_id}"
                if self.model_id
                else f"runs:/{self.run_id}/{self.artifact_path}"
            ),
            model_uuid=self.model_uuid,
            run_id=self.run_id,
            saved_input_example_info=self.saved_input_example_info,
            signature=self.signature,
            utc_time_created=self.utc_time_created,
            mlflow_version=self.mlflow_version,
            metadata=self.metadata,
            prompts=self.prompts,
            env_vars=self.env_vars,
            logged_model=logged_model,
        )

    def get_tags_dict(self) -> dict[str, Any]:
        result = self.to_dict()

        tags = {
            key: value
            for key, value in result.items()
            if key in ["run_id", "utc_time_created", "artifact_path", "model_uuid"]
        }

        tags["flavors"] = {
            flavor: (
                {k: v for k, v in config.items() if k != "config"}
                if isinstance(config, dict)
                else config
            )
            for flavor, config in result.get("flavors", {}).items()
        }

        return tags

    def to_dict(self) -> dict[str, Any]:
        """Serialize the model to a dictionary."""
        res = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        databricks_runtime = get_databricks_runtime_version()
        if databricks_runtime:
            res["databricks_runtime"] = databricks_runtime
        if self.signature is not None:
            res["signature"] = self.signature.to_dict()
            res["is_signature_from_type_hint"] = self.signature._is_signature_from_type_hint
            res["type_hint_from_example"] = self.signature._is_type_hint_from_example
        if self.saved_input_example_info is not None:
            res["saved_input_example_info"] = self.saved_input_example_info
        if self.mlflow_version is None and _MLFLOW_VERSION_KEY in res:
            res.pop(_MLFLOW_VERSION_KEY)
        if self.metadata is not None:
            res["metadata"] = self.metadata
        if self.prompts is not None:
            res["prompts"] = self.prompts
        if self.resources is not None:
            res["resources"] = self.resources
        if self.model_size_bytes is not None:
            res["model_size_bytes"] = self.model_size_bytes
        if self.auth_policy is not None:
            res["auth_policy"] = self.auth_policy
        # Exclude null fields in case MLmodel file consumers such as Model Serving may not
        # handle them correctly.
        if self.artifact_path is None:
            res.pop("artifact_path", None)
        if self.run_id is None:
            res.pop("run_id", None)
        if self.env_vars is not None:
            res["env_vars"] = self.env_vars
        return res

    def to_yaml(self, stream=None) -> str:
        """Write the model as yaml string."""
        return yaml.safe_dump(self.to_dict(), stream=stream, default_flow_style=False)

    def __str__(self):
        return self.to_yaml()

    def to_json(self) -> str:
        """Write the model as json."""
        return json.dumps(self.to_dict())

    def save(self, path) -> None:
        """Write the model as a local YAML file."""
        with open(path, "w") as out:
            self.to_yaml(out)

    @classmethod
    def load(cls, path) -> "Model":
        """
        Load a model from its YAML representation.

        Args:
            path: A local filesystem path or URI referring to the MLmodel YAML file
                representation of the Model object or to the directory containing
                the MLmodel YAML file representation.

        Returns:
            An instance of Model.

        .. code-block:: python
            :caption: example

            from mlflow.models import Model

            # Load the Model object from a local MLmodel file
            model1 = Model.load("~/path/to/my/MLmodel")

            # Load the Model object from a remote model directory
            model2 = Model.load("s3://mybucket/path/to/my/model")
        """

        # Check if the path is a local directory and not remote
        sep = os.path.sep
        path = str(path).rstrip(sep)
        path_scheme = urlparse(path).scheme
        if (not path_scheme or path_scheme == "file") and not os.path.exists(path):
            raise MlflowException(
                f'Could not find an "{MLMODEL_FILE_NAME}" configuration file at "{path}"',
                RESOURCE_DOES_NOT_EXIST,
            )

        if ModelsArtifactRepository._is_logged_model_uri(path):
            path = ModelsArtifactRepository.get_underlying_uri(path)

        is_model_dir = path.rsplit(sep, maxsplit=1)[-1] != MLMODEL_FILE_NAME
        mlmodel_file_path = f"{path}/{MLMODEL_FILE_NAME}" if is_model_dir else path
        mlmodel_local_path = _download_artifact_from_uri(artifact_uri=mlmodel_file_path)
        with open(mlmodel_local_path) as f:
            model_dict = yaml.safe_load(f)
        return cls.from_dict(model_dict)

    @classmethod
    def from_dict(cls, model_dict) -> "Model":
        """Load a model from its YAML representation."""

        from mlflow.models.signature import ModelSignature

        model_dict = model_dict.copy()
        if "signature" in model_dict and isinstance(model_dict["signature"], dict):
            signature = ModelSignature.from_dict(model_dict["signature"])
            if "is_signature_from_type_hint" in model_dict:
                signature._is_signature_from_type_hint = model_dict.pop(
                    "is_signature_from_type_hint"
                )
            if "type_hint_from_example" in model_dict:
                signature._is_type_hint_from_example = model_dict.pop("type_hint_from_example")
            model_dict["signature"] = signature

        if "model_uuid" not in model_dict:
            model_dict["model_uuid"] = None

        if _MLFLOW_VERSION_KEY not in model_dict:
            model_dict[_MLFLOW_VERSION_KEY] = None
        return cls(**model_dict)

    # MLflow 2.x log_model API. Only spark flavors uses this API.
    # https://github.com/mlflow/mlflow/blob/fd2d9861fa52eeca178825c871d5d29fbb3b95c4/mlflow/models/model.py#L773-L982
    @format_docstring(LOG_MODEL_PARAM_DOCS)
    @classmethod
    def _log_v2(
        cls,
        artifact_path,
        flavor,
        registered_model_name=None,
        await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
        metadata=None,
        run_id=None,
        resources=None,
        auth_policy=None,
        prompts=None,
        **kwargs,
    ) -> ModelInfo:
        """
        Log model using supplied flavor module. If no run is active, this method will create a new
        active run.

        Args:
            artifact_path: Run relative path identifying the model.
            flavor: Flavor module to save the model with. The module must have
                the ``save_model`` function that will persist the model as a valid
                MLflow model.
            registered_model_name: If given, create a model version under
                ``registered_model_name``, also creating a registered model if
                one with the given name does not exist.
            await_registration_for: Number of seconds to wait for the model version to finish
                being created and is in ``READY`` status. By default, the
                function waits for five minutes. Specify 0 or None to skip
                waiting.
            metadata: {{ metadata }}
            run_id: The run ID to associate with this model. If not provided,
                a new run will be started.
            resources: {{ resources }}
            auth_policy: {{ auth_policy }}
            prompts: {{ prompts }}
            kwargs: Extra args passed to the model flavor.

        Returns:
            A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
            metadata of the logged model.
        """

        # Only one of Auth policy and resources should be defined

        if resources is not None and auth_policy is not None:
            raise ValueError("Only one of `resources`, and `auth_policy` can be specified.")

        from mlflow.utils.model_utils import _validate_and_get_model_config_from_file

        registered_model = None
        with TempDir() as tmp:
            local_path = tmp.path("model")
            if run_id is None:
                run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
            if prompts is not None:
                # Convert to URIs for serialization
                prompts = [pr.uri if isinstance(pr, PromptVersion) else pr for pr in prompts]
            mlflow_model = cls(
                artifact_path=artifact_path,
                run_id=run_id,
                metadata=metadata,
                resources=resources,
                auth_policy=auth_policy,
                prompts=prompts,
            )
            flavor.save_model(path=local_path, mlflow_model=mlflow_model, **kwargs)
            # `save_model` calls `load_model` to infer the model requirements, which may result in
            # __pycache__ directories being created in the model directory.
            for pycache in Path(local_path).rglob("__pycache__"):
                shutil.rmtree(pycache, ignore_errors=True)

            if is_in_databricks_runtime():
                _copy_model_metadata_for_uc_sharing(local_path, flavor)

            tracking_uri = _resolve_tracking_uri()
            serving_input = mlflow_model.get_serving_input(local_path)
            # We check signature presence here as some flavors have a default signature as a
            # fallback when not provided by user, which is set during flavor's save_model() call.
            if mlflow_model.signature is None:
                if serving_input is None:
                    _logger.warning(
                        _LOG_MODEL_MISSING_INPUT_EXAMPLE_WARNING, extra={"color": "red"}
                    )
                elif tracking_uri == "databricks" or get_uri_scheme(tracking_uri) == "databricks":
                    _logger.warning(_LOG_MODEL_MISSING_SIGNATURE_WARNING, extra={"color": "red"})

            env_vars = None
            # validate input example works for serving when logging the model
            if serving_input and kwargs.get("validate_serving_input", True):
                from mlflow.models import validate_serving_input
                from mlflow.utils.model_utils import RECORD_ENV_VAR_ALLOWLIST, env_var_tracker

                with env_var_tracker() as tracked_env_names:
                    try:
                        validate_serving_input(
                            model_uri=local_path,
                            serving_input=serving_input,
                        )
                    except Exception as e:
                        serving_input_msg = (
                            serving_input[:50] + "..." if len(serving_input) > 50 else serving_input
                        )
                        _logger.warning(
                            f"Failed to validate serving input example {serving_input_msg}. "
                            "Alternatively, you can avoid passing input example and pass model "
                            "signature instead when logging the model. To ensure the input example "
                            "is valid prior to serving, please try calling "
                            "`mlflow.models.validate_serving_input` on the model uri and serving "
                            "input example. A serving input example can be generated from model "
                            "input example using "
                            "`mlflow.models.convert_input_example_to_serving_input` function.\n"
                            f"Got error: {e}",
                            exc_info=_logger.isEnabledFor(logging.DEBUG),
                        )
                    env_vars = (
                        sorted(
                            x
                            for x in tracked_env_names
                            if any(env_var in x for env_var in RECORD_ENV_VAR_ALLOWLIST)
                        )
                        or None
                    )
            if env_vars:
                # Keep the environment variable file as it serves as a check
                # for displaying tips in Databricks serving endpoint
                env_var_path = Path(local_path, ENV_VAR_FILE_NAME)
                env_var_path.write_text(ENV_VAR_FILE_HEADER + "\n".join(env_vars) + "\n")
                if len(env_vars) <= 3:
                    env_var_info = "[" + ", ".join(env_vars) + "]"
                else:
                    env_var_info = "[" + ", ".join(env_vars[:3]) + ", ... " + "]"
                    f"(check file {ENV_VAR_FILE_NAME} in the model's artifact folder for full list"
                    " of environment variable names)"
                _logger.info(
                    "Found the following environment variables used during model inference: "
                    f"{env_var_info}. Please check if you need to set them when deploying the "
                    "model. To disable this message, set environment variable "
                    f"`{MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING.name}` to `false`."
                )
                mlflow_model.env_vars = env_vars
                # mlflow_model is updated, rewrite the MLmodel file
                mlflow_model.save(os.path.join(local_path, MLMODEL_FILE_NAME))

            # Associate prompts to the model Run
            if prompts:
                client = mlflow.MlflowClient()
                for prompt in prompts:
                    client.link_prompt_version_to_run(run_id, prompt)

            mlflow.tracking.fluent.log_artifacts(local_path, mlflow_model.artifact_path, run_id)

            # if the model_config kwarg is passed in, then log the model config as an params
            if model_config := kwargs.get("model_config"):
                if isinstance(model_config, str):
                    try:
                        file_extension = os.path.splitext(model_config)[1].lower()
                        if file_extension == ".json":
                            with open(model_config) as f:
                                model_config = json.load(f)
                        elif file_extension in [".yaml", ".yml"]:
                            model_config = _validate_and_get_model_config_from_file(model_config)
                        else:
                            _logger.warning(
                                "Unsupported file format for model config: %s. "
                                "Failed to load model config.",
                                model_config,
                            )
                    except Exception as e:
                        _logger.warning("Failed to load model config from %s: %s", model_config, e)

                try:
                    from mlflow.models.utils import _flatten_nested_params

                    # We are using the `/` separator to flatten the nested params
                    # since we are using the same separator to log nested metrics.
                    params_to_log = _flatten_nested_params(model_config, sep="/")
                except Exception as e:
                    _logger.warning("Failed to flatten nested params: %s", str(e))
                    params_to_log = model_config

                try:
                    mlflow.tracking.fluent.log_params(params_to_log or {}, run_id=run_id)
                except Exception as e:
                    _logger.warning("Failed to log model config as params: %s", str(e))

            try:
                mlflow.tracking.fluent._record_logged_model(mlflow_model, run_id)
            except MlflowException:
                # We need to swallow all mlflow exceptions to maintain backwards compatibility with
                # older tracking servers. Only print out a warning for now.
                _logger.warning(_LOG_MODEL_METADATA_WARNING_TEMPLATE, mlflow.get_artifact_uri())
                _logger.debug("", exc_info=True)

            if registered_model_name is not None:
                registered_model = mlflow.tracking._model_registry.fluent._register_model(
                    f"runs:/{run_id}/{mlflow_model.artifact_path}",
                    registered_model_name,
                    await_registration_for=await_registration_for,
                    local_model_path=local_path,
                )

            model_info = mlflow_model.get_model_info()
            if registered_model is not None:
                model_info.registered_model_version = registered_model.version

        # If the model signature is Mosaic AI Agent compatible, render a recipe for evaluation.
        from mlflow.models.display_utils import maybe_render_agent_eval_recipe

        maybe_render_agent_eval_recipe(model_info)

        return model_info

    @format_docstring(LOG_MODEL_PARAM_DOCS)
    @classmethod
    def log(
        cls,
        artifact_path,
        flavor,
        registered_model_name=None,
        await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
        metadata=None,
        run_id=None,
        resources=None,
        auth_policy=None,
        prompts=None,
        name: str | None = None,
        model_type: str | None = None,
        params: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
        step: int = 0,
        model_id: str | None = None,
        **kwargs,
    ) -> ModelInfo:
        """
        Log model using supplied flavor module. If no run is active, this method will create a new
        active run.

        Args:
            artifact_path: Deprecated. Use `name` instead.
            flavor: Flavor module to save the model with. The module must have
                the ``save_model`` function that will persist the model as a valid
                MLflow model.
            registered_model_name: If given, create a model version under
                ``registered_model_name``, also creating a registered model if
                one with the given name does not exist.
            await_registration_for: Number of seconds to wait for the model version to finish
                being created and is in ``READY`` status. By default, the
                function waits for five minutes. Specify 0 or None to skip
                waiting.
            metadata: {{ metadata }}
            run_id: The run ID to associate with this model.
            resources: {{ resources }}
            auth_policy: {{ auth_policy }}
            prompts: {{ prompts }}
            name: The name of the model.
            model_type: {{ model_type }}
            params: {{ params }}
            tags: {{ tags }}
            step: {{ step }}
            model_id: {{ model_id }}
            kwargs: Extra args passed to the model flavor.

        Returns:
            A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
            metadata of the logged model.
        """
        if name is not None and artifact_path is not None:
            raise MlflowException.invalid_parameter_value(
                "Both `artifact_path` (deprecated) and `name` parameters were specified. "
                "Please only specify `name`."
            )
        elif artifact_path is not None:
            _logger.warning("`artifact_path` is deprecated. Please use `name` instead.")

        name = name or artifact_path

        def log_model_metrics_for_step(client, model_id, run_id, step):
            metric_names = client.get_run(run_id).data.metrics.keys()
            metrics_for_step = []
            for metric_name in metric_names:
                history = client.get_metric_history(run_id, metric_name)
                metrics_for_step.extend(
                    [
                        Metric(
                            key=metric.key,
                            value=metric.value,
                            timestamp=metric.timestamp,
                            step=metric.step,
                            dataset_name=metric.dataset_name,
                            dataset_digest=metric.dataset_digest,
                            run_id=metric.run_id,
                            model_id=model_id,
                        )
                        for metric in history
                        if metric.step == step and metric.model_id is None
                    ]
                )
            client.log_batch(run_id=run_id, metrics=metrics_for_step)

        # Only one of Auth policy and resources should be defined

        if resources is not None and auth_policy is not None:
            raise ValueError("Only one of `resources`, and `auth_policy` can be specified.")

        registered_model = None
        with TempDir() as tmp:
            local_path = tmp.path("model")

            tracking_uri = _resolve_tracking_uri()
            client = mlflow.MlflowClient(tracking_uri)
            if not run_id:
                run_id = active_run.info.run_id if (active_run := mlflow.active_run()) else None

            if model_id is not None:
                model = client.get_logged_model(model_id)
            else:
                params = {
                    **(params or {}),
                    **(client.get_run(run_id).data.params if run_id else {}),
                }
                model = _create_logged_model(
                    # TODO: Update model name
                    name=name,
                    source_run_id=run_id,
                    model_type=model_type,
                    params={key: str(value) for key, value in params.items()},
                    tags={key: str(value) for key, value in tags.items()}
                    if tags is not None
                    else None,
                    flavor=flavor.__name__ if hasattr(flavor, "__name__") else "custom",
                )
                _last_logged_model_id.set(model.model_id)
                if (
                    MLFLOW_PRINT_MODEL_URLS_ON_CREATION.get()
                    and is_databricks_uri(tracking_uri)
                    and (workspace_url := get_workspace_url())
                ):
                    logged_model_url = _construct_databricks_logged_model_url(
                        workspace_url,
                        model.experiment_id,
                        model.model_id,
                        get_workspace_id(),
                    )
                    eprint(f"🔗 View Logged Model at: {logged_model_url}")

            with _use_logged_model(model=model):
                if run_id is not None:
                    client.log_outputs(
                        run_id=run_id, models=[LoggedModelOutput(model.model_id, step=step)]
                    )
                    log_model_metrics_for_step(
                        client=client, model_id=model.model_id, run_id=run_id, step=step
                    )

                if prompts is not None:
                    # Convert to URIs for serialization
                    prompts = [pr.uri if isinstance(pr, PromptVersion) else pr for pr in prompts]

                mlflow_model = cls(
                    artifact_path=model.artifact_location,
                    model_uuid=model.model_id,
                    run_id=run_id,
                    metadata=metadata,
                    resources=resources,
                    auth_policy=auth_policy,
                    prompts=prompts,
                    model_id=model.model_id,
                )
                flavor.save_model(path=local_path, mlflow_model=mlflow_model, **kwargs)
                # `save_model` calls `load_model` to infer the model requirements, which may result
                # in __pycache__ directories being created in the model directory.
                for pycache in Path(local_path).rglob("__pycache__"):
                    shutil.rmtree(pycache, ignore_errors=True)

                if is_in_databricks_runtime():
                    _copy_model_metadata_for_uc_sharing(local_path, flavor)

                serving_input = mlflow_model.get_serving_input(local_path)
                # We check signature presence here as some flavors have a default signature as a
                # fallback when not provided by user, which is set during flavor's save_model()
                # call.
                if mlflow_model.signature is None:
                    if serving_input is None:
                        _logger.warning(
                            _LOG_MODEL_MISSING_INPUT_EXAMPLE_WARNING, extra={"color": "red"}
                        )
                    elif (
                        tracking_uri == "databricks" or get_uri_scheme(tracking_uri) == "databricks"
                    ):
                        _logger.warning(
                            _LOG_MODEL_MISSING_SIGNATURE_WARNING, extra={"color": "red"}
                        )

                env_vars = None
                # validate input example works for serving when logging the model
                if serving_input and kwargs.get("validate_serving_input", True):
                    from mlflow.models import validate_serving_input
                    from mlflow.utils.model_utils import RECORD_ENV_VAR_ALLOWLIST, env_var_tracker

                    with env_var_tracker() as tracked_env_names:
                        try:
                            validate_serving_input(
                                model_uri=local_path,
                                serving_input=serving_input,
                            )
                        except Exception as e:
                            serving_input_msg = (
                                serving_input[:50] + "..."
                                if len(serving_input) > 50
                                else serving_input
                            )
                            _logger.warning(
                                f"Failed to validate serving input example {serving_input_msg}. "
                                "Alternatively, you can avoid passing input example and pass model "
                                "signature instead when logging the model. To ensure the input "
                                "example is valid prior to serving, please try calling "
                                "`mlflow.models.validate_serving_input` on the model uri and "
                                "serving input example. A serving input example can be generated "
                                "from model input example using "
                                "`mlflow.models.convert_input_example_to_serving_input` function.\n"
                                f"Got error: {e}",
                                exc_info=_logger.isEnabledFor(logging.DEBUG),
                            )
                        env_vars = (
                            sorted(
                                x
                                for x in tracked_env_names
                                if any(env_var in x for env_var in RECORD_ENV_VAR_ALLOWLIST)
                            )
                            or None
                        )
                if env_vars:
                    # Keep the environment variable file as it serves as a check
                    # for displaying tips in Databricks serving endpoint
                    env_var_path = Path(local_path, ENV_VAR_FILE_NAME)
                    env_var_path.write_text(ENV_VAR_FILE_HEADER + "\n".join(env_vars) + "\n")
                    if len(env_vars) <= 3:
                        env_var_info = "[" + ", ".join(env_vars) + "]"
                    else:
                        env_var_info = "[" + ", ".join(env_vars[:3]) + ", ... " + "]"
                        f"(check file {ENV_VAR_FILE_NAME} in the model's artifact folder for full "
                        "list of environment variable names)"
                    _logger.info(
                        "Found the following environment variables used during model inference: "
                        f"{env_var_info}. Please check if you need to set them when deploying the "
                        "model. To disable this message, set environment variable "
                        f"`{MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING.name}` to `false`."
                    )
                    mlflow_model.env_vars = env_vars
                    # mlflow_model is updated, rewrite the MLmodel file
                    mlflow_model.save(os.path.join(local_path, MLMODEL_FILE_NAME))

                client.log_model_artifacts(model.model_id, local_path)
                # If the model was previously identified as external, delete the tag because
                # the model now has artifacts in MLflow Model format
                if model.tags.get(MLFLOW_MODEL_IS_EXTERNAL, "false").lower() == "true":
                    client.delete_logged_model_tag(model.model_id, MLFLOW_MODEL_IS_EXTERNAL)
                # client.finalize_logged_model(model.model_id, status=LoggedModelStatus.READY)

                # Associate prompts to the model Run
                if prompts and run_id:
                    client = mlflow.MlflowClient()
                    for prompt in prompts:
                        client.link_prompt_version_to_run(run_id, prompt)

                # if the model_config kwarg is passed in, then log the model config as an params
                if model_config := kwargs.get("model_config"):
                    if isinstance(model_config, str):
                        try:
                            file_extension = os.path.splitext(model_config)[1].lower()
                            if file_extension == ".json":
                                with open(model_config) as f:
                                    model_config = json.load(f)
                            elif file_extension in [".yaml", ".yml"]:
                                from mlflow.utils.model_utils import (
                                    _validate_and_get_model_config_from_file,
                                )

                                model_config = _validate_and_get_model_config_from_file(
                                    model_config
                                )
                            else:
                                _logger.warning(
                                    "Unsupported file format for model config: %s. "
                                    "Failed to load model config.",
                                    model_config,
                                )
                        except Exception as e:
                            _logger.warning(
                                "Failed to load model config from %s: %s", model_config, e
                            )

                    try:
                        from mlflow.models.utils import _flatten_nested_params

                        # We are using the `/` separator to flatten the nested params
                        # since we are using the same separator to log nested metrics.
                        params_to_log = _flatten_nested_params(model_config, sep="/")
                    except Exception as e:
                        _logger.warning("Failed to flatten nested params: %s", str(e))
                        params_to_log = model_config

                    try:
                        # do not log params to run if run_id is None, since that could trigger
                        # a new run to be created
                        if run_id:
                            mlflow.tracking.fluent.log_params(params_to_log or {}, run_id=run_id)
                    except Exception as e:
                        _logger.warning("Failed to log model config as params: %s", str(e))

            if registered_model_name is not None:
                registered_model = mlflow.tracking._model_registry.fluent._register_model(
                    f"models:/{model.model_id}",
                    registered_model_name,
                    await_registration_for=await_registration_for,
                    local_model_path=local_path,
                )
            model_info = mlflow_model.get_model_info(model)
            if registered_model is not None:
                model_info.registered_model_version = registered_model.version

        # If the model signature is Mosaic AI Agent compatible, render a recipe for evaluation.
        from mlflow.models.display_utils import maybe_render_agent_eval_recipe

        maybe_render_agent_eval_recipe(model_info)

        return model_info


def _copy_model_metadata_for_uc_sharing(local_path: str, flavor) -> None:
    """
    Copy model metadata files to a sub-directory 'metadata',
    For Databricks Unity Catalog sharing use-cases.

    Args:
        local_path: Local path to the model directory.
        flavor: Flavor module to save the model with.
    """
    from mlflow.models.wheeled_model import _ORIGINAL_REQ_FILE_NAME, WheeledModel

    metadata_path = os.path.join(local_path, "metadata")
    if isinstance(flavor, WheeledModel):
        # wheeled model updates several metadata files in original model directory
        # copy these updated metadata files to the 'metadata' subdirectory
        os.makedirs(metadata_path, exist_ok=True)
        for file_name in METADATA_FILES + [
            _ORIGINAL_REQ_FILE_NAME,
        ]:
            src_file_path = os.path.join(local_path, file_name)
            if os.path.exists(src_file_path):
                dest_file_path = os.path.join(metadata_path, file_name)
                shutil.copyfile(src_file_path, dest_file_path)
    else:
        os.makedirs(metadata_path, exist_ok=True)
        for file_name in METADATA_FILES:
            src_file_path = os.path.join(local_path, file_name)
            if os.path.exists(src_file_path):
                dest_file_path = os.path.join(metadata_path, file_name)
                shutil.copyfile(src_file_path, dest_file_path)


def get_model_info(model_uri: str) -> ModelInfo:
    """
    Get metadata for the specified model, such as its input/output signature.

    Args:
        model_uri: The location, in URI format, of the MLflow model. For example:

            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
            - ``models:/<model_name>/<model_version>``
            - ``models:/<model_name>/<stage>``
            - ``mlflow-artifacts:/path/to/model``

            For more information about supported URI schemes, see
            `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
            artifact-locations>`_.

    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
        metadata of the logged model.

    .. code-block:: python
        :caption: Example usage of get_model_info

        import mlflow.models
        import mlflow.sklearn
        from sklearn.ensemble import RandomForestRegressor

        with mlflow.start_run() as run:
            params = {"n_estimators": 3, "random_state": 42}
            X = [[0, 1]]
            y = [1]
            signature = mlflow.models.infer_signature(X, y)
            rfr = RandomForestRegressor(**params).fit(X, y)
            mlflow.log_params(params)
            mlflow.sklearn.log_model(rfr, name="sklearn-model", signature=signature)

        model_uri = f"runs:/{run.info.run_id}/sklearn-model"
        # Get model info with model_uri
        model_info = mlflow.models.get_model_info(model_uri)
        # Get model signature directly
        model_signature = model_info.signature
        assert model_signature == signature
    """
    return Model.load(model_uri).get_model_info()


class Files(NamedTuple):
    requirements: Path
    conda: Path


def get_model_requirements_files(resolved_uri: str) -> Files:
    requirements_txt_file = _download_artifact_from_uri(
        artifact_uri=append_to_uri_path(resolved_uri, _REQUIREMENTS_FILE_NAME)
    )
    conda_yaml_file = _download_artifact_from_uri(
        artifact_uri=append_to_uri_path(resolved_uri, _CONDA_ENV_FILE_NAME)
    )

    return Files(
        Path(requirements_txt_file),
        Path(conda_yaml_file),
    )


def update_model_requirements(
    model_uri: str,
    operation: Literal["add", "remove"],
    requirement_list: list[str],
) -> None:
    """
    Add or remove requirements from a model's conda.yaml and requirements.txt files.

    The process involves downloading these two files from the model artifacts
    (if they're non-local), updating them with the specified requirements,
    and then overwriting the existing files. Should the artifact repository
    associated with the model artifacts disallow overwriting, this function will
    fail.

    Note that model registry URIs (i.e. URIs in the form ``models:/``) are not
    supported, as artifacts in the model registry are intended to be read-only.

    If adding requirements, the function will overwrite any existing requirements
    that overlap, or else append the new requirements to the existing list.

    If removing requirements, the function will ignore any version specifiers,
    and remove all the specified package names. Any requirements that are not
    found in the existing files will be ignored.

    Args:
        model_uri: The location, in URI format, of the MLflow model. For example:

            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
            - ``mlflow-artifacts:/path/to/model``

            For more information about supported URI schemes, see
            `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
            artifact-locations>`_.

        operation: The operation to perform. Must be one of "add" or "remove".

        requirement_list: A list of requirements to add or remove from the model.
            For example: ["numpy==1.20.3", "pandas>=1.3.3"]
    """
    resolved_uri = model_uri
    if ModelsArtifactRepository.is_models_uri(model_uri):
        if not ModelsArtifactRepository._is_logged_model_uri(model_uri):
            raise MlflowException(
                f'Failed to set requirements on "{model_uri}". '
                + "Model URIs with the `models:/` scheme are not supported.",
                INVALID_PARAMETER_VALUE,
            )
        resolved_uri = ModelsArtifactRepository.get_underlying_uri(model_uri)
    elif RunsArtifactRepository.is_runs_uri(model_uri):
        resolved_uri = RunsArtifactRepository.get_underlying_uri(model_uri)

    _logger.info(f"Retrieving model requirements files from {resolved_uri}...")
    local_paths = get_model_requirements_files(resolved_uri)
    conda_yaml_path = local_paths.conda
    requirements_txt_path = local_paths.requirements

    old_conda_reqs = _get_requirements_from_file(conda_yaml_path)
    old_requirements_reqs = _get_requirements_from_file(requirements_txt_path)

    requirements = []
    invalid_requirements = {}
    for s in requirement_list:
        try:
            requirements.append(Requirement(s.strip().lower()))
        except InvalidRequirement as e:
            invalid_requirements[s] = e
    if invalid_requirements:
        raise MlflowException.invalid_parameter_value(
            f"Found invalid requirements: {invalid_requirements}"
        )
    if operation == "add":
        updated_conda_reqs = _add_or_overwrite_requirements(requirements, old_conda_reqs)
        updated_requirements_reqs = _add_or_overwrite_requirements(
            requirements, old_requirements_reqs
        )
    else:
        updated_conda_reqs = _remove_requirements(requirements, old_conda_reqs)
        updated_requirements_reqs = _remove_requirements(requirements, old_requirements_reqs)

    _write_requirements_to_file(conda_yaml_path, updated_conda_reqs)
    _write_requirements_to_file(requirements_txt_path, updated_requirements_reqs)

    # just print conda reqs here to avoid log spam
    # it should be the same as requirements.txt anyway
    _logger.info(
        "Done updating requirements!\n\n"
        f"Old requirements:\n{pformat([str(req) for req in old_conda_reqs])}\n\n"
        f"Updated requirements:\n{pformat(updated_conda_reqs)}\n"
    )

    _logger.info(f"Uploading updated requirements files to {resolved_uri}...")
    _upload_artifact_to_uri(conda_yaml_path, resolved_uri)
    _upload_artifact_to_uri(requirements_txt_path, resolved_uri)


__mlflow_model__ = None


def _validate_langchain_model(model):
    from langchain_core.runnables.base import Runnable

    from mlflow.models.utils import _validate_and_get_model_code_path

    if isinstance(model, str):
        return _validate_and_get_model_code_path(model, None)

    if not isinstance(model, Runnable):
        raise MlflowException.invalid_parameter_value(
            "Model must be a Langchain Runnable type or path to a Langchain model, "
            f"got {type(model)}"
        )

    return model


def _validate_llama_index_model(model):
    from mlflow.llama_index.model import _validate_and_prepare_llama_index_model_or_path

    return _validate_and_prepare_llama_index_model_or_path(model, None)


def set_model(model) -> None:
    """
    When logging model as code, this function can be used to set the model object
    to be logged.

    Args:
        model: The model object to be logged. Supported model types are:

                - A Python function or callable object.
                - A Langchain model or path to a Langchain model.
                - A Llama Index model or path to a Llama Index model.
    """
    from mlflow.pyfunc import PythonModel

    if isinstance(model, str):
        raise mlflow.MlflowException(SET_MODEL_ERROR)

    if isinstance(model, PythonModel) or callable(model):
        globals()["__mlflow_model__"] = model
        return

    for validate_function in [_validate_langchain_model, _validate_llama_index_model]:
        try:
            globals()["__mlflow_model__"] = validate_function(model)
            return
        except Exception:
            pass

    raise mlflow.MlflowException(SET_MODEL_ERROR)


def _update_active_model_id_based_on_mlflow_model(mlflow_model: Model):
    """
    Update the current active model ID based on the provided MLflow model.
    Only set the active model ID if it is not already set by the user.
    This is useful for setting the active model ID when loading a model
    to ensure traces generated are associated with the loaded model.
    """
    if mlflow_model.model_id is None:
        return
    amc = _get_active_model_context()
    # only set the active model if the model is not set by the user
    if amc.model_id != mlflow_model.model_id and not amc.set_by_user:
        _set_active_model_id(model_id=mlflow_model.model_id)
