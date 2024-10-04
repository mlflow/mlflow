import json
import logging
import os
import shutil
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any, Callable, Dict, List, Literal, NamedTuple, Optional, Union
from urllib.parse import urlparse

import yaml

import mlflow
from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.models.resources import Resource, ResourceType, _ResourceBuilder
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_DOES_NOT_EXIST
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking._tracking_service.utils import _resolve_tracking_uri
from mlflow.tracking.artifact_utils import _download_artifact_from_uri, _upload_artifact_to_uri
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import get_databricks_runtime_version, is_in_databricks_runtime
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
from mlflow.utils.uri import (
    append_to_uri_path,
    get_uri_scheme,
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


class ModelInfo:
    """
    The metadata of a logged MLflow Model.
    """

    def __init__(
        self,
        artifact_path: str,
        flavors: Dict[str, Any],
        model_uri: str,
        model_uuid: str,
        run_id: str,
        saved_input_example_info: Optional[Dict[str, Any]],
        signature,  # Optional[ModelSignature]
        utc_time_created: str,
        mlflow_version: str,
        signature_dict: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        registered_model_version: Optional[int] = None,
    ):
        self._artifact_path = artifact_path
        self._flavors = flavors
        self._model_uri = model_uri
        self._model_uuid = model_uuid
        self._run_id = run_id
        self._saved_input_example_info = saved_input_example_info
        self._signature_dict = signature_dict
        self._signature = signature
        self._utc_time_created = utc_time_created
        self._mlflow_version = mlflow_version
        self._metadata = metadata
        self._registered_model_version = registered_model_version

    @property
    def artifact_path(self):
        """
        Run relative path identifying the logged model.

        :getter: Retrieves the relative path of the logged model.
        :type: str
        """
        return self._artifact_path

    @property
    def flavors(self):
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
    def model_uri(self):
        """
        The ``model_uri`` of the logged model in the format ``'runs:/<run_id>/<artifact_path>'``.

        :getter: Gets the uri path of the logged model from the uri `runs:/<run_id>` path
            encapsulation
        :type: str
        """
        return self._model_uri

    @property
    def model_uuid(self):
        """
        The ``model_uuid`` of the logged model,
        e.g., ``'39ca11813cfc46b09ab83972740b80ca'``.

        :getter: [Legacy] Gets the model_uuid (run_id) of a logged model
        :type: str
        """
        return self._model_uuid

    @property
    def run_id(self):
        """
        The ``run_id`` associated with the logged model,
        e.g., ``'8ede7df408dd42ed9fc39019ef7df309'``

        :getter: Gets the run_id identifier for the logged model
        :type: str
        """
        return self._run_id

    @property
    def saved_input_example_info(self):
        """
        A dictionary that contains the metadata of the saved input example, e.g.,
        ``{"artifact_path": "input_example.json", "type": "dataframe", "pandas_orient": "split"}``.

        :getter: Gets the input example if specified during model logging
        :type: Optional[Dict[str, str]]
        """
        return self._saved_input_example_info

    @property
    def signature_dict(self):
        """
        A dictionary that describes the model input and output generated by
        :py:meth:`ModelSignature.to_dict() <mlflow.models.ModelSignature.to_dict>`.

        :getter: Gets the model signature as a dictionary
        :type: Optional[Dict[str, Any]]
        """
        warnings.warn(
            "Field signature_dict is deprecated since v1.28.1. Use signature instead.",
            category=FutureWarning,
            stacklevel=2,
        )
        return self._signature_dict

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
    def utc_time_created(self):
        """
        The UTC time that the logged model is created, e.g., ``'2022-01-12 05:17:31.634689'``.

        :getter: Gets the UTC formatted timestamp for when the model was logged
        :type: str
        """
        return self._utc_time_created

    @property
    def mlflow_version(self):
        """
        Version of MLflow used to log the model

        :getter: Gets the version of MLflow that was installed when a model was logged
        :type: str
        """
        return self._mlflow_version

    @experimental
    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
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
                    "iris_rf",
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
    def registered_model_version(self) -> Optional[int]:
        """
        The registered model version, if the model is registered.

        :getter: Gets the registered model version, if the model is registered in Model Registry.
        :setter: Sets the registered model version.
        :type: Optional[int]
        """
        return self._registered_model_version

    @registered_model_version.setter
    def registered_model_version(self, value):
        self._registered_model_version = value


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
        saved_input_example_info: Optional[Dict[str, Any]] = None,
        model_uuid: Union[str, Callable, None] = lambda: uuid.uuid4().hex,
        mlflow_version: Union[str, None] = mlflow.version.VERSION,
        metadata: Optional[Dict[str, Any]] = None,
        model_size_bytes: Optional[int] = None,
        resources: Optional[Union[str, List[Resource]]] = None,
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
        self.model_size_bytes = model_size_bytes
        self.resources = resources
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

    def get_serving_input(self, path: str):
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

    def load_input_example(self, path: str):
        """
        Load the input example saved along a model. Returns None if there is no example metadata
        (i.e. the model was saved without example). Raises FileNotFoundError if there is model
        metadata but the example file is missing.

        Args:
            path: Path to the model directory.

        Returns:
            Input example (NumPy ndarray, SciPy csc_matrix, SciPy csr_matrix,
            pandas DataFrame, dict) or None if the model has no example.
        """

        # Just-in-time import to only load example-parsing libraries (e.g. numpy, pandas, etc.) if
        # example is requested.
        from mlflow.models.utils import _read_example

        return _read_example(self, path)

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

    def add_flavor(self, name, **params):
        """Add an entry for how to serve the model in a given format."""
        self.flavors[name] = params
        return self

    @experimental
    @property
    def metadata(self) -> Optional[Dict[str, Any]]:
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
                    "iris_rf",
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

    @experimental
    @metadata.setter
    def metadata(self, value: Optional[Dict[str, Any]]):
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
    def signature(self, value):
        # signature cannot be set to `False`, which is used in `log_model` and `save_model` calls
        # to disable automatic signature inference
        if value is not False:
            self._signature = value

    @property
    def saved_input_example_info(self) -> Optional[Dict[str, Any]]:
        """
        A dictionary that contains the metadata of the saved input example, e.g.,
        ``{"artifact_path": "input_example.json", "type": "dataframe", "pandas_orient": "split"}``.
        """
        return self._saved_input_example_info

    @saved_input_example_info.setter
    def saved_input_example_info(self, value: Dict[str, Any]):
        self._saved_input_example_info = value

    @property
    def model_size_bytes(self) -> Optional[int]:
        """
        An optional integer that represents the model size in bytes

        :getter: Retrieves the model size if it's calculated when the model is saved
        :setter: Sets the model size to a model instance
        :type: Optional[int]
        """
        return self._model_size_bytes

    @model_size_bytes.setter
    def model_size_bytes(self, value: Optional[int]):
        self._model_size_bytes = value

    @experimental
    @property
    def resources(self) -> Dict[str, Dict[ResourceType, List[Dict]]]:
        """
        An optional dictionary that contains the resources required to serve the model.

        :getter: Retrieves the resources required to serve the model
        :setter: Sets the resources required to serve the model
        :type: Dict[str, Dict[ResourceType, List[Dict]]]
        """
        return self._resources

    @experimental
    @resources.setter
    def resources(self, value: Optional[Union[str, List[Resource]]]):
        if isinstance(value, (Path, str)):
            serialized_resource = _ResourceBuilder.from_yaml_file(value)
        elif isinstance(value, List) and all(isinstance(resource, Resource) for resource in value):
            serialized_resource = _ResourceBuilder.from_resources(value)
        else:
            serialized_resource = value
        self._resources = serialized_resource

    def get_model_info(self):
        """
        Create a :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
        model metadata.
        """
        return ModelInfo(
            artifact_path=self.artifact_path,
            flavors=self.flavors,
            model_uri=f"runs:/{self.run_id}/{self.artifact_path}",
            model_uuid=self.model_uuid,
            run_id=self.run_id,
            saved_input_example_info=self.saved_input_example_info,
            signature_dict=self.signature.to_dict() if self.signature else None,
            signature=self.signature,
            utc_time_created=self.utc_time_created,
            mlflow_version=self.mlflow_version,
            metadata=self.metadata,
        )

    def get_tags_dict(self):
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

    def to_dict(self):
        """Serialize the model to a dictionary."""
        res = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        databricks_runtime = get_databricks_runtime_version()
        if databricks_runtime:
            res["databricks_runtime"] = databricks_runtime
        if self.signature is not None:
            res["signature"] = self.signature.to_dict()
        if self.saved_input_example_info is not None:
            res["saved_input_example_info"] = self.saved_input_example_info
        if self.mlflow_version is None and _MLFLOW_VERSION_KEY in res:
            res.pop(_MLFLOW_VERSION_KEY)
        if self.metadata is not None:
            res["metadata"] = self.metadata
        if self.resources is not None:
            res["resources"] = self.resources
        if self.model_size_bytes is not None:
            res["model_size_bytes"] = self.model_size_bytes
        # Exclude null fields in case MLmodel file consumers such as Model Serving may not
        # handle them correctly.
        if self.artifact_path is None:
            res.pop("artifact_path", None)
        if self.run_id is None:
            res.pop("run_id", None)
        return res

    def to_yaml(self, stream=None):
        """Write the model as yaml string."""
        return yaml.safe_dump(self.to_dict(), stream=stream, default_flow_style=False)

    def __str__(self):
        return self.to_yaml()

    def to_json(self):
        """Write the model as json."""
        return json.dumps(self.to_dict())

    def save(self, path):
        """Write the model as a local YAML file."""
        with open(path, "w") as out:
            self.to_yaml(out)

    @classmethod
    def load(cls, path):
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
        path_scheme = urlparse(str(path)).scheme
        if (not path_scheme or path_scheme == "file") and not os.path.exists(path):
            raise MlflowException(
                f'Could not find an "{MLMODEL_FILE_NAME}" configuration file at "{path}"',
                RESOURCE_DOES_NOT_EXIST,
            )

        path = download_artifacts(artifact_uri=path)
        if os.path.isdir(path):
            path = os.path.join(path, MLMODEL_FILE_NAME)
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f.read()))

    @classmethod
    def from_dict(cls, model_dict):
        """Load a model from its YAML representation."""

        from mlflow.models.signature import ModelSignature

        model_dict = model_dict.copy()
        if "signature" in model_dict and isinstance(model_dict["signature"], dict):
            model_dict["signature"] = ModelSignature.from_dict(model_dict["signature"])

        if "model_uuid" not in model_dict:
            model_dict["model_uuid"] = None

        if _MLFLOW_VERSION_KEY not in model_dict:
            model_dict[_MLFLOW_VERSION_KEY] = None

        return cls(**model_dict)

    @format_docstring(LOG_MODEL_PARAM_DOCS)
    @classmethod
    def log(  # noqa: D417
        cls,
        artifact_path,
        flavor,
        registered_model_name=None,
        await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
        metadata=None,
        run_id=None,
        resources=None,
        **kwargs,
    ):
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
            signature: {{ signature }}
            input_example: {{ input_example }}
            await_registration_for: Number of seconds to wait for the model version to finish
                being created and is in ``READY`` status. By default, the
                function waits for five minutes. Specify 0 or None to skip
                waiting.
            metadata: {{ metadata }}
            resources: {{ resources }}
            kwargs: Extra args passed to the model flavor.

        Returns:
            A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
            metadata of the logged model.
        """
        from mlflow.utils.model_utils import _validate_and_get_model_config_from_file

        registered_model = None
        with TempDir() as tmp:
            local_path = tmp.path("model")
            if run_id is None:
                run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
            mlflow_model = cls(
                artifact_path=artifact_path, run_id=run_id, metadata=metadata, resources=resources
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
                    _logger.warning(_LOG_MODEL_MISSING_INPUT_EXAMPLE_WARNING)
                elif tracking_uri == "databricks" or get_uri_scheme(tracking_uri) == "databricks":
                    _logger.warning(_LOG_MODEL_MISSING_SIGNATURE_WARNING)
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

            # validate input example works for serving when logging the model
            if serving_input and kwargs.get("validate_serving_input", True):
                from mlflow.models import validate_serving_input

                try:
                    validate_serving_input(model_info.model_uri, serving_input)
                except Exception as e:
                    _logger.warning(
                        f"Failed to validate serving input example {serving_input}. "
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

        return model_info


def _copy_model_metadata_for_uc_sharing(local_path, flavor):
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
            X, y = [[0, 1]], [1]
            signature = mlflow.models.infer_signature(X, y)
            rfr = RandomForestRegressor(**params).fit(X, y)
            mlflow.log_params(params)
            mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model", signature=signature)

        model_uri = f"runs:/{run.info.run_id}/sklearn-model"
        # Get model info with model_uri
        model_info = mlflow.models.get_model_info(model_uri)
        # Get model signature directly
        model_signature = model_info.signature
        assert model_signature == signature
    """
    from mlflow.pyfunc import _download_artifact_from_uri

    meta_file_uri = model_uri.rstrip("/") + "/" + MLMODEL_FILE_NAME
    meta_local_path = _download_artifact_from_uri(artifact_uri=meta_file_uri)
    model_meta = Model.load(meta_local_path)
    return ModelInfo(
        artifact_path=model_meta.artifact_path,
        flavors=model_meta.flavors,
        model_uri=model_uri,
        model_uuid=model_meta.model_uuid,
        run_id=model_meta.run_id,
        saved_input_example_info=model_meta.saved_input_example_info,
        signature_dict=model_meta.signature.to_dict() if model_meta.signature else None,
        signature=model_meta.signature,
        utc_time_created=model_meta.utc_time_created,
        mlflow_version=model_meta.mlflow_version,
        metadata=model_meta.metadata,
    )


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
    requirement_list: List[str],
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
        model_uri (str): The location, in URI format, of the MLflow model. For example:

            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
            - ``mlflow-artifacts:/path/to/model``

            For more information about supported URI schemes, see
            `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
            artifact-locations>`_.

        operation (Literal["add", "remove]): The operation to perform.
            Must be one of "add" or "remove".

        requirement_list (List[str]): A list of requirements to add or remove from the model.
            For example: ["numpy==1.20.3", "pandas>=1.3.3"]
    """
    if ModelsArtifactRepository.is_models_uri(model_uri):
        raise MlflowException(
            f'Failed to set requirements on "{model_uri}". '
            + "Model URIs with the `models:/` scheme are not supported.",
            INVALID_PARAMETER_VALUE,
        )

    resolved_uri = model_uri
    if RunsArtifactRepository.is_runs_uri(model_uri):
        resolved_uri = RunsArtifactRepository.get_underlying_uri(model_uri)

    _logger.info(f"Retrieving model requirements files from {resolved_uri}...")
    local_paths = get_model_requirements_files(resolved_uri)
    conda_yaml_path = local_paths.conda
    requirements_txt_path = local_paths.requirements

    old_conda_reqs = _get_requirements_from_file(conda_yaml_path)
    old_requirements_reqs = _get_requirements_from_file(requirements_txt_path)

    if operation == "add":
        updated_conda_reqs = _add_or_overwrite_requirements(requirement_list, old_conda_reqs)
        updated_requirements_reqs = _add_or_overwrite_requirements(
            requirement_list, old_requirements_reqs
        )
    else:
        updated_conda_reqs = _remove_requirements(requirement_list, old_conda_reqs)
        updated_requirements_reqs = _remove_requirements(requirement_list, old_requirements_reqs)

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
    from mlflow.langchain import _validate_and_prepare_lc_model_or_path

    return _validate_and_prepare_lc_model_or_path(model, None)


def _validate_llama_index_model(model):
    from mlflow.llama_index import _validate_and_prepare_llama_index_model_or_path

    return _validate_and_prepare_llama_index_model_or_path(model, None)


@experimental
def set_model(model):
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
