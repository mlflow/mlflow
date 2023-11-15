import json
import logging
import os
import uuid
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Union

import yaml

import mlflow
from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking._tracking_service.utils import _resolve_tracking_uri
from mlflow.utils.annotations import experimental
from mlflow.utils.databricks_utils import get_databricks_runtime
from mlflow.utils.file_utils import TempDir
from mlflow.utils.uri import get_uri_scheme

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
    "Model logged without a signature. Signatures will be required for upcoming model registry "
    "features as they validate model inputs and denote the expected schema of model outputs. "
    f"Please visit https://www.mlflow.org/docs/{mlflow.__version__.replace('.dev0', '')}/"
    "models.html#set-signature-on-logged-model for instructions on setting a model signature on "
    "your logged model."
)
# NOTE: The _MLFLOW_VERSION_KEY constant is considered @developer_stable
_MLFLOW_VERSION_KEY = "mlflow_version"


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
        **kwargs,
    ):
        # store model id instead of run_id and path to avoid confusion when model gets exported
        if run_id:
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

    def load_input_example(self, path: str):
        """
        Load the input example saved along a model. Returns None if there is no example metadata
        (i.e. the model was saved without example). Raises FileNotFoundError if there is model
        metadata but the example file is missing.

        :param path: Path to the model directory.

        :return: Input example (NumPy ndarray, SciPy csc_matrix, SciPy csr_matrix,
                 pandas DataFrame, dict) or None if the model has no example.
        """

        # Just-in-time import to only load example-parsing libraries (e.g. numpy, pandas, etc.) if
        # example is requested.
        from mlflow.models.utils import _read_example

        return _read_example(self, path)

    def load_input_example_params(self, path: str):
        """
        Load the params of input example saved along a model. Returns None if there is no params in
        the input_example.

        :param path: Path to the model directory.

        :return: params (dict) or None if the model has no params.
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

        :return: A Dictionary of user-defined metadata iff defined.

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
        # pylint: disable=attribute-defined-outside-init
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
            # pylint: disable=attribute-defined-outside-init
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
        # pylint: disable=attribute-defined-outside-init
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
        # pylint: disable=attribute-defined-outside-init
        self._model_size_bytes = value

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

    def to_dict(self):
        """Serialize the model to a dictionary."""
        res = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        databricks_runtime = get_databricks_runtime()
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
        if self.model_size_bytes is not None:
            res["model_size_bytes"] = self.model_size_bytes
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

        :param path: A local filesystem path or URI referring to the MLmodel YAML file
                     representation of the :py:class:`Model` object or to the directory containing
                     the MLmodel YAML file representation.
        :return: An instance of :py:class:`Model`.


        .. code-block:: python
            :caption: example

            from mlflow.models import Model

            # Load the Model object from a local MLmodel file
            model1 = Model.load("~/path/to/my/MLmodel")

            # Load the Model object from a remote model directory
            model2 = Model.load("s3://mybucket/path/to/my/model")
        """
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

    @classmethod
    def log(
        cls,
        artifact_path,
        flavor,
        registered_model_name=None,
        await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
        metadata=None,
        **kwargs,
    ):
        """
        Log model using supplied flavor module. If no run is active, this method will create a new
        active run.

        :param artifact_path: Run relative path identifying the model.
        :param flavor: Flavor module to save the model with. The module must have
                       the ``save_model`` function that will persist the model as a valid
                       MLflow model.
        :param registered_model_name: If given, create a model version under
                                      ``registered_model_name``, also creating a registered model if
                                      one with the given name does not exist.
        :param signature: :py:class:`ModelSignature` describes model input
                          and output :py:class:`Schema <mlflow.types.Schema>`. The model signature
                          can be :py:func:`inferred <infer_signature>` from datasets representing
                          valid model input (e.g. the training dataset) and valid model output
                          (e.g. model predictions generated on the training dataset), for example:

                          .. code-block:: python

                            from mlflow.models import infer_signature

                            train = df.drop_column("target_label")
                            signature = infer_signature(train, model.predict(train))

        :param input_example: one or several instances of valid model input. The input example is
                            used as a hint of what data to feed the model. It will be converted to
                            a Pandas DataFrame and then serialized to json using the Pandas
                            split-oriented format, or a numpy array where the example will be
                            serialized to json by converting it to a list. If input example is a
                            tuple, then the first element must be a valid model input, and the
                            second element must be a valid params dictionary that can optionally
                            be used during model inference. Bytes are base64-encoded. When the
                            ``signature`` parameter is ``None``, the input example is used to infer
                            a model's signature. If an input example containing params is provided,
                            and signature is inferred from the input example, then params will be
                            used as default params during model inference if no extra params are
                            passed at inference time.

        :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.

        :param metadata: Custom metadata dictionary passed to the model and stored in
                         the MLmodel file.

                         .. Note:: Experimental: This parameter may change or be removed in a
                                                 future release without warning.

        :param kwargs: Extra args passed to the model flavor.

        :return: A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
                 metadata of the logged model.
        """
        with TempDir() as tmp:
            local_path = tmp.path("model")
            run_id = mlflow.tracking.fluent._get_or_start_run().info.run_id
            mlflow_model = cls(artifact_path=artifact_path, run_id=run_id, metadata=metadata)
            tracking_uri = _resolve_tracking_uri()
            if (
                (tracking_uri == "databricks" or get_uri_scheme(tracking_uri) == "databricks")
                and kwargs.get("signature") is None
                and kwargs.get("input_example") is None
            ):
                _logger.warning(_LOG_MODEL_MISSING_SIGNATURE_WARNING)
            flavor.save_model(path=local_path, mlflow_model=mlflow_model, **kwargs)
            mlflow.tracking.fluent.log_artifacts(local_path, mlflow_model.artifact_path)
            try:
                mlflow.tracking.fluent._record_logged_model(mlflow_model)
            except MlflowException:
                # We need to swallow all mlflow exceptions to maintain backwards compatibility with
                # older tracking servers. Only print out a warning for now.
                _logger.warning(_LOG_MODEL_METADATA_WARNING_TEMPLATE, mlflow.get_artifact_uri())
                _logger.debug("", exc_info=True)
            if registered_model_name is not None:
                run_id = mlflow.tracking.fluent.active_run().info.run_id
                mlflow.tracking._model_registry.fluent._register_model(
                    f"runs:/{run_id}/{mlflow_model.artifact_path}",
                    registered_model_name,
                    await_registration_for=await_registration_for,
                    local_model_path=local_path,
                )
        return mlflow_model.get_model_info()


def get_model_info(model_uri: str) -> ModelInfo:
    """
    Get metadata for the specified model, such as its input/output signature.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

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

    :return: A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
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

    local_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=None)
    model_meta = Model.load(os.path.join(local_path, MLMODEL_FILE_NAME))
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
