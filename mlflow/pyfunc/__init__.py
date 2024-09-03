"""
The ``python_function`` model flavor serves as a default model interface for MLflow Python models.
Any MLflow Python model is expected to be loadable as a ``python_function`` model.

In addition, the ``mlflow.pyfunc`` module defines a generic :ref:`filesystem format
<pyfunc-filesystem-format>` for Python models and provides utilities for saving to and loading from
this format. The format is self contained in the sense that it includes all necessary information
for anyone to load it and use it. Dependencies are either stored directly with the model or
referenced via a Conda environment.

The ``mlflow.pyfunc`` module also defines utilities for creating custom ``pyfunc`` models
using frameworks and inference logic that may not be natively included in MLflow. See
:ref:`pyfunc-create-custom`.

.. _pyfunc-inference-api:

*************
Inference API
*************

Python function models are loaded as an instance of :py:class:`PyFuncModel
<mlflow.pyfunc.PyFuncModel>`, which is an MLflow wrapper around the model implementation and model
metadata (MLmodel file). You can score the model by calling the :py:func:`predict()
<mlflow.pyfunc.PyFuncModel.predict>` method, which has the following signature::

  predict(
    model_input: [pandas.DataFrame, numpy.ndarray, scipy.sparse.(csc_matrix | csr_matrix),
    List[Any], Dict[str, Any], pyspark.sql.DataFrame]
  ) -> [numpy.ndarray | pandas.(Series | DataFrame) | List | Dict | pyspark.sql.DataFrame]

All PyFunc models will support `pandas.DataFrame` as input and PyFunc deep learning models will
also support tensor inputs in the form of Dict[str, numpy.ndarray] (named tensors) and
`numpy.ndarrays` (unnamed tensors).

Here are some examples of supported inference types, assuming we have the correct ``model`` object
loaded.

.. list-table::
    :widths: 30 70
    :header-rows: 1
    :class: wrap-table

    * - Input Type
      - Example
    * - ``pandas.DataFrame``
      -
        .. code-block:: python

            import pandas as pd

            x_new = pd.DataFrame(dict(x1=[1, 2, 3], x2=[4, 5, 6]))
            model.predict(x_new)

    * - ``numpy.ndarray``
      -
        .. code-block:: python

            import numpy as np

            x_new = np.array([[1, 4][2, 5], [3, 6]])
            model.predict(x_new)

    * - ``scipy.sparse.csc_matrix`` or ``scipy.sparse.csr_matrix``
      -
        .. code-block:: python

            import scipy

            x_new = scipy.sparse.csc_matrix([[1, 2, 3], [4, 5, 6]])
            model.predict(x_new)

            x_new = scipy.sparse.csr_matrix([[1, 2, 3], [4, 5, 6]])
            model.predict(x_new)

    * - python ``List``
      -
        .. code-block:: python

            x_new = [[1, 4], [2, 5], [3, 6]]
            model.predict(x_new)

    * - python ``Dict``
      -
        .. code-block:: python

            x_new = dict(x1=[1, 2, 3], x2=[4, 5, 6])
            model.predict(x_new)

    * - ``pyspark.sql.DataFrame``
      -
        .. code-block:: python

            from pyspark.sql import SparkSession

            spark = SparkSession.builder.getOrCreate()

            data = [(1, 4), (2, 5), (3, 6)]  # List of tuples
            x_new = spark.createDataFrame(data, ["x1", "x2"])  # Specify column name
            model.predict(x_new)

.. _pyfunc-filesystem-format:

*****************
Filesystem format
*****************

The Pyfunc format is defined as a directory structure containing all required data, code, and
configuration::

    ./dst-path/
        ./MLmodel: configuration
        <code>: code packaged with the model (specified in the MLmodel file)
        <data>: data packaged with the model (specified in the MLmodel file)
        <env>: Conda environment definition (specified in the MLmodel file)

The directory structure may contain additional contents that can be referenced by the ``MLmodel``
configuration.

.. _pyfunc-model-config:

MLModel configuration
#####################

A Python model contains an ``MLmodel`` file in **python_function** format in its root with the
following parameters:

- loader_module [required]:
         Python module that can load the model. Expected as module identifier
         e.g. ``mlflow.sklearn``, it will be imported using ``importlib.import_module``.
         The imported module must contain a function with the following signature::

          _load_pyfunc(path: string) -> <pyfunc model implementation>

         The path argument is specified by the ``data`` parameter and may refer to a file or
         directory. The model implementation is expected to be an object with a
         ``predict`` method with the following signature::

          predict(
            model_input: [pandas.DataFrame, numpy.ndarray,
            scipy.sparse.(csc_matrix | csr_matrix), List[Any], Dict[str, Any]],
            pyspark.sql.DataFrame
          ) -> [numpy.ndarray | pandas.(Series | DataFrame) | List | Dict | pyspark.sql.DataFrame]

- code [optional]:
        Relative path to a directory containing the code packaged with this model.
        All files and directories inside this directory are added to the Python path
        prior to importing the model loader.

- data [optional]:
         Relative path to a file or directory containing model data.
         The path is passed to the model loader.

- env [optional]:
         Relative path to an exported Conda environment. If present this environment
         should be activated prior to running the model.

- Optionally, any additional parameters necessary for interpreting the serialized model in
  ``pyfunc`` format.

.. rubric:: Example

::

    tree example/sklearn_iris/mlruns/run1/outputs/linear-lr

::

  ├── MLmodel
  ├── code
  │   ├── sklearn_iris.py
  │
  ├── data
  │   └── model.pkl
  └── mlflow_env.yml

::

    cat example/sklearn_iris/mlruns/run1/outputs/linear-lr/MLmodel

::

  python_function:
    code: code
    data: data/model.pkl
    loader_module: mlflow.sklearn
    env: mlflow_env.yml
    main: sklearn_iris

.. _pyfunc-create-custom:

**********************************
Models From Code for Custom Models
**********************************

.. tip::

    MLflow 2.12.2 introduced the feature "models from code", which greatly simplifies the process
    of serializing and deploying custom models through the use of script serialization. It is
    strongly recommended to migrate custom model implementations to this new paradigm to avoid the
    limitations and complexity of serializing with cloudpickle.
    You can learn more about models from code within the
    `Models From Code Guide <../model/models-from-code.html>`_.

The section below illustrates the process of using the legacy serializer for custom Pyfunc models.
Models from code will provide a far simpler experience for logging of your models.

******************************
Creating custom Pyfunc models
******************************

MLflow's persistence modules provide convenience functions for creating models with the
``pyfunc`` flavor in a variety of machine learning frameworks (scikit-learn, Keras, Pytorch, and
more); however, they do not cover every use case. For example, you may want to create an MLflow
model with the ``pyfunc`` flavor using a framework that MLflow does not natively support.
Alternatively, you may want to build an MLflow model that executes custom logic when evaluating
queries, such as preprocessing and postprocessing routines. Therefore, ``mlflow.pyfunc``
provides utilities for creating ``pyfunc`` models from arbitrary code and model data.

The :meth:`save_model()` and :meth:`log_model()` methods are designed to support multiple workflows
for creating custom ``pyfunc`` models that incorporate custom inference logic and artifacts
that the logic may require.

An `artifact` is a file or directory, such as a serialized model or a CSV. For example, a
serialized TensorFlow graph is an artifact. An MLflow model directory is also an artifact.

.. _pyfunc-create-custom-workflows:

Workflows
#########

:meth:`save_model()` and :meth:`log_model()` support the following workflows:

1. Programmatically defining a new MLflow model, including its attributes and artifacts.

   Given a set of artifact URIs, :meth:`save_model()` and :meth:`log_model()` can
   automatically download artifacts from their URIs and create an MLflow model directory.

   In this case, you must define a Python class which inherits from :class:`~PythonModel`,
   defining ``predict()`` and, optionally, ``load_context()``. An instance of this class is
   specified via the ``python_model`` parameter; it is automatically serialized and deserialized
   as a Python class, including all of its attributes.

2. Interpreting pre-existing data as an MLflow model.

   If you already have a directory containing model data, :meth:`save_model()` and
   :meth:`log_model()` can import the data as an MLflow model. The ``data_path`` parameter
   specifies the local filesystem path to the directory containing model data.

   In this case, you must provide a Python module, called a `loader module`. The
   loader module defines a ``_load_pyfunc()`` method that performs the following tasks:

   - Load data from the specified ``data_path``. For example, this process may include
     deserializing pickled Python objects or models or parsing CSV files.

   - Construct and return a pyfunc-compatible model wrapper. As in the first
     use case, this wrapper must define a ``predict()`` method that is used to evaluate
     queries. ``predict()`` must adhere to the :ref:`pyfunc-inference-api`.

   The ``loader_module`` parameter specifies the name of your loader module.

   For an example loader module implementation, refer to the `loader module
   implementation in mlflow.sklearn <https://github.com/mlflow/mlflow/blob/
   74d75109aaf2975f5026104d6125bb30f4e3f744/mlflow/sklearn.py#L200-L205>`_.

.. _pyfunc-create-custom-selecting-workflow:

Which workflow is right for my use case?
########################################

We consider the first workflow to be more user-friendly and generally recommend it for the
following reasons:

- It automatically resolves and collects specified model artifacts.

- It automatically serializes and deserializes the ``python_model`` instance and all of
  its attributes, reducing the amount of user logic that is required to load the model

- You can create Models using logic that is defined in the ``__main__`` scope. This allows
  custom models to be constructed in interactive environments, such as notebooks and the Python
  REPL.

You may prefer the second, lower-level workflow for the following reasons:

- Inference logic is always persisted as code, rather than a Python object. This makes logic
  easier to inspect and modify later.

- If you have already collected all of your model data in a single location, the second
  workflow allows it to be saved in MLflow format directly, without enumerating constituent
  artifacts.

******************************************
Function-based Model vs Class-based Model
******************************************

When creating custom PyFunc models, you can choose between two different interfaces:
a function-based model and a class-based model. In short, a function-based model is simply a
python function that does not take additional params. The class-based model, on the other hand,
is subclass of ``PythonModel`` that supports several required and optional
methods. If your use case is simple and fits within a single predict function, a funcion-based
approach is recommended. If you need more power, such as custom serialization, custom data
processing, or to override additional methods, you should use the class-based implementation.

Before looking at code examples, it's important to note that both methods are serialized via
`cloudpickle <https://github.com/cloudpipe/cloudpickle>`_. cloudpickle can serialize Python
functions, lambda functions, and locally defined classes and functions inside other functions. This
makes cloudpickle especially useful for parallel and distributed computing where code objects need
to be sent over network to execute on remote workers, which is a common deployment paradigm for
MLflow.

That said, cloudpickle has some limitations.

- **Environment Dependency**: cloudpickle does not capture the full execution environment, so in
  MLflow we must pass ``pip_requirements``, ``extra_pip_requirements``, or an ``input_example``,
  the latter of which is used to infer environment dependencies. For more, refer to
  `the model dependency docs <https://mlflow.org/docs/latest/model/dependencies.html>`_.

- **Object Support**: cloudpickle does not serialize objects outside of the Python data model.
  Some relevant examples include raw files and database connections. If your program depends on
  these, be sure to log ways to reference these objects along with your model.

Function-based Model
####################
If you're looking to serialize a simple python function without additional dependent methods, you
can simply log a predict method via the keyword argument ``python_model``.

.. note::

    Function-based model only supports a function with a single input argument. If you would like
    to pass more arguments or additional inference parameters, please use the class-based model
    below.

.. code-block:: python

    import mlflow
    import pandas as pd


    # Define a simple function to log
    def predict(model_input):
        return model_input.apply(lambda x: x * 2)


    # Save the function as a model
    with mlflow.start_run():
        mlflow.pyfunc.log_model("model", python_model=predict, pip_requirements=["pandas"])
        run_id = mlflow.active_run().info.run_id

    # Load the model from the tracking server and perform inference
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    x_new = pd.Series([1, 2, 3])

    prediction = model.predict(x_new)
    print(prediction)


Class-based Model
#################
If you're looking to serialize a more complex object, for instance a class that handles
preprocessing, complex prediction logic, or custom serialization, you should subclass the
``PythonModel`` class. MLflow has tutorials on building custom PyFunc models, as shown
`here <https://mlflow.org/docs/latest/traditional-ml/creating-custom-pyfunc/index.html>`_,
so instead of duplicating that information, in this example we'll recreate the above functionality
to highlight the differences. Note that this PythonModel implementation is overly complex and
we would recommend using the functional-based Model instead for this simple case.

.. code-block:: python

    import mlflow
    import pandas as pd


    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input, params=None):
            return [x * 2 for x in model_input]


    # Save the function as a model
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            "model", python_model=MyModel(), pip_requirements=["pandas"]
        )
        run_id = mlflow.active_run().info.run_id

    # Load the model from the tracking server and perform inference
    model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    x_new = pd.Series([1, 2, 3])

    print(f"Prediction:\n\t{model.predict(x_new)}")

The primary difference between the this implementation and the function-based implementation above
is that the predict method is wrapped with a class, has the ``self`` parameter,
and has the ``params`` parameter that defaults to None. Note that function-based models don't
support additional params.

In summary, use the function-based Model when you have a simple function to serialize.
If you need more power, use  the class-based model.
"""
import collections
import functools
import importlib
import inspect
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import warnings
from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas
import yaml
from packaging.version import Version

import mlflow
import mlflow.pyfunc.loaders
import mlflow.pyfunc.model
from mlflow.environment_variables import (
    _MLFLOW_IN_CAPTURE_MODULE_PROCESS,
    _MLFLOW_TESTING,
    MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT,
)
from mlflow.exceptions import MlflowException
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.dependencies_schemas import (
    _clear_dependencies_schemas,
    _get_dependencies_schemas,
)
from mlflow.models.flavor_backend_registry import get_flavor_backend
from mlflow.models.model import (
    _DATABRICKS_FS_LOADER_MODULE,
    MLMODEL_FILE_NAME,
    MODEL_CODE_PATH,
    MODEL_CONFIG,
)
from mlflow.models.resources import Resource, _ResourceBuilder
from mlflow.models.signature import (
    _infer_signature_from_input_example,
    _infer_signature_from_type_hints,
)
from mlflow.models.utils import (
    PyFuncInput,
    PyFuncLLMOutputChunk,
    PyFuncLLMSingleInput,
    PyFuncOutput,
    _convert_llm_input_data,
    _enforce_params_schema,
    _enforce_schema,
    _load_model_code_path,
    _save_example,
    _split_input_data_and_params,
    _validate_and_get_model_code_path,
)
from mlflow.protos.databricks_pb2 import (
    BAD_REQUEST,
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
    Entity,
    Job,
    LineageHeaderInfo,
    Notebook,
)
from mlflow.pyfunc.context import Context, set_prediction_context
from mlflow.pyfunc.model import (
    ChatModel,
    PythonModel,
    PythonModelContext,
    _log_warning_if_params_not_in_predict_signature,
    _PythonModelPyfuncWrapper,
    get_default_conda_env,  # noqa: F401
    get_default_pip_requirements,
)
from mlflow.tracing.provider import trace_disabled
from mlflow.tracing.utils import _try_get_prediction_context
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types.llm import (
    CHAT_MODEL_INPUT_EXAMPLE,
    CHAT_MODEL_INPUT_SCHEMA,
    CHAT_MODEL_OUTPUT_SCHEMA,
    ChatMessage,
    ChatParams,
    ChatResponse,
)
from mlflow.utils import (
    PYTHON_VERSION,
    _is_in_ipython_notebook,
    check_port_connectivity,
    databricks_utils,
    find_free_port,
    get_major_minor_py_version,
    insecure_hash,
)
from mlflow.utils import env_manager as _EnvManager
from mlflow.utils._spark_utils import modified_environ
from mlflow.utils.annotations import deprecated, developer_stable, experimental
from mlflow.utils.databricks_utils import is_in_databricks_runtime
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _process_conda_env,
    _process_pip_requirements,
    _PythonEnv,
    _validate_env_arguments,
)
from mlflow.utils.file_utils import (
    _copy_file_or_tree,
    get_or_create_nfs_tmp_dir,
    get_or_create_tmp_dir,
    get_total_file_size,
    write_to,
)
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _get_flavor_configuration_from_ml_model_file,
    _get_overridden_pyfunc_model_config,
    _validate_and_copy_file_to_directory,
    _validate_and_get_model_config_from_file,
    _validate_and_prepare_target_save_path,
    _validate_infer_and_copy_code_paths,
    _validate_pyfunc_model_config,
)
from mlflow.utils.nfs_on_spark import get_nfs_cache_root_dir
from mlflow.utils.requirements_utils import (
    _parse_requirements,
    warn_dependency_requirement_mismatches,
)

try:
    from pyspark.sql import DataFrame as SparkDataFrame

    HAS_PYSPARK = True
except ImportError:
    HAS_PYSPARK = False
FLAVOR_NAME = "python_function"
MAIN = "loader_module"
CODE = "code"
DATA = "data"
ENV = "env"

_MODEL_DATA_SUBPATH = "data"


class EnvType:
    CONDA = "conda"
    VIRTUALENV = "virtualenv"

    def __init__(self):
        raise NotImplementedError("This class is not meant to be instantiated.")


PY_VERSION = "python_version"


_logger = logging.getLogger(__name__)


def add_to_model(
    model,
    loader_module,
    data=None,
    code=None,
    conda_env=None,
    python_env=None,
    model_config=None,
    model_code_path=None,
    **kwargs,
):
    """
    Add a ``pyfunc`` spec to the model configuration.

    Defines ``pyfunc`` configuration schema. Caller can use this to create a valid ``pyfunc`` model
    flavor out of an existing directory structure. For example, other model flavors can use this to
    specify how to use their output as a ``pyfunc``.

    NOTE:

        All paths are relative to the exported model root directory.

    Args:
        model: Existing model.
        loader_module: The module to be used to load the model.
        data: Path to the model data.
        code: Path to the code dependencies.
        conda_env: Conda environment.
        python_env: Python environment.
        req: pip requirements file.
        kwargs: Additional key-value pairs to include in the ``pyfunc`` flavor specification.
                Values must be YAML-serializable.
        model_config: The model configuration to apply to the model. This configuration
                      is available during model loading.

                      .. Note:: Experimental: This parameter may change or be removed in a future
                                              release without warning.

    Returns:
        Updated model configuration.
    """
    params = deepcopy(kwargs)
    params[MAIN] = loader_module
    params[PY_VERSION] = PYTHON_VERSION
    if code:
        params[CODE] = code
    if data:
        params[DATA] = data
    if conda_env or python_env:
        params[ENV] = {}
        if conda_env:
            params[ENV][EnvType.CONDA] = conda_env
        if python_env:
            params[ENV][EnvType.VIRTUALENV] = python_env
    if model_config:
        params[MODEL_CONFIG] = model_config
    if model_code_path:
        params[MODEL_CODE_PATH] = model_code_path
    return model.add_flavor(FLAVOR_NAME, **params)


def _extract_conda_env(env):
    # In MLflow < 2.0.0, the 'env' field in a pyfunc configuration is a string containing the path
    # to a conda.yaml file.
    return env if isinstance(env, str) else env[EnvType.CONDA]


def _load_model_env(path):
    """
    Get ENV file string from a model configuration stored in Python Function format.
    Returned value is a model-relative path to a Conda Environment file,
    or None if none was specified at model save time
    """
    return _get_flavor_configuration(model_path=path, flavor_name=FLAVOR_NAME).get(ENV, None)


def _validate_params(params, model_metadata):
    if hasattr(model_metadata, "get_params_schema"):
        params_schema = model_metadata.get_params_schema()
        return _enforce_params_schema(params, params_schema)
    if params:
        raise MlflowException.invalid_parameter_value(
            "This model was not logged with a params schema and does not support "
            "providing the params argument."
            "Please log the model with mlflow >= 2.6.0 and specify a params schema.",
        )
    return


def _validate_prediction_input(data: PyFuncInput, params, input_schema, params_schema, flavor=None):
    """
    Internal helper function to transform and validate input data and params for prediction.
    Any additional transformation logics related to input data and params should be added here.
    """
    if input_schema is not None:
        try:
            data = _enforce_schema(data, input_schema, flavor)
        except Exception as e:
            # Include error in message for backwards compatibility
            raise MlflowException.invalid_parameter_value(
                f"Failed to enforce schema of data '{data}' "
                f"with schema '{input_schema}'. "
                f"Error: {e}",
            )
    params = _enforce_params_schema(params, params_schema)
    if HAS_PYSPARK and isinstance(data, SparkDataFrame):
        _logger.warning(
            "Input data is a Spark DataFrame. Note that behaviour for "
            "Spark DataFrames is model dependent."
        )
    return data, params


class PyFuncModel:
    """
    MLflow 'python function' model.

    Wrapper around model implementation and metadata. This class is not meant to be constructed
    directly. Instead, instances of this class are constructed and returned from
    :py:func:`load_model() <mlflow.pyfunc.load_model>`.

    ``model_impl`` can be any Python object that implements the `Pyfunc interface
    <https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#pyfunc-inference-api>`_, and is
    returned by invoking the model's ``loader_module``.

    ``model_meta`` contains model metadata loaded from the MLmodel file.
    """

    def __init__(
        self,
        model_meta: Model,
        model_impl: Any,
        predict_fn: str = "predict",
        predict_stream_fn: Optional[str] = None,
    ):
        if not hasattr(model_impl, predict_fn):
            raise MlflowException(f"Model implementation is missing required {predict_fn} method.")
        if not model_meta:
            raise MlflowException("Model is missing metadata.")
        self._model_meta = model_meta
        self.__model_impl = model_impl
        self._predict_fn = getattr(model_impl, predict_fn)
        if predict_stream_fn:
            if not hasattr(model_impl, predict_stream_fn):
                raise MlflowException(
                    f"Model implementation is missing required {predict_stream_fn} method."
                )
            self._predict_stream_fn = getattr(model_impl, predict_stream_fn)
        else:
            self._predict_stream_fn = None

    @property
    @developer_stable
    def _model_impl(self) -> Any:
        """
        The underlying model implementation object.

        NOTE: This is a stable developer API.
        """
        return self.__model_impl

    def _update_dependencies_schemas_in_prediction_context(self, context: Context):
        if self._model_meta and self._model_meta.metadata:
            dependencies_schemas = self._model_meta.metadata.get("dependencies_schemas", {})
            context.update(
                dependencies_schemas={
                    dependency: json.dumps(schema)
                    for dependency, schema in dependencies_schemas.items()
                }
            )

    @contextmanager
    def _try_get_or_generate_prediction_context(self):
        # set context for prediction if it's not set
        # NB: in model serving the prediction context must be set
        # with a request_id
        context = _try_get_prediction_context() or Context()
        with set_prediction_context(context):
            yield context

    def predict(self, data: PyFuncInput, params: Optional[Dict[str, Any]] = None) -> PyFuncOutput:
        with self._try_get_or_generate_prediction_context() as context:
            self._update_dependencies_schemas_in_prediction_context(context)
            return self._predict(data, params)

    def _predict(self, data: PyFuncInput, params: Optional[Dict[str, Any]] = None) -> PyFuncOutput:
        """
        Generates model predictions.

        If the model contains signature, enforce the input schema first before calling the model
        implementation with the sanitized input. If the pyfunc model does not include model schema,
        the input is passed to the model implementation as is. See `Model Signature Enforcement
        <https://www.mlflow.org/docs/latest/models.html#signature-enforcement>`_ for more details.

        Args:
            data: LLM Model single input as one of pandas.DataFrame, numpy.ndarray,
                scipy.sparse.(csc_matrix | csr_matrix), List[Any], or
                Dict[str, numpy.ndarray].
                For model signatures with tensor spec inputs
                (e.g. the Tensorflow core / Keras model), the input data type must be one of
                `numpy.ndarray`, `List[numpy.ndarray]`, `Dict[str, numpy.ndarray]` or
                `pandas.DataFrame`. If data is of `pandas.DataFrame` type and the model
                contains a signature with tensor spec inputs, the corresponding column values
                in the pandas DataFrame will be reshaped to the required shape with 'C' order
                (i.e. read / write the elements using C-like index order), and DataFrame
                column values will be cast as the required tensor spec type. For Pyspark
                DataFrame inputs, MLflow will only enforce the schema on a subset
                of the data rows.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Model predictions as one of pandas.DataFrame, pandas.Series, numpy.ndarray or list.
        """
        # fetch the schema from metadata to avoid signature change after model is loaded
        self.input_schema = self.metadata.get_input_schema()
        self.params_schema = self.metadata.get_params_schema()
        data, params = _validate_prediction_input(
            data, params, self.input_schema, self.params_schema, self.loader_module
        )
        params_arg = inspect.signature(self._predict_fn).parameters.get("params")
        if params_arg and params_arg.kind != inspect.Parameter.VAR_KEYWORD:
            return self._predict_fn(data, params=params)

        _log_warning_if_params_not_in_predict_signature(_logger, params)
        return self._predict_fn(data)

    def predict_stream(
        self, data: PyFuncLLMSingleInput, params: Optional[Dict[str, Any]] = None
    ) -> Iterator[PyFuncLLMOutputChunk]:
        with self._try_get_or_generate_prediction_context() as context:
            self._update_dependencies_schemas_in_prediction_context(context)
            return self._predict_stream(data, params)

    def _predict_stream(
        self, data: PyFuncLLMSingleInput, params: Optional[Dict[str, Any]] = None
    ) -> Iterator[PyFuncLLMOutputChunk]:
        """
        Generates streaming model predictions. Only LLM suports this method.

        If the model contains signature, enforce the input schema first before calling the model
        implementation with the sanitized input. If the pyfunc model does not include model schema,
        the input is passed to the model implementation as is. See `Model Signature Enforcement
        <https://www.mlflow.org/docs/latest/models.html#signature-enforcement>`_ for more details.

        Args:
            data: LLM Model single input as one of dict, str, bool, bytes, float, int, str type.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Model predictions as an iterator of chunks. The chunks in the iterator must be type of
            dict or string. Chunk dict fields are determined by the model implementation.
        """

        if self._predict_stream_fn is None:
            raise MlflowException("This model does not support predict_stream method.")

        self.input_schema = self.metadata.get_input_schema()
        self.params_schema = self.metadata.get_params_schema()
        data, params = _validate_prediction_input(
            data, params, self.input_schema, self.params_schema, self.loader_module
        )
        data = _convert_llm_input_data(data)
        if isinstance(data, list):
            # `predict_stream` only accepts single input.
            # but `enforce_schema` might convert single input into a list like `[single_input]`
            # so extract the first element in the list.
            if len(data) != 1:
                raise MlflowException(
                    f"'predict_stream' requires single input, but it got input data {data}"
                )
            data = data[0]

        if inspect.signature(self._predict_stream_fn).parameters.get("params"):
            return self._predict_stream_fn(data, params=params)

        _log_warning_if_params_not_in_predict_signature(_logger, params)
        return self._predict_stream_fn(data)

    @experimental
    def unwrap_python_model(self):
        """
        Unwrap the underlying Python model object.

        This method is useful for accessing custom model functions, while still being able to
        leverage the MLflow designed workflow through the `predict()` method.

        Returns:
            The underlying wrapped model object

        .. code-block:: python
            :test:
            :caption: Example

            import mlflow


            # define a custom model
            class MyModel(mlflow.pyfunc.PythonModel):
                def predict(self, context, model_input, params=None):
                    return self.my_custom_function(model_input, params)

                def my_custom_function(self, model_input, params=None):
                    # do something with the model input
                    return 0


            some_input = 1
            # save the model
            with mlflow.start_run():
                model_info = mlflow.pyfunc.log_model(artifact_path="model", python_model=MyModel())

            # load the model
            loaded_model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
            print(type(loaded_model))  # <class 'mlflow.pyfunc.model.PyFuncModel'>
            unwrapped_model = loaded_model.unwrap_python_model()
            print(type(unwrapped_model))  # <class '__main__.MyModel'>

            # does not work, only predict() is exposed
            # print(loaded_model.my_custom_function(some_input))
            print(unwrapped_model.my_custom_function(some_input))  # works
            print(loaded_model.predict(some_input))  # works

            # works, but None is needed for context arg
            print(unwrapped_model.predict(None, some_input))
        """
        try:
            python_model = self._model_impl.python_model
            if python_model is None:
                raise AttributeError("Expected python_model attribute not to be None.")
        except AttributeError as e:
            raise MlflowException("Unable to retrieve base model object from pyfunc.") from e
        return python_model

    def __eq__(self, other):
        if not isinstance(other, PyFuncModel):
            return False
        return self._model_meta == other._model_meta

    @property
    def metadata(self):
        """Model metadata."""
        if self._model_meta is None:
            raise MlflowException("Model is missing metadata.")
        return self._model_meta

    @experimental
    @property
    def model_config(self):
        """Model's flavor configuration"""
        return self._model_meta.flavors[FLAVOR_NAME].get(MODEL_CONFIG, {})

    @experimental
    @property
    def loader_module(self):
        """Model's flavor configuration"""
        if self._model_meta.flavors.get(FLAVOR_NAME) is None:
            return None
        return self._model_meta.flavors[FLAVOR_NAME].get(MAIN)

    def __repr__(self):
        info = {}
        if self._model_meta is not None:
            if hasattr(self._model_meta, "run_id") and self._model_meta.run_id is not None:
                info["run_id"] = self._model_meta.run_id
            if (
                hasattr(self._model_meta, "artifact_path")
                and self._model_meta.artifact_path is not None
            ):
                info["artifact_path"] = self._model_meta.artifact_path
            info["flavor"] = self._model_meta.flavors[FLAVOR_NAME]["loader_module"]
        return yaml.safe_dump({"mlflow.pyfunc.loaded_model": info}, default_flow_style=False)

    @experimental
    def get_raw_model(self):
        """
        Get the underlying raw model if the model wrapper implemented `get_raw_model` function.
        """
        if hasattr(self._model_impl, "get_raw_model"):
            return self._model_impl.get_raw_model()
        raise NotImplementedError("`get_raw_model` is not implemented by the underlying model")


def _get_pip_requirements_from_model_path(model_path: str):
    req_file_path = os.path.join(model_path, _REQUIREMENTS_FILE_NAME)
    if not os.path.exists(req_file_path):
        return []

    return [req.req_str for req in _parse_requirements(req_file_path, is_constraint=False)]


@trace_disabled  # Suppress traces while loading model
def load_model(
    model_uri: str,
    suppress_warnings: bool = False,
    dst_path: Optional[str] = None,
    model_config: Optional[Union[str, Path, Dict[str, Any]]] = None,
) -> PyFuncModel:
    """
    Load a model stored in Python function format.

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
        suppress_warnings: If ``True``, non-fatal warning messages associated with the model
            loading process will be suppressed. If ``False``, these warning messages will be
            emitted.
        dst_path: The local filesystem path to which to download the model artifact.
            This directory must already exist. If unspecified, a local output
            path will be created.
        model_config: The model configuration to apply to the model. This configuration
            is available during model loading. The configuration can be passed as a file path,
            or a dict with string keys.

            .. Note:: Experimental: This parameter may change or be removed in a future
                release without warning.
    """

    lineage_header_info = None
    if (
        not _MLFLOW_IN_CAPTURE_MODULE_PROCESS.get()
    ) and databricks_utils.is_in_databricks_runtime():
        entity_list = []
        # Get notebook id and job id, pack them into lineage_header_info
        if databricks_utils.is_in_databricks_notebook() and (
            notebook_id := databricks_utils.get_notebook_id()
        ):
            notebook_entity = Notebook(id=notebook_id)
            entity_list.append(Entity(notebook=notebook_entity))

        if databricks_utils.is_in_databricks_job() and (job_id := databricks_utils.get_job_id()):
            job_entity = Job(id=job_id)
            entity_list.append(Entity(job=job_entity))

        lineage_header_info = LineageHeaderInfo(entities=entity_list) if entity_list else None

    local_path = _download_artifact_from_uri(
        artifact_uri=model_uri, output_path=dst_path, lineage_header_info=lineage_header_info
    )

    if not suppress_warnings:
        model_requirements = _get_pip_requirements_from_model_path(local_path)
        warn_dependency_requirement_mismatches(model_requirements)

    model_meta = Model.load(os.path.join(local_path, MLMODEL_FILE_NAME))

    conf = model_meta.flavors.get(FLAVOR_NAME)
    if conf is None:
        raise MlflowException(
            f'Model does not have the "{FLAVOR_NAME}" flavor',
            RESOURCE_DOES_NOT_EXIST,
        )
    model_py_version = conf.get(PY_VERSION)
    if not suppress_warnings:
        _warn_potentially_incompatible_py_version_if_necessary(model_py_version=model_py_version)

    _add_code_from_conf_to_system_path(local_path, conf, code_key=CODE)
    data_path = os.path.join(local_path, conf[DATA]) if (DATA in conf) else local_path

    if isinstance(model_config, str):
        model_config = _validate_and_get_model_config_from_file(model_config)

    model_config = _get_overridden_pyfunc_model_config(
        conf.get(MODEL_CONFIG, None), model_config, _logger
    )

    try:
        if model_config:
            model_impl = importlib.import_module(conf[MAIN])._load_pyfunc(data_path, model_config)
        else:
            model_impl = importlib.import_module(conf[MAIN])._load_pyfunc(data_path)
    except ModuleNotFoundError as e:
        # This error message is particularly for the case when the error is caused by module
        # "databricks.feature_store.mlflow_model". But depending on the environment, the offending
        # module might be "databricks", "databricks.feature_store" or full package. So we will
        # raise the error with the following note if "databricks" presents in the error. All non-
        # databricks moduel errors will just be re-raised.
        if conf[MAIN] == _DATABRICKS_FS_LOADER_MODULE and e.name.startswith("databricks"):
            raise MlflowException(
                f"{e.msg}; "
                "Note: mlflow.pyfunc.load_model is not supported for Feature Store models. "
                "spark_udf() and predict() will not work as expected. Use "
                "score_batch for offline predictions.",
                BAD_REQUEST,
            ) from None
        raise e
    finally:
        # clean up the dependencies schema which is set to global state after loading the model.
        # This avoids the schema being used by other models loaded in the same process.
        _clear_dependencies_schemas()
    predict_fn = conf.get("predict_fn", "predict")
    streamable = conf.get("streamable", False)
    predict_stream_fn = conf.get("predict_stream_fn", "predict_stream") if streamable else None

    return PyFuncModel(
        model_meta=model_meta,
        model_impl=model_impl,
        predict_fn=predict_fn,
        predict_stream_fn=predict_stream_fn,
    )


class _ServedPyFuncModel(PyFuncModel):
    def __init__(self, model_meta: Model, client: Any, server_pid: int, env_manager="local"):
        super().__init__(model_meta=model_meta, model_impl=client, predict_fn="invoke")
        self._client = client
        self._server_pid = server_pid
        # We need to set `env_manager` attribute because it is used by Databricks runtime
        # evaluate usage logging to log 'env_manager' tag in `_evaluate` function patching.
        self._env_manager = env_manager

    def predict(self, data, params=None):
        """
        Args:
            data: Model input data.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Model predictions.
        """
        if inspect.signature(self._client.invoke).parameters.get("params"):
            result = self._client.invoke(data, params=params).get_predictions()
        else:
            _log_warning_if_params_not_in_predict_signature(_logger, params)
            result = self._client.invoke(data).get_predictions()
        if isinstance(result, pandas.DataFrame):
            result = result[result.columns[0]]
        return result

    @property
    def pid(self):
        if self._server_pid is None:
            raise MlflowException("Served PyFunc Model is missing server process ID.")
        return self._server_pid

    @property
    def env_manager(self):
        return self._env_manager

    @env_manager.setter
    def env_manager(self, value):
        self._env_manager = value


def _load_model_or_server(
    model_uri: str, env_manager: str, model_config: Optional[Dict[str, Any]] = None
):
    """
    Load a model with env restoration. If a non-local ``env_manager`` is specified, prepare an
    independent Python environment with the training time dependencies of the specified model
    installed and start a MLflow Model Scoring Server process with that model in that environment.
    Return a _ServedPyFuncModel that invokes the scoring server for prediction. Otherwise, load and
    return the model locally as a PyFuncModel using :py:func:`mlflow.pyfunc.load_model`.

    Args:
        model_uri: The uri of the model.
        env_manager: The environment manager to load the model.
        model_config: The model configuration to use by the model, only if the model
                      accepts it.

    Returns:
        A _ServedPyFuncModel for non-local ``env_manager``s or a PyFuncModel otherwise.
    """
    from mlflow.pyfunc.scoring_server.client import (
        ScoringServerClient,
        StdinScoringServerClient,
    )

    if env_manager == _EnvManager.LOCAL:
        return load_model(model_uri, model_config=model_config)

    _logger.info("Starting model server for model environment restoration.")

    local_path = _download_artifact_from_uri(artifact_uri=model_uri)
    model_meta = Model.load(os.path.join(local_path, MLMODEL_FILE_NAME))

    is_port_connectable = check_port_connectivity()
    pyfunc_backend = get_flavor_backend(
        local_path,
        env_manager=env_manager,
        install_mlflow=os.environ.get("MLFLOW_HOME") is not None,
        create_env_root_dir=not is_port_connectable,
    )
    _logger.info("Restoring model environment. This can take a few minutes.")
    # Set capture_output to True in Databricks so that when environment preparation fails, the
    # exception message of the notebook cell output will include child process command execution
    # stdout/stderr output.
    pyfunc_backend.prepare_env(model_uri=local_path, capture_output=is_in_databricks_runtime())
    if is_port_connectable:
        server_port = find_free_port()
        scoring_server_proc = pyfunc_backend.serve(
            model_uri=local_path,
            port=server_port,
            host="127.0.0.1",
            timeout=MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT.get(),
            enable_mlserver=False,
            synchronous=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        client = ScoringServerClient("127.0.0.1", server_port)
    else:
        scoring_server_proc = pyfunc_backend.serve_stdin(local_path)
        client = StdinScoringServerClient(scoring_server_proc)

    _logger.info(f"Scoring server process started at PID: {scoring_server_proc.pid}")
    try:
        client.wait_server_ready(timeout=90, scoring_server_proc=scoring_server_proc)
    except Exception as e:
        if scoring_server_proc.poll() is None:
            # the scoring server is still running but client can't connect to it.
            # kill the server.
            scoring_server_proc.kill()
        server_output, _ = scoring_server_proc.communicate(timeout=15)
        if isinstance(server_output, bytes):
            server_output = server_output.decode("UTF-8")
        raise MlflowException(
            "MLflow model server failed to launch, server process stdout and stderr are:\n"
            + server_output
        ) from e

    return _ServedPyFuncModel(
        model_meta=model_meta,
        client=client,
        server_pid=scoring_server_proc.pid,
        env_manager=env_manager,
    )


def _get_model_dependencies(model_uri, format="pip"):
    model_dir = _download_artifact_from_uri(model_uri)

    def get_conda_yaml_path():
        model_config = _get_flavor_configuration_from_ml_model_file(
            os.path.join(model_dir, MLMODEL_FILE_NAME), flavor_name=FLAVOR_NAME
        )
        return os.path.join(model_dir, _extract_conda_env(model_config[ENV]))

    if format == "pip":
        requirements_file = os.path.join(model_dir, _REQUIREMENTS_FILE_NAME)
        if os.path.exists(requirements_file):
            return requirements_file

        _logger.info(
            f"{_REQUIREMENTS_FILE_NAME} is not found in the model directory. Falling back to"
            f" extracting pip requirements from the model's 'conda.yaml' file. Conda"
            " dependencies will be ignored."
        )

        with open(get_conda_yaml_path()) as yf:
            conda_yaml = yaml.safe_load(yf)

        conda_deps = conda_yaml.get("dependencies", [])
        for index, dep in enumerate(conda_deps):
            if isinstance(dep, dict) and "pip" in dep:
                pip_deps_index = index
                break
        else:
            raise MlflowException(
                "No pip section found in conda.yaml file in the model directory.",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        pip_deps = conda_deps.pop(pip_deps_index)["pip"]
        tmp_dir = tempfile.mkdtemp()
        pip_file_path = os.path.join(tmp_dir, _REQUIREMENTS_FILE_NAME)
        with open(pip_file_path, "w") as f:
            f.write("\n".join(pip_deps) + "\n")

        if len(conda_deps) > 0:
            _logger.warning(
                "The following conda dependencies have been excluded from the environment file:"
                f" {', '.join(conda_deps)}."
            )

        return pip_file_path

    elif format == "conda":
        return get_conda_yaml_path()
    else:
        raise MlflowException(
            f"Illegal format argument '{format}'.", error_code=INVALID_PARAMETER_VALUE
        )


def get_model_dependencies(model_uri, format="pip"):
    """
    Downloads the model dependencies and returns the path to requirements.txt or conda.yaml file.

    .. warning::
        This API downloads all the model artifacts to the local filesystem. This may take
        a long time for large models. To avoid this overhead, use
        ``mlflow.artifacts.download_artifacts("<model_uri>/requirements.txt")`` or
        ``mlflow.artifacts.download_artifacts("<model_uri>/conda.yaml")`` instead.

    Args:
        model_uri: The uri of the model to get dependencies from.
        format: The format of the returned dependency file. If the ``"pip"`` format is
            specified, the path to a pip ``requirements.txt`` file is returned.
            If the ``"conda"`` format is specified, the path to a ``"conda.yaml"``
            file is returned . If the ``"pip"`` format is specified but the model
            was not saved with a ``requirements.txt`` file, the ``pip`` section
            of the model's ``conda.yaml`` file is extracted instead, and any
            additional conda dependencies are ignored. Default value is ``"pip"``.

    Returns:
        The local filesystem path to either a pip ``requirements.txt`` file
        (if ``format="pip"``) or a ``conda.yaml`` file (if ``format="conda"``)
        specifying the model's dependencies.
    """
    dep_file = _get_model_dependencies(model_uri, format)

    if format == "pip":
        prefix = "%" if _is_in_ipython_notebook() else ""
        _logger.info(
            "To install the dependencies that were used to train the model, run the "
            f"following command: '{prefix}pip install -r {dep_file}'."
        )
    return dep_file


@deprecated("mlflow.pyfunc.load_model", 1.0)
def load_pyfunc(model_uri, suppress_warnings=False):
    """
    Load a model stored in Python function format.

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

        suppress_warnings: If ``True``, non-fatal warning messages associated with the model
            loading process will be suppressed. If ``False``, these warning messages will be
            emitted.
    """
    return load_model(model_uri, suppress_warnings)


def _warn_potentially_incompatible_py_version_if_necessary(model_py_version=None):
    """
    Compares the version of Python that was used to save a given model with the version
    of Python that is currently running. If a major or minor version difference is detected,
    logs an appropriate warning.
    """
    if model_py_version is None:
        _logger.warning(
            "The specified model does not have a specified Python version. It may be"
            " incompatible with the version of Python that is currently running: Python %s",
            PYTHON_VERSION,
        )
    elif get_major_minor_py_version(model_py_version) != get_major_minor_py_version(PYTHON_VERSION):
        _logger.warning(
            "The version of Python that the model was saved in, `Python %s`, differs"
            " from the version of Python that is currently running, `Python %s`,"
            " and may be incompatible",
            model_py_version,
            PYTHON_VERSION,
        )


def _create_model_downloading_tmp_dir(should_use_nfs):
    root_tmp_dir = get_or_create_nfs_tmp_dir() if should_use_nfs else get_or_create_tmp_dir()

    root_model_cache_dir = os.path.join(root_tmp_dir, "models")
    os.makedirs(root_model_cache_dir, exist_ok=True)

    tmp_model_dir = tempfile.mkdtemp(dir=root_model_cache_dir)
    # mkdtemp creates a directory with permission 0o700
    # change it to be 0o770 to ensure it can be seen in spark UDF
    os.chmod(tmp_model_dir, 0o770)
    return tmp_model_dir


_MLFLOW_SERVER_OUTPUT_TAIL_LINES_TO_KEEP = 200


def _convert_spec_type_to_spark_type(spec_type):
    from pyspark.sql.types import ArrayType, StructField, StructType

    from mlflow.types.schema import Array, DataType, Object

    if isinstance(spec_type, DataType):
        return spec_type.to_spark()

    if isinstance(spec_type, Array):
        return ArrayType(_convert_spec_type_to_spark_type(spec_type.dtype))

    if isinstance(spec_type, Object):
        return StructType(
            [
                StructField(
                    property.name,
                    _convert_spec_type_to_spark_type(property.dtype),
                    # we set nullable to True for all properties
                    # to avoid some errors like java.lang.NullPointerException
                    # when the signature is not inferred based on correct data.
                )
                for property in spec_type.properties
            ]
        )


def _cast_output_spec_to_spark_type(spec):
    from pyspark.sql.types import ArrayType

    from mlflow.types.schema import ColSpec, DataType, TensorSpec

    # TODO: handle optional output columns.
    if isinstance(spec, ColSpec):
        return _convert_spec_type_to_spark_type(spec.type)
    elif isinstance(spec, TensorSpec):
        data_type = DataType.from_numpy_type(spec.type)
        if data_type is None:
            raise MlflowException(
                f"Model output tensor spec type {spec.type} is not supported in spark_udf.",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if len(spec.shape) == 1:
            return ArrayType(data_type.to_spark())
        elif len(spec.shape) == 2:
            return ArrayType(ArrayType(data_type.to_spark()))
        else:
            raise MlflowException(
                "Only 1D or 2D tensors are supported as spark_udf "
                f"return value, but model output '{spec.name}' has shape {spec.shape}.",
                error_code=INVALID_PARAMETER_VALUE,
            )
    else:
        raise MlflowException(
            f"Unknown schema output spec {spec}.", error_code=INVALID_PARAMETER_VALUE
        )


def _infer_spark_udf_return_type(model_output_schema):
    from pyspark.sql.types import StructField, StructType

    if len(model_output_schema.inputs) == 1:
        return _cast_output_spec_to_spark_type(model_output_schema.inputs[0])

    return StructType(
        [
            StructField(name=spec.name or str(i), dataType=_cast_output_spec_to_spark_type(spec))
            for i, spec in enumerate(model_output_schema.inputs)
        ]
    )


def _parse_spark_datatype(datatype: str):
    from pyspark.sql.functions import udf
    from pyspark.sql.session import SparkSession

    return_type = "boolean" if datatype == "bool" else datatype
    parsed_datatype = udf(lambda x: x, returnType=return_type).returnType

    if parsed_datatype.typeName() == "unparseddata":
        # For spark 3.5.x, `udf(lambda x: x, returnType=return_type).returnType`
        # returns UnparsedDataType, which is not compatible with signature inference.
        # Note: SparkSession.active only exists for spark >= 3.5.0
        schema = (
            SparkSession.active()
            .range(0)
            .select(udf(lambda x: x, returnType=return_type)("id"))
            .schema
        )
        return schema[0].dataType

    return parsed_datatype


def _is_none_or_nan(value):
    # The condition `isinstance(value, float)` is needed to avoid error
    # from `np.isnan(value)` if value is a non-numeric type.
    return value is None or isinstance(value, float) and np.isnan(value)


def _convert_array_values(values, result_type):
    """
    Convert list or numpy array values to spark dataframe column values.
    """
    from pyspark.sql.types import ArrayType, StructType

    if not isinstance(result_type, ArrayType):
        raise MlflowException.invalid_parameter_value(
            f"result_type must be ArrayType, got {result_type.simpleString()}",
        )

    spark_primitive_type_to_np_type = _get_spark_primitive_type_to_np_type()

    if type(result_type.elementType) in spark_primitive_type_to_np_type:
        np_type = spark_primitive_type_to_np_type[type(result_type.elementType)]
        # For array type result values, if provided value is None or NaN, regard it as a null array.
        # see https://github.com/mlflow/mlflow/issues/8986
        return None if _is_none_or_nan(values) else np.array(values, dtype=np_type)
    if isinstance(result_type.elementType, ArrayType):
        return [_convert_array_values(v, result_type.elementType) for v in values]
    if isinstance(result_type.elementType, StructType):
        return [_convert_struct_values(v, result_type.elementType) for v in values]

    raise MlflowException.invalid_parameter_value(
        "Unsupported array type field with element type "
        f"{result_type.elementType.simpleString()} in Array type.",
    )


@lru_cache
def _get_spark_primitive_types():
    from pyspark.sql import types

    return (
        types.IntegerType,
        types.LongType,
        types.FloatType,
        types.DoubleType,
        types.StringType,
        types.BooleanType,
    )


@lru_cache
def _get_spark_primitive_type_to_np_type():
    from pyspark.sql import types

    return {
        types.IntegerType: np.int32,
        types.LongType: np.int64,
        types.FloatType: np.float32,
        types.DoubleType: np.float64,
        types.BooleanType: np.bool_,
        types.StringType: np.str_,
    }


def _check_udf_return_struct_type(struct_type):
    from pyspark.sql.types import ArrayType, StructType

    primitive_types = _get_spark_primitive_types()

    for field in struct_type.fields:
        field_type = field.dataType
        if isinstance(field_type, primitive_types):
            continue

        if isinstance(field_type, ArrayType) and _check_udf_return_array_type(
            field_type, allow_struct=True
        ):
            continue

        if isinstance(field_type, StructType) and _check_udf_return_struct_type(field_type):
            continue

        return False

    return True


def _check_udf_return_array_type(array_type, allow_struct):
    from pyspark.sql.types import ArrayType, StructType

    elem_type = array_type.elementType
    primitive_types = _get_spark_primitive_types()

    if isinstance(elem_type, primitive_types):
        return True

    if isinstance(elem_type, ArrayType):
        return _check_udf_return_array_type(elem_type, allow_struct)

    if isinstance(elem_type, StructType):
        if allow_struct:
            # Array of struct values.
            return _check_udf_return_struct_type(elem_type)

        return False

    return False


def _check_udf_return_type(data_type):
    from pyspark.sql.types import ArrayType, StructType

    primitive_types = _get_spark_primitive_types()
    if isinstance(data_type, primitive_types):
        return True

    if isinstance(data_type, ArrayType):
        return _check_udf_return_array_type(data_type, allow_struct=True)

    if isinstance(data_type, StructType):
        return _check_udf_return_struct_type(data_type)

    return False


def _convert_struct_values(
    result: Union[pandas.DataFrame, Dict[str, Any]],
    result_type,
):
    """
    Convert spark StructType values to spark dataframe column values.
    """

    from pyspark.sql.types import ArrayType, StructType

    if not isinstance(result_type, StructType):
        raise MlflowException.invalid_parameter_value(
            f"result_type must be StructType, got {result_type.simpleString()}",
        )

    if not isinstance(result, (dict, pandas.DataFrame)):
        raise MlflowException.invalid_parameter_value(
            f"Unsupported result type {type(result)}, expected dict or pandas DataFrame",
        )

    spark_primitive_type_to_np_type = _get_spark_primitive_type_to_np_type()
    is_pandas_df = isinstance(result, pandas.DataFrame)
    result_dict = {}
    for field_name in result_type.fieldNames():
        field_type = result_type[field_name].dataType
        field_values = result[field_name]

        if type(field_type) in spark_primitive_type_to_np_type:
            np_type = spark_primitive_type_to_np_type[type(field_type)]
            if is_pandas_df:
                field_values = field_values.astype(np_type)
            else:
                field_values = (
                    None
                    if _is_none_or_nan(field_values)
                    else np.array(field_values, dtype=np_type).item()
                )
        elif isinstance(field_type, ArrayType):
            if is_pandas_df:
                field_values = pandas.Series(
                    _convert_array_values(field_value, field_type) for field_value in field_values
                )
            else:
                field_values = _convert_array_values(field_values, field_type)
        elif isinstance(field_type, StructType):
            if is_pandas_df:
                field_values = pandas.Series(
                    [
                        _convert_struct_values(field_value, field_type)
                        for field_value in field_values
                    ]
                )
            else:
                field_values = _convert_struct_values(field_values, field_type)
        else:
            raise MlflowException.invalid_parameter_value(
                f"Unsupported field type {field_type.simpleString()} in struct type.",
            )
        result_dict[field_name] = field_values

    if is_pandas_df:
        return pandas.DataFrame(result_dict)
    return result_dict


def _is_spark_connect():
    try:
        from pyspark.sql.utils import is_remote
    except ImportError:
        return False
    return is_remote()


def spark_udf(
    spark,
    model_uri,
    result_type=None,
    env_manager=_EnvManager.LOCAL,
    params: Optional[Dict[str, Any]] = None,
    extra_env: Optional[Dict[str, str]] = None,
):
    """
    A Spark UDF that can be used to invoke the Python function formatted model.

    Parameters passed to the UDF are forwarded to the model as a DataFrame where the column names
    are ordinals (0, 1, ...). On some versions of Spark (3.0 and above), it is also possible to
    wrap the input in a struct. In that case, the data will be passed as a DataFrame with column
    names given by the struct definition (e.g. when invoked as my_udf(struct('x', 'y')), the model
    will get the data as a pandas DataFrame with 2 columns 'x' and 'y').

    If a model contains a signature with tensor spec inputs, you will need to pass a column of
    array type as a corresponding UDF argument. The column values of which must be one dimensional
    arrays. The UDF will reshape the column values to the required shape with 'C' order
    (i.e. read / write the elements using C-like index order) and cast the values as the required
    tensor spec type.

    If a model contains a signature, the UDF can be called without specifying column name
    arguments. In this case, the UDF will be called with column names from signature, so the
    evaluation dataframe's column names must match the model signature's column names.

    The predictions are filtered to contain only the columns that can be represented as the
    ``result_type``. If the ``result_type`` is string or array of strings, all predictions are
    converted to string. If the result type is not an array type, the left most column with
    matching type is returned.

    NOTE: Inputs of type ``pyspark.sql.types.DateType`` are not supported on earlier versions of
    Spark (2.4 and below).

    .. code-block:: python
        :caption: Example

        from pyspark.sql.functions import struct

        predict = mlflow.pyfunc.spark_udf(spark, "/my/local/model")
        df.withColumn("prediction", predict(struct("name", "age"))).show()

    Args:
        spark: A SparkSession object.
        model_uri: The location, in URI format, of the MLflow model with the
            :py:mod:`mlflow.pyfunc` flavor. For example:

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

        result_type: the return type of the user-defined function. The value can be either a
            ``pyspark.sql.types.DataType`` object or a DDL-formatted type string. Only a primitive
            type, an array ``pyspark.sql.types.ArrayType`` of primitive type, or a struct type
            containing fields of above 2 kinds of types are allowed.
            If unspecified, it tries to infer result type from model signature
            output schema, if model output schema is not available, it fallbacks to use ``double``
            type.

            The following classes of result type are supported:

            - "int" or ``pyspark.sql.types.IntegerType``: The leftmost integer that can fit in an
              ``int32`` or an exception if there is none.

            - "long" or ``pyspark.sql.types.LongType``: The leftmost long integer that can fit in an
              ``int64`` or an exception if there is none.

            - ``ArrayType(IntegerType|LongType)``: All integer columns that can fit into the
              requested size.

            - "float" or ``pyspark.sql.types.FloatType``: The leftmost numeric result cast to
              ``float32`` or an exception if there is none.

            - "double" or ``pyspark.sql.types.DoubleType``: The leftmost numeric result cast to
              ``double`` or an exception if there is none.

            - ``ArrayType(FloatType|DoubleType)``: All numeric columns cast to the requested type or
              an exception if there are no numeric columns.

            - "string" or ``pyspark.sql.types.StringType``: The leftmost column converted to
              ``string``.

            - "boolean" or "bool" or ``pyspark.sql.types.BooleanType``: The leftmost column
              converted to ``bool`` or an exception if there is none.

            - ``ArrayType(StringType)``: All columns converted to ``string``.

            - "field1 FIELD1_TYPE, field2 FIELD2_TYPE, ...": A struct type containing multiple
              fields separated by comma, each field type must be one of types listed above.

        env_manager: The environment manager to use in order to create the python environment
            for model inference. Note that environment is only restored in the context
            of the PySpark UDF; the software environment outside of the UDF is
            unaffected. Default value is ``local``, and the following values are
            supported:

            - ``virtualenv``: Use virtualenv to restore the python environment that
              was used to train the model.
            - ``conda``: (Recommended) Use Conda to restore the software environment
              that was used to train the model.
            - ``local``: Use the current Python environment for model inference, which
              may differ from the environment used to train the model and may lead to
              errors or invalid predictions.

        params: Additional parameters to pass to the model for inference.

        extra_env: Extra environment variables to pass to the UDF executors.

    Returns:
        Spark UDF that applies the model's ``predict`` method to the data and returns a
        type specified by ``result_type``, which by default is a double.
    """

    # Scope Spark import to this method so users don't need pyspark to use non-Spark-related
    # functionality.
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import (
        ArrayType,
        BooleanType,
        DoubleType,
        FloatType,
        IntegerType,
        LongType,
        StringType,
    )
    from pyspark.sql.types import StructType as SparkStructType

    from mlflow.pyfunc.spark_model_cache import SparkModelCache
    from mlflow.utils._spark_utils import _SparkDirectoryDistributor

    is_spark_connect = _is_spark_connect()
    # Used in test to force install local version of mlflow when starting a model server
    mlflow_home = os.environ.get("MLFLOW_HOME")
    openai_env_vars = mlflow.openai._OpenAIEnvVar.read_environ()
    mlflow_testing = _MLFLOW_TESTING.get_raw()

    _EnvManager.validate(env_manager)

    if is_spark_connect:
        is_spark_in_local_mode = False
    else:
        # Check whether spark is in local or local-cluster mode
        # this case all executors and driver share the same filesystem
        is_spark_in_local_mode = spark.conf.get("spark.master").startswith("local")

    nfs_root_dir = get_nfs_cache_root_dir()
    should_use_nfs = nfs_root_dir is not None
    should_use_spark_to_broadcast_file = not (
        is_spark_in_local_mode or should_use_nfs or is_spark_connect
    )

    # For spark connect mode,
    # If client code is executed in databricks runtime and NFS is available,
    # we save model to NFS temp directory in the driver
    # and load the model in the executor.
    should_spark_connect_use_nfs = is_in_databricks_runtime() and should_use_nfs

    if (
        is_spark_connect
        and env_manager in (_EnvManager.VIRTUALENV, _EnvManager.CONDA)
        and not should_spark_connect_use_nfs
    ):
        raise MlflowException.invalid_parameter_value(
            f"Environment manager {env_manager!r} is not supported in Spark connect mode "
            "when either non-Databricks environment is in use or NFS is unavailable.",
        )

    local_model_path = _download_artifact_from_uri(
        artifact_uri=model_uri,
        output_path=_create_model_downloading_tmp_dir(should_use_nfs),
    )

    if env_manager == _EnvManager.LOCAL:
        # Assume spark executor python environment is the same with spark driver side.
        model_requirements = _get_pip_requirements_from_model_path(local_model_path)
        warn_dependency_requirement_mismatches(model_requirements)
        _logger.warning(
            'Calling `spark_udf()` with `env_manager="local"` does not recreate the same '
            "environment that was used during training, which may lead to errors or inaccurate "
            'predictions. We recommend specifying `env_manager="conda"`, which automatically '
            "recreates the environment that was used to train the model and performs inference "
            "in the recreated environment."
        )
    else:
        _logger.info(
            f"This UDF will use {env_manager} to recreate the model's software environment for "
            "inference. This may take extra time during execution."
        )
        if not sys.platform.startswith("linux"):
            # TODO: support killing mlflow server launched in UDF task when spark job canceled
            #  for non-linux system.
            #  https://stackoverflow.com/questions/53208/how-do-i-automatically-destroy-child-processes-in-windows
            _logger.warning(
                "In order to run inference code in restored python environment, PySpark UDF "
                "processes spawn MLflow Model servers as child processes. Due to system "
                "limitations with handling SIGKILL signals, these MLflow Model server child "
                "processes cannot be cleaned up if the Spark Job is canceled."
            )
    pyfunc_backend = get_flavor_backend(
        local_model_path,
        env_manager=env_manager,
        install_mlflow=os.environ.get("MLFLOW_HOME") is not None,
        create_env_root_dir=True,
    )
    if not should_use_spark_to_broadcast_file:
        # Prepare restored environment in driver side if possible.
        # Note: In databricks runtime, because databricks notebook cell output cannot capture
        # child process output, so that set capture_output to be True so that when `conda prepare
        # env` command failed, the exception message will include command stdout/stderr output.
        # Otherwise user have to check cluster driver log to find command stdout/stderr output.
        # In non-databricks runtime, set capture_output to be False, because the benefit of
        # "capture_output=False" is the output will be printed immediately, otherwise you have
        # to wait conda command fail and suddenly get all output printed (included in error
        # message).
        if env_manager != _EnvManager.LOCAL:
            pyfunc_backend.prepare_env(
                model_uri=local_model_path, capture_output=is_in_databricks_runtime()
            )
    else:
        # Broadcast local model directory to remote worker if needed.
        archive_path = SparkModelCache.add_local_model(spark, local_model_path)

    model_metadata = Model.load(os.path.join(local_model_path, MLMODEL_FILE_NAME))

    if result_type is None:
        if model_output_schema := model_metadata.get_output_schema():
            result_type = _infer_spark_udf_return_type(model_output_schema)
        else:
            _logger.warning(
                "No 'result_type' provided for spark_udf and the model does not "
                "have an output schema. 'result_type' is set to 'double' type."
            )
            result_type = DoubleType()
    else:
        if isinstance(result_type, str):
            result_type = _parse_spark_datatype(result_type)

    if not _check_udf_return_type(result_type):
        raise MlflowException.invalid_parameter_value(
            f"""Invalid 'spark_udf' result type: {result_type}.
It must be one of the following types:
Primitive types:
 - int
 - long
 - float
 - double
 - string
 - boolean
Compound types:
 - ND array of primitives / structs.
 - struct<field: primitive | array<primitive> | array<array<primitive>>, ...>:
   A struct with primitive, ND array<primitive/structs>,
   e.g., struct<a:int, b:array<int>>.
"""
        )
    params = _validate_params(params, model_metadata)

    def _predict_row_batch(predict_fn, args):
        input_schema = model_metadata.get_input_schema()
        args = list(args)
        if len(args) == 1 and isinstance(args[0], pandas.DataFrame):
            pdf = args[0]
        else:
            if input_schema is None:
                names = [str(i) for i in range(len(args))]
            else:
                names = input_schema.input_names()
                required_names = input_schema.required_input_names()
                if len(args) > len(names):
                    args = args[: len(names)]
                if len(args) < len(required_names):
                    raise MlflowException(
                        f"Model input is missing required columns. Expected {len(names)} required"
                        f" input columns {names}, but the model received only {len(args)} "
                        "unnamed input columns (Since the columns were passed unnamed they are"
                        " expected to be in the order specified by the schema)."
                    )
            pdf = pandas.DataFrame(
                data={
                    names[i]: arg
                    if isinstance(arg, pandas.Series)
                    # pandas_udf receives a StructType column as a pandas DataFrame.
                    # We need to convert it back to a dict of pandas Series.
                    else arg.apply(lambda row: row.to_dict(), axis=1)
                    for i, arg in enumerate(args)
                },
                columns=names,
            )

        result = predict_fn(pdf, params)

        if isinstance(result, dict):
            result = {k: list(v) for k, v in result.items()}

        if isinstance(result_type, ArrayType) and isinstance(result_type.elementType, ArrayType):
            result_values = _convert_array_values(result, result_type)
            return pandas.Series(result_values)

        if not isinstance(result, pandas.DataFrame):
            result = pandas.DataFrame([result]) if np.isscalar(result) else pandas.DataFrame(result)

        if isinstance(result_type, SparkStructType):
            return _convert_struct_values(result, result_type)

        elem_type = result_type.elementType if isinstance(result_type, ArrayType) else result_type

        if type(elem_type) == IntegerType:
            result = result.select_dtypes(
                [np.byte, np.ubyte, np.short, np.ushort, np.int32]
            ).astype(np.int32)

        elif type(elem_type) == LongType:
            result = result.select_dtypes([np.byte, np.ubyte, np.short, np.ushort, int]).astype(
                np.int64
            )

        elif type(elem_type) == FloatType:
            result = result.select_dtypes(include=(np.number,)).astype(np.float32)

        elif type(elem_type) == DoubleType:
            result = result.select_dtypes(include=(np.number,)).astype(np.float64)

        elif type(elem_type) == BooleanType:
            result = result.select_dtypes([bool, np.bool_]).astype(bool)

        if len(result.columns) == 0:
            raise MlflowException(
                message="The model did not produce any values compatible with the requested "
                f"type '{elem_type}'. Consider requesting udf with StringType or "
                "Arraytype(StringType).",
                error_code=INVALID_PARAMETER_VALUE,
            )

        if type(elem_type) == StringType:
            if Version(pandas.__version__) >= Version("2.1.0"):
                result = result.map(str)
            else:
                result = result.applymap(str)

        if type(result_type) == ArrayType:
            return pandas.Series(result.to_numpy().tolist())
        else:
            return result[result.columns[0]]

    result_type_hint = (
        pandas.DataFrame if isinstance(result_type, SparkStructType) else pandas.Series
    )

    tracking_uri = mlflow.get_tracking_uri()

    @pandas_udf(result_type)
    def udf(
        iterator: Iterator[Tuple[Union[pandas.Series, pandas.DataFrame], ...]],
    ) -> Iterator[result_type_hint]:
        # importing here to prevent circular import
        from mlflow.pyfunc.scoring_server.client import (
            ScoringServerClient,
            StdinScoringServerClient,
        )

        # Note: this is a pandas udf function in iteration style, which takes an iterator of
        # tuple of pandas.Series and outputs an iterator of pandas.Series.
        update_envs = {}
        if mlflow_home is not None:
            update_envs["MLFLOW_HOME"] = mlflow_home
        if openai_env_vars:
            update_envs.update(openai_env_vars)
        if mlflow_testing:
            update_envs[_MLFLOW_TESTING.name] = mlflow_testing
        if extra_env:
            update_envs.update(extra_env)

        #  use `modified_environ` to temporarily set the envs and restore them finally
        with modified_environ(update=update_envs):
            scoring_server_proc = None
            # set tracking_uri inside udf so that with spark_connect
            # we can load the model from correct path
            mlflow.set_tracking_uri(tracking_uri)

            if env_manager != _EnvManager.LOCAL:
                if should_use_spark_to_broadcast_file:
                    local_model_path_on_executor = _SparkDirectoryDistributor.get_or_extract(
                        archive_path
                    )
                    # Call "prepare_env" in advance in order to reduce scoring server launch time.
                    # So that we can use a shorter timeout when call `client.wait_server_ready`,
                    # otherwise we have to set a long timeout for `client.wait_server_ready` time,
                    # this prevents spark UDF task failing fast if other exception raised
                    # when scoring server launching.
                    # Set "capture_output" so that if "conda env create" command failed, the command
                    # stdout/stderr output will be attached to the exception message and included in
                    # driver side exception.
                    pyfunc_backend.prepare_env(
                        model_uri=local_model_path_on_executor, capture_output=True
                    )
                else:
                    local_model_path_on_executor = None

                if check_port_connectivity():
                    # launch scoring server
                    server_port = find_free_port()
                    host = "127.0.0.1"
                    scoring_server_proc = pyfunc_backend.serve(
                        model_uri=local_model_path_on_executor or local_model_path,
                        port=server_port,
                        host=host,
                        timeout=MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT.get(),
                        enable_mlserver=False,
                        synchronous=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )

                    client = ScoringServerClient(host, server_port)
                else:
                    scoring_server_proc = pyfunc_backend.serve_stdin(
                        model_uri=local_model_path_on_executor or local_model_path,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )
                    client = StdinScoringServerClient(scoring_server_proc)

                _logger.info("Using %s", client.__class__.__name__)

                server_tail_logs = collections.deque(
                    maxlen=_MLFLOW_SERVER_OUTPUT_TAIL_LINES_TO_KEEP
                )

                def server_redirect_log_thread_func(child_stdout):
                    for line in child_stdout:
                        decoded = line.decode() if isinstance(line, bytes) else line
                        server_tail_logs.append(decoded)
                        sys.stdout.write("[model server] " + decoded)

                server_redirect_log_thread = threading.Thread(
                    target=server_redirect_log_thread_func,
                    args=(scoring_server_proc.stdout,),
                    daemon=True,
                )
                server_redirect_log_thread.start()

                try:
                    client.wait_server_ready(timeout=90, scoring_server_proc=scoring_server_proc)
                except Exception as e:
                    err_msg = (
                        "During spark UDF task execution, mlflow model server failed to launch. "
                    )
                    if len(server_tail_logs) == _MLFLOW_SERVER_OUTPUT_TAIL_LINES_TO_KEEP:
                        err_msg += (
                            f"Last {_MLFLOW_SERVER_OUTPUT_TAIL_LINES_TO_KEEP} "
                            "lines of MLflow model server output:\n"
                        )
                    else:
                        err_msg += "MLflow model server output:\n"
                    err_msg += "".join(server_tail_logs)
                    raise MlflowException(err_msg) from e

                def batch_predict_fn(pdf, params=None):
                    if inspect.signature(client.invoke).parameters.get("params"):
                        return client.invoke(pdf, params=params).get_predictions()
                    _log_warning_if_params_not_in_predict_signature(_logger, params)
                    return client.invoke(pdf).get_predictions()

            elif env_manager == _EnvManager.LOCAL:
                if is_spark_connect and not should_spark_connect_use_nfs:
                    model_path = os.path.join(
                        tempfile.gettempdir(),
                        "mlflow",
                        insecure_hash.sha1(model_uri.encode()).hexdigest(),
                        # Use pid to avoid conflict when multiple spark UDF tasks
                        str(os.getpid()),
                    )
                    try:
                        loaded_model = mlflow.pyfunc.load_model(model_path)
                    except Exception:
                        os.makedirs(model_path, exist_ok=True)
                        loaded_model = mlflow.pyfunc.load_model(model_uri, dst_path=model_path)
                elif should_use_spark_to_broadcast_file:
                    loaded_model, _ = SparkModelCache.get_or_load(archive_path)
                else:
                    loaded_model = mlflow.pyfunc.load_model(local_model_path)

                def batch_predict_fn(pdf, params=None):
                    if inspect.signature(loaded_model.predict).parameters.get("params"):
                        return loaded_model.predict(pdf, params=params)
                    _log_warning_if_params_not_in_predict_signature(_logger, params)
                    return loaded_model.predict(pdf)

            try:
                for input_batch in iterator:
                    # If the UDF is called with only multiple arguments,
                    # the `input_batch` is a tuple which composes of several pd.Series/pd.DataFrame
                    # objects.
                    # If the UDF is called with only one argument,
                    # the `input_batch` instance will be an instance of `pd.Series`/`pd.DataFrame`,
                    if isinstance(input_batch, (pandas.Series, pandas.DataFrame)):
                        # UDF is called with only one argument
                        row_batch_args = (input_batch,)
                    else:
                        row_batch_args = input_batch

                    if len(row_batch_args[0]) > 0:
                        yield _predict_row_batch(batch_predict_fn, row_batch_args)
            finally:
                if scoring_server_proc is not None:
                    os.kill(scoring_server_proc.pid, signal.SIGTERM)

    udf.metadata = model_metadata

    @functools.wraps(udf)
    def udf_with_default_cols(*args):
        if len(args) == 0:
            input_schema = model_metadata.get_input_schema()
            if input_schema and len(input_schema.optional_input_names()) > 0:
                raise MlflowException(
                    message="Cannot apply UDF without column names specified when"
                    " model signature contains optional columns.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
            if input_schema and len(input_schema.inputs) > 0:
                if input_schema.has_input_names():
                    input_names = input_schema.input_names()
                    return udf(*input_names)
                else:
                    raise MlflowException(
                        message="Cannot apply udf because no column names specified. The udf "
                        f"expects {len(input_schema.inputs)} columns with types: "
                        "{input_schema.inputs}. Input column names could not be inferred from the"
                        " model signature (column names not found).",
                        error_code=INVALID_PARAMETER_VALUE,
                    )
            else:
                raise MlflowException(
                    "Attempting to apply udf on zero columns because no column names were "
                    "specified as arguments or inferred from the model signature.",
                    error_code=INVALID_PARAMETER_VALUE,
                )
        else:
            return udf(*args)

    return udf_with_default_cols


def _validate_function_python_model(python_model):
    if not (isinstance(python_model, PythonModel) or callable(python_model)):
        raise MlflowException(
            "`python_model` must be a PythonModel instance, callable object, or path to a script "
            "that uses set_model() to set a PythonModel instance or callable object.",
            error_code=INVALID_PARAMETER_VALUE,
        )

    if callable(python_model):
        num_args = len(inspect.signature(python_model).parameters)
        if num_args != 1:
            raise MlflowException(
                "When `python_model` is a callable object, it must accept exactly one argument. "
                f"Found {num_args} arguments.",
                error_code=INVALID_PARAMETER_VALUE,
            )


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name="scikit-learn"))
@trace_disabled  # Suppress traces for internal predict calls while saving model
def save_model(
    path,
    loader_module=None,
    data_path=None,
    code_path=None,  # deprecated
    code_paths=None,
    infer_code_paths=False,
    conda_env=None,
    mlflow_model=None,
    python_model=None,
    artifacts=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    model_config=None,
    example_no_conversion=None,
    streamable=None,
    resources: Optional[Union[str, List[Resource]]] = None,
    **kwargs,
):
    """
    save_model(path, loader_module=None, data_path=None, code_path=None, conda_env=None,\
               mlflow_model=Model(), python_model=None, artifacts=None)

    Save a Pyfunc model with custom inference logic and optional data dependencies to a path on the
    local filesystem.

    For information about the workflows that this method supports, please see :ref:`"workflows for
    creating custom pyfunc models" <pyfunc-create-custom-workflows>` and
    :ref:`"which workflow is right for my use case?" <pyfunc-create-custom-selecting-workflow>`.
    Note that the parameters for the second workflow: ``loader_module``, ``data_path`` and the
    parameters for the first workflow: ``python_model``, ``artifacts``, cannot be
    specified together.

    Args:
        path: The path to which to save the Python model.
        loader_module: The name of the Python module that is used to load the model
            from ``data_path``. This module must define a method with the prototype
            ``_load_pyfunc(data_path)``. If not ``None``, this module and its
            dependencies must be included in one of the following locations:

            - The MLflow library.
            - Package(s) listed in the model's Conda environment, specified by
              the ``conda_env`` parameter.
            - One or more of the files specified by the ``code_path`` parameter.

        data_path: Path to a file or directory containing model data.
        code_path: **Deprecated** The legacy argument for defining dependent code. This argument is
            replaced by ``code_paths`` and will be removed in a future version of MLflow.
        code_paths: {{ code_paths_pyfunc }}
        infer_code_paths: {{ infer_code_paths }}
        conda_env: {{ conda_env }}
        mlflow_model: :py:mod:`mlflow.models.Model` configuration to which to add the
            **python_function** flavor.
        python_model:
            An instance of a subclass of :class:`~PythonModel` or a callable object with a single
            argument (see the examples below). The passed-in object is serialized using the
            CloudPickle library. The python_model can also be a file path to the PythonModel
            which defines the model from code artifact rather than serializing the model object.
            Any dependencies of the class should be included in one of the
            following locations:

            - The MLflow library.
            - Package(s) listed in the model's Conda environment, specified by the ``conda_env``
              parameter.
            - One or more of the files specified by the ``code_path`` parameter.

            Note: If the class is imported from another module, as opposed to being defined in the
            ``__main__`` scope, the defining module should also be included in one of the listed
            locations.

            **Examples**

            Class model

            .. code-block:: python

                from typing import List, Dict
                import mlflow


                class MyModel(mlflow.pyfunc.PythonModel):
                    def predict(self, context, model_input: List[str], params=None) -> List[str]:
                        return [i.upper() for i in model_input]


                mlflow.pyfunc.save_model("model", python_model=MyModel(), input_example=["a"])
                model = mlflow.pyfunc.load_model("model")
                print(model.predict(["a", "b", "c"]))  # -> ["A", "B", "C"]

            Functional model

            .. note::
                Experimental: Functional model support is experimental and may change or be removed
                in a future release without warning.

            .. code-block:: python

                from typing import List
                import mlflow


                def predict(model_input: List[str]) -> List[str]:
                    return [i.upper() for i in model_input]


                mlflow.pyfunc.save_model("model", python_model=predict, input_example=["a"])
                model = mlflow.pyfunc.load_model("model")
                print(model.predict(["a", "b", "c"]))  # -> ["A", "B", "C"]

            Model from code

            .. note::
                Experimental: Model from code model support is experimental and may change or
                be removed in a future release without warning.

            .. code-block:: python

                # code.py
                from typing import List
                import mlflow


                class MyModel(mlflow.pyfunc.PythonModel):
                    def predict(self, context, model_input: List[str], params=None) -> List[str]:
                        return [i.upper() for i in model_input]


                mlflow.models.set_model(MyModel())

                # log_model.py
                import mlflow

                with mlflow.start_run():
                    model_info = mlflow.pyfunc.log_model(
                        artifact_path="model",
                        python_model="code.py",
                    )

            If the `predict` method or function has type annotations, MLflow automatically
            constructs a model signature based on the type annotations (unless the ``signature``
            argument is explicitly specified), and converts the input value to the specified type
            before passing it to the function. Currently, the following type annotations are
            supported:

                - ``List[str]``
                - ``List[Dict[str, str]]``

        artifacts: A dictionary containing ``<name, artifact_uri>`` entries. Remote artifact URIs
            are resolved to absolute filesystem paths, producing a dictionary of
            ``<name, absolute_path>`` entries. ``python_model`` can reference these
            resolved entries as the ``artifacts`` property of the ``context`` parameter
            in :func:`PythonModel.load_context() <mlflow.pyfunc.PythonModel.load_context>`
            and :func:`PythonModel.predict() <mlflow.pyfunc.PythonModel.predict>`.
            For example, consider the following ``artifacts`` dictionary::

                {
                    "my_file": "s3://my-bucket/path/to/my/file"
                }

            In this case, the ``"my_file"`` artifact is downloaded from S3. The
            ``python_model`` can then refer to ``"my_file"`` as an absolute filesystem
            path via ``context.artifacts["my_file"]``.

            If ``None``, no artifacts are added to the model.

        signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
            describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
            The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
            from datasets with valid model input (e.g. the training dataset with target
            column omitted) and valid model output (e.g. model predictions generated on
            the training dataset), for example:

            .. code-block:: python

                from mlflow.models import infer_signature

                train = df.drop_column("target_label")
                predictions = ...  # compute model predictions
                signature = infer_signature(train, predictions)
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}
        model_config: The model configuration to apply to the model. This configuration
            is available during model loading.

            .. Note:: Experimental: This parameter may change or be removed in a future
                                    release without warning.
        example_no_conversion: {{ example_no_conversion }}
        resources: A list of model resources or a resources.yaml file containing a list of
                    resources required to serve the model.

            .. Note:: Experimental: This parameter may change or be removed in a future
                                    release without warning.
    """
    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)
    _validate_pyfunc_model_config(model_config)
    _validate_and_prepare_target_save_path(path)

    with tempfile.TemporaryDirectory() as temp_dir:
        model_code_path = None
        if python_model:
            if isinstance(model_config, Path):
                model_config = os.fspath(model_config)

            if isinstance(model_config, str):
                model_config = _validate_and_get_model_config_from_file(model_config)

            if isinstance(python_model, Path):
                python_model = os.fspath(python_model)

            if isinstance(python_model, str):
                model_code_path = _validate_and_get_model_code_path(python_model, temp_dir)
                _validate_and_copy_file_to_directory(model_code_path, path, "code")
                python_model = _load_model_code_path(model_code_path, model_config)

            _validate_function_python_model(python_model)
            if callable(python_model) and all(
                a is None for a in (input_example, pip_requirements, extra_pip_requirements)
            ):
                raise MlflowException(
                    "If `python_model` is a callable object, at least one of `input_example`, "
                    "`pip_requirements`, or `extra_pip_requirements` must be specified."
                )

    mlflow_model = kwargs.pop("model", mlflow_model)
    if len(kwargs) > 0:
        raise TypeError(f"save_model() got unexpected keyword arguments: {kwargs}")

    if code_path is not None and code_paths is not None:
        raise MlflowException(
            "Both `code_path` and `code_paths` have been specified, which is not permitted."
        )
    if code_path is not None:
        # Alias for `code_path` deprecation
        code_paths = code_path
        warnings.warn(
            "The `code_path` argument is replaced by `code_paths` and is deprecated "
            "as of MLflow version 2.12.0. This argument will be removed in a future "
            "release of MLflow."
        )

    if code_paths is not None:
        if not isinstance(code_paths, list):
            raise TypeError(f"Argument code_path should be a list, not {type(code_paths)}")

    first_argument_set = {
        "loader_module": loader_module,
        "data_path": data_path,
    }
    second_argument_set = {
        "artifacts": artifacts,
        "python_model": python_model,
    }
    first_argument_set_specified = any(item is not None for item in first_argument_set.values())
    second_argument_set_specified = any(item is not None for item in second_argument_set.values())
    if first_argument_set_specified and second_argument_set_specified:
        raise MlflowException(
            message=(
                f"The following sets of parameters cannot be specified together:"
                f" {first_argument_set.keys()}  and {second_argument_set.keys()}."
                " All parameters in one set must be `None`. Instead, found"
                f" the following values: {first_argument_set} and {second_argument_set}"
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )
    elif (loader_module is None) and (python_model is None):
        msg = (
            "Either `loader_module` or `python_model` must be specified. A `loader_module` "
            "should be a python module. A `python_model` should be a subclass of PythonModel"
        )
        raise MlflowException(message=msg, error_code=INVALID_PARAMETER_VALUE)

    if mlflow_model is None:
        mlflow_model = Model()
    saved_example = None

    hints = None
    if signature is not None:
        if isinstance(python_model, ChatModel):
            raise MlflowException(
                "ChatModel subclasses have a standard signature that is set "
                "automatically. Please remove the `signature` parameter from "
                "the call to log_model() or save_model().",
                error_code=INVALID_PARAMETER_VALUE,
            )
        mlflow_model.signature = signature
    elif python_model is not None:
        if callable(python_model):
            input_arg_index = 0  # first argument
            if signature := _infer_signature_from_type_hints(
                python_model, input_arg_index, input_example=input_example
            ):
                mlflow_model.signature = signature
        elif isinstance(python_model, ChatModel):
            mlflow_model.signature = ModelSignature(
                CHAT_MODEL_INPUT_SCHEMA,
                CHAT_MODEL_OUTPUT_SCHEMA,
            )
            input_example = input_example or CHAT_MODEL_INPUT_EXAMPLE
            input_example, input_params = _split_input_data_and_params(input_example)

            if isinstance(input_example, list):
                params = ChatParams()
                messages = []
                for each_message in input_example:
                    if isinstance(each_message, ChatMessage):
                        messages.append(each_message)
                    else:
                        messages.append(ChatMessage(**each_message))
            else:
                # If the input example is a dictionary, convert it to ChatMessage format
                messages = [ChatMessage(**m) for m in input_example["messages"]]
                params = ChatParams(**{k: v for k, v in input_example.items() if k != "messages"})
            input_example = (
                {"messages": [m.to_dict() for m in messages]},
                {**params.to_dict(), **(input_params or {})},
            )

            # call load_context() first, as predict may depend on it
            _logger.info("Predicting on input example to validate output")
            context = PythonModelContext(artifacts, model_config)
            python_model.load_context(context)
            output = python_model.predict(context, messages, params)
            if not isinstance(output, ChatResponse):
                raise MlflowException(
                    "Failed to save ChatModel. Please ensure that the model's predict() method "
                    "returns a ChatResponse object. If your predict() method currently returns "
                    "a dict, you can instantiate a ChatResponse by unpacking the output, e.g. "
                    "`ChatResponse(**output)`",
                )
        elif isinstance(python_model, PythonModel):
            saved_example = _save_example(mlflow_model, input_example, path, example_no_conversion)
            input_arg_index = 1  # second argument
            if signature := _infer_signature_from_type_hints(
                python_model.predict,
                input_arg_index=input_arg_index,
                input_example=input_example,
            ):
                mlflow_model.signature = signature
            elif saved_example is not None:
                try:
                    context = PythonModelContext(artifacts, model_config)
                    python_model.load_context(context)
                    mlflow_model.signature = _infer_signature_from_input_example(
                        saved_example,
                        _PythonModelPyfuncWrapper(python_model, None, None),
                    )
                except Exception as e:
                    _logger.warning(f"Failed to infer model signature from input example. {e}")

    if metadata is not None:
        mlflow_model.metadata = metadata
    if saved_example is None:
        saved_example = _save_example(mlflow_model, input_example, path, example_no_conversion)

    with _get_dependencies_schemas() as dependencies_schemas:
        schema = dependencies_schemas.to_dict()
        if schema is not None:
            if mlflow_model.metadata is None:
                mlflow_model.metadata = {}
            mlflow_model.metadata.update(schema)

    if resources is not None:
        if isinstance(resources, (Path, str)):
            serialized_resource = _ResourceBuilder.from_yaml_file(resources)
        else:
            serialized_resource = _ResourceBuilder.from_resources(resources)

        mlflow_model.resources = serialized_resource

    if first_argument_set_specified:
        return _save_model_with_loader_module_and_data_path(
            path=path,
            loader_module=loader_module,
            data_path=data_path,
            code_paths=code_paths,
            conda_env=conda_env,
            mlflow_model=mlflow_model,
            pip_requirements=pip_requirements,
            extra_pip_requirements=extra_pip_requirements,
            model_config=model_config,
            streamable=streamable,
            infer_code_paths=infer_code_paths,
        )
    elif second_argument_set_specified:
        return mlflow.pyfunc.model._save_model_with_class_artifacts_params(
            path=path,
            signature=signature,
            hints=hints,
            python_model=python_model,
            artifacts=artifacts,
            conda_env=conda_env,
            code_paths=code_paths,
            mlflow_model=mlflow_model,
            pip_requirements=pip_requirements,
            extra_pip_requirements=extra_pip_requirements,
            model_config=model_config,
            streamable=streamable,
            model_code_path=model_code_path,
            infer_code_paths=infer_code_paths,
        )


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name="scikit-learn"))
@trace_disabled  # Suppress traces for internal predict calls while logging model
def log_model(
    artifact_path,
    loader_module=None,
    data_path=None,
    code_path=None,  # deprecated
    code_paths=None,
    infer_code_paths=False,
    conda_env=None,
    python_model=None,
    artifacts=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    model_config=None,
    example_no_conversion=None,
    streamable=None,
    resources: Optional[Union[str, List[Resource]]] = None,
):
    """
    Log a Pyfunc model with custom inference logic and optional data dependencies as an MLflow
    artifact for the current run.

    For information about the workflows that this method supports, see :ref:`Workflows for
    creating custom pyfunc models <pyfunc-create-custom-workflows>` and
    :ref:`Which workflow is right for my use case? <pyfunc-create-custom-selecting-workflow>`.
    You cannot specify the parameters for the second workflow: ``loader_module``, ``data_path``
    and the parameters for the first workflow: ``python_model``, ``artifacts`` together.

    Args:
        artifact_path: The run-relative artifact path to which to log the Python model.
        loader_module: The name of the Python module that is used to load the model
            from ``data_path``. This module must define a method with the prototype
            ``_load_pyfunc(data_path)``. If not ``None``, this module and its
            dependencies must be included in one of the following locations:

            - The MLflow library.
            - Package(s) listed in the model's Conda environment, specified by
              the ``conda_env`` parameter.
            - One or more of the files specified by the ``code_path`` parameter.

        data_path: Path to a file or directory containing model data.
        code_path: **Deprecated** The legacy argument for defining dependent code. This argument is
            replaced by ``code_paths`` and will be removed in a future version of MLflow.
        code_paths: {{ code_paths_pyfunc }}
        infer_code_paths: {{ infer_code_paths }}
        conda_env: {{ conda_env }}
        python_model:
            An instance of a subclass of :class:`~PythonModel` or a callable object with a single
            argument (see the examples below). The passed-in object is serialized using the
            CloudPickle library. The python_model can also be a file path to the PythonModel
            which defines the model from code artifact rather than serializing the model object.
            Any dependencies of the class should be included in one of the
            following locations:

            - The MLflow library.
            - Package(s) listed in the model's Conda environment, specified by the ``conda_env``
              parameter.
            - One or more of the files specified by the ``code_path`` parameter.

            Note: If the class is imported from another module, as opposed to being defined in the
            ``__main__`` scope, the defining module should also be included in one of the listed
            locations.

            **Examples**

            Class model

            .. code-block:: python

                from typing import List
                import mlflow


                class MyModel(mlflow.pyfunc.PythonModel):
                    def predict(self, context, model_input: List[str], params=None) -> List[str]:
                        return [i.upper() for i in model_input]


                with mlflow.start_run():
                    model_info = mlflow.pyfunc.log_model(
                        artifact_path="model",
                        python_model=MyModel(),
                    )


                loaded_model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
                print(loaded_model.predict(["a", "b", "c"]))  # -> ["A", "B", "C"]

            Functional model

            .. note::
                Experimental: Functional model support is experimental and may change or be removed
                in a future release without warning.

            .. code-block:: python

                from typing import List
                import mlflow


                def predict(model_input: List[str]) -> List[str]:
                    return [i.upper() for i in model_input]


                with mlflow.start_run():
                    model_info = mlflow.pyfunc.log_model(
                        artifact_path="model", python_model=predict, input_example=["a"]
                    )


                loaded_model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
                print(loaded_model.predict(["a", "b", "c"]))  # -> ["A", "B", "C"]

            Model from code

            .. note::
                Experimental: Model from code model support is experimental and may change or
                be removed in a future release without warning.

            .. code-block:: python

                # code.py
                from typing import List
                import mlflow


                class MyModel(mlflow.pyfunc.PythonModel):
                    def predict(self, context, model_input: List[str], params=None) -> List[str]:
                        return [i.upper() for i in model_input]


                mlflow.models.set_model(MyModel())

                # log_model.py
                import mlflow

                with mlflow.start_run():
                    model_info = mlflow.pyfunc.log_model(
                        artifact_path="model",
                        python_model="code.py",
                    )

            If the `predict` method or function has type annotations, MLflow automatically
            constructs a model signature based on the type annotations (unless the ``signature``
            argument is explicitly specified), and converts the input value to the specified type
            before passing it to the function. Currently, the following type annotations are
            supported:

                - ``List[str]``
                - ``List[Dict[str, str]]``

        artifacts: A dictionary containing ``<name, artifact_uri>`` entries. Remote artifact URIs
            are resolved to absolute filesystem paths, producing a dictionary of
            ``<name, absolute_path>`` entries. ``python_model`` can reference these
            resolved entries as the ``artifacts`` property of the ``context`` parameter
            in :func:`PythonModel.load_context() <mlflow.pyfunc.PythonModel.load_context>`
            and :func:`PythonModel.predict() <mlflow.pyfunc.PythonModel.predict>`.
            For example, consider the following ``artifacts`` dictionary::

                {"my_file": "s3://my-bucket/path/to/my/file"}

            In this case, the ``"my_file"`` artifact is downloaded from S3. The
            ``python_model`` can then refer to ``"my_file"`` as an absolute filesystem
            path via ``context.artifacts["my_file"]``.

            If ``None``, no artifacts are added to the model.
        registered_model_name: This argument may change or be removed in a
            future release without warning. If given, create a model
            version under ``registered_model_name``, also creating a
            registered model if one with the given name does not exist.

        signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
            describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
            The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
            from datasets with valid model input (e.g. the training dataset with target
            column omitted) and valid model output (e.g. model predictions generated on
            the training dataset), for example:

            .. code-block:: python

                from mlflow.models import infer_signature

                train = df.drop_column("target_label")
                predictions = ...  # compute model predictions
                signature = infer_signature(train, predictions)

        input_example: {{ input_example }}
        await_registration_for: Number of seconds to wait for the model version to finish
            being created and is in ``READY`` status. By default, the function
            waits for five minutes. Specify 0 or None to skip waiting.
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}

        model_config: The model configuration to apply to the model. This configuration
            is available during model loading.

            .. Note:: Experimental: This parameter may change or be removed in a future
                                    release without warning.
        example_no_conversion: {{ example_no_conversion }}
        resources: A list of model resources or a resources.yaml file containing a list of
                    resources required to serve the model.

            .. Note:: Experimental: This parameter may change or be removed in a future
                                    release without warning.


        streamable: A boolean value indicating if the model supports streaming prediction,
                    If None, MLflow will try to inspect if the model supports streaming
                    by checking if `predict_stream` method exists. Default None.

    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
        metadata of the logged model.
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.pyfunc,
        loader_module=loader_module,
        data_path=data_path,
        code_path=code_path,  # deprecated
        code_paths=code_paths,
        python_model=python_model,
        artifacts=artifacts,
        conda_env=conda_env,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
        model_config=model_config,
        example_no_conversion=example_no_conversion,
        streamable=streamable,
        resources=resources,
        infer_code_paths=infer_code_paths,
    )


def _save_model_with_loader_module_and_data_path(
    path,
    loader_module,
    data_path=None,
    code_paths=None,
    conda_env=None,
    mlflow_model=None,
    pip_requirements=None,
    extra_pip_requirements=None,
    model_config=None,
    streamable=None,
    infer_code_paths=False,
):
    """
    Export model as a generic Python function model.

    Args:
        path: The path to which to save the Python model.
        loader_module: The name of the Python module that is used to load the model
            from ``data_path``. This module must define a method with the prototype
            ``_load_pyfunc(data_path)``.
        data_path: Path to a file or directory containing model data.
        code_paths: A list of local filesystem paths to Python file dependencies (or directories
            containing file dependencies). These files are *prepended* to the system
            path before the model is loaded.
        conda_env: Either a dictionary representation of a Conda environment or the path to a
            Conda environment yaml file. If provided, this describes the environment
            this model should be run in.
        streamable: A boolean value indicating if the model supports streaming prediction,
                    None value also means not streamable.

    Returns:
        Model configuration containing model info.
    """

    data = None

    if data_path is not None:
        model_file = _copy_file_or_tree(src=data_path, dst=path, dst_dir="data")
        data = model_file

    if mlflow_model is None:
        mlflow_model = Model()

    streamable = streamable or False
    mlflow.pyfunc.add_to_model(
        mlflow_model,
        loader_module=loader_module,
        code=None,
        data=data,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        model_config=model_config,
        streamable=streamable,
    )
    if size := get_total_file_size(path):
        mlflow_model.model_size_bytes = size
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    code_dir_subpath = _validate_infer_and_copy_code_paths(
        code_paths, path, infer_code_paths, FLAVOR_NAME
    )
    mlflow_model.flavors[FLAVOR_NAME][CODE] = code_dir_subpath

    # `mlflow_model.code` is updated, re-generate `MLmodel` file.
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            # To ensure `_load_pyfunc` can successfully load the model during the dependency
            # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
            inferred_reqs = mlflow.models.infer_pip_requirements(
                path,
                FLAVOR_NAME,
                fallback=default_reqs,
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs,
            pip_requirements,
            extra_pip_requirements,
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `constraints.txt` if necessary
    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    # Save `requirements.txt`
    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))
    return mlflow_model
