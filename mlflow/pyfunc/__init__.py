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
    model_input: [pandas.DataFrame, Dict[str, numpy.ndarray], numpy.ndarray]
  ) -> [numpy.ndarray | pandas.(Series | DataFrame)]

All PyFunc models will support `pandas.DataFrame` as input and DL PyFunc models will also support
tensor inputs in the form of Dict[str, numpy.ndarray] (named tensors) and `numpy.ndarrays`
(unnamed tensors).


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
              model_input: [pandas.DataFrame, Dict[str, numpy.ndarray], numpy.ndarray]
          ) -> [numpy.ndarray | pandas.(Series | DataFrame)]

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
   implementation in mlflow.keras <https://github.com/mlflow/mlflow/blob/
   74d75109aaf2975f5026104d6125bb30f4e3f744/mlflow/keras.py#L157-L187>`_.

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
"""

import importlib

import numpy as np
import os
import pandas
import yaml
from copy import deepcopy
import logging

from typing import Any, Union, List, Dict
import mlflow
import mlflow.pyfunc.model
import mlflow.pyfunc.utils
from mlflow.models import Model, ModelSignature, ModelInputExample
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.utils import _save_example
from mlflow.pyfunc.model import (  # pylint: disable=unused-import
    PythonModel,
    PythonModelContext,
    get_default_conda_env,
)
from mlflow.pyfunc.model import get_default_pip_requirements
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.types import DataType, Schema, TensorSpec
from mlflow.types.utils import clean_tensor_type
from mlflow.utils import PYTHON_VERSION, get_major_minor_py_version
from mlflow.utils.annotations import deprecated
from mlflow.utils.file_utils import TempDir, _copy_file_or_tree, write_to
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.utils.environment import (
    _validate_env_arguments,
    _process_pip_requirements,
    _process_conda_env,
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
)
from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS
from mlflow.exceptions import MlflowException
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)

FLAVOR_NAME = "python_function"
MAIN = "loader_module"
CODE = "code"
DATA = "data"
ENV = "env"
PY_VERSION = "python_version"

_logger = logging.getLogger(__name__)
PyFuncInput = Union[pandas.DataFrame, np.ndarray, List[Any], Dict[str, Any]]
PyFuncOutput = Union[pandas.DataFrame, pandas.Series, np.ndarray, list]


def add_to_model(model, loader_module, data=None, code=None, env=None, **kwargs):
    """
    Add a ``pyfunc`` spec to the model configuration.

    Defines ``pyfunc`` configuration schema. Caller can use this to create a valid ``pyfunc`` model
    flavor out of an existing directory structure. For example, other model flavors can use this to
    specify how to use their output as a ``pyfunc``.

    NOTE:

        All paths are relative to the exported model root directory.

    :param model: Existing model.
    :param loader_module: The module to be used to load the model.
    :param data: Path to the model data.
    :param code: Path to the code dependencies.
    :param env: Conda environment.
    :param req: pip requirements file.
    :param kwargs: Additional key-value pairs to include in the ``pyfunc`` flavor specification.
                   Values must be YAML-serializable.
    :return: Updated model configuration.
    """
    params = deepcopy(kwargs)
    params[MAIN] = loader_module
    params[PY_VERSION] = PYTHON_VERSION
    if code:
        params[CODE] = code
    if data:
        params[DATA] = data
    if env:
        params[ENV] = env

    return model.add_flavor(FLAVOR_NAME, **params)


def _load_model_env(path):
    """
    Get ENV file string from a model configuration stored in Python Function format.
    Returned value is a model-relative path to a Conda Environment file,
    or None if none was specified at model save time
    """
    return _get_flavor_configuration(model_path=path, flavor_name=FLAVOR_NAME).get(ENV, None)


def _enforce_mlflow_datatype(name, values: pandas.Series, t: DataType):
    """
    Enforce the input column type matches the declared in model input schema.

    The following type conversions are allowed:

    1. np.object -> string
    2. int -> long (upcast)
    3. float -> double (upcast)
    4. int -> double (safe conversion)
    5. np.datetime64[x] -> datetime (any precision)
    6. np.object -> datetime

    Any other type mismatch will raise error.
    """
    if values.dtype == np.object and t not in (DataType.binary, DataType.string):
        values = values.infer_objects()

    if t == DataType.string and values.dtype == np.object:
        #  NB: strings are by default parsed and inferred as objects, but it is
        # recommended to use StringDtype extension type if available. See
        #
        # `https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html`
        #
        # for more detail.
        try:
            return values.astype(t.to_pandas(), errors="raise")
        except ValueError:
            raise MlflowException(
                "Failed to convert column {0} from type {1} to {2}.".format(name, values.dtype, t)
            )

    # NB: Comparison of pandas and numpy data type fails when numpy data type is on the left hand
    # side of the comparison operator. It works, however, if pandas type is on the left hand side.
    # That is because pandas is aware of numpy.
    if t.to_pandas() == values.dtype or t.to_numpy() == values.dtype:
        # The types are already compatible => conversion is not necessary.
        return values

    if t == DataType.binary and values.dtype.kind == t.binary.to_numpy().kind:
        # NB: bytes in numpy have variable itemsize depending on the length of the longest
        # element in the array (column). Since MLflow binary type is length agnostic, we ignore
        # itemsize when matching binary columns.
        return values

    if t == DataType.datetime and values.dtype.kind == t.to_numpy().kind:
        # NB: datetime values have variable precision denoted by brackets, e.g. datetime64[ns]
        # denotes nanosecond precision. Since MLflow datetime type is precision agnostic, we
        # ignore precision when matching datetime columns.
        return values

    if t == DataType.datetime and values.dtype == np.object:
        # NB: Pyspark date columns get converted to np.object when converted to a pandas
        # DataFrame. To respect the original typing, we convert the column to datetime.
        try:
            return values.astype(np.datetime64, errors="raise")
        except ValueError:
            raise MlflowException(
                "Failed to convert column {0} from type {1} to {2}.".format(name, values.dtype, t)
            )

    numpy_type = t.to_numpy()
    if values.dtype.kind == numpy_type.kind:
        is_upcast = values.dtype.itemsize <= numpy_type.itemsize
    elif values.dtype.kind == "u" and numpy_type.kind == "i":
        is_upcast = values.dtype.itemsize < numpy_type.itemsize
    elif values.dtype.kind in ("i", "u") and numpy_type == np.float64:
        # allow (u)int => double conversion
        is_upcast = values.dtype.itemsize <= 6
    else:
        is_upcast = False

    if is_upcast:
        return values.astype(numpy_type, errors="raise")
    else:
        # NB: conversion between incompatible types (e.g. floats -> ints or
        # double -> float) are not allowed. While supported by pandas and numpy,
        # these conversions alter the values significantly.
        def all_ints(xs):
            return all([pandas.isnull(x) or int(x) == x for x in xs])

        hint = ""
        if (
            values.dtype == np.float64
            and numpy_type.kind in ("i", "u")
            and values.hasnans
            and all_ints(values)
        ):
            hint = (
                " Hint: the type mismatch is likely caused by missing values. "
                "Integer columns in python can not represent missing values and are therefore "
                "encoded as floats. The best way to avoid this problem is to infer the model "
                "schema based on a realistic data sample (training dataset) that includes missing "
                "values. Alternatively, you can declare integer columns as doubles (float64) "
                "whenever these columns may have missing values. See `Handling Integers With "
                "Missing Values <https://www.mlflow.org/docs/latest/models.html#"
                "handling-integers-with-missing-values>`_ for more details."
            )

        raise MlflowException(
            "Incompatible input types for column {0}. "
            "Can not safely convert {1} to {2}.{3}".format(name, values.dtype, numpy_type, hint)
        )


def _enforce_tensor_spec(values: np.ndarray, tensor_spec: TensorSpec):
    """
    Enforce the input tensor shape and type matches the provided tensor spec.
    """
    expected_shape = tensor_spec.shape
    actual_shape = values.shape
    if len(expected_shape) != len(actual_shape):
        raise MlflowException(
            "Shape of input {0} does not match expected shape {1}.".format(
                actual_shape, expected_shape
            )
        )
    for expected, actual in zip(expected_shape, actual_shape):
        if expected == -1:
            continue
        if expected != actual:
            raise MlflowException(
                "Shape of input {0} does not match expected shape {1}.".format(
                    actual_shape, expected_shape
                )
            )
    if clean_tensor_type(values.dtype) != tensor_spec.type:
        raise MlflowException(
            "dtype of input {0} does not match expected dtype {1}".format(
                values.dtype, tensor_spec.type
            )
        )
    return values


def _enforce_col_schema(pfInput: PyFuncInput, input_schema: Schema):
    """Enforce the input columns conform to the model's column-based signature."""
    if input_schema.has_input_names():
        input_names = input_schema.input_names()
    else:
        input_names = pfInput.columns[: len(input_schema.inputs)]
    input_types = input_schema.input_types()
    new_pfInput = pandas.DataFrame()
    for i, x in enumerate(input_names):
        new_pfInput[x] = _enforce_mlflow_datatype(x, pfInput[x], input_types[i])
    return new_pfInput


def _enforce_tensor_schema(pfInput: PyFuncInput, input_schema: Schema):
    """Enforce the input tensor(s) conforms to the model's tensor-based signature."""
    if input_schema.has_input_names():
        if isinstance(pfInput, dict):
            new_pfInput = dict()
            for col_name, tensor_spec in zip(input_schema.input_names(), input_schema.inputs):
                if not isinstance(pfInput[col_name], np.ndarray):
                    raise MlflowException(
                        "This model contains a tensor-based model signature with input names,"
                        " which suggests a dictionary input mapping input name to a numpy"
                        " array, but a dict with value type {0} was found.".format(
                            type(pfInput[col_name])
                        )
                    )
                new_pfInput[col_name] = _enforce_tensor_spec(pfInput[col_name], tensor_spec)
        elif isinstance(pfInput, pandas.DataFrame):
            new_pfInput = dict()
            for col_name, tensor_spec in zip(input_schema.input_names(), input_schema.inputs):
                new_pfInput[col_name] = _enforce_tensor_spec(
                    np.array(pfInput[col_name], dtype=tensor_spec.type), tensor_spec
                )
        else:
            raise MlflowException(
                "This model contains a tensor-based model signature with input names, which"
                " suggests a dictionary input mapping input name to tensor, but an input of"
                " type {0} was found.".format(type(pfInput))
            )
    else:
        if isinstance(pfInput, pandas.DataFrame):
            new_pfInput = _enforce_tensor_spec(pfInput.to_numpy(), input_schema.inputs[0])
        elif isinstance(pfInput, np.ndarray):
            new_pfInput = _enforce_tensor_spec(pfInput, input_schema.inputs[0])
        else:
            raise MlflowException(
                "This model contains a tensor-based model signature with no input names,"
                " which suggests a numpy array input, but an input of type {0} was"
                " found.".format(type(pfInput))
            )
    return new_pfInput


def _enforce_schema(pfInput: PyFuncInput, input_schema: Schema):
    """
    Enforces the provided input matches the model's input schema,

    For signatures with input names, we check there are no missing inputs and reorder the inputs to
    match the ordering declared in schema if necessary. Any extra columns are ignored.

    For column-based signatures, we make sure the types of the input match the type specified in
    the schema or if it can be safely converted to match the input schema.

    For tensor-based signatures, we make sure the shape and type of the input matches the shape
    and type specified in model's input schema.
    """
    if not input_schema.is_tensor_spec():
        if isinstance(pfInput, (list, np.ndarray, dict)):
            try:
                pfInput = pandas.DataFrame(pfInput)
            except Exception as e:
                raise MlflowException(
                    "This model contains a column-based signature, which suggests a DataFrame"
                    " input. There was an error casting the input data to a DataFrame:"
                    " {0}".format(str(e))
                )
        if not isinstance(pfInput, pandas.DataFrame):
            raise MlflowException(
                "Expected input to be DataFrame or list. Found: %s" % type(pfInput).__name__
            )

    if input_schema.has_input_names():
        # make sure there are no missing columns
        input_names = input_schema.input_names()
        expected_cols = set(input_names)
        actual_cols = set()
        if len(expected_cols) == 1 and isinstance(pfInput, np.ndarray):
            # for schemas with a single column, match input with column
            pfInput = {input_names[0]: pfInput}
            actual_cols = expected_cols
        elif isinstance(pfInput, pandas.DataFrame):
            actual_cols = set(pfInput.columns)
        elif isinstance(pfInput, dict):
            actual_cols = set(pfInput.keys())
        missing_cols = expected_cols - actual_cols
        extra_cols = actual_cols - expected_cols
        # Preserve order from the original columns, since missing/extra columns are likely to
        # be in same order.
        missing_cols = [c for c in input_names if c in missing_cols]
        extra_cols = [c for c in actual_cols if c in extra_cols]
        if missing_cols:
            raise MlflowException(
                "Model is missing inputs {0}."
                " Note that there were extra inputs: {1}".format(missing_cols, extra_cols)
            )
    elif not input_schema.is_tensor_spec():
        # The model signature does not specify column names => we can only verify column count.
        num_actual_columns = len(pfInput.columns)
        if num_actual_columns < len(input_schema.inputs):
            raise MlflowException(
                "Model inference is missing inputs. The model signature declares "
                "{0} inputs  but the provided value only has "
                "{1} inputs. Note: the inputs were not named in the signature so we can "
                "only verify their count.".format(len(input_schema.inputs), num_actual_columns)
            )

    return (
        _enforce_tensor_schema(pfInput, input_schema)
        if input_schema.is_tensor_spec()
        else _enforce_col_schema(pfInput, input_schema)
    )


class PyFuncModel(object):
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

    def __init__(self, model_meta: Model, model_impl: Any):
        if not hasattr(model_impl, "predict"):
            raise MlflowException("Model implementation is missing required predict method.")
        if not model_meta:
            raise MlflowException("Model is missing metadata.")
        self._model_meta = model_meta
        self._model_impl = model_impl

    def predict(self, data: PyFuncInput) -> PyFuncOutput:
        """
        Generate model predictions.

        If the model contains signature, enforce the input schema first before calling the model
        implementation with the sanitized input. If the pyfunc model does not include model schema,
        the input is passed to the model implementation as is. See `Model Signature Enforcement
        <https://www.mlflow.org/docs/latest/models.html#signature-enforcement>`_ for more details."

        :param data: Model input as one of pandas.DataFrame, numpy.ndarray, or
                     Dict[str, numpy.ndarray]
        :return: Model predictions as one of pandas.DataFrame, pandas.Series, numpy.ndarray or list.
        """
        input_schema = self.metadata.get_input_schema()
        if input_schema is not None:
            data = _enforce_schema(data, input_schema)
        return self._model_impl.predict(data)

    @property
    def metadata(self):
        """Model metadata."""
        if self._model_meta is None:
            raise MlflowException("Model is missing metadata.")
        return self._model_meta

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


def load_model(model_uri: str, suppress_warnings: bool = True, dst_path: str = None) -> PyFuncModel:
    """
    Load a model stored in Python function format.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                      artifact-locations>`_.
    :param suppress_warnings: If ``True``, non-fatal warning messages associated with the model
                              loading process will be suppressed. If ``False``, these warning
                              messages will be emitted.
    :param dst_path: The local filesystem path to which to download the model artifact.
                     This directory must already exist. If unspecified, a local output
                     path will be created.
    """
    local_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    model_meta = Model.load(os.path.join(local_path, MLMODEL_FILE_NAME))

    conf = model_meta.flavors.get(FLAVOR_NAME)
    if conf is None:
        raise MlflowException(
            'Model does not have the "{flavor_name}" flavor'.format(flavor_name=FLAVOR_NAME),
            RESOURCE_DOES_NOT_EXIST,
        )
    model_py_version = conf.get(PY_VERSION)
    if not suppress_warnings:
        _warn_potentially_incompatible_py_version_if_necessary(model_py_version=model_py_version)
    if CODE in conf and conf[CODE]:
        code_path = os.path.join(local_path, conf[CODE])
        mlflow.pyfunc.utils._add_code_to_system_path(code_path=code_path)
    data_path = os.path.join(local_path, conf[DATA]) if (DATA in conf) else local_path
    model_impl = importlib.import_module(conf[MAIN])._load_pyfunc(data_path)
    return PyFuncModel(model_meta=model_meta, model_impl=model_impl)


@deprecated("mlflow.pyfunc.load_model", 1.0)
def load_pyfunc(model_uri, suppress_warnings=False):
    """
    Load a model stored in Python function format.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                      artifact-locations>`_.

    :param suppress_warnings: If ``True``, non-fatal warning messages associated with the model
                              loading process will be suppressed. If ``False``, these warning
                              messages will be emitted.
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


def spark_udf(spark, model_uri, result_type="double"):
    """
    A Spark UDF that can be used to invoke the Python function formatted model.

    Parameters passed to the UDF are forwarded to the model as a DataFrame where the column names
    are ordinals (0, 1, ...). On some versions of Spark (3.0 and above), it is also possible to
    wrap the input in a struct. In that case, the data will be passed as a DataFrame with column
    names given by the struct definition (e.g. when invoked as my_udf(struct('x', 'y')), the model
    will get the data as a pandas DataFrame with 2 columns 'x' and 'y').

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

    :param spark: A SparkSession object.
    :param model_uri: The location, in URI format, of the MLflow model with the
                      :py:mod:`mlflow.pyfunc` flavor. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                      artifact-locations>`_.

    :param result_type: the return type of the user-defined function. The value can be either a
        ``pyspark.sql.types.DataType`` object or a DDL-formatted type string. Only a primitive
        type or an array ``pyspark.sql.types.ArrayType`` of primitive type are allowed.
        The following classes of result type are supported:

        - "int" or ``pyspark.sql.types.IntegerType``: The leftmost integer that can fit in an
          ``int32`` or an exception if there is none.

        - "long" or ``pyspark.sql.types.LongType``: The leftmost long integer that can fit in an
          ``int64`` or an exception if there is none.

        - ``ArrayType(IntegerType|LongType)``: All integer columns that can fit into the requested
          size.

        - "float" or ``pyspark.sql.types.FloatType``: The leftmost numeric result cast to
          ``float32`` or an exception if there is none.

        - "double" or ``pyspark.sql.types.DoubleType``: The leftmost numeric result cast to
          ``double`` or an exception if there is none.

        - ``ArrayType(FloatType|DoubleType)``: All numeric columns cast to the requested type or
          an exception if there are no numeric columns.

        - "string" or ``pyspark.sql.types.StringType``: The leftmost column converted to ``string``.

        - ``ArrayType(StringType)``: All columns converted to ``string``.

    :return: Spark UDF that applies the model's ``predict`` method to the data and returns a
             type specified by ``result_type``, which by default is a double.
    """

    # Scope Spark import to this method so users don't need pyspark to use non-Spark-related
    # functionality.
    import functools
    from mlflow.pyfunc.spark_model_cache import SparkModelCache
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import _parse_datatype_string
    from pyspark.sql.types import ArrayType, DataType as SparkDataType
    from pyspark.sql.types import DoubleType, IntegerType, FloatType, LongType, StringType

    if not isinstance(result_type, SparkDataType):
        result_type = _parse_datatype_string(result_type)

    elem_type = result_type
    if isinstance(elem_type, ArrayType):
        elem_type = elem_type.elementType

    supported_types = [IntegerType, LongType, FloatType, DoubleType, StringType]

    if not any([isinstance(elem_type, x) for x in supported_types]):
        raise MlflowException(
            message="Invalid result_type '{}'. Result type can only be one of or an array of one "
            "of the following types: {}".format(str(elem_type), str(supported_types)),
            error_code=INVALID_PARAMETER_VALUE,
        )

    with TempDir() as local_tmpdir:
        local_model_path = _download_artifact_from_uri(
            artifact_uri=model_uri, output_path=local_tmpdir.path()
        )
        archive_path = SparkModelCache.add_local_model(spark, local_model_path)
        model_metadata = Model.load(os.path.join(local_model_path, MLMODEL_FILE_NAME))

    def predict(*args):
        model = SparkModelCache.get_or_load(archive_path)
        input_schema = model.metadata.get_input_schema()
        pdf = None

        for x in args:
            if type(x) == pandas.DataFrame:
                if len(args) != 1:
                    raise Exception(
                        "If passing a StructType column, there should be only one "
                        "input column, but got %d" % len(args)
                    )
                pdf = x
        if pdf is None:
            args = list(args)
            if input_schema is None:
                names = [str(i) for i in range(len(args))]
            else:
                names = input_schema.input_names()
                if len(args) > len(names):
                    args = args[: len(names)]
                if len(args) < len(names):
                    raise MlflowException(
                        "Model input is missing columns. Expected {0} input columns {1},"
                        " but the model received only {2} unnamed input columns"
                        " (Since the columns were passed unnamed they are expected to be in"
                        " the order specified by the schema).".format(len(names), names, len(args))
                    )
            pdf = pandas.DataFrame(data={names[i]: x for i, x in enumerate(args)}, columns=names)

        result = model.predict(pdf)

        if not isinstance(result, pandas.DataFrame):
            result = pandas.DataFrame(data=result)

        elem_type = result_type.elementType if isinstance(result_type, ArrayType) else result_type

        if type(elem_type) == IntegerType:
            result = result.select_dtypes(
                [np.byte, np.ubyte, np.short, np.ushort, np.int32]
            ).astype(np.int32)
        elif type(elem_type) == LongType:
            result = result.select_dtypes([np.byte, np.ubyte, np.short, np.ushort, np.int, np.long])

        elif type(elem_type) == FloatType:
            result = result.select_dtypes(include=(np.number,)).astype(np.float32)

        elif type(elem_type) == DoubleType:
            result = result.select_dtypes(include=(np.number,)).astype(np.float64)

        if len(result.columns) == 0:
            raise MlflowException(
                message="The the model did not produce any values compatible with the requested "
                "type '{}'. Consider requesting udf with StringType or "
                "Arraytype(StringType).".format(str(elem_type)),
                error_code=INVALID_PARAMETER_VALUE,
            )

        if type(elem_type) == StringType:
            result = result.applymap(str)

        if type(result_type) == ArrayType:
            return pandas.Series(result.to_numpy().tolist())
        else:
            return result[result.columns[0]]

    udf = pandas_udf(predict, result_type)
    udf.metadata = model_metadata

    @functools.wraps(udf)
    def udf_with_default_cols(*args):
        if len(args) == 0:
            input_schema = model_metadata.get_input_schema()

            if input_schema and len(input_schema.inputs) > 0:
                if input_schema.has_input_names():
                    input_names = input_schema.input_names()
                    return udf(*input_names)
                else:
                    raise MlflowException(
                        message="Cannot apply udf because no column names specified. The udf "
                        "expects {} columns with types: {}. Input column names could not be "
                        "inferred from the model signature (column names not found).".format(
                            len(input_schema.inputs),
                            input_schema.inputs,
                        ),
                        error_code=INVALID_PARAMETER_VALUE,
                    )
            else:
                _logger.warning(
                    "Attempting to apply udf on zero columns because no column names were "
                    "specified as arguments or inferred from the model signature."
                )
                return udf()  # pylint: disable=no-value-for-parameter
        else:
            return udf(*args)

    return udf_with_default_cols


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name="scikit-learn"))
def save_model(
    path,
    loader_module=None,
    data_path=None,
    code_path=None,
    conda_env=None,
    mlflow_model=None,
    python_model=None,
    artifacts=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
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

    :param path: The path to which to save the Python model.
    :param loader_module: The name of the Python module that is used to load the model
                          from ``data_path``. This module must define a method with the prototype
                          ``_load_pyfunc(data_path)``. If not ``None``, this module and its
                          dependencies must be included in one of the following locations:

                          - The MLflow library.
                          - Package(s) listed in the model's Conda environment, specified by
                            the ``conda_env`` parameter.
                          - One or more of the files specified by the ``code_path`` parameter.

    :param data_path: Path to a file or directory containing model data.
    :param code_path: A list of local filesystem paths to Python file dependencies (or directories
                      containing file dependencies). These files are *prepended* to the system
                      path before the model is loaded.
    :param conda_env: {{ conda_env }}
    :param mlflow_model: :py:mod:`mlflow.models.Model` configuration to which to add the
                         **python_function** flavor.
    :param python_model: An instance of a subclass of :class:`~PythonModel`. This class is
                         serialized using the CloudPickle library. Any dependencies of the class
                         should be included in one of the following locations:

                            - The MLflow library.
                            - Package(s) listed in the model's Conda environment, specified by
                              the ``conda_env`` parameter.
                            - One or more of the files specified by the ``code_path`` parameter.

                         Note: If the class is imported from another module, as opposed to being
                         defined in the ``__main__`` scope, the defining module should also be
                         included in one of the listed locations.
    :param artifacts: A dictionary containing ``<name, artifact_uri>`` entries. Remote artifact URIs
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

    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example can be a Pandas DataFrame where the given
                          example will be serialized to json using the Pandas split-oriented
                          format, or a numpy array where the example will be serialized to json
                          by converting it to a list. Bytes are base64-encoded.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    """
    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    mlflow_model = kwargs.pop("model", mlflow_model)
    if len(kwargs) > 0:
        raise TypeError("save_model() got unexpected keyword arguments: {}".format(kwargs))
    if code_path is not None:
        if not isinstance(code_path, list):
            raise TypeError("Argument code_path should be a list, not {}".format(type(code_path)))

    first_argument_set = {
        "loader_module": loader_module,
        "data_path": data_path,
    }
    second_argument_set = {
        "artifacts": artifacts,
        "python_model": python_model,
    }
    first_argument_set_specified = any([item is not None for item in first_argument_set.values()])
    second_argument_set_specified = any([item is not None for item in second_argument_set.values()])
    if first_argument_set_specified and second_argument_set_specified:
        raise MlflowException(
            message=(
                "The following sets of parameters cannot be specified together: {first_set_keys}"
                " and {second_set_keys}. All parameters in one set must be `None`. Instead, found"
                " the following values: {first_set_entries} and {second_set_entries}".format(
                    first_set_keys=first_argument_set.keys(),
                    second_set_keys=second_argument_set.keys(),
                    first_set_entries=first_argument_set,
                    second_set_entries=second_argument_set,
                )
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )
    elif (loader_module is None) and (python_model is None):
        msg = (
            "Either `loader_module` or `python_model` must be specified. A `loader_module` "
            "should be a python module. A `python_model` should be a subclass of PythonModel"
        )
        raise MlflowException(message=msg, error_code=INVALID_PARAMETER_VALUE)

    if os.path.exists(path):
        raise MlflowException(
            message="Path '{}' already exists".format(path), error_code=RESOURCE_ALREADY_EXISTS
        )
    os.makedirs(path)
    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    if first_argument_set_specified:
        return _save_model_with_loader_module_and_data_path(
            path=path,
            loader_module=loader_module,
            data_path=data_path,
            code_paths=code_path,
            conda_env=conda_env,
            mlflow_model=mlflow_model,
            pip_requirements=pip_requirements,
            extra_pip_requirements=extra_pip_requirements,
        )
    elif second_argument_set_specified:
        return mlflow.pyfunc.model._save_model_with_class_artifacts_params(
            path=path,
            python_model=python_model,
            artifacts=artifacts,
            conda_env=conda_env,
            code_paths=code_path,
            mlflow_model=mlflow_model,
            pip_requirements=pip_requirements,
            extra_pip_requirements=extra_pip_requirements,
        )


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name="scikit-learn"))
def log_model(
    artifact_path,
    loader_module=None,
    data_path=None,
    code_path=None,
    conda_env=None,
    python_model=None,
    artifacts=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
):
    """
    Log a Pyfunc model with custom inference logic and optional data dependencies as an MLflow
    artifact for the current run.

    For information about the workflows that this method supports, see :ref:`Workflows for
    creating custom pyfunc models <pyfunc-create-custom-workflows>` and
    :ref:`Which workflow is right for my use case? <pyfunc-create-custom-selecting-workflow>`.
    You cannot specify the parameters for the second workflow: ``loader_module``, ``data_path``
    and the parameters for the first workflow: ``python_model``, ``artifacts`` together.

    :param artifact_path: The run-relative artifact path to which to log the Python model.
    :param loader_module: The name of the Python module that is used to load the model
                          from ``data_path``. This module must define a method with the prototype
                          ``_load_pyfunc(data_path)``. If not ``None``, this module and its
                          dependencies must be included in one of the following locations:

                          - The MLflow library.
                          - Package(s) listed in the model's Conda environment, specified by
                            the ``conda_env`` parameter.
                          - One or more of the files specified by the ``code_path`` parameter.

    :param data_path: Path to a file or directory containing model data.
    :param code_path: A list of local filesystem paths to Python file dependencies (or directories
                      containing file dependencies). These files are *prepended* to the system
                      path before the model is loaded.
    :param conda_env: {{ conda_env }}
    :param python_model: An instance of a subclass of :class:`~PythonModel`. This class is
                         serialized using the CloudPickle library. Any dependencies of the class
                         should be included in one of the following locations:

                            - The MLflow library.
                            - Package(s) listed in the model's Conda environment, specified by
                              the ``conda_env`` parameter.
                            - One or more of the files specified by the ``code_path`` parameter.

                         Note: If the class is imported from another module, as opposed to being
                         defined in the ``__main__`` scope, the defining module should also be
                         included in one of the listed locations.
    :param artifacts: A dictionary containing ``<name, artifact_uri>`` entries. Remote artifact URIs
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
    :param registered_model_name: This argument may change or be removed in a
                                  future release without warning. If given, create a model
                                  version under ``registered_model_name``, also creating a
                                  registered model if one with the given name does not exist.

    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature
                        train = df.drop_column("target_label")
                        predictions = ... # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example can be a Pandas DataFrame where the given
                          example will be serialized to json using the Pandas split-oriented
                          format, or a numpy array where the example will be serialized to json
                          by converting it to a list. Bytes are base64-encoded.
    :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.pyfunc,
        loader_module=loader_module,
        data_path=data_path,
        code_path=code_path,
        python_model=python_model,
        artifacts=artifacts,
        conda_env=conda_env,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
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
):
    """
    Export model as a generic Python function model.
    :param path: The path to which to save the Python model.
    :param loader_module: The name of the Python module that is used to load the model
                          from ``data_path``. This module must define a method with the prototype
                          ``_load_pyfunc(data_path)``.
    :param data_path: Path to a file or directory containing model data.
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                      containing file dependencies). These files are *prepended* to the system
                      path before the model is loaded.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decsribes the environment
                      this model should be run in.
    :return: Model configuration containing model info.
    """

    code = None
    data = None

    if data_path is not None:
        model_file = _copy_file_or_tree(src=data_path, dst=path, dst_dir="data")
        data = model_file

    if code_paths is not None:
        for code_path in code_paths:
            _copy_file_or_tree(src=code_path, dst=path, dst_dir="code")
        code = "code"

    if mlflow_model is None:
        mlflow_model = Model()

    mlflow.pyfunc.add_to_model(
        mlflow_model, loader_module=loader_module, code=code, data=data, env=_CONDA_ENV_FILE_NAME
    )
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
    return mlflow_model


loader_template = """

import importlib
import os
import sys

def load_pyfunc():
    {update_path}return importlib.import_module('{main}')._load_pyfunc('{data_path}')

"""
