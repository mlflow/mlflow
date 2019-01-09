# -*- coding: utf-8 -*-

"""
The ``mlflow.pyfunc`` module defines a generic filesystem format for Python models and provides
utilities for saving to and loading from this format. The format is self contained in the sense
that it includes all necessary information for anyone to load it and use it. Dependencies
are either stored directly with the model or referenced via a Conda environment.

The convention for pyfunc models is to have a ``predict`` method or function with the following
signature::

    predict(data: pandas.DataFrame) -> numpy.ndarray | pandas.Series | pandas.DataFrame

This convention is relied on by other MLflow components.

Pyfunc model format is defined as a directory structure containing all required data, code, and
configuration::

    ./dst-path/
        ./MLmodel: configuration
        <code>: code packaged with the model (specified in the MLmodel file)
        <data>: data packaged with the model (specified in the MLmodel file)
        <env>: Conda environment definition (specified in the MLmodel file)

* The directory structure may contain additional contents that can be referenced by the
  ``MLmodel`` configuration.

A Python model contains an ``MLmodel`` file in "python_function" format in its root with the
following parameters:

- loader_module [required]:
         Python module that can load the model. Expected as module identifier
         e.g. ``mlflow.sklearn``, it will be imported via ``importlib.import_module``.
         The imported module must contain function with the following signature::

          _load_pyfunc(path: string) -> <pyfunc model>

         The path argument is specified by the ``data`` parameter and may refer to a file or
         directory.

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

- **Optionally, any additional parameters necessary for interpreting the serialized model in pyfunc
  format.**

.. rubric:: Example

>>> tree example/sklearn_iris/mlruns/run1/outputs/linear-lr

::

  ├── MLmodel
  ├── code
  │   ├── sklearn_iris.py
  │  
  ├── data
  │   └── model.pkl
  └── mlflow_env.yml

>>> cat example/sklearn_iris/mlruns/run1/outputs/linear-lr/MLmodel

::

  python_function:
    code: code
    data: data/model.pkl
    loader_module: mlflow.sklearn
    env: mlflow_env.yml
    main: sklearn_iris
"""

import importlib
import logging
import numpy as np
import os
import pandas
import shutil
import sys
from copy import deepcopy

import mlflow
import mlflow.pyfunc.model
from mlflow.tracking.fluent import active_run, log_artifacts
from mlflow import tracking
from mlflow.models import Model
from mlflow.pyfunc.model import PythonModel, PythonModelContext,\
    DEFAULT_CONDA_ENV
from mlflow.utils import PYTHON_VERSION, get_major_minor_py_version
from mlflow.utils.file_utils import TempDir
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

FLAVOR_NAME = "python_function"
MAIN = "loader_module"
CODE = "code"
DATA = "data"
ENV = "env"
PY_VERSION = "python_version"

_logger = logging.getLogger(__name__)


def add_to_model(model, loader_module, data=None, code=None, env=None, **kwargs):
    """
    Add a pyfunc spec to the model configuration.

    Defines pyfunc configuration schema. Caller can use this to create a valid pyfunc model flavor
    out of an existing directory structure. For example, other model flavors can use this to specify
    how to use their output as a pyfunc.

    NOTE:

        All paths are relative to the exported model root directory.

    :param model: Existing model.
    :param loader_module: The module to be used to load the model.
    :param data: Path to the model data.
    :param code: Path to the code dependencies.
    :param env: Conda environment.
    :param kwargs: Additional key-value pairs to include in the pyfunc flavor specification.
                   Values must be YAML-serializable.
    :return: Updated model configuration.
    """
    parms = deepcopy(kwargs)
    parms[MAIN] = loader_module
    parms[PY_VERSION] = PYTHON_VERSION
    if code:
        parms[CODE] = code
    if data:
        parms[DATA] = data
    if env:
        parms[ENV] = env
    return model.add_flavor(FLAVOR_NAME, **parms)


def _load_model_env(path, run_id=None):
    """
    Get ENV file string from a model configuration stored in Python Function format.
    Returned value is a model-relative path to a Conda Environment file,
    or None if none was specified at model save time
    """
    if run_id is not None:
        path = tracking.utils._get_model_log_dir(path, run_id)
    return _get_flavor_configuration(model_path=path, flavor_name=FLAVOR_NAME).get(ENV, None)


def save_model(path, loader_module=None, data_path=None, code_paths=None, conda_env=None,
               mlflow_model=Model(), artifacts=None, parameters=None, model_class=None, **kwargs):
    deprecated_args_mapping = {
        "dst_path": "path",
        "code_path": "code_paths",
        "model": "mlflow_model",
    }
    for deprecated_arg_name, new_arg_name in deprecated_args_mapping.items():
        if deprecated_arg_name in kwargs:
            if locals()[new_arg_name] is not None:
                raise MlflowException(
                    "Deprecated argument with name `{deprecated_arg_name}` was specified along with"
                    " its equivalent: `{new_arg_name}`. Please use the new argument name"
                    " exclusively.".format(deprecated_arg_name=deprecated_arg_name,
                                           new_arg_name=new_arg_name))
            else:
                _logger.warn("The argument with name `{deprecated_arg_name}` has been deprecated"
                             " and will be removed in MLflow 0.9.0. Please use the equivalent"
                             " argument, `{new_arg_name}` instead.")
                locals()[new_arg_name] = kwargs[deprecated_arg_name]


    first_argument_set = {
        "loader_module": loader_module,
        "data_path": data_path,
    }
    second_argument_set = {
        "artifacts": artifacts,
        "parameters": parameters,
        "model_class": model_class,
    }
    first_argument_set_specified = any([item is not None for item in first_argument_set.values()])
    second_argument_set_specified = any([item is not None for item in second_argument_set.values()])
    if first_argument_set_specified and second_argument_set_specified:
        raise MlflowException(
            message=(
                "The following sets of arguments cannot be specified together: {first_set_keys}"
                " and {second_set_keys}. All arguments in one set must be `None`. Instead, found the"
                " following values: {first_set_entries} and {second_set_entries}".format(
                    first_set_keys=first_argument_set.keys(),
                    second_set_keys=second_argument_set.keys(),
                    first_set_entries=first_argument_set,
                    second_set_entries=second_argument_set)),
            error_code=INVALID_PARAMETER_VALUE)

    if loader_module is not None:
        return mlflow.pyfunc.model._save_model_with_loader_module_and_data_path(
            path=path, loader_module=loader_module, data_path=data_path,
            code_paths=code_paths, conda_env=conda_env, mlflow_model=mlflow_model)
    elif model_class is not None:
        return mlflow.pyfunc.model._save_model_with_class_artifacts_params(
            path=path, model_class=model_class, artifacts=artifacts, parameters=parameters,
            conda_env=conda_env, code_paths=code_paths, mlflow_model=mlflow_model)
    elif data_path is not None:
        raise MlflowException(
            message="`data_path` was specified, but the `loader_module` argument was not provided.",
            error_code=INVALID_PARAMETER_VALUE)
    elif artifacts is not None or parameters is not None:
        raise MlflowException(
            message=("`artifacts` or `parameters` was specified, but the `model_class` argument was"
                     " not provided."),
            error_code=INVALID_PARAMETER_VALUE)


def log_model(artifact_path, model_class, artifacts=None, parameters=None, conda_env=None,
              code_paths=None):
    """
    :param path: The run-relative artifact path to which to log the Python model.
    :param model_class: A ``type`` object referring to a subclass of
                        :class:`~PythonModel`, or the fully-qualified name of such a subclass.
                        ``model_class`` defines how the model is loaded and how it performs
                        inference.
    :param artifacts: A dictionary containing ``<name, artifact_uri>`` entries. Remote artifact URIs
                      will be resolved to absolute filesystem paths, producing a dictionary of
                      ``<name, absolute_path>`` entries. ``model_class`` can reference these
                      resolved entries as the ``artifacts`` property of the ``context`` attribute.
                      For example, consider the following ``artifacts`` dictionary::

                        {
                            "my_file": "s3://my-bucket/path/to/my/file"
                        }

                      In this case, the ``"my_file"`` artifact will be downloaded from S3. The
                      ``model_class`` can then refer to ``"my_file"`` as an absolute filesystem path
                      via ``self.context.artifacts["my_file"]``.
    :param parameters: A dictionary containing ``<name, python_object>`` entries. ``python_object``
                       may be any Python object that is serializable with CloudPickle.
                       ``model_class`` can reference these resolved entries as the ``parameters``
                       property of the ``context`` attribute. For example, consider the following
                       ``parameters`` dictionary::

                         {
                             "my_list": range(10)
                         }

                       The ``model_class`` can refer to the Python list named ``"my_list"`` as
                       ``self.context.parameters["my_list"]``.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :data:`mlflow.pyfunc.DEFAULT_CONDA_ENV`. If `None`, the default
                      :data:`mlflow.pyfunc.DEFAULT_CONDA_ENV` environment will be added to the
                      model. The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'cloudpickle=0.5.8'
                            ]
                        }
    :param code_paths: A list of paths to Python file dependencies that are required by
                       instances of ``model_class``.
    :param mlflow_model: The model configuration to which to add the ``mlflow.pyfunc`` flavor.
    """
    return Model.log(artifact_path=artifact_path, flavor=mlflow.pyfunc, artifacts=artifacts,
                     parameters=parameters, model_class=model_class, conda_env=conda_env,
                     code_paths=code_paths)


def load_pyfunc(path, run_id=None, suppress_warnings=False):
    """
    Load a model stored in Python function format.

    :param path: Path to the model.
    :param run_id: MLflow run ID.
    :param suppress_warnings: If True, non-fatal warning messages associated with the model
                              loading process will be suppressed. If False, these warning messages
                              will be emitted.
    """
    if run_id is not None:
        path = tracking.utils._get_model_log_dir(path, run_id)
    conf = _get_flavor_configuration(model_path=path, flavor_name=FLAVOR_NAME)
    model_py_version = conf.get(PY_VERSION)
    if not suppress_warnings:
        _warn_potentially_incompatible_py_version_if_necessary(model_py_version=model_py_version)
    if CODE in conf and conf[CODE]:
        code_path = os.path.join(path, conf[CODE])
        sys.path = [code_path] + _get_code_dirs(code_path) + sys.path
    data_path = os.path.join(path, conf[DATA]) if (DATA in conf) else path
    return importlib.import_module(conf[MAIN])._load_pyfunc(data_path)


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
            PYTHON_VERSION)
    elif get_major_minor_py_version(model_py_version) != get_major_minor_py_version(PYTHON_VERSION):
        _logger.warning(
            "The version of Python that the model was saved in, `Python %s`, differs"
            " from the version of Python that is currently running, `Python %s`,"
            " and may be incompatible",
            model_py_version, PYTHON_VERSION)


def _get_code_dirs(src_code_path, dst_code_path=None):
    """
    Obtains the names of the subdirectories contained under the specified source code
    path and joins them with the specified destination code path.

    :param src_code_path: The path of the source code directory for which to list subdirectories.
    :param dst_code_path: The destination directory path to which subdirectory names should be
                          joined.
    """
    if not dst_code_path:
        dst_code_path = src_code_path
    return [(os.path.join(dst_code_path, x)) for x in os.listdir(src_code_path)
            if os.path.isdir(x) and not x == "__pycache__"]


def spark_udf(spark, path, run_id=None, result_type="double"):
    """
    A Spark UDF that can be used to invoke the Python function formatted model.

    Parameters passed to the UDF are forwarded to the model as a DataFrame where the names are
    ordinals (0, 1, ...).

    The predictions are filtered to contain only the columns that can be represented as the
    ``result_type``. If the ``result_type`` is string or array of strings, all predictions are
    converted to string. If the result type is not an array type, the left most column with
    matching type will be returned.

    >>> predict = mlflow.pyfunc.spark_udf(spark, "/my/local/model")
    >>> df.withColumn("prediction", predict("name", "age")).show()

    :param spark: A SparkSession object.
    :param path: A path containing a :py:mod:`mlflow.pyfunc` model.
    :param run_id: ID of the run that produced this model. If provided, ``run_id`` is used to
                   retrieve the model logged with MLflow.
    :param result_type: the return type of the user-defined function. The value can be either a
                        :class:`pyspark.sql.types.DataType` object or a DDL-formatted type string.
                        Only a primitive type or an array (pyspark.sql.types.ArrayType) of primitive
                        types are allowed. The following classes of result type are supported:
                        - "int" or pyspark.sql.types.IntegerType: The leftmost integer that can fit
                          in int32 result is returned or exception is raised if there is none.
                        - "long" or pyspark.sql.types.LongType: The leftmost long integer that can
                          fit in int64 result is returned or exception is raised if there is none.
                        - ArrayType(IntegerType|LongType): Return all integer columns that can fit
                          into the requested size.
                        - "float" or pyspark.sql.types.FloatType: The leftmost numeric result cast
                          to float32 is returned or exception is raised if there is none.
                        - "double" or pyspark.sql.types.DoubleType: The leftmost numeric result cast
                          to double is returned or exception is raised if there is none..
                        - ArrayType(FloatType|DoubleType): Return all numeric columns cast to the
                          requested type. Exception is raised if there are no numeric columns.
                        - "string" or pyspark.sql.types.StringType: Result is the leftmost column
                          converted to string.
                        - ArrayType(StringType): Return all columns converted to string.

    :return: Spark UDF which will apply model's prediction method to the data. Default double.
    """

    # Scope Spark import to this method so users don't need pyspark to use non-Spark-related
    # functionality.
    from mlflow.pyfunc.spark_model_cache import SparkModelCache
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import _parse_datatype_string
    from pyspark.sql.types import ArrayType, DataType
    from pyspark.sql.types import DoubleType, IntegerType, FloatType, LongType, StringType

    if not isinstance(result_type, DataType):
        result_type = _parse_datatype_string(result_type)

    elem_type = result_type
    if isinstance(elem_type, ArrayType):
        elem_type = elem_type.elementType

    supported_types = [IntegerType, LongType, FloatType, DoubleType, StringType]

    if not any([isinstance(elem_type, x) for x in supported_types]):
        raise MlflowException(
            message="Invalid result_type '{}'. Result type can only be one of or an array of one "
                    "of the following types types: {}".format(str(elem_type), str(supported_types)),
            error_code=INVALID_PARAMETER_VALUE)

    if run_id:
        path = tracking.utils._get_model_log_dir(path, run_id)

    archive_path = SparkModelCache.add_local_model(spark, path)

    def predict(*args):
        model = SparkModelCache.get_or_load(archive_path)
        schema = {str(i): arg for i, arg in enumerate(args)}
        # Explicitly pass order of columns to avoid lexicographic ordering (i.e., 10 < 2)
        columns = [str(i) for i, _ in enumerate(args)]
        pdf = pandas.DataFrame(schema, columns=columns)
        result = model.predict(pdf)
        if not isinstance(result, pandas.DataFrame):
            result = pandas.DataFrame(data=result)

        elif type(elem_type) == IntegerType:
            result = result.select_dtypes([np.byte, np.ubyte, np.short, np.ushort,
                                           np.int32]).astype(np.int32)

        elif type(elem_type) == LongType:
            result = result.select_dtypes([np.byte, np.ubyte, np.short, np.ushort, np.int, np.long])

        elif type(elem_type) == FloatType:
            result = result.select_dtypes(include=np.number).astype(np.float32)

        elif type(elem_type) == DoubleType:
            result = result.select_dtypes(include=np.number).astype(np.float64)

        if len(result.columns) == 0:
            raise MlflowException(
                message="The the model did not produce any values compatible with the requested "
                        "type '{}'. Consider requesting udf with StringType or "
                        "Arraytype(StringType).".format(str(elem_type)),
                error_code=INVALID_PARAMETER_VALUE)

        if type(elem_type) == StringType:
            result = result.applymap(str)

        if type(result_type) == ArrayType:
            return pandas.Series([row[1].values for row in result.iterrows()])
        else:
            return result[result.columns[0]]

    return pandas_udf(predict, result_type)
