# -*- coding: utf-8 -*-

"""
The ``mlflow.pyfunc`` module defines a generic :ref:`filesystem format <pyfunc-filesystem-format>`
for Python models and provides utilities for saving to and loading from this format. The format is
self contained in the sense that it includes all necessary information for anyone to load it and
use it. Dependencies are either stored directly with the model or referenced via a Conda
environment.

The ``mlflow.pyfunc`` module also defines utilities for creating custom ``pyfunc`` models
using frameworks and inference logic that may not be natively included in MLflow. See
:ref:`pyfunc-create-custom`.

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

- Optionally, any additional parameters necessary for interpreting the serialized model in
  ``pyfunc`` format.

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

.. _pyfunc-inference-api:

*************
Inference API
*************

The convention for pyfunc models is to have a ``predict`` method or function with the following
signature::

    predict(model_input: pandas.DataFrame) -> [numpy.ndarray | pandas.Series | pandas.DataFrame]

This convention is relied on by other MLflow components.

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

- If the user has already collected all of their model data in a single location, the second
  workflow allows it to be saved in MLflow format directly, without enumerating constituent
  artifacts.
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
import mlflow.pyfunc.utils
from mlflow.tracking.fluent import active_run, log_artifacts
from mlflow import tracking
from mlflow.models import Model
from mlflow.pyfunc.model import PythonModel, PythonModelContext,\
    DEFAULT_CONDA_ENV
from mlflow.utils import PYTHON_VERSION, get_major_minor_py_version
from mlflow.utils.file_utils import TempDir, _copy_file_or_tree
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, RESOURCE_ALREADY_EXISTS

FLAVOR_NAME = "python_function"
MAIN = "loader_module"
CODE = "code"
DATA = "data"
ENV = "env"
PY_VERSION = "python_version"

_logger = logging.getLogger(__name__)


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
    :param kwargs: Additional key-value pairs to include in the ``pyfunc`` flavor specification.
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
        mlflow.pyfunc.utils._add_code_to_system_path(code_path=code_path)
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


def save_model(dst_path, loader_module=None, data_path=None, code_path=None, conda_env=None,
               model=Model(), python_model=None, artifacts=None):
    """
    Create a custom Pyfunc model, incorporating custom inference logic and data dependencies.

    For information about the workflows that this method supports, please see :ref:`"workflows for
    creating custom pyfunc models" <pyfunc-create-custom-workflows>` and
    :ref:`"which workflow is right for my use case?" <pyfunc-create-custom-selecting-workflow>`.
    Note that the parameters for the first workflow: ``loader_module``, ``data_path`` and the
    parameters for the second workflow: ``python_model``, ``artifacts``, cannot be
    specified together.

    :param dst_path: The path to which to save the Python model.
    :param loader_module: The name of the Python module that will be used to load the model
                          from ``data_path``. This module must define a method with the prototype
                          ``_load_pyfunc(data_path)``. If not *None*, this module and its
                          dependencies must be included in one of the following locations:

                          - The MLflow library.
                          - Package(s) listed in the model's Conda environment, specified by
                            the ``conda_env`` parameter.
                          - One or more of the files specified by the ``code_path`` parameter.

    :param data_path: Path to a file or directory containing model data.
    :param code_path: A list of local filesystem paths to Python file dependencies (or directories
                      containing file dependencies). These files will be *prepended* to the system
                      path before the model is loaded.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. This decribes the environment this model should
                      be run in. If ``python_model`` is not ``None``, the Conda environment must
                      at least specify the dependencies contained in
                      :data:`mlflow.pyfunc.DEFAULT_CONDA_ENV`. If `None`, the default
                      :data:`mlflow.pyfunc.DEFAULT_CONDA_ENV` environment will be added to the
                      model. The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'cloudpickle==0.5.8'
                            ]
                        }

    :param python_model: An instance of a subclass of :class:`~PythonModel`. This class will be
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
                      will be resolved to absolute filesystem paths, producing a dictionary of
                      ``<name, absolute_path>`` entries. ``python_model`` can reference these
                      resolved entries as the ``artifacts`` property of the ``context`` parameter
                      in :func:`PythonModel.load_context() <mlflow.pyfunc.PythonModel.load_context>`
                      and :func:`PythonModel.predict() <mlflow.pyfunc.PythonModel.predict>`.
                      For example, consider the following ``artifacts`` dictionary::

                        {
                            "my_file": "s3://my-bucket/path/to/my/file"
                        }

                      In this case, the ``"my_file"`` artifact will be downloaded from S3. The
                      ``python_model`` can then refer to ``"my_file"`` as an absolute filesystem
                      path via ``context.artifacts["my_file"]``.

                      If *None*, no artifacts will be added to the model.
    """
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
                    second_set_entries=second_argument_set)),
            error_code=INVALID_PARAMETER_VALUE)
    elif (loader_module is None) and (python_model is None):
        raise MlflowException(
            message="Either `loader_module` or `python_model` must be specified!",
            error_code=INVALID_PARAMETER_VALUE)

    if first_argument_set_specified:
        return _save_model_with_loader_module_and_data_path(
                path=dst_path, loader_module=loader_module, data_path=data_path,
                code_paths=code_path, conda_env=conda_env, mlflow_model=model)
    elif second_argument_set_specified:
        return mlflow.pyfunc.model._save_model_with_class_artifacts_params(
            path=dst_path, python_model=python_model, artifacts=artifacts, conda_env=conda_env,
            code_paths=code_path, mlflow_model=model)


def log_model(artifact_path, loader_module=None, data_path=None, code_path=None, conda_env=None,
              python_model=None, artifacts=None):
    """
    Create a custom Pyfunc model, incorporating custom inference logic and data dependencies.

    For information about the workflows that this method supports, see :ref:`Workflows for
    creating custom pyfunc models <pyfunc-create-custom-workflows>` and
    :ref:`Which workflow is right for my use case? <pyfunc-create-custom-selecting-workflow>`.
    You cannot specify the parameters for the first workflow: ``loader_module``, ``data_path``
    and the parameters for the second workflow: ``python_model``, ``artifacts`` together.

    :param artifact_path: The run-relative artifact path to which to log the Python model.
    :param loader_module: The name of the Python module that will be used to load the model
                          from ``data_path``. This module must define a method with the prototype
                          ``_load_pyfunc(data_path)``. If not *None*, this module and its
                          dependencies must be included in one of the following locations:

                          - The MLflow library.
                          - Package(s) listed in the model's Conda environment, specified by
                            the ``conda_env`` parameter.
                          - One or more of the files specified by the ``code_path`` parameter.

    :param data_path: Path to a file or directory containing model data.
    :param code_path: A list of local filesystem paths to Python file dependencies (or directories
                      containing file dependencies). These files will be *prepended* to the system
                      path before the model is loaded.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. This decribes the environment this model should
                      be run in. If ``python_model`` is not *None*, the Conda environment must
                      at least specify the dependencies contained in
                      :data:`mlflow.pyfunc.DEFAULT_CONDA_ENV`. If `None`, the default
                      :data:`mlflow.pyfunc.DEFAULT_CONDA_ENV` environment will be added to the
                      model. The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'cloudpickle==0.5.8'
                            ]
                        }

    :param python_model: An instance of a subclass of :class:`~PythonModel`. This class will be
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
                      will be resolved to absolute filesystem paths, producing a dictionary of
                      ``<name, absolute_path>`` entries. ``python_model`` can reference these
                      resolved entries as the ``artifacts`` property of the ``context`` parameter
                      in :func:`PythonModel.load_context() <mlflow.pyfunc.PythonModel.load_context>`
                      and :func:`PythonModel.predict() <mlflow.pyfunc.PythonModel.predict>`.
                      For example, consider the following ``artifacts`` dictionary::

                        {
                            "my_file": "s3://my-bucket/path/to/my/file"
                        }

                      In this case, the ``"my_file"`` artifact will be downloaded from S3. The
                      ``python_model`` can then refer to ``"my_file"`` as an absolute filesystem
                      path via ``context.artifacts["my_file"]``.

                      If *None*, no artifacts will be added to the model.
    """
    with TempDir() as tmp:
        local_path = tmp.path(artifact_path)
        run_id = active_run().info.run_uuid
        save_model(dst_path=local_path, model=Model(artifact_path=artifact_path, run_id=run_id),
                   loader_module=loader_module, data_path=data_path, code_path=code_path,
                   conda_env=conda_env, python_model=python_model, artifacts=artifacts)
        log_artifacts(local_path, artifact_path)


def _save_model_with_loader_module_and_data_path(path, loader_module, data_path=None,
                                                 code_paths=None, conda_env=None,
                                                 mlflow_model=Model()):
    """
    Export model as a generic Python function model.
    :param path: The path to which to save the Python model.
    :param loader_module: The name of the Python module that will be used to load the model
                          from ``data_path``. This module must define a method with the prototype
                          ``_load_pyfunc(data_path)``.
    :param data_path: Path to a file or directory containing model data.
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                      containing file dependencies). These files will be *prepended* to the system
                      path before the model is loaded.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decribes the environment
                      this model should be run in.
    :return: Model configuration containing model info.
    """
    if os.path.exists(path):
        raise MlflowException(
                message="Path '{}' already exists".format(path),
                error_code=RESOURCE_ALREADY_EXISTS)
    os.makedirs(path)

    code = None
    data = None
    env = None

    if data_path is not None:
        model_file = _copy_file_or_tree(src=data_path, dst=path, dst_dir="data")
        data = model_file

    if code_paths is not None:
        for code_path in code_paths:
            _copy_file_or_tree(src=code_path, dst=path, dst_dir="code")
        code = "code"

    if conda_env is not None:
        shutil.copy(src=conda_env, dst=os.path.join(path, "mlflow_env.yml"))
        env = "mlflow_env.yml"

    mlflow.pyfunc.add_to_model(
        mlflow_model, loader_module=loader_module, code=code, data=data, env=env)
    mlflow_model.save(os.path.join(path, 'MLmodel'))
    return mlflow_model


def get_module_loader_src(src_path, dst_path):
    """
    Generate Python source of the model loader.

    Model loader contains ``load_pyfunc`` method with no parameters. It hardcodes model
    loading of the given model into a Python source. This is done so that the exported model has no
    unnecessary dependencies on MLflow or any other configuration file format or parsing library.

    :param src_path: Current path to the model.
    :param dst_path: Relative or absolute path where the model will be stored in the deployment
                     environment.
    :return: Python source code of the model loader as string.

    """
    conf_path = os.path.join(src_path, "MLmodel")
    model = Model.load(conf_path)
    if FLAVOR_NAME not in model.flavors:
        raise Exception("Format '{format}' not found not in {path}.".format(format=FLAVOR_NAME,
                                                                            path=conf_path))
    conf = model.flavors[FLAVOR_NAME]
    update_path = ""
    if CODE in conf and conf[CODE]:
        src_code_path = os.path.join(src_path, conf[CODE])
        dst_code_path = os.path.join(dst_path, conf[CODE])
        code_path = ["os.path.abspath('%s')" % x for x in [dst_code_path] +
                     mlflow.pyfunc.utils._get_code_dirs(src_code_path, dst_code_path)]
        update_path = "sys.path = {} + sys.path; ".format("[%s]" % ",".join(code_path))

    data_path = os.path.join(dst_path, conf[DATA]) if (DATA in conf) else dst_path
    return loader_template.format(update_path=update_path, main=conf[MAIN], data_path=data_path)


loader_template = """

import importlib
import os
import sys

def load_pyfunc():
    {update_path}return importlib.import_module('{main}')._load_pyfunc('{data_path}')

"""
