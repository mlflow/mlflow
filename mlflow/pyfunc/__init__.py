# -*- coding: utf-8 -*-

"""Export / Import of generic python models.

This module defines generic filesystem format for python models and provides utilities
for saving and loading to and from this format. The format is self contained in a sense
that it includes all necessary information for anyone to load it and use it. Dependencies
are either stored directly with the model or referenced via a conda environment.

The convention for pyfunc models is to have a predict method or function with the following
signature

predict(data: pandas.DataFrame) -> numpy.ndarray | pandas.Series | pandas.DataFrame

This convention is relied upon by other mlflow components.

Pyfunc model format is defined as a directory structure containing all required data, code and
configuration:

./dst-path/

    ./MLmodel - config
    <code> - any code packaged with the model (specified in the conf file, see below)
    <data> - any data packaged with the model (specified in the conf file, see below)
    <env>  - conda environment definition (specified in the conf file, see below)

It must contain MLmodel file in its root with "python_function" format with the following
parameters:

- loader_module [required]:
         Python module that can load the model. Expected as module identifier
         e.g. ``mlflow.sklearn``, it will be imported via importlib.import_module.
         The imported module must contain function with the following signature:

              load_pyfunc(path: string) -> <pyfunc model>

         The path argument is specified by the data parameter and may refer to a file or directory.

- code [optional]:
        relative path to a directory containing the code packaged with this model.
        All files and directories inside this directory are added to the python path
        prior to importing the model loader.

- data [optional]:
         relative path to a file or directory containing model data.
         the path is passed to the model loader.

- env [optional]:
         relative path to an exported conda environment. If present this environment
         should be activated prior to running the model.

Example:

.. code:: shell

  >tree example/sklearn_iris/mlruns/run1/outputs/linear-lr
  ├── MLmodel
  ├── code
  │   ├── sklearn_iris.py
  │  
  ├── data
  │   └── model.pkl
  └── mlflow_env.yml

  >cat example/sklearn_iris/mlruns/run1/outputs/linear-lr/MLmodel
  python_function:
    code: code
    data: data/model.pkl
    env: mlflow_env.yml
    main: sklearn_iris
"""

import importlib
import os
import shutil
import sys
import pandas

from mlflow import tracking
from mlflow.models import Model
from mlflow.utils.file_utils import TempDir

FLAVOR_NAME = "python_function"
MAIN = "loader_module"
CODE = "code"
DATA = "data"
ENV = "env"


def add_to_model(model, loader_module, data=None, code=None, env=None):
    """ Add pyfunc spec to the model configuration.

    Defines pyfunc configuration schema. Caller can use this to create a valid pyfunc model flavor
    out of an existing directory structure. For example, other model flavors can use this to specify
    how to use their output as a pyfunc.

    NOTE: all paths are relative to the exported model root directory.

    :param loader_module:
    :param model: Existing servable
    :param data: to the model data
    :param code: path to the code dependencies
    :param env: conda environment
    :return: updated model configuration.
    """
    parms = {MAIN: loader_module}
    if code:
        parms[CODE] = code
    if data:
        parms[DATA] = data
    if env:
        parms[ENV] = env
    return model.add_flavor(FLAVOR_NAME, **parms)


def load_pyfunc(path, run_id=None):
    """ Load model stored in python-function format.
    """
    if run_id:
        path = tracking._get_model_log_dir(path, run_id)
    conf_path = os.path.join(path, "MLmodel")
    model = Model.load(conf_path)
    if FLAVOR_NAME not in model.flavors:
        raise Exception("Format '{format}' not found not in {path}.".format(format=FLAVOR_NAME,
                                                                            path=conf_path))
    conf = model.flavors[FLAVOR_NAME]
    if CODE in conf and conf[CODE]:
        code_path = os.path.join(path, conf[CODE])
        sys.path = [code_path] + _get_code_dirs(code_path) + sys.path
    data_path = os.path.join(path, conf[DATA]) if (DATA in conf) else path
    return importlib.import_module(conf[MAIN]).load_pyfunc(data_path)


def _get_code_dirs(src_code_path, dst_code_path=None):
    if not dst_code_path:
        dst_code_path = src_code_path
    return [(os.path.join(dst_code_path, x))
            for x in os.listdir(src_code_path) if not x.endswith(".py") and not x.endswith(".pyc")
            and not x == "__pycache__"]


def spark_udf(spark, path, run_id=None, result_type="double"):
    """Returns a Spark UDF that can be used to invoke the python-function formatted model.

    Note that parameters passed to the UDF will be forwarded to the model as a DataFrame
    where the names are simply ordinals (0, 1, ...).

    Example:
        predict = mlflow.pyfunc.spark_udf(spark, "/my/local/model")
        df.withColumn("prediction", predict("name", "age")).show()

    Args:
        spark (SparkSession): a SparkSession object
        path (str): A path containing a pyfunc model.
        run_id: Id of the run that produced this model.
        If provided, run_id is used to retrieve the model logged with mlflow.
        result_type (str): Spark UDF type returned by the model's prediction method. Default double
    """

    # Scope Spark import to this method so users don't need pyspark to use non-Spark-related
    # functionality.
    from mlflow.pyfunc.spark_model_cache import SparkModelCache
    from pyspark.sql.functions import pandas_udf

    if run_id:
        path = tracking._get_model_log_dir(path, run_id)

    archive_path = SparkModelCache.add_local_model(spark, path)

    def predict(*args):
        model = SparkModelCache.get_or_load(archive_path)
        schema = {str(i): arg for i, arg in enumerate(args)}
        # Explicitly pass order of columns to avoid lexicographic ordering (i.e., 10 < 2)
        columns = [str(i) for i, _ in enumerate(args)]
        pdf = pandas.DataFrame(schema, columns=columns)
        result = model.predict(pdf)
        return pandas.Series(result)

    return pandas_udf(predict, result_type)


def _copy_file_or_tree(src, dst, dst_dir):
    name = os.path.join(dst_dir, os.path.basename(os.path.abspath(src)))
    if dst_dir:
        os.mkdir(os.path.join(dst, dst_dir))
    if os.path.isfile(src):
        shutil.copy(src=src, dst=os.path.join(dst, name))
    else:
        shutil.copytree(src=src, dst=os.path.join(dst, name))
    return name


def save_model(dst_path, loader_module, data_path=None, code_path=(), conda_env=None,
               model=Model()):
    """Export model as a generic python-function model.

   Args:
       dst_path (str): path where the model is gonna be stored.
       loader_module (str): the module to be used to load the model.
       data_path (str): path to a file or directory containing model data.
       code_path (list[str]): list of paths (file or dir)
           contains code dependencies not present in the environment.
           every path in the code_path is added to the python path
           before the model is loaded.
       conda_env (str): path to the conda environment definition (.yml).
           This environment will be activated prior to running model code.
   Returns:
       model config (Servable) containing model info.
       :param dst_path:
       :param loader_module:
       :param data_path:
       :param code_path:
       :param conda_env:
       :param model:
    """
    if os.path.exists(dst_path):
        raise Exception("Path '{}' already exists".format(dst_path))
    os.makedirs(dst_path)
    code = None
    data = None
    env = None

    if data_path:
        model_file = _copy_file_or_tree(src=data_path, dst=dst_path, dst_dir="data")
        data = model_file

    if code_path:
        for path in code_path:
            _copy_file_or_tree(src=path, dst=dst_path, dst_dir="code")
        code = "code"

    if conda_env:
        shutil.copy(src=conda_env, dst=os.path.join(dst_path, "mlflow_env.yml"))
        env = "mlflow_env.yml"

    add_to_model(model, loader_module=loader_module, code=code, data=data, env=env)
    model.save(os.path.join(dst_path, 'MLmodel'))
    return model


def log_model(artifact_path, **kwargs):
    """Export the model in python-function form and log it with current mlflow tracking service.

    Model is exported by calling @save_model and logs the result with @tracking.log_output_files
    """
    with TempDir() as tmp:
        local_path = tmp.path(artifact_path)
        run_id = tracking.active_run().info.run_uuid
        if 'model' in kwargs:
            raise Exception("Unused argument 'model'. log_model creates a new model object")

        save_model(dst_path=local_path, model=Model(artifact_path=artifact_path, run_id=run_id),
                   **kwargs)
        tracking.log_artifacts(local_path, artifact_path)


def get_module_loader_src(src_path, dst_path):
    """ Generate python source of the model loader.

    Model loader contains load_pyfunc method with no parameters. It basically hardcodes model
    loading of the given model into a python source. This is done so that the exported model has no
    unnecessary dependencies on mlflow or any other configuration file format or parsing library.

    :param src_path: current path to the model
    :param dst_path: relative or absolute path where the model will be stored
                     in the deployment environment
    :return: python source code of the model loader as string.
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
        code_path = ["os.path.abspath('%s')" % x
                     for x in [dst_code_path] + _get_code_dirs(src_code_path, dst_code_path)]
        update_path = "sys.path = {} + sys.path; ".format("[%s]" % ",".join(code_path))

    data_path = os.path.join(dst_path, conf[DATA]) if (DATA in conf) else dst_path
    return loader_template.format(update_path=update_path, main=conf[MAIN], data_path=data_path)


loader_template = """

import importlib
import os
import sys

def load_pyfunc():
    {update_path}return importlib.import_module('{main}').load_pyfunc('{data_path}')

"""
