"""
The ``mlflow.onnx`` module provides APIs for logging and loading ONNX models in the MLflow Model
format. This module exports MLflow Models with the following flavors:

ONNX (native) format
    This is the main flavor that can be loaded back as an ONNX model object.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""

from __future__ import absolute_import

import os
import yaml
import numpy as np

import pandas as pd

from mlflow import pyfunc
from mlflow.models import Model
import mlflow.tracking
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import experimental
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration

FLAVOR_NAME = "onnx"


@experimental
def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    import onnx
    import onnxruntime
    return _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=[
            "onnx=={}".format(onnx.__version__),
            # The ONNX pyfunc representation requires the OnnxRuntime
            # inference engine. Therefore, the conda environment must
            # include OnnxRuntime
            "onnxruntime=={}".format(onnxruntime.__version__),
        ],
        additional_conda_channels=None,
    )


@experimental
def save_model(onnx_model, path, conda_env=None, mlflow_model=Model()):
    """
    Save an ONNX model to a path on the local file system.

    :param onnx_model: ONNX model to be saved.
    :param path: Local path where the model is to be saved.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If `None`, the default
                      :func:`get_default_conda_env()` environment is added to the model.
                      The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.6.0',
                                'onnx=1.4.1',
                                'onnxruntime=0.3.0'
                            ]
                        }

    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
    """
    import onnx

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException(
            message="Path '{}' already exists".format(path),
            error_code=RESOURCE_ALREADY_EXISTS)
    os.makedirs(path)
    model_data_subpath = "model.onnx"
    model_data_path = os.path.join(path, model_data_subpath)

    # Save onnx-model
    onnx.save_model(onnx_model, model_data_path)

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = get_default_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.onnx",
                        data=model_data_subpath, env=conda_env_subpath)
    mlflow_model.add_flavor(FLAVOR_NAME, onnx_version=onnx.__version__, data=model_data_subpath)
    mlflow_model.save(os.path.join(path, "MLmodel"))


def _load_model(model_file):
    import onnx

    onnx_model = onnx.load(model_file)
    # Check Formation
    onnx.checker.check_model(onnx_model)
    return onnx_model


class _OnnxModelWrapper:
    def __init__(self, path):
        import onnxruntime
        self.rt = onnxruntime.InferenceSession(path)
        assert len(self.rt.get_inputs()) >= 1
        self.inputs = [
            (inp.name, inp.type) for inp in self.rt.get_inputs()
        ]
        self.output_names = [
            outp.name for outp in self.rt.get_outputs()
        ]

    @staticmethod
    def _cast_float64_to_float32(dataframe, column_names):
        for input_name in column_names:
            if dataframe[input_name].values.dtype == np.float64:
                dataframe[input_name] = dataframe[input_name].values.astype(np.float32)
        return dataframe

    @experimental
    def predict(self, dataframe):
        """
        :param dataframe: A Pandas DataFrame that is converted to a collection of ONNX Runtime
                          inputs. If the underlying ONNX model only defines a *single* input
                          tensor, the DataFrame's values are converted to a NumPy array
                          representation using the `DataFrame.values()
                          <https://pandas.pydata.org/pandas-docs/stable/reference/api/
                          pandas.DataFrame.values.html#pandas.DataFrame.values>`_ method. If the
                          underlying ONNX model defines *multiple* input tensors, each column
                          of the DataFrame is converted to a NumPy array representation.
                          The corresponding NumPy array representation is then passed to the
                          ONNX Runtime. For more information about the ONNX Runtime, see
                          `<https://github.com/microsoft/onnxruntime>`_.
        :return: A Pandas DataFrame output. Each column of the DataFrame corresponds to an
                 output tensor produced by the underlying ONNX model.
        """
        # ONNXRuntime throws the following exception for some operators when the input
        # dataframe contains float64 values. Unfortunately, even if the original user-supplied
        # dataframe did not contain float64 values, the serialization/deserialization between the
        # client and the scoring server can introduce 64-bit floats. This is being tracked in
        # https://github.com/mlflow/mlflow/issues/1286. Meanwhile, we explicitly cast the input to
        # 32-bit floats when needed. TODO: Remove explicit casting when issue #1286 is fixed.
        if len(self.inputs) > 1:
            cols = [name for (name, type) in self.inputs if type == 'tensor(float)']
        else:
            cols = dataframe.columns if self.inputs[0][1] == 'tensor(float)' else []

        dataframe = _OnnxModelWrapper._cast_float64_to_float32(dataframe, cols)
        if len(self.inputs) > 1:
            feed_dict = {
                name: dataframe[name].values
                for (name, _) in self.inputs
            }
        else:
            feed_dict = {self.inputs[0][0]: dataframe.values}

        predicted = self.rt.run(self.output_names, feed_dict)
        return pd.DataFrame.from_dict(
            {c: p.reshape(-1) for (c, p) in zip(self.output_names, predicted)})


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    return _OnnxModelWrapper(path)


@experimental
def load_model(model_uri):
    """
    Load an ONNX model from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model, for example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``

                      For more information about supported URI schemes, see the
                      `Artifacts Documentation <https://www.mlflow.org/docs/latest/
                      tracking.html#artifact-stores>`_.

    :return: An ONNX model instance.

    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    onnx_model_artifacts_path = os.path.join(local_model_path, flavor_conf["data"])
    return _load_model(model_file=onnx_model_artifacts_path)


@experimental
def log_model(onnx_model, artifact_path, conda_env=None, registered_model_name=None):
    """
    Log an ONNX model as an MLflow artifact for the current run.

    :param onnx_model: ONNX model to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If `None`, the default
                      :func:`get_default_conda_env()` environment is added to the model.
                      The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.6.0',
                                'onnx=1.4.1',
                                'onnxruntime=0.3.0'
                            ]
                        }
    :param registered_model_name: Note:: Experimental: This argument may change or be removed in a
                                  future release without warning. If given, create a model
                                  version under ``registered_model_name``, also creating a
                                  registered model if one with the given name does not exist.
    """
    Model.log(artifact_path=artifact_path, flavor=mlflow.onnx,
              onnx_model=onnx_model, conda_env=conda_env,
              registered_model_name=registered_model_name)
