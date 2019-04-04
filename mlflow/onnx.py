from __future__ import absolute_import

import os
import yaml
import numpy as np
import onnx
import onnxruntime as onnxrt

import pandas as pd

from mlflow import pyfunc
from mlflow.models import Model
import mlflow.tracking
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration

FLAVOR_NAME = "onnx"

DEFAULT_CONDA_ENV = _mlflow_conda_env(
    additional_conda_deps=None,
    additional_pip_deps=[
        "onnx=={}".format(onnx.__version__),
        # The ONNX pyfunc representation requires the OnnxRuntime
        # inference engine. Therefore, the conda environment must
        # include OnnxRuntime
        "onnxruntime=={}".format(onnxrt.__version__),
    ],
    additional_conda_channels=None,
)


def save_model(onnx_model, path, conda_env=None, mlflow_model=Model()):
    """
    Save an ONNX model to a path on the local file system.

    :param onnx_model: ONNX model to be saved.
    :param path: Local path where the model is to be saved.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in ``mlflow.onnx.DEFAULT_CONDA_ENV``. If `None`, the default
                      ``mlflow.onnx.DEFAULT_CONDA_ENV`` environment will be added to the model.
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
    path = os.path.abspath(path)
    if os.path.exists(path):
        raise Exception("Path '{}' already exists".format(path))
    os.makedirs(path)
    model_data_subpath = "model.onnx"
    model_data_path = os.path.join(path, model_data_subpath)

    # Save onnx-model
    onnx.save_model(onnx_model, model_data_path)

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = DEFAULT_CONDA_ENV
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
    onnx_model = onnx.load(model_file)
    # Check Formation
    onnx.checker.check_model(onnx_model)
    return onnx_model


class _OnnxModelWrapper:
    def __init__(self, path):
        self.rt = onnxrt.InferenceSession(path)
        assert len(self.rt.get_inputs()) >= 1
        self.inputs = [
            (inp.name, inp.type) for inp in self.rt.get_inputs()
        ]
        self.output_names = [
            outp.name for outp in self.rt.get_outputs()
        ]

    def predict(self, dataframe):

        # ONNXRuntime throws the following exception for some operators when the input
        # dataframe contains float64 values. Unfortunately, even if the original user-supplied
        # dataframe did not contain float64 values, the serialization/deserialization between the
        # client and the scoring server can introduce 64-bit floats. This is being tracked in
        # issue #1286. Meanwhile we explicitly cast the input to 32-bit floats when needed.
        # TODO: Remove explicit casting when issue #1286 is fixed
        if len(self.inputs) > 1:
            cols = [name for (name, type) in self.inputs if type == 'tensor(float)']
        else:
            cols = dataframe.columns if self.inputs[0][1] == 'tensor(float)' else []

        dataframe = cast_float64_to_float32(dataframe, cols)
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


def load_model(model_uri):
    """
    Load an ONNX model from a local file (if ``run_id`` is None) or a run.

    :param model_uri: The location, in URI format, of the MLflow model, for example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``

                      For more information about supported URI schemes, see the
                      `Artifacts Documentation <https://www.mlflow.org/docs/latest/tracking.html#
                      supported-artifact-stores>`_.

    :return: An ONNX model instance.

    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    onnx_model_artifacts_path = os.path.join(local_model_path, flavor_conf["data"])
    return _load_model(model_file=onnx_model_artifacts_path)


def log_model(onnx_model, artifact_path, conda_env=None):
    """
    Log an ONNX model as an MLflow artifact for the current run.

    :param onnx_model: ONNX model to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in ``mlflow.onnx.DEFAULT_CONDA_ENV``. If `None`, the default
                      ``mlflow.onnx.DEFAULT_CONDA_ENV`` environment will be added to the model.
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
    """
    Model.log(artifact_path=artifact_path, flavor=mlflow.onnx,
              onnx_model=onnx_model, conda_env=conda_env)


def cast_float64_to_float32(dataframe, column_names):
    for input_name in column_names:
        if dataframe[input_name].values.dtype == np.float64:
            dataframe[input_name] = dataframe[input_name].values.astype(np.float32)
    return dataframe
