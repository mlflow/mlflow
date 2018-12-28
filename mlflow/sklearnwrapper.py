"""
The ``mlflow.sklearnwrapper`` module provides an API for logging and loading scikit-learn models
that can have additional code injected into the `predict()` method.

This module exports scikit-learn models with the following flavors:

:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""

from __future__ import absolute_import

import os
import pickle
import yaml
import copy
import pandas as pd

import sklearn

from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, INTERNAL_ERROR
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS
import mlflow.tracking
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration

FLAVOR_NAME = "sklearnwrapper"

DEFAULT_CONDA_ENV = _mlflow_conda_env(
    additional_conda_deps=[
        "scikit-learn={}".format(sklearn.__version__),
    ],
    additional_pip_deps=None,
    additional_conda_channels=None,
)

SERIALIZATION_FORMAT_PICKLE = "pickle"
SERIALIZATION_FORMAT_CLOUDPICKLE = "cloudpickle"

SUPPORTED_SERIALIZATION_FORMATS = [
    SERIALIZATION_FORMAT_PICKLE,
    SERIALIZATION_FORMAT_CLOUDPICKLE
]


def save_model(model_wrapper, path, conda_env=None, mlflow_model=Model(),
               serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE):
    """
    Save a sklearn model wrapper to a path on the local file system.

    :param model_wrapper: model wrapper to be saved.
    :param path: Local path where the model is to be saved.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in ``mlflow.sklearn.DEFAULT_CONDA_ENV``. If `None`, the default
                      ``mlflow.sklearn.DEFAULT_CONDA_ENV`` environment will be added to the model.
                      The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'scikit-learn=0.19.2'
                            ]
                        }

    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
    :param serialization_format: The format in which to serialize the model. This should be one of
                                 the formats listed in
                                 ``mlflow.sklearn.SUPPORTED_SERIALIZATION_FORMATS``. The Cloudpickle
                                 format, ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``,
                                 provides better cross-system compatibility by identifying and
                                 packaging code dependencies with the serialized model.

    >>> import mlflow.sklearnwrapper
    >>> from sklearn.datasets import load_iris
    >>> from sklearn import tree
    >>> iris = load_iris()
    >>> sk_model = tree.DecisionTreeClassifier()
    >>> sk_model = sk_model.fit(iris.data, iris.target)
    >>> model_wrapper = SKLearnModelWrapper(sk_model)
    >>> #Save the model in cloudpickle format
    >>> #set path to location for persistence
    >>> sk_path_dir_1 = ...
    >>> mlflow.sklearnwrapper.save_model(
    >>>         model_wrapper, sk_path_dir_1,
    >>>         serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
    >>>
    >>> #Save the model in pickle format
    >>> #set path to location for persistence
    >>> sk_path_dir_2 = ...
    >>> mlflow.sklearnwrapper.save_model(model_wrapper, sk_path_dir_2,
    >>>                           serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)
    """
    if serialization_format not in SUPPORTED_SERIALIZATION_FORMATS:
        raise MlflowException(
                message=(
                    "Unrecognized serialization format: {serialization_format}. Please specify one"
                    " of the following supported formats: {supported_formats}.".format(
                        serialization_format=serialization_format,
                        supported_formats=SUPPORTED_SERIALIZATION_FORMATS)),
                error_code=INVALID_PARAMETER_VALUE)

    if os.path.exists(path):
        raise MlflowException(message="Path '{}' already exists".format(path),
                              error_code=RESOURCE_ALREADY_EXISTS)
    os.makedirs(path)
    model_data_subpath = "model.pkl"
    _save_model(model_wrapper=model_wrapper, output_path=os.path.join(path, model_data_subpath),
                serialization_format=serialization_format)

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = copy.deepcopy(DEFAULT_CONDA_ENV)
        if serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE:
            import cloudpickle
            conda_env["dependencies"].append("cloudpickle=={}".format(cloudpickle.__version__))
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearnwrapper",
                        data=model_data_subpath, env=conda_env_subpath)
    mlflow_model.add_flavor(FLAVOR_NAME,
                            pickled_model=model_data_subpath,
                            sklearn_version=sklearn.__version__,
                            serialization_format=serialization_format)
    mlflow_model.save(os.path.join(path, "MLmodel"))


def log_model(model_wrapper, artifact_path, conda_env=None,
              serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE):
    """
    Log a scikit-learn model wrapper as an MLflow artifact for the current run.

    :param model_wrapper: scikit-learn model wrapper to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in ``mlflow.sklearn.DEFAULT_CONDA_ENV``. If `None`, the default
                      ``mlflow.sklearn.DEFAULT_CONDA_ENV`` environment will be added to the model.
                      The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'scikit-learn=0.19.2'
                            ]
                        }

    :param serialization_format: The format in which to serialize the model. This should be one of
                                 the formats listed in
                                 ``mlflow.sklearn.SUPPORTED_SERIALIZATION_FORMATS``. The Cloudpickle
                                 format, ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``,
                                 provides better cross-system compatibility by identifying and
                                 packaging code dependencies with the serialized model.

    >>> import mlflow
    >>> import mlflow.sklearnwrapper
    >>> from sklearn.datasets import load_iris
    >>> from sklearn import tree
    >>> iris = load_iris()
    >>> sk_model = tree.DecisionTreeClassifier()
    >>> sk_model = sk_model.fit(iris.data, iris.target)
    >>> model_wrapper = SKLearnModelWrapper(sk_model)
    >>> #set the artifact_path to location where experiment artifacts will be saved
    >>> #log model params
    >>> mlflow.log_param("criterion", sk_model.criterion)
    >>> mlflow.log_param("splitter", sk_model.splitter)
    >>> #log model
    >>> mlflow.sklearnwrapper.log_model(model_wrapper, "model_wrappers")
    """
    return Model.log(artifact_path=artifact_path,
                     flavor=mlflow.sklearnwrapper,
                     model_wrapper=model_wrapper,
                     conda_env=conda_env,
                     serialization_format=serialization_format)


def _load_model_from_local_file(path):
    """Load a scikit-learn model wrapper saved as an MLflow artifact on the local file system."""
    # TODO: we could validate the SciKit-Learn version here
    with open(path, "rb") as f:
        # Models serialized with Cloudpickle can be deserialized using Pickle; in fact,
        # Cloudpickle.load() is just a redefinition of pickle.load(). Therefore, we do
        # not need to check the serialization format of the model before deserializing.
        return pickle.load(f)


def _load_pyfunc(path):
    """Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``. """
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_model(model_wrapper, output_path, serialization_format):
    """
    :param model_wrapper: The Scikit-learn model wrapper to serialize.
    :param output_path: The file path to which to write the serialized model.
    :param serialization_format: The format in which to serialize the model. This should be one of
                                 the following: `mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE`,
                                 `mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE`.
    """
    with open(output_path, "wb") as out:
        if serialization_format == SERIALIZATION_FORMAT_PICKLE:
            pickle.dump(model_wrapper, out)
        elif serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE:
            import cloudpickle
            cloudpickle.dump(model_wrapper, out)
        else:
            raise MlflowException(
                    message="Unrecognized serialization format: {serialization_format}".format(
                        serialization_format=serialization_format),
                    error_code=INTERNAL_ERROR)

def load_model(path, run_id=None):
    """
    Load a sklearn model wrapper from a local file (if ``run_id`` is None) or a run.

    :param path: Local filesystem path or run-relative artifact path to the model saved
                 by :py:func:`mlflow.sklearn.save_model`.
    :param run_id: Run ID. If provided, combined with ``path`` to identify the model.

    >>> import mlflow.sklearn
    >>> model_wrapper = mlflow.sklearn.load_model("model_wrappers", run_id="96771d893a5e46159d9f3b49bf9013e2")
    >>> #use Pandas DataFrame to make predictions
    >>> pandas_df = ...
    >>> predictions = model_wrapper.predict(pandas_df)
    """
    if run_id is not None:
        path = mlflow.tracking.utils._get_model_log_dir(model_name=path, run_id=run_id)
    path = os.path.abspath(path)
    flavor_conf = _get_flavor_configuration(model_path=path, flavor_name=FLAVOR_NAME)
    sklearn_model_artifacts_path = os.path.join(path, flavor_conf['pickled_model'])
    return _load_model_from_local_file(path=sklearn_model_artifacts_path)

def transform_input(input_df, coltypes, features=None, fit=True):
    """Transforms raw input into model features.

    >>> coltypes = {
    >>>     'cont_num_cols': ['age', 'income'],
    >>>     'disc_num_cols': ['count'],
    >>>     'categoric_cols': ['day_of_week'],
    >>> }
    >>> X = mlflow.sklearnwrapper.transform_input(X, coltypes, fit=True)
    """
    df = input_df.copy()
    for col in coltypes['cont_num_cols']:
        if col in df.columns:
            df[col] = df[col].astype('float64')
            df[col] = df[col].fillna(df[col].mean())
    for col in coltypes['disc_num_cols']:
        if col in df.columns:
            df[col] = df[col].astype('int64')
            df[col] = df[col].fillna(0)
    for col in coltypes['categoric_cols']:
        if col in df.columns:
            df[col] = df[col].astype('object')
            df[col] = df[col].fillna('UNKNOWN')
    df = pd.get_dummies(df)
    if not fit:
        missing_cols = set(features) - set(df.columns)
        for c in missing_cols:
            df[c] = 0
        extra_cols = set( df.columns ) - set(features)
        print('extra columns: {}'.format(extra_cols))
        df = df[features]
    return df


class SKLearnModelWrapper:

    def __init__(self, model, coltypes, features):
        """
        A wrapper around an scikit-learn model with ability to transform raw input in the same
        manner as in training.

        :param model: sckit-learn model
        :param coltypes: dict used to transform raw inputs into model features
        :param features: list of all feature names (after one-hot encoding)

        >>> clf = LogisticRegression(penalty='l1', C=0.1, random_state=123)
        >>> clf.fit(X_train, y_train)
        >>> model_wrapper = mlflow.sklearnwrapper.SKLearnModelWrapper(clf, coltypes,
                                                                  X_train.columns.values)
        """
        self.model = model
        self.coltypes = coltypes
        self.features = features

    # def transform(self, input_df):
    #     """Transforms input when serving the model."""
    #     df = input_df.copy()
    #     for col in self.coltypes['cont_num_cols']:
    #         if col in df.columns:
    #             df[col] = df[col].astype('float64')
    #             df[col] = df[col].fillna(df[col].mean())
    #     for col in self.coltypes['disc_num_cols']:
    #         if col in df.columns:
    #             df[col] = df[col].astype('int64')
    #             df[col] = df[col].fillna(0)
    #     for col in self.coltypes['categoric_cols']:
    #         if col in df.columns:
    #             df[col] = df[col].astype('object')
    #             df[col] = df[col].fillna('UNKNOWN')
    #     df = pd.get_dummies(df)
    #     missing_cols = set(self.features) - set(df.columns)
    #     for c in missing_cols:
    #         df[c] = 0
    #     extra_cols = set( df.columns ) - set(self.features)
    #     print('extra columns: {}'.format(extra_cols))
    #     df = df[self.features]
    #     return df

    def predict(self, df):
        transformed = transform_input(df, self.coltypes, self.features, fit=False)
        return self.model.predict_proba(transformed)[:,1]
