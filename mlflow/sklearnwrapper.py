"""
The ``mlflow.sklearnwrapper`` module provides an API for logging and loading scikit-learn models
that can have additional code injected into the `predict()` method.

This module exports scikit-learn models with the following flavors:

:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""

from __future__ import absolute_import

from collections import defaultdict
import os
import pickle
import yaml
import copy
import pandas as pd
import numpy as np

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin

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


def save_model(pipeline_wrapper, path, conda_env=None, mlflow_model=Model(),
               serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE):
    """
    Save a sklearn pipeline wrapper to a path on the local file system.

    :param pipeline_wrapper: pipeline wrapper to be saved.
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
    >>> pipeline_wrapper = SKLearnPipelineWrapper(sk_model)
    >>> #Save the model in cloudpickle format
    >>> #set path to location for persistence
    >>> sk_path_dir_1 = ...
    >>> mlflow.sklearnwrapper.save_model(
    >>>         pipeline_wrapper, sk_path_dir_1,
    >>>         serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
    >>>
    >>> #Save the model in pickle format
    >>> #set path to location for persistence
    >>> sk_path_dir_2 = ...
    >>> mlflow.sklearnwrapper.save_model(pipeline_wrapper, sk_path_dir_2,
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
    _save_model(pipeline_wrapper=pipeline_wrapper, output_path=os.path.join(path, model_data_subpath),
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


def log_model(pipeline_wrapper, artifact_path, conda_env=None,
              serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE):
    """
    Log a scikit-learn pipeline wrapper as an MLflow artifact for the current run.

    :param pipeline_wrapper: scikit-learn pipeline wrapper to be saved.
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
    >>> pipeline_wrapper = SKLearnPipelineWrapper(sk_model)
    >>> #set the artifact_path to location where experiment artifacts will be saved
    >>> #log model params
    >>> mlflow.log_param("criterion", sk_model.criterion)
    >>> mlflow.log_param("splitter", sk_model.splitter)
    >>> #log model
    >>> mlflow.sklearnwrapper.log_model(pipeline_wrapper, "pipeline_wrappers")
    """
    return Model.log(artifact_path=artifact_path,
                     flavor=mlflow.sklearnwrapper,
                     pipeline_wrapper=pipeline_wrapper,
                     conda_env=conda_env,
                     serialization_format=serialization_format)


def _load_model_from_local_file(path):
    """Load a scikit-learn pipeline wrapper saved as an MLflow artifact on the local file system."""
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


def _save_model(pipeline_wrapper, output_path, serialization_format):
    """
    :param pipeline_wrapper: The Scikit-learn pipeline wrapper to serialize.
    :param output_path: The file path to which to write the serialized model.
    :param serialization_format: The format in which to serialize the model. This should be one of
                                 the following: `mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE`,
                                 `mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE`.
    """
    with open(output_path, "wb") as out:
        if serialization_format == SERIALIZATION_FORMAT_PICKLE:
            pickle.dump(pipeline_wrapper, out)
        elif serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE:
            import cloudpickle
            cloudpickle.dump(pipeline_wrapper, out)
        else:
            raise MlflowException(
                    message="Unrecognized serialization format: {serialization_format}".format(
                        serialization_format=serialization_format),
                    error_code=INTERNAL_ERROR)

def load_model(path, run_id=None):
    """
    Load a sklearn pipeline wrapper from a local file (if ``run_id`` is None) or a run.

    :param path: Local filesystem path or run-relative artifact path to the model saved
                 by :py:func:`mlflow.sklearn.save_model`.
    :param run_id: Run ID. If provided, combined with ``path`` to identify the model.

    >>> import mlflow.sklearn
    >>> pipeline_wrapper = mlflow.sklearn.load_model("pipeline_wrappers", run_id="96771d893a5e46159d9f3b49bf9013e2")
    >>> #use Pandas DataFrame to make predictions
    >>> pandas_df = ...
    >>> predictions = pipeline_wrapper.predict(pandas_df)
    """
    if run_id is not None:
        path = mlflow.tracking.utils._get_model_log_dir(model_name=path, run_id=run_id)
    path = os.path.abspath(path)
    flavor_conf = _get_flavor_configuration(model_path=path, flavor_name=FLAVOR_NAME)
    sklearn_model_artifacts_path = os.path.join(path, flavor_conf['pickled_model'])
    return _load_model_from_local_file(path=sklearn_model_artifacts_path)

class DtypeTransform(BaseEstimator, TransformerMixin):

    def __init__(self, coltypes):
        """
        Sets column dtypes.

        >>> coltypes = {
        >>>     'cont_num_cols': ['age', 'income'],
        >>>     'disc_num_cols': ['count'],
        >>>     'categoric_cols': ['day_of_week'],
        >>> }
        >>> trans = mlflow.sklearnwrapper.DtypeTransform(coltypes)
        >>> df = trans.fit_transform(df)

        """
        self.coltypes = coltypes

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):
        df = X.copy()
        for col in self.coltypes['cont_num_cols']:
            if col in df.columns:
                df[col] = df[col].astype('float64')
                df[col] = df[col].fillna(df[col].mean())
        for col in self.coltypes['disc_num_cols']:
            if col in df.columns:
                df[col] = df[col].astype('int64')
                df[col] = df[col].fillna(0)
        for col in self.coltypes['categoric_cols']:
            if col in df.columns:
                df[col] = df[col].astype('object')
                df[col] = df[col].fillna('UNKNOWN')

        return df

class BucketTransform(BaseEstimator, TransformerMixin):

    def __init__(self, id_col, target_col, min_thresh, pctiles=np.linspace(0,90,10, dtype=int)):
        self.id_col = id_col
        self.target_col = target_col
        self.min_thresh = min_thresh
        self.pctiles = pctiles
        self.bucket_lookup = defaultdict(lambda: '{}_NA'.format(self.id_col))

    def fit(self, X, y, **fit_params):
        X = X.copy()
        y = y.copy()

        if self.id_col not in X.columns:
            print('WARNING: {} not in X'.format(self.id_col))
            return self

        df = X.join(y)
        # Ensure percentiles are sorted in descending order
        self.pctiles[::-1].sort() # sorts inplace
        buckets = df.copy().groupby(self.id_col, as_index=False)[self.target_col].agg(['count','sum'])
        buckets[self.target_col] = buckets['sum'] / buckets['count']
        buckets[self.id_col] = buckets.index.values
        lbl_prfx = '{}_'.format(self.id_col)
        pctile_lbls = []
        pctile_targets = []
        for pctile in self.pctiles:
            lbl = '{}{}'.format(lbl_prfx, pctile)
            target = np.percentile(buckets[self.target_col], pctile)
            pctile_lbls.append(lbl)
            pctile_targets.append(target)
        def get_bucket_lbl(row):
            for lbl, target in zip(pctile_lbls, pctile_targets):
                if (row['count'] > self.min_thresh) and (row[self.target_col] >= target):
                    return lbl
            return '{}NA'.format(lbl_prfx)
        buckets['bucket'] = buckets.apply(get_bucket_lbl, axis=1)

        # Build bucket lookup dict
        for _id in buckets[self.id_col].unique():
            bucket = buckets.loc[buckets[self.id_col] == _id]['bucket'].values[0]
            self.bucket_lookup[str(_id)] = bucket

        self.categories_ = np.unique(list(self.bucket_lookup.values()))
        self.buckets_ = buckets

        return self

    def transform(self, X, y=None, **transform_params):
        X = X.copy()

        if self.id_col not in X.columns:
            print('WARNING: {} not in X'.format(self.id_col))
            return X

        def get_bucket(row):
            _id = row[self.id_col]
            return self.bucket_lookup[str(_id)]

        bucket_col = '{}_bucket'.format(self.id_col)
        X[bucket_col] = X.apply(get_bucket, axis=1)
        dummies = pd.get_dummies(X[bucket_col])

        # Add missing columns (only needed for test/predict sets)
        missing_cols = set(self.categories_) - set(dummies.columns)
        for c in missing_cols:
            dummies[c] = 0

        # Join new columns back to original data
        X = X.join(dummies)

        # Drop original and intermediate cols (e.g. affiliate_id and affiliate_id_bucket)
        X = X.drop(columns=[self.id_col, bucket_col])

        # Need to sort columns since adding missing cols does not guarantee order
        X = X.reindex(sorted(X.columns), axis=1)

        return X


class EmptyTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **transform_params):
        return X


class SKLearnPipelineWrapper:

    def __init__(self, pipeline):
        """
        A wrapper around an scikit-learn pipeline with ability to transform raw input in the same
        manner as in training.

        :param pipeline: an sklearn.pipeline

        >>> # Build pipeline
        >>> dt_trans = mlflow.sklearnwrapper.DtypeTransform(coltypes)
        >>> numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        >>> categoric_transormer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
        >>> numeric_cols = coltypes['cont_num_cols'] + coltypes['disc_num_cols']
        >>> preprocessor = ColumnTransformer(transformers=[
        >>>     ('categoric_transformer', categoric_transormer, coltypes['categoric_cols']),
        >>>     ('numeric_transformer', numeric_transformer, numeric_cols),
        >>> ])
        >>> clf = LogisticRegression(penalty=penalty, C=C, random_state=123)
        >>> pipe = Pipeline(steps=[
        >>>     ('dtype_trans', dt_trans),
        >>>     ('preprocessor', preprocessor),
        >>>     ('clf', clf),
        >>> ])
        >>> pipe.fit(X_train, y_train)
        >>>
        >>> # Create pipeline wrapper
        >>> pipeline_wrapper = mlflow.sklearnwrapper.SKLearnPipelineWrapper(pipe)
        >>> pipeline_wrapper.fit(X_train, y_train)
        >>>
        >>> # Evaluate Metrics
        >>> preds = pipeline_wrapper._predict(X_test)
        >>> pred_probas = pipeline_wrapper._predict_proba(X_test)
        >>> (acc, auc) = eval_metrics(y_test, preds, pred_probas)
        """
        self.pipeline = pipeline

    def fit(self, X, y):
        return self.pipeline.fit(X, y)

    def predict(self, df):
        """Used for serving pipeline in production."""
        return self._predict_proba(df)

    def _predict(self, df):
        """Used during training for model evaluation."""
        return self.pipeline.predict(df)

    def _predict_proba(self, df):
        """Used during training for model evaluation."""
        return self.pipeline.predict_proba(df)[:,1]
