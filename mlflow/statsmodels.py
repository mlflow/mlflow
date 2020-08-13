"""
The ``mlflow.statsmodels`` module provides an API for logging and loading statsmodels models.
This module exports statsmodels models with the following flavors:

statsmodels (native) format
    This is the main flavor that can be loaded back into statsmodels, which relies on pickle
    internally to serialize a model.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.

.. _statsmodels.base.model:
    https://www.statsmodels.org/stable/dev/generated/statsmodels.base.model.Model.html

"""
import os
import yaml
import logging
import gorilla
import pandas as pd
import numpy as np

import mlflow
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import try_mlflow_log, log_fn_args_as_params

from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.tsa.base.tsa_model import TimeSeriesModel
from statsmodels.tsa.arima.model import ARIMA

FLAVOR_NAME = "statsmodels"
STATSMODELS_DATA_SUBPATH = "model.statsmodels"

_logger = logging.getLogger(__name__)


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    import statsmodels
    return _mlflow_conda_env(
        additional_conda_deps=[
            "statsmodels={}".format(statsmodels.__version__),
        ],
        # statsmodels is not yet available via the default conda channels, so we install it via pip
        additional_pip_deps=None,
        additional_conda_channels=None)


def save_model(statsmodels_model, path, conda_env=None, mlflow_model=None,
               remove_data: bool = False,
               signature: ModelSignature = None, input_example: ModelInputExample = None):
    """
    Save a statsmodels model to a path on the local file system.

    :param statsmodels_model: statsmodels model (an instance of `statsmodels.base.model`_)
                              to be saved.
    :param path: Local path where the model is to be saved.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this describes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If ``None``, the default
                      :func:`get_default_conda_env()` environment is added to the model.
                      The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'statsmodels=0.11.1'
                            ]
                        }

    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
    :param remove_data: bool. If False (default), then the instance is pickled without changes.
                        If True, then all arrays with length nobs are set to None before
                        pickling. See the remove_data method.
                        In some cases not all arrays will be set to None.

    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
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
    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    """
    import statsmodels

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException("Path '{}' already exists".format(path))
    model_data_path = os.path.join(path, STATSMODELS_DATA_SUBPATH)
    os.makedirs(path)
    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    # Save a statsmodels model
    statsmodels_model.save(model_data_path, remove_data)

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = get_default_conda_env()
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.statsmodels",
                        data=STATSMODELS_DATA_SUBPATH, env=conda_env_subpath)
    mlflow_model.add_flavor(FLAVOR_NAME, statsmodels_version=statsmodels.__version__,
                            data=STATSMODELS_DATA_SUBPATH)
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


def log_model(statsmodels_model, artifact_path, conda_env=None, registered_model_name=None,
              remove_data: bool = False,
              signature: ModelSignature = None, input_example: ModelInputExample = None,
              **kwargs):
    """
    Log a statsmodels model as an MLflow artifact for the current run.

    :param statsmodels_model: statsmodels model (an instance of `statsmodels.base.model.Model`_)
                              to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this describes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If ``None``, the default
                      :func:`get_default_conda_env()` environment is added to the model.
                      The following is an *example* dictionary representation of a Conda
                      environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'statsmodels=0.11.1'
                            ]
                        }
    :param registered_model_name: (Experimental) If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

    :param remove_data: bool. If False (default), then the instance is pickled without changes.
                        If True, then all arrays with length nobs are set to None before
                        pickling. See the remove_data method.
                        In some cases not all arrays will be set to None.

    :param signature: (Experimental) :py:class:`ModelSignature <mlflow.models.ModelSignature>`
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
    :param input_example: (Experimental) Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    """
    Model.log(artifact_path=artifact_path, flavor=mlflow.statsmodels,
              registered_model_name=registered_model_name,
              statsmodels_model=statsmodels_model, conda_env=conda_env,
              signature=signature, input_example=input_example,
              remove_data=remove_data)


def _load_model(path):
    import statsmodels.iolib.api as smio
    return smio.load_pickle(path)


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.

    :param path: Local filesystem path to the MLflow Model with the ``statsmodels`` flavor.
    """
    return _StatsmodelsModelWrapper(_load_model(path))


def load_model(model_uri):
    """
    Load a statsmodels model from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
                      artifact-locations>`_.

    :return: A statsmodels model (an instance of `statsmodels.base.model.Model`_).
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    statsmodels_model_file_path = os.path.join(local_model_path,
                                               flavor_conf.get("data", STATSMODELS_DATA_SUBPATH))
    return _load_model(path=statsmodels_model_file_path)


class _StatsmodelsModelWrapper:
    def __init__(self, statsmodels_model):
        self.statsmodels_model = statsmodels_model

    def predict(self, dataframe):
        model = self.statsmodels_model.model
        print(model.__class__)
        if(isinstance(model, TimeSeriesModel) or
           isinstance(model, ARIMA)):
            # Assume the inference df has columns "start" and "end", and just one row
            # TODO: move this to a specific mlflow.statsmodels.tsa flavor since time series models
            # expect slightly different arguments to make predictions
            start_date = dataframe["start"][0]
            end_date = dataframe["end"][0]
            return self.statsmodels_model.predict(start=start_date, end=end_date)
        else:
            return self.statsmodels_model.predict(dataframe)


@experimental
def autolog():
    """
    Enables automatic logging from statsmodels to MLflow. Logs the following.

    - results metrics returned by `statsmodels.base.model.Model.fit`_.
    - trained model.

    Note that the `scikit-learn API`_ is not supported.
    """
    import statsmodels
    import numpy as np

    @gorilla.patch(statsmodels.base.model.Model)
    def fit(*args, **kwargs):

        if not mlflow.active_run():
            try_mlflow_log(mlflow.start_run)
            auto_end_run = True
        else:
            auto_end_run = False

        original = gorilla.get_original_attribute(statsmodels.base.model.Model, 'fit')

        # training model
        model = original(*args, **kwargs)

        # Log the model
        try_mlflow_log(log_model, model, artifact_path='model')

        # Log the most common metrics
        metrics_dict = _results_to_dict(model)
        try_mlflow_log(mlflow.log_metrics, metrics_dict)

        if auto_end_run:
            try_mlflow_log(mlflow.end_run)
        return model

    settings = gorilla.Settings(allow_hit=True, store_hit=True)
    gorilla.apply(gorilla.Patch(statsmodels.base.model.Model, 'fit', fit, settings=settings))


def try_log_dict(name: str, value: dict, separator: str = '.'):
    """
    This code has been taken from https://github.com/nyanp/nyaggle/issues/63
    Logs a (nested) dictionary as parameter with flatten format.
    Args:
        name: Parameter name
        value: Parameter value
        separator: Separating character used to concatanate keys
    Examples:
        >>> with Experiment('./') as e:
        >>>     e.log_dict('a', {'b': 1, 'c': 'd'})
        >>>     print(e.params)
        { 'a.b': 1, 'a.c': 'd' }
    """

    if value is None:
        try_mlflow_log(mlflow.log_metric, name, value)
        return

    def _flatten(d: dict, prefix: str, separator: str) -> dict:
        items = []
        for k, v in d.items():
            child_key = prefix + separator + str(k) if prefix else str(k)
            if isinstance(v, dict) and v:
                items.extend(_flatten(v, child_key, separator).items())
            else:
                items.append((child_key, v))
        return dict(items)

    value = _flatten(value, name, separator)
    try_mlflow_log(mlflow.log_metrics, value)


def _prepend_to_keys(dictionary: dict, preffix="_"):
    """
        Modifies all keys of a dictionary by adding a preffix string to all of them
        Returns a new dictionary where all keys have been modified. No changes are
        made to the input dictionary
    """
    import re
    keys = list(dictionary.keys())
    d2 = {}
    for k in keys:
        newkey = re.sub("\(|\)|\[|\]|\.|\+", "_", preffix + k)
        d2[newkey] = dictionary.get(k)
    return d2


def _results_to_dict(results: RegressionResultsWrapper):
    nfeat = getattr(results, "params").shape[0]
    results_dict = {}
    for f in dir(results):
        # Get all fields except covariances and private ones
        if not callable(getattr(results, f)) and \
                not f.startswith('__') and \
                not f.startswith('_') and \
                not f.startswith('cov_'):
            field = getattr(results, f)
            if isinstance(field, np.ndarray) and \
                    field.ndim == 1 and field.shape[0] == nfeat:
                d = field.to_dict()
                renamed_keys_dict = _prepend_to_keys(d, f)
                results_dict.update(renamed_keys_dict)
            elif isinstance(field, (int, float)):
                results_dict[f] = field

    return results_dict