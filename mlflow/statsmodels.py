"""
The ``mlflow.statsmodels`` module provides an API for logging and loading statsmodels models.
This module exports statsmodels models with the following flavors:

statsmodels (native) format
    This is the main flavor that can be loaded back into statsmodels, which relies on pickle
    internally to serialize a model.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.

.. _statsmodels.base.model.Results:
    https://www.statsmodels.org/stable/_modules/statsmodels/base/model.html#Results

"""
import os
import yaml
import logging
import gorilla
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

import itertools
import inspect

FLAVOR_NAME = "statsmodels"
STATSMODELS_DATA_SUBPATH = "model.statsmodels"

_logger = logging.getLogger(__name__)

# monkey patching should be done only the first time the user calls autolog()
_patching_accomplished = False


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
        additional_pip_deps=None,
        additional_conda_channels=None)


def save_model(statsmodels_model, path, conda_env=None, mlflow_model=None,
               remove_data: bool = False,
               signature: ModelSignature = None, input_example: ModelInputExample = None):
    """
    Save a statsmodels model to a path on the local file system.

    :param statsmodels_model: statsmodels model (an instance of `statsmodels.base.model.Results`_)
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

    :param statsmodels_model: statsmodels model (an instance of `statsmodels.base.model.Results`_)
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

    :return: A statsmodels model (an instance of `statsmodels.base.model.Results`_).
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
        from statsmodels.tsa.base.tsa_model import TimeSeriesModel
        model = self.statsmodels_model.model
        if isinstance(model, TimeSeriesModel):
            # Assume the inference dataframe has columns "start" and "end", and just one row
            # TODO: move this to a specific mlflow.statsmodels.tsa flavor? Time series models
            # often expect slightly different arguments to make predictions
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
    """
    import statsmodels

    def find_subclasses(klass):
        """
        Recursively return a (non-nested) list of the class object and all its subclasses
        :param klass: the class whose class subtree we want to retrieve
        :return: a list of classes that includes the argument in the first position
        """
        subclasses = klass.__subclasses__()
        if subclasses:
            subclass_lists = [find_subclasses(c) for c in subclasses]
            chain = itertools.chain.from_iterable(subclass_lists)
            result = [klass] + list(chain)
            return result
        else:
            return [klass]

    def overrides(klass, function_name):
        """
        Returns True when the class passed as first argument overrides the function_name
        Based on https://stackoverflow.com/a/62303206/5726057
        :param klass: the class we are inspecting
        :param function_name: a string with the name of the method we want to check overriding
        :return:
        """
        try:
            superclass = inspect.getmro(klass)[1]
            overriden = getattr(klass, function_name) is not getattr(superclass, function_name)
            return overriden
        except (IndexError, AttributeError):
            return False

    def apply_gorilla_patch(patch, force_backup=True):
        """
        Apply a patch, even if the backup method already exists.
        Adapted from gorilla.py in the gorilla package
        """
        settings = gorilla.Settings() if patch.settings is None else patch.settings

        # When a hit occurs due to an attribute at the destination already existing
        # with the patch's name, the existing attribute is referred to as 'target'.
        try:
            target = gorilla.get_attribute(patch.destination, patch.name)
        except AttributeError:
            pass
        else:
            if not settings.allow_hit:
                raise RuntimeError(
                    "An attribute named '%s' already exists at the destination "
                    "'%s'. Set a different name through the patch object to avoid "
                    "a name clash or set the setting 'allow_hit' to True to "
                    "overwrite the attribute. In the latter case, it is "
                    "recommended to also set the 'store_hit' setting to True in "
                    "order to store the original attribute under a different "
                    "name so it can still be accessed."
                    % (patch.name, patch.destination.__name__))

            if settings.store_hit:
                original_name = gorilla._ORIGINAL_NAME % (patch.name,)
                # This condition is different from gorilla.apply as it now includes force_backup
                if force_backup or not hasattr(patch.destination, original_name):
                    setattr(patch.destination, original_name, target)

        setattr(patch.destination, patch.name, patch.obj)

    def prepend_to_keys(dictionary: dict, preffix="_"):
        """
        Modifies all keys of a dictionary by adding a preffix string to all of them
        and make them compliant with mlflow params & metrics naming rules.
        :param dictionary:
        :param preffix: a string to be prepended to existing keys, using _ as separator
        :return: a new dictionary where all keys have been modified. No changes are
            made to the input dictionary
        """
        import re
        keys = list(dictionary.keys())
        d2 = {}
        for k in keys:
            newkey = re.sub("\(|\)|\[|\]|\.|\+", "_", preffix + "_" + k)
            d2[newkey] = dictionary.get(k)
        return d2

    def results_to_dict(results):
        """
        Turns a statsmodels.regression.linear_model.RegressionResultsWrapper into a python dict
        :param results: instance of a RegressionResultsWrapper returned by a call to fit()
        :return: a python dictionary with those metrics that are (a) a real number, or (b) an array
                 of the same length of the number of coefficients
        """
        has_features = False
        features = results.model.exog_names
        if features is not None:
            has_features = True
            nfeat = len(features)

        results_dict = {}
        for f in dir(results):
            try:
                field = getattr(results, f)
                # Get all fields except covariances and private ones
                if not callable(field) and \
                        not f.startswith('__') and \
                        not f.startswith('_') and \
                        not f.startswith('cov_'):

                    if has_features and isinstance(field, np.ndarray) and \
                            field.ndim == 1 and field.shape[0] == nfeat:

                        d = dict(zip(features, field))
                        renamed_keys_dict = prepend_to_keys(d, f)
                        results_dict.update(renamed_keys_dict)

                    elif isinstance(field, (int, float)):
                        results_dict[f] = field

            except AttributeError:
                pass

        return results_dict

    def patch_class_tree(klass):
        """
        Patches all subclasses that override any auto-loggable method via monkey patching using
        the gorilla package, taking the argument as the tree root in the class hierarchy. Every
        auto-loggable method found in any of the subclasses is replaced by the patched version.
        :param klass: root in the class hierarchy to be analyzed and patched recursively
        """

        # TODO: add more autologgable methods here (e.g. fit_regularized, from_formula, etc)
        # See https://www.statsmodels.org/dev/api.html
        autolog_supported_func = {
            "fit": wrapper_fit
        }

        glob_settings = gorilla.Settings(allow_hit=True, store_hit=True)
        glob_subclasses = set(find_subclasses(klass))

        # Create a patch for every method that needs to be patched, i.e. those
        # which actually override an autologgable method
        patches_list = [
            # Link the patched function with the original via a local variable in the closure
            # to allow invoking superclass methods in the context of the subclass, and not
            # losing the trace of the true original method
            gorilla.Patch(c, method_name, wrapper_func(getattr(c, method_name)),
                          settings=glob_settings)
            for c in glob_subclasses
            for (method_name, wrapper_func) in autolog_supported_func.items()
            if overrides(c, method_name)]

        for p in patches_list:
            apply_gorilla_patch(p)

    def wrapper_fit(original_method):
        """
        External function to generate customized versions of fit with the proper value of the
        original function (set externally in parameter). This enables a more accurate link
        between the patched and the original function than using gorilla.get_original_attribute
        in corner cases where `self` still refers to the subclass but the method we want to invoke
        (in the context of the subclass) belongs to a superclass

        :param original_method: the original function object that will be replaced by this function
        :return: the new fit function, from which we will be doing a call to the original fit
                 method at some point
        """
        def fit(self, *args, **kwargs):
            if not mlflow.active_run():
                try_mlflow_log(mlflow.start_run)
                auto_end_run = True
            else:
                auto_end_run = False

            # training model
            model = original_method(self, *args, **kwargs)

            # Log the model
            try_mlflow_log(log_model, model, artifact_path='model')

            # Log the most common metrics
            if isinstance(model, statsmodels.base.wrapper.ResultsWrapper):
                metrics_dict = results_to_dict(model)
                try_mlflow_log(mlflow.log_metrics, metrics_dict)
                # This may generate warnings due to collisions in already-logged param names
                log_fn_args_as_params(original_method, args, kwargs)

            if auto_end_run:
                try_mlflow_log(mlflow.end_run)

            return model

        return fit

    global _patching_accomplished

    if not _patching_accomplished:
        _patching_accomplished = True
        patch_class_tree(statsmodels.base.model.Model)
