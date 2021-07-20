"""
The ``mlflow.sklearn`` module provides an API for logging and loading scikit-learn models. This
module exports scikit-learn models with the following flavors:

Python (native) `pickle <https://scikit-learn.org/stable/modules/model_persistence.html>`_ format
    This is the main flavor that can be loaded back into scikit-learn.

:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
    NOTE: The `mlflow.pyfunc` flavor is only added for scikit-learn models that define `predict()`,
    since `predict()` is required for pyfunc model inference.
"""
import inspect
import os
import logging
import numpy as np
import pickle
import yaml
import warnings
import weakref
from collections import defaultdict, OrderedDict

import mlflow
from mlflow import pyfunc
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE, INTERNAL_ERROR
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import _inspect_original_var_name
from mlflow.utils.annotations import experimental
from mlflow.utils.environment import _mlflow_conda_env, _log_pip_requirements
from mlflow.utils.mlflow_tags import MLFLOW_AUTOLOGGING
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.utils.autologging_utils import (
    autologging_integration,
    safe_patch,
    INPUT_EXAMPLE_SAMPLE_ROWS,
    resolve_input_example_and_signature,
    _get_new_training_session_class,
    MlflowAutologgingQueueingClient,
)
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

FLAVOR_NAME = "sklearn"

SERIALIZATION_FORMAT_PICKLE = "pickle"
SERIALIZATION_FORMAT_CLOUDPICKLE = "cloudpickle"

SUPPORTED_SERIALIZATION_FORMATS = [SERIALIZATION_FORMAT_PICKLE, SERIALIZATION_FORMAT_CLOUDPICKLE]

_logger = logging.getLogger(__name__)
_SklearnTrainingSession = _get_new_training_session_class()


def get_default_pip_requirements(include_cloudpickle=False):
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    import sklearn

    pip_deps = ["scikit-learn=={}".format(sklearn.__version__)]
    if include_cloudpickle:
        import cloudpickle

        pip_deps += ["cloudpickle=={}".format(cloudpickle.__version__)]

    return pip_deps


def get_default_conda_env(include_cloudpickle=False):
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements(include_cloudpickle))


def save_model(
    sk_model,
    path,
    conda_env=None,
    mlflow_model=None,
    serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
):
    """
    Save a scikit-learn model to a path on the local file system. Produces an MLflow Model
    containing the following flavors:

        - :py:mod:`mlflow.sklearn`
        - :py:mod:`mlflow.pyfunc`. NOTE: This flavor is only included for scikit-learn models
          that define `predict()`, since `predict()` is required for pyfunc model inference.

    :param sk_model: scikit-learn model to be saved.
    :param path: Local path where the model is to be saved.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decsribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If `None`, the default
                      :func:`get_default_conda_env()` environment is added to the model.
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


    .. code-block:: python
        :caption: Example

        import mlflow.sklearn
        from sklearn.datasets import load_iris
        from sklearn import tree

        iris = load_iris()
        sk_model = tree.DecisionTreeClassifier()
        sk_model = sk_model.fit(iris.data, iris.target)

        # Save the model in cloudpickle format
        # set path to location for persistence
        sk_path_dir_1 = ...
        mlflow.sklearn.save_model(
                sk_model, sk_path_dir_1,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)

        # save the model in pickle format
        # set path to location for persistence
        sk_path_dir_2 = ...
        mlflow.sklearn.save_model(sk_model, sk_path_dir_2,
                                  serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)
    """
    import sklearn

    if serialization_format not in SUPPORTED_SERIALIZATION_FORMATS:
        raise MlflowException(
            message=(
                "Unrecognized serialization format: {serialization_format}. Please specify one"
                " of the following supported formats: {supported_formats}.".format(
                    serialization_format=serialization_format,
                    supported_formats=SUPPORTED_SERIALIZATION_FORMATS,
                )
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )

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

    model_data_subpath = "model.pkl"
    _save_model(
        sk_model=sk_model,
        output_path=os.path.join(path, model_data_subpath),
        serialization_format=serialization_format,
    )

    conda_env_subpath = "conda.yaml"
    if conda_env is None:
        conda_env = get_default_conda_env(
            include_cloudpickle=serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE
        )
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, conda_env_subpath), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    _log_pip_requirements(conda_env, path)

    # `PyFuncModel` only works for sklearn models that define `predict()`.
    if hasattr(sk_model, "predict"):
        pyfunc.add_to_model(
            mlflow_model,
            loader_module="mlflow.sklearn",
            model_path=model_data_subpath,
            env=conda_env_subpath,
        )
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        pickled_model=model_data_subpath,
        sklearn_version=sklearn.__version__,
        serialization_format=serialization_format,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


def log_model(
    sk_model,
    artifact_path,
    conda_env=None,
    serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
):
    """
    Log a scikit-learn model as an MLflow artifact for the current run. Produces an MLflow Model
    containing the following flavors:

        - :py:mod:`mlflow.sklearn`
        - :py:mod:`mlflow.pyfunc`. NOTE: This flavor is only included for scikit-learn models
          that define `predict()`, since `predict()` is required for pyfunc model inference.

    :param sk_model: scikit-learn model to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decsribes the environment
                      this model should be run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If `None`, the default
                      :func:`get_default_conda_env()` environment is added to the model.
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
    :param registered_model_name: (Experimental) If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

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
    :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.


    .. code-block:: python
        :caption: Example

        import mlflow
        import mlflow.sklearn
        from sklearn.datasets import load_iris
        from sklearn import tree

        iris = load_iris()
        sk_model = tree.DecisionTreeClassifier()
        sk_model = sk_model.fit(iris.data, iris.target)
        # set the artifact_path to location where experiment artifacts will be saved

        #log model params
        mlflow.log_param("criterion", sk_model.criterion)
        mlflow.log_param("splitter", sk_model.splitter)

        # log model
        mlflow.sklearn.log_model(sk_model, "sk_models")
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.sklearn,
        sk_model=sk_model,
        conda_env=conda_env,
        serialization_format=serialization_format,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
    )


def _load_model_from_local_file(path, serialization_format):
    """Load a scikit-learn model saved as an MLflow artifact on the local file system.

    :param path: Local filesystem path to the MLflow Model saved with the ``sklearn`` flavor
    :param serialization_format: The format in which the model was serialized. This should be one of
                                 the following: ``mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE`` or
                                 ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``.
    """
    # TODO: we could validate the scikit-learn version here
    if serialization_format not in SUPPORTED_SERIALIZATION_FORMATS:
        raise MlflowException(
            message=(
                "Unrecognized serialization format: {serialization_format}. Please specify one"
                " of the following supported formats: {supported_formats}.".format(
                    serialization_format=serialization_format,
                    supported_formats=SUPPORTED_SERIALIZATION_FORMATS,
                )
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )
    with open(path, "rb") as f:
        # Models serialized with Cloudpickle cannot necessarily be deserialized using Pickle;
        # That's why we check the serialization format of the model before deserializing
        if serialization_format == SERIALIZATION_FORMAT_PICKLE:
            return pickle.load(f)
        elif serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE:
            import cloudpickle

            return cloudpickle.load(f)


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.

    :param path: Local filesystem path to the MLflow Model with the ``sklearn`` flavor.
    """
    if os.path.isfile(path):
        # Scikit-learn models saved in older versions of MLflow (<= 1.9.1) specify the ``data``
        # field within the pyfunc flavor configuration. For these older models, the ``path``
        # parameter of ``_load_pyfunc()`` refers directly to a serialized scikit-learn model
        # object. In this case, we assume that the serialization format is ``pickle``, since
        # the model loading procedure in older versions of MLflow used ``pickle.load()``.
        serialization_format = SERIALIZATION_FORMAT_PICKLE
    else:
        # In contrast, scikit-learn models saved in versions of MLflow > 1.9.1 do not
        # specify the ``data`` field within the pyfunc flavor configuration. For these newer
        # models, the ``path`` parameter of ``load_pyfunc()`` refers to the top-level MLflow
        # Model directory. In this case, we parse the model path from the MLmodel's pyfunc
        # flavor configuration and attempt to fetch the serialization format from the
        # scikit-learn flavor configuration
        try:
            sklearn_flavor_conf = _get_flavor_configuration(
                model_path=path, flavor_name=FLAVOR_NAME
            )
            serialization_format = sklearn_flavor_conf.get(
                "serialization_format", SERIALIZATION_FORMAT_PICKLE
            )
        except MlflowException:
            _logger.warning(
                "Could not find scikit-learn flavor configuration during model loading process."
                " Assuming 'pickle' serialization format."
            )
            serialization_format = SERIALIZATION_FORMAT_PICKLE

        pyfunc_flavor_conf = _get_flavor_configuration(
            model_path=path, flavor_name=pyfunc.FLAVOR_NAME
        )
        path = os.path.join(path, pyfunc_flavor_conf["model_path"])

    return _load_model_from_local_file(path=path, serialization_format=serialization_format)


def _save_model(sk_model, output_path, serialization_format):
    """
    :param sk_model: The scikit-learn model to serialize.
    :param output_path: The file path to which to write the serialized model.
    :param serialization_format: The format in which to serialize the model. This should be one of
                                 the following: ``mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE`` or
                                 ``mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE``.
    """
    with open(output_path, "wb") as out:
        if serialization_format == SERIALIZATION_FORMAT_PICKLE:
            pickle.dump(sk_model, out)
        elif serialization_format == SERIALIZATION_FORMAT_CLOUDPICKLE:
            import cloudpickle

            cloudpickle.dump(sk_model, out)
        else:
            raise MlflowException(
                message="Unrecognized serialization format: {serialization_format}".format(
                    serialization_format=serialization_format
                ),
                error_code=INTERNAL_ERROR,
            )


def load_model(model_uri):
    """
    Load a scikit-learn model from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model, for example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                      artifact-locations>`_.

    :return: A scikit-learn model.

    .. code-block:: python
        :caption: Example

        import mlflow.sklearn
        sk_model = mlflow.sklearn.load_model("runs:/96771d893a5e46159d9f3b49bf9013e2/sk_models")

        # use Pandas DataFrame to make predictions
        pandas_df = ...
        predictions = sk_model.predict(pandas_df)
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    sklearn_model_artifacts_path = os.path.join(local_model_path, flavor_conf["pickled_model"])
    serialization_format = flavor_conf.get("serialization_format", SERIALIZATION_FORMAT_PICKLE)
    return _load_model_from_local_file(
        path=sklearn_model_artifacts_path, serialization_format=serialization_format
    )


class _AutologTrainingStatus:
    """
    This class is designed for holding information which is used by autologging metrics
    It will hold information of:
    (1) a map of "prediction result object id" to a tuple of dataset name(the dataset is
       the one which generate the prediction result) and run_id.
    (2) _log_post_training_metrics_enabled flag, in the following method scope:
       `model.fit`, `eval_and_log_metrics`, `model.score`,
       in order to avoid nested/duplicated autologging metric, when run into these scopes,
       we need temporarily disable the metric autologging.
    (3) eval_dataset_info_map, it is a double level map:
       `eval_dataset_info_map[run_id][eval_dataset_var_name]` will get a list, each
       element in the list is an id of "eval_dataset" instance.
       This data structure is used for:
        * generating unique dataset name key when autologging metric. For each eval dataset object,
          if they have the same eval_dataset_var_name, but object ids are different,
          then they will be assigned different name (via appending index to the
          eval_dataset_var_name) when autologging.
        * storing the dataset row samples, we need log them into metric_info artifact file.
    (4) _metric_api_call_info, it is a double level map:
       `_metric_api_call_info[run_id][metric_name]` wil get a list of tuples, each tuple is:
         (logged_metric_key, metric_call_command_string)
        each call command string is like `metric_fn(arg1, arg2, ...)`
        This data structure is used for:
         * storing the call arguments dict for each metric call, we need log them into metric_info
           artifact file.

    Note: this class is not thread-safe.
    Design rule for this class:
     Because this class instance is a global instance, in order to prevent memory leak, it should
     only holds IDs and other small objects references. This class internal data structure should
     avoid reference to user dataset variables or model variables.
    """
    def __init__(self):
        self._pred_result_id_to_dataset_name_and_run_id = {}
        self._eval_dataset_info_map = defaultdict(lambda: defaultdict(list))
        self._metric_api_call_info = defaultdict(lambda: defaultdict(list))
        self._log_post_training_metrics_enabled = True
        self._metric_info_artifact_need_update = defaultdict(lambda: False)

    def should_log_post_training_metrics(self):
        """
        Check whether we should run patching code for autologging post training metrics.
        This checking should surround the whole patched code due to the safe guard checking,
        See following note.

        Note: It includes checking `_SklearnTrainingSession.is_active()`, This is a safe guarding
        for meta-estimator (e.g. GridSearchCV) case:
          running GridSearchCV.fit, the nested `estimator.fit` will be called in parallel,
          but, the _autolog_training_status is a global status without thread-safe lock protecting.
          This safe guarding will prevent code run into this case.
        """
        return _SklearnTrainingSession.is_active() and self._log_post_training_metrics_enabled

    def disable_log_post_training_metrics(self):

        class LogPostTrainingMetricsDisabledScope:
            def __enter__(inner_self):
                inner_self.old_status = self._log_post_training_metrics_enabled
                self._log_post_training_metrics_enabled = False

            def __exit__(inner_self, exc_type, exc_val, exc_tb):
                self._log_post_training_metrics_enabled = inner_self.old_status

        return LogPostTrainingMetricsDisabledScope()

    @staticmethod
    def get_run_id_for_model(model):
        return getattr(model, '_mlflow_run_id', None)

    @staticmethod
    def is_metrics_value_loggable(metric_value):
        return np.isscalar(metric_value)

    def register_model(self, model, run_id):
        """
        In `patched_fit`, we need register the model with the run_id used in `patched_fit`
        So that in following metric autologging, the metric will be logged into the registered
        run_id
        """
        model._mlflow_run_id = run_id

    @staticmethod
    def gen_name_with_index(name, index):
        assert index >= 0
        if index == 0:
            return name
        else:
            # Use '-' as the separator between name and index,
            # The '-' is not valid character in python var name
            # so it can prevent name conflicts after appending index.
            return f'{name}-{index + 1}'

    def register_eval_dataset(self, model, eval_dataset):
        """
        Register eval dataset into eval_dataset_info_map, it will do:
         1. inspect eval dataset var name.
         2. check whether eval_dataset_info_map already registered this eval dataset.
            will check by object id and comparing row samples.
            Note: only check object id is not sufficient, id may be reused.
         3. register eval dataset (with id and row samples information)
         4. return eval dataset name with index.

        Note: this method include inspecting argument variable name.
         So should be called directly from the "patched method", to ensure it capture
         correct argument variable name.
        """
        eval_dataset_name = _inspect_original_var_name(
            eval_dataset, fallback_name='unknown_dataset')
        eval_dataset_id = id(eval_dataset)

        run_id = self.get_run_id_for_model(model)
        registered_dataset_list = self._eval_dataset_info_map[run_id][eval_dataset_name]

        index = len(registered_dataset_list)
        for i, id_i in enumerate(registered_dataset_list):
            if eval_dataset_id == id_i:
                index = i
                break

        if index == len(registered_dataset_list):
            # register new eval dataset
            registered_dataset_list.append(eval_dataset_id)

        return self.gen_name_with_index(eval_dataset_name, index)

    def register_prediction_result(self, model, eval_dataset_name, predict_result):
        """
        In `patched_predict`, register the prediction result instance with the model id and eval
        dataset name. e.g.

        ```
        prediction_result = model_1.predict(eval_X)
        ```
        then we need register an item into the `pred_result_id_to_dataset_name_and_run_id` map like
        below:
        id(prediction_result) --> (inspected_original_var_name_of(eval_X), id(model_1))

        With this map `dataset_id_to_dataset_name_and_run_id`, in following patched metric API,
        we can query the eval dataset name and the model id via the `y_pred` argument instance.
        """
        value = (eval_dataset_name, self.get_run_id_for_model(model))
        prediction_result_id = id(predict_result)
        self._pred_result_id_to_dataset_name_and_run_id[prediction_result_id] = value

        def clean_id(id_):
            _get_autolog_training_status()._pred_result_id_to_dataset_name_and_run_id \
                .pop(id_, None)

        # When the `predict_result` object being GCed, its ID may be reused, so register a finalizer
        # to clear the ID from the dict for preventing wrong ID mapping.
        weakref.finalize(predict_result, clean_id, prediction_result_id)

    @staticmethod
    def gen_metric_call_command(call_fn_name, *call_pos_args, **call_kwargs):
        """
        Generate metric function call command string like `metric_fn(arg1, arg2, ...)`
        Note: this method include inspecting argument variable name.
         So should be called directly from the "patched method", to ensure it capture
         correct argument variable name.
        """

        arg_list = []

        def arg_to_str(arg):
            if arg is not None and not np.isscalar(arg):
                # dataset arguments
                return _inspect_original_var_name(arg, fallback_name=f'<{arg.__class__.__name__}>')
            elif np.isscalar(arg):
                return str(arg)
            else:
                return arg.__class__.__name__

        for arg in call_pos_args:
            arg_list.append(arg_to_str(arg))

        for arg_name, arg in call_kwargs.items():
            arg_list.append(f'{arg_name}={arg_to_str(arg)}')

        arg_list_str = ','.join(arg_list)
        return f'{call_fn_name}({arg_list_str})'

    def register_metric_info(self, run_id, metric_name, dataset_name, call_command):

        call_cmd_list = self._metric_api_call_info[run_id][metric_name]

        index = len(call_cmd_list)
        metric_name_with_index = self.gen_name_with_index(metric_name, index)
        metric_key = f'{metric_name_with_index}_{dataset_name}'

        call_cmd_list.append((metric_name_with_index, call_command))

        self._metric_info_artifact_need_update[run_id] = True
        return metric_key

    def register_metric_api_call(self, metric_name, call_command, call_pos_args, call_kwargs):
        """
        Given a metric api call (include the called metric function, and call arguments)
        Register the call information (arguments dict) into the `metric_api_call_arg_dict_list_map`
        and return a tuple of (is_ok, run_id, metric_key)
        - metric key format is: {eval_dataset_name}_{metric_name}
          metric_name is generated by metric function name, if there're multiple calls with
          different call arguments, then appending index to the name.
        """
        call_arg_list = list(call_pos_args) + list(call_kwargs.values())

        dataset_id_list = self._pred_result_id_to_dataset_name_and_run_id.keys()
        print(f'DGB: get_eval_dataset_name: # 1')

        dataset_name = None
        run_id = None

        # Note: some metric API the arguments is not like `y_true`, `y_pred`
        #  e.g.
        #    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
        #    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score
        for arg in call_arg_list:
            if arg is not None and not np.isscalar(arg) and id(arg) in dataset_id_list:
                print(f'DGB: get_eval_dataset_name: # 3')
                dataset_name, run_id = self._pred_result_id_to_dataset_name_and_run_id[id(arg)]
                break

        if dataset_name is None or run_id is None:
            return False, None, None

        metric_key = self.register_metric_info(
            run_id, metric_name, dataset_name, call_command
        )

        return True, run_id, metric_key

    def log_eval_metric(self, run_id, key, value):
        # Note: if the case log the same metric key multiple times,
        #  newer value will overwrite old value
        client = mlflow.tracking.MlflowClient()
        client.log_metric(
            run_id=run_id,
            key=key,
            value=value
        )
        if self._metric_info_artifact_need_update[run_id]:
            call_commands_list = []
            for _, v in self._metric_api_call_info[run_id].items():
                call_commands_list.extend(v)

            call_commands_list.sort(key=lambda x: x[0])
            dict_to_log = OrderedDict(call_commands_list)
            client.log_dict(
                run_id=run_id, dictionary=dict_to_log, artifact_file='metric_info.json'
            )
            self._metric_info_artifact_need_update[run_id] = False


_autolog_training_status = _AutologTrainingStatus()


def _get_autolog_training_status():
    return _autolog_training_status


_metric_api_excluding_list = [
    'check_scoring', 'get_scorer', 'make_scorer'
]


@experimental
@autologging_integration(FLAVOR_NAME)
def autolog(
    log_input_examples=False,
    log_model_signatures=True,
    log_models=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    max_tuning_runs=5,
):  # pylint: disable=unused-argument
    """
    Enables (or disables) and configures autologging for scikit-learn estimators.

    **When is autologging performed?**
      Autologging is performed when you call:

      - ``estimator.fit()``
      - ``estimator.fit_predict()``
      - ``estimator.fit_transform()``

    **Logged information**
      **Parameters**
        - Parameters obtained by ``estimator.get_params(deep=True)``. Note that ``get_params``
          is called with ``deep=True``. This means when you fit a meta estimator that chains
          a series of estimators, the parameters of these child estimators are also logged.

      **Metrics**
        - A training score obtained by ``estimator.score``. Note that the training score is
          computed using parameters given to ``fit()``.
        - Common metrics for classifier:

          - `precision score`_

          .. _precision score:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html

          - `recall score`_

          .. _recall score:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html

          - `f1 score`_

          .. _f1 score:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

          - `accuracy score`_

          .. _accuracy score:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

          If the classifier has method ``predict_proba``, we additionally log:

          - `log loss`_

          .. _log loss:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html

          - `roc auc score`_

          .. _roc auc score:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

        - Common metrics for regressor:

          - `mean squared error`_

          .. _mean squared error:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html

          - root mean squared error

          - `mean absolute error`_

          .. _mean absolute error:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html

          - `r2 score`_

          .. _r2 score:
              https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html

      **Tags**
        - An estimator class name (e.g. "LinearRegression").
        - A fully qualified estimator class name
          (e.g. "sklearn.linear_model._base.LinearRegression").

      **Artifacts**
        - An MLflow Model with the :py:mod:`mlflow.sklearn` flavor containing a fitted estimator
          (logged by :py:func:`mlflow.sklearn.log_model()`). The Model also contains the
          :py:mod:`mlflow.pyfunc` flavor when the scikit-learn estimator defines `predict()`.

    **How does autologging work for meta estimators?**
      When a meta estimator (e.g. `Pipeline`_, `GridSearchCV`_) calls ``fit()``, it internally calls
      ``fit()`` on its child estimators. Autologging does NOT perform logging on these constituent
      ``fit()`` calls.

      **Parameter search**
          In addition to recording the information discussed above, autologging for parameter
          search meta estimators (`GridSearchCV`_ and `RandomizedSearchCV`_) records child runs
          with metrics for each set of explored parameters, as well as artifacts and parameters
          for the best model (if available).

    **Supported estimators**
      - All estimators obtained by `sklearn.utils.all_estimators`_ (including meta estimators).
      - `Pipeline`_
      - Parameter search estimators (`GridSearchCV`_ and `RandomizedSearchCV`_)

    .. _sklearn.utils.all_estimators:
        https://scikit-learn.org/stable/modules/generated/sklearn.utils.all_estimators.html

    .. _Pipeline:
        https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

    .. _GridSearchCV:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

    .. _RandomizedSearchCV:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html

    **Example**

    `See more examples <https://github.com/mlflow/mlflow/blob/master/examples/sklearn_autolog>`_

    .. code-block:: python

        from pprint import pprint
        import numpy as np
        from sklearn.linear_model import LinearRegression
        import mlflow

        def fetch_logged_data(run_id):
            client = mlflow.tracking.MlflowClient()
            data = client.get_run(run_id).data
            tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
            artifacts = [f.path for f in client.list_artifacts(run_id, "model")]
            return data.params, data.metrics, tags, artifacts

        # enable autologging
        mlflow.sklearn.autolog()

        # prepare training data
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3

        # train a model
        model = LinearRegression()
        with mlflow.start_run() as run:
            model.fit(X, y)

        # fetch logged data
        params, metrics, tags, artifacts = fetch_logged_data(run.info.run_id)

        pprint(params)
        # {'copy_X': 'True',
        #  'fit_intercept': 'True',
        #  'n_jobs': 'None',
        #  'normalize': 'False'}

        pprint(metrics)
        # {'training_score': 1.0,
           'training_mae': 2.220446049250313e-16,
           'training_mse': 1.9721522630525295e-31,
           'training_r2_score': 1.0,
           'training_rmse': 4.440892098500626e-16}

        pprint(tags)
        # {'estimator_class': 'sklearn.linear_model._base.LinearRegression',
        #  'estimator_name': 'LinearRegression'}

        pprint(artifacts)
        # ['model/MLmodel', 'model/conda.yaml', 'model/model.pkl']

    :param log_input_examples: If ``True``, input examples from training datasets are collected and
                               logged along with scikit-learn model artifacts during training. If
                               ``False``, input examples are not logged.
                               Note: Input examples are MLflow model attributes
                               and are only collected if ``log_models`` is also ``True``.
    :param log_model_signatures: If ``True``,
                                 :py:class:`ModelSignatures <mlflow.models.ModelSignature>`
                                 describing model inputs and outputs are collected and logged along
                                 with scikit-learn model artifacts during training. If ``False``,
                                 signatures are not logged.
                                 Note: Model signatures are MLflow model attributes
                                 and are only collected if ``log_models`` is also ``True``.
    :param log_models: If ``True``, trained models are logged as MLflow model artifacts.
                       If ``False``, trained models are not logged.
                       Input examples and model signatures, which are attributes of MLflow models,
                       are also omitted when ``log_models`` is ``False``.
    :param disable: If ``True``, disables the scikit-learn autologging integration. If ``False``,
                    enables the scikit-learn autologging integration.
    :param exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
                      If ``False``, autologged content is logged to the active fluent run,
                      which may be user-created.
    :param disable_for_unsupported_versions: If ``True``, disable autologging for versions of
                      scikit-learn that have not been tested against this version of the MLflow
                      client or are incompatible.
    :param silent: If ``True``, suppress all event logs and warnings from MLflow during scikit-learn
                   autologging. If ``False``, show all events and warnings during scikit-learn
                   autologging.
    :param max_tuning_runs: The maximum number of child Mlflow runs created for hyperparameter
                            search estimators. To create child runs for the best `k` results from
                            the search, set `max_tuning_runs` to `k`. The default value is to track
                            the best 5 search parameter sets. If `max_tuning_runs=None`, then
                            a child run is created for each search parameter set. Note: The best k
                            results is based on ordering in `rank_test_score`. In the case of
                            multi-metric evaluation with a custom scorer, the first scorer’s
                            `rank_test_score_<scorer_name>` will be used to select the best k
                            results. To change metric used for selecting best k results, change
                            ordering of dict passed as `scoring` parameter for estimator.
    """
    import pandas as pd
    import sklearn

    from mlflow.models import infer_signature
    from mlflow.sklearn.utils import (
        _MIN_SKLEARN_VERSION,
        _TRAINING_PREFIX,
        _is_supported_version,
        _get_args_for_metrics,
        _log_estimator_content,
        _all_estimators,
        _get_arg_names,
        _get_estimator_info_tags,
        _get_meta_estimators_for_autologging,
        _is_parameter_search_estimator,
        _log_parameter_search_results_as_artifact,
        _create_child_runs_for_parameter_search,
    )
    from mlflow.tracking.context import registry as context_registry

    if max_tuning_runs is not None and max_tuning_runs < 0:
        raise MlflowException(
            message=(
                "`max_tuning_runs` must be non-negative, instead got {}.".format(max_tuning_runs)
            ),
            error_code=INVALID_PARAMETER_VALUE,
        )

    if not _is_supported_version():
        warnings.warn(
            "Autologging utilities may not work properly on scikit-learn < {} ".format(
                _MIN_SKLEARN_VERSION
            )
            + "(current version: {})".format(sklearn.__version__),
            stacklevel=2,
        )

    def fit_mlflow(original, self, *args, **kwargs):
        """
        Autologging function that performs model training by executing the training method
        referred to be `func_name` on the instance of `clazz` referred to by `self` & records
        MLflow parameters, metrics, tags, and artifacts to a corresponding MLflow Run.
        """
        autologging_client = MlflowAutologgingQueueingClient()
        _log_pretraining_metadata(autologging_client, self, *args, **kwargs)
        params_logging_future = autologging_client.flush(synchronous=False)
        fit_output = original(self, *args, **kwargs)
        _log_posttraining_metadata(autologging_client, self, *args, **kwargs)
        autologging_client.flush(synchronous=True)
        params_logging_future.await_completion()
        return fit_output

    def _log_pretraining_metadata(
        autologging_client, estimator, *args, **kwargs
    ):  # pylint: disable=unused-argument
        """
        Records metadata (e.g., params and tags) for a scikit-learn estimator prior to training.
        This is intended to be invoked within a patched scikit-learn training routine
        (e.g., `fit()`, `fit_transform()`, ...) and assumes the existence of an active
        MLflow run that can be referenced via the fluent Tracking API.

        :param autologging_client: An instance of `MlflowAutologgingQueueingClient` used for
                                   efficiently logging run data to MLflow Tracking.
        :param estimator: The scikit-learn estimator for which to log metadata.
        :param args: The arguments passed to the scikit-learn training routine (e.g.,
                     `fit()`, `fit_transform()`, ...).
        :param kwargs: The keyword arguments passed to the scikit-learn training routine.
        """
        # Deep parameter logging includes parameters from children of a given
        # estimator. For some meta estimators (e.g., pipelines), recording
        # these parameters is desirable. For parameter search estimators,
        # however, child estimators act as seeds for the parameter search
        # process; accordingly, we avoid logging initial, untuned parameters
        # for these seed estimators.
        should_log_params_deeply = not _is_parameter_search_estimator(estimator)
        run_id = mlflow.active_run().info.run_id
        autologging_client.log_params(
            run_id=mlflow.active_run().info.run_id,
            params=estimator.get_params(deep=should_log_params_deeply),
        )
        autologging_client.set_tags(
            run_id=run_id, tags=_get_estimator_info_tags(estimator),
        )

    def _log_posttraining_metadata(autologging_client, estimator, *args, **kwargs):
        """
        Records metadata for a scikit-learn estimator after training has completed.
        This is intended to be invoked within a patched scikit-learn training routine
        (e.g., `fit()`, `fit_transform()`, ...) and assumes the existence of an active
        MLflow run that can be referenced via the fluent Tracking API.

        :param autologging_client: An instance of `MlflowAutologgingQueueingClient` used for
                                   efficiently logging run data to MLflow Tracking.
        :param estimator: The scikit-learn estimator for which to log metadata.
        :param args: The arguments passed to the scikit-learn training routine (e.g.,
                     `fit()`, `fit_transform()`, ...).
        :param kwargs: The keyword arguments passed to the scikit-learn training routine.
        """

        def infer_model_signature(input_example):
            if not hasattr(estimator, "predict"):
                raise Exception(
                    "the trained model does not specify a `predict` function, "
                    + "which is required in order to infer the signature"
                )

            return infer_signature(input_example, estimator.predict(input_example))

        (X, y_true, sample_weight) = _get_args_for_metrics(estimator.fit, args, kwargs)

        # log common metrics and artifacts for estimators (classifier, regressor)
        logged_metrics = _log_estimator_content(
            autologging_client=autologging_client,
            estimator=estimator,
            prefix=_TRAINING_PREFIX,
            run_id=mlflow.active_run().info.run_id,
            X=X,
            y_true=y_true,
            sample_weight=sample_weight,
        )
        if y_true is None and not logged_metrics:
            _logger.warning(
                "Training metrics will not be recorded because training labels were not specified."
                " To automatically record training metrics, provide training labels as inputs to"
                " the model training function."
            )

        def get_input_example():
            # Fetch an input example using the first several rows of the array-like
            # training data supplied to the training routine (e.g., `fit()`)
            input_example = X[:INPUT_EXAMPLE_SAMPLE_ROWS]
            return input_example

        if log_models:
            # Will only resolve `input_example` and `signature` if `log_models` is `True`.
            input_example, signature = resolve_input_example_and_signature(
                get_input_example,
                infer_model_signature,
                log_input_examples,
                log_model_signatures,
                _logger,
            )

            log_model(
                estimator, artifact_path="model", signature=signature, input_example=input_example,
            )

        if _is_parameter_search_estimator(estimator):
            if hasattr(estimator, "best_estimator_") and log_models:
                log_model(
                    estimator.best_estimator_,
                    artifact_path="best_estimator",
                    signature=signature,
                    input_example=input_example,
                )

            if hasattr(estimator, "best_score_"):
                autologging_client.log_metrics(
                    run_id=mlflow.active_run().info.run_id,
                    metrics={"best_cv_score": estimator.best_score_},
                )

            if hasattr(estimator, "best_params_"):
                best_params = {
                    "best_{param_name}".format(param_name=param_name): param_value
                    for param_name, param_value in estimator.best_params_.items()
                }
                autologging_client.log_params(
                    run_id=mlflow.active_run().info.run_id, params=best_params,
                )

            if hasattr(estimator, "cv_results_"):
                try:
                    # Fetch environment-specific tags (e.g., user and source) to ensure that lineage
                    # information is consistent with the parent run
                    child_tags = context_registry.resolve_tags()
                    child_tags.update({MLFLOW_AUTOLOGGING: FLAVOR_NAME})
                    _create_child_runs_for_parameter_search(
                        autologging_client=autologging_client,
                        cv_estimator=estimator,
                        parent_run=mlflow.active_run(),
                        max_tuning_runs=max_tuning_runs,
                        child_tags=child_tags,
                    )
                except Exception as e:

                    msg = (
                        "Encountered exception during creation of child runs for parameter search."
                        " Child runs may be missing. Exception: {}".format(str(e))
                    )
                    _logger.warning(msg)

                try:
                    cv_results_df = pd.DataFrame.from_dict(estimator.cv_results_)
                    _log_parameter_search_results_as_artifact(
                        cv_results_df, mlflow.active_run().info.run_id
                    )
                except Exception as e:

                    msg = (
                        "Failed to log parameter search results as an artifact."
                        " Exception: {}".format(str(e))
                    )
                    _logger.warning(msg)

    def patched_fit(original, self, *args, **kwargs):
        """
        Autologging patch function to be applied to a sklearn model class that defines a `fit`
        method and inherits from `BaseEstimator` (thereby defining the `get_params()` method)

        :param clazz: The scikit-learn model class to which this patch function is being applied for
                      autologging (e.g., `sklearn.linear_model.LogisticRegression`)
        :param func_name: The function name on the specified `clazz` that this patch is overriding
                          for autologging (e.g., specify "fit" in order to indicate that
                          `sklearn.linear_model.LogisticRegression.fit()` is being patched)
        """
        status = _get_autolog_training_status()
        with _SklearnTrainingSession(clazz=self.__class__, allow_children=False) as t:
            if t.should_log():
                with status.disable_log_post_training_metrics():
                    result = fit_mlflow(original, self, *args, **kwargs)
                status.register_model(self, mlflow.active_run().info.run_id)
                return result
            else:
                return original(self, *args, **kwargs)

    def patched_predict(original, self, *args, **kwargs):
        status = _get_autolog_training_status()
        if status.should_log_post_training_metrics() and status.get_run_id_for_model(self):
            print(f'DBG: patched_predict # 1')
            predict_result = original(self, *args, **kwargs)
            print(f'DBG: patched_predict # 1.5')
            eval_dataset = args[0] if len(args) >= 1 else kwargs.get('X')
            eval_dataset_name = status.register_eval_dataset(self, eval_dataset)
            status.register_prediction_result(
                self, eval_dataset_name, predict_result)
            return predict_result
        else:
            print(f'DBG: patched_predict # 3')
            return original(self, *args, **kwargs)

    def patched_metric_api(original, *args, **kwargs):
        status = _get_autolog_training_status()
        if status.should_log_post_training_metrics():
            # one metric api may call another metric api,
            # to avoid this, call disable_log_post_training_metrics to avoid nested patch
            with status.disable_log_post_training_metrics():
                metric = original(*args, **kwargs)

            if status.is_metrics_value_loggable(metric):
                print('DGB: run patched_metric_api #2')

                metric_name = original.__name__
                if metric_name.strip() == '<lambda>':
                    metric_name = 'unknown_metric'

                call_command = status.gen_metric_call_command(metric_name, *args, **kwargs)

                is_register_ok, run_id, metric_key = \
                    status.register_metric_api_call(metric_name, call_command, args, kwargs)
                if is_register_ok:
                    print('DGB: run patched_metric_api #4')
                    status.log_eval_metric(run_id, metric_key, metric)

            return metric
        else:
            return original(*args, **kwargs)

    # we need patch model.score method because:
    #  some model.score() implementation won't call metric APIs in `sklearn.metrics`
    #  e.g.
    #  https://github.com/scikit-learn/scikit-learn/blob/82df48934eba1df9a1ed3be98aaace8eada59e6e/sklearn/covariance/_empirical_covariance.py#L220
    def patched_model_score(original, self, *args, **kwargs):
        status = _get_autolog_training_status()
        if status.should_log_post_training_metrics() and status.get_run_id_for_model(self):
            with status.disable_log_post_training_metrics():
                score_value = original(self, *args, **kwargs)

            if status.is_metrics_value_loggable(score_value):
                metric_name = f'{self.__class__.__name__}_score'
                call_fn_name = f'{self.__class__.__name__}.score'
                call_command = status.gen_metric_call_command(call_fn_name, *args, **kwargs)

                eval_dataset = args[0] if len(args) >= 1 else kwargs.get('X')
                eval_dataset_name = status.register_eval_dataset(self, eval_dataset)
                run_id = status.get_run_id_for_model(self)
                metric_key = status.register_metric_info(
                    run_id, metric_name, eval_dataset_name, call_command)

                status.log_eval_metric(
                    status.get_run_id_for_model(self), metric_key, score_value
                )

            return score_value
        else:
            return original(self, *args, **kwargs)

    _, estimators_to_patch = zip(*_all_estimators())
    # Ensure that relevant meta estimators (e.g. GridSearchCV, Pipeline) are selected
    # for patching if they are not already included in the output of `all_estimators()`
    estimators_to_patch = set(estimators_to_patch).union(
        set(_get_meta_estimators_for_autologging())
    )
    # Exclude certain preprocessing & feature manipulation estimators from patching. These
    # estimators represent data manipulation routines (e.g., normalization, label encoding)
    # rather than ML algorithms. Accordingly, we should not create MLflow runs and log
    # parameters / metrics for these routines, unless they are captured as part of an ML pipeline
    # (via `sklearn.pipeline.Pipeline`)
    excluded_module_names = [
        "sklearn.preprocessing",
        "sklearn.impute",
        "sklearn.feature_extraction",
        "sklearn.feature_selection",
    ]

    excluded_class_names = [
        "sklearn.compose._column_transformer.ColumnTransformer",
    ]

    estimators_to_patch = [
        estimator
        for estimator in estimators_to_patch
        if not any(
            estimator.__module__.startswith(excluded_module_name)
            or (estimator.__module__ + "." + estimator.__name__) in excluded_class_names
            for excluded_module_name in excluded_module_names
        )
    ]

    for class_def in estimators_to_patch:
        for func_name in ["fit", "fit_transform", "fit_predict"]:
            if hasattr(class_def, func_name):
                original = getattr(class_def, func_name)

                # A couple of estimators use property methods to return fitting functions,
                # rather than defining the fitting functions on the estimator class directly.
                #
                # Example: https://github.com/scikit-learn/scikit-learn/blob/0.23.2/sklearn/neighbors/_lof.py#L183  # noqa
                #
                # We currently exclude these property fitting methods from patching because
                # it's challenging to patch them correctly.
                #
                # Excluded fitting methods:
                # - sklearn.cluster._agglomerative.FeatureAgglomeration.fit_predict
                # - sklearn.neighbors._lof.LocalOutlierFactor.fit_predict
                #
                # You can list property fitting methods by inserting "print(class_def, func_name)"
                # in the if clause below.
                if isinstance(original, property):
                    continue

                safe_patch(
                    FLAVOR_NAME, class_def, func_name, patched_fit, manage_run=True,
                )

        for func_name in ["predict", "transform", "predict_proba"]:
            if hasattr(class_def, func_name):
                safe_patch(
                    FLAVOR_NAME, class_def, func_name, patched_predict, manage_run=False,
                )

        if hasattr(class_def, 'score'):
            safe_patch(
                FLAVOR_NAME, class_def, 'score', patched_model_score, manage_run=False,
            )

    from sklearn import metrics
    for metric_method_name in metrics.__all__:
        # excludes plot_* methods
        # exclude class (e.g. metrics.ConfusionMatrixDisplay)
        metric_method = getattr(metrics, metric_method_name)
        if metric_method_name not in _metric_api_excluding_list \
                and not inspect.isclass(metric_method) \
                and callable(metric_method) \
                and not metric_method_name.startswith('plot_'):
            safe_patch(FLAVOR_NAME, metrics, metric_method_name, patched_metric_api, manage_run=False)


def eval_and_log_metrics(model, X, y_true, *, prefix, sample_weight=None):
    """
    Computes and logs metrics (and artifacts) for the given model and labeled dataset.
    The metrics/artifacts mirror what is auto-logged when training a model
    (see mlflow.sklearn.autolog).

    :param model: The model to be evaluated.
    :param X: The features for the evaluation dataset.
    :param y_true: The labels for the evaluation dataset.
    :param prefix: Prefix used to name metrics and artifacts.
    :param sample_weight: Per-sample weights to apply in the computation of metrics/artifacts.
    :return: The dict of logged metrics. Artifacts can be retrieved by inspecting the run.

    ** Example **

    .. code-block:: python

        from sklearn.linear_model import LinearRegression
        import mlflow

        # enable autologging
        mlflow.sklearn.autolog()

        # prepare training data
        X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        y = np.dot(X, np.array([1, 2])) + 3

        # prepare evaluation data
        X_eval = np.array([[3, 3], [3, 4]])
        y_eval = np.dot(X_eval, np.array([1,2])) + 3

        # train a model
        model = LinearRegression()
        with mlflow.start_run() as run:
            model.fit(X, y)
            metrics = mlflow.sklearn.eval_and_log_metrics(model, X_eval, y_eval, prefix="val_")


    Each metric's and artifact's name is prefixed with `prefix`, e.g., in the previous example the
    metrics and artifacts are named 'val_XXXXX'. Note that training-time metrics are auto-logged
    as 'training_XXXXX'. Metrics and artifacts are logged under the currently active run if one
    exists, otherwise a new run is started and left active.

    Raises an error if:
      - prefix is empty
      - model is not an sklearn estimator or does not support the 'predict' method
    """
    status = _get_autolog_training_status()
    with status.disable_log_post_training_metrics():
        return _eval_and_log_metrics_impl(
            model, X, y_true, prefix=prefix, sample_weight=sample_weight)


def _eval_and_log_metrics_impl(model, X, y_true, *, prefix, sample_weight=None):
    from mlflow.sklearn.utils import _log_estimator_content
    from sklearn.base import BaseEstimator

    if prefix is None or prefix == "":
        raise ValueError("Must specify a non-empty prefix")

    if not isinstance(model, BaseEstimator):
        raise ValueError(
            "The provided model was not a sklearn estimator. Please ensure the passed-in model is "
            "a sklearn estimator subclassing sklearn.base.BaseEstimator"
        )

    if not hasattr(model, "predict"):
        raise ValueError(
            "Model does not support predictions. Please pass a model object defining a predict() "
            "method"
        )

    active_run = mlflow.active_run()
    run = active_run if active_run is not None else mlflow.start_run()

    with MlflowAutologgingQueueingClient() as autologging_client:
        metrics = _log_estimator_content(
            autologging_client=autologging_client,
            estimator=model,
            run_id=run.info.run_id,
            prefix=prefix,
            X=X,
            y_true=y_true,
            sample_weight=sample_weight,
        )

    return metrics
