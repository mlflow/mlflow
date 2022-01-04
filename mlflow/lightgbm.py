"""
The ``mlflow.lightgbm`` module provides an API for logging and loading LightGBM models.
This module exports LightGBM models with the following flavors:

LightGBM (native) format
    This is the main flavor that can be loaded back into LightGBM.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.

.. _lightgbm.Booster:
    https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html#lightgbm.Booster
.. _lightgbm.Booster.save_model:
    https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html
    #lightgbm.Booster.save_model
.. _lightgbm.train:
    https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html#lightgbm-train
.. _scikit-learn API:
    https://lightgbm.readthedocs.io/en/latest/Python-API.html#scikit-learn-api
"""
import os
import yaml
import json
import tempfile
import shutil
import logging
import functools
from copy import deepcopy
from packaging.version import Version

import mlflow
from mlflow import pyfunc
from mlflow.models import Model, infer_signature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils import _get_fully_qualified_class_name
from mlflow.utils.environment import (
    _mlflow_conda_env,
    _validate_env_arguments,
    _process_pip_requirements,
    _process_conda_env,
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.file_utils import write_to
from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.exceptions import MlflowException
from mlflow.utils.arguments_utils import _get_arg_names
from mlflow.utils.autologging_utils import (
    autologging_integration,
    safe_patch,
    picklable_exception_safe_function,
    get_mlflow_run_params_for_fn_args,
    INPUT_EXAMPLE_SAMPLE_ROWS,
    resolve_input_example_and_signature,
    InputExampleInfo,
    ENSURE_AUTOLOGGING_ENABLED_TEXT,
    batch_metrics_logger,
    MlflowAutologgingQueueingClient,
)
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

FLAVOR_NAME = "lightgbm"

_logger = logging.getLogger(__name__)


def get_default_pip_requirements(include_cloudpickle=False):
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    pip_deps = [_get_pinned_requirement("lightgbm")]
    if include_cloudpickle:
        pip_deps.append(_get_pinned_requirement("cloudpickle"))
    return pip_deps


def get_default_conda_env(include_cloudpickle=False):
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements(include_cloudpickle))


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    lgb_model,
    path,
    conda_env=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
):
    """
    Save a LightGBM model to a path on the local file system.

    :param lgb_model: LightGBM model (an instance of `lightgbm.Booster`_) to be saved.
                      Note that models that implement the `scikit-learn API`_  are not supported.
    :param path: Local path where the model is to be saved.
    :param conda_env: {{ conda_env }}
    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.

    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
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
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    """
    import lightgbm as lgb

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException("Path '{}' already exists".format(path))
    model_data_subpath = "model.lgb" if isinstance(lgb_model, lgb.Booster) else "model.pkl"
    model_data_path = os.path.join(path, model_data_subpath)
    os.makedirs(path)
    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    # Save a LightGBM model
    _save_model(lgb_model, model_data_path)

    lgb_model_class = _get_fully_qualified_class_name(lgb_model)
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.lightgbm",
        data=model_data_subpath,
        env=_CONDA_ENV_FILE_NAME,
    )
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        lgb_version=lgb.__version__,
        data=model_data_subpath,
        model_class=lgb_model_class,
    )
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements(
                include_cloudpickle=not isinstance(lgb_model, lgb.Booster)
            )
            # To ensure `_load_pyfunc` can successfully load the model during the dependency
            # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
            inferred_reqs = mlflow.models.infer_pip_requirements(
                path,
                FLAVOR_NAME,
                fallback=default_reqs,
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs,
            pip_requirements,
            extra_pip_requirements,
        )
    else:
        conda_env, pip_requirements, pip_constraints = _process_conda_env(conda_env)

    with open(os.path.join(path, _CONDA_ENV_FILE_NAME), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # Save `constraints.txt` if necessary
    if pip_constraints:
        write_to(os.path.join(path, _CONSTRAINTS_FILE_NAME), "\n".join(pip_constraints))

    # Save `requirements.txt`
    write_to(os.path.join(path, _REQUIREMENTS_FILE_NAME), "\n".join(pip_requirements))


def _save_model(lgb_model, model_path):
    """
    LightGBM Boosters are saved using the built-in method `save_model()`,
    whereas LightGBM scikit-learn models are serialized using Cloudpickle.
    """
    import lightgbm as lgb

    if isinstance(lgb_model, lgb.Booster):
        lgb_model.save_model(model_path)
    else:
        import cloudpickle

        with open(model_path, "wb") as out:
            cloudpickle.dump(lgb_model, out)


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    lgb_model,
    artifact_path,
    conda_env=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    **kwargs,
):
    """
    Log a LightGBM model as an MLflow artifact for the current run.

    :param lgb_model: LightGBM model (an instance of `lightgbm.Booster`_) to be saved.
                      Note that models that implement the `scikit-learn API`_  are not supported.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: {{ conda_env }}
    :param registered_model_name: If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
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
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example will be converted to a Pandas DataFrame and then
                          serialized to json using the Pandas split-oriented format. Bytes are
                          base64-encoded.
    :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param kwargs: kwargs to pass to `lightgbm.Booster.save_model`_ method.
    """
    Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.lightgbm,
        registered_model_name=registered_model_name,
        lgb_model=lgb_model,
        conda_env=conda_env,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        **kwargs,
    )


def _load_model(path):
    """
    Load Model Implementation.
    :param path: Local filesystem path to
                    the MLflow Model with the ``lightgbm`` flavor (MLflow < 1.23.0) or
                    the top-level MLflow Model directory (MLflow >= 1.23.0).
    """

    model_dir = os.path.dirname(path) if os.path.isfile(path) else path
    flavor_conf = _get_flavor_configuration(model_path=model_dir, flavor_name=FLAVOR_NAME)

    model_class = flavor_conf.get("model_class", "lightgbm.basic.Booster")
    lgb_model_path = os.path.join(model_dir, flavor_conf.get("data"))

    if model_class == "lightgbm.basic.Booster":
        import lightgbm as lgb

        model = lgb.Booster(model_file=lgb_model_path)
    else:
        # LightGBM scikit-learn models are deserialized using Cloudpickle.
        import cloudpickle

        with open(lgb_model_path, "rb") as f:
            model = cloudpickle.load(f)

    return model


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.

    :param path: Local filesystem path to the MLflow Model with the ``lightgbm`` flavor.
    """
    return _LGBModelWrapper(_load_model(path))


def load_model(model_uri, dst_path=None):
    """
    Load a LightGBM model from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
                      artifact-locations>`_.
    :param dst_path: The local filesystem path to which to download the model artifact.
                     This directory must already exist. If unspecified, a local output
                     path will be created.

    :return: A LightGBM model (an instance of `lightgbm.Booster`_) or a LightGBM scikit-learn
             model, depending on the saved model class specification.
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    return _load_model(path=local_model_path)


class _LGBModelWrapper:
    def __init__(self, lgb_model):
        self.lgb_model = lgb_model

    def predict(self, dataframe):
        return self.lgb_model.predict(dataframe)


def _autolog_callback(env, metrics_logger, eval_results):
    res = {}
    for data_name, eval_name, value, _ in env.evaluation_result_list:
        key = data_name + "-" + eval_name
        res[key] = value
    metrics_logger.record_metrics(res, env.iteration)
    eval_results.append(res)


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_input_examples=False,
    log_model_signatures=True,
    log_models=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
):  # pylint: disable=unused-argument
    """
    Enables (or disables) and configures autologging from LightGBM to MLflow. Logs the following:

    - parameters specified in `lightgbm.train`_.
    - metrics on each iteration (if ``valid_sets`` specified).
    - metrics at the best iteration (if ``early_stopping_rounds`` specified or ``early_stopping``
        callback is set).
    - feature importance (both "split" and "gain") as JSON files and plots.
    - trained model, including:
        - an example of valid input.
        - inferred signature of the inputs and outputs of the model.

    Note that the `scikit-learn API`_ is not supported.

    :param log_input_examples: If ``True``, input examples from training datasets are collected and
                               logged along with LightGBM model artifacts during training. If
                               ``False``, input examples are not logged.
                               Note: Input examples are MLflow model attributes
                               and are only collected if ``log_models`` is also ``True``.
    :param log_model_signatures: If ``True``,
                                 :py:class:`ModelSignatures <mlflow.models.ModelSignature>`
                                 describing model inputs and outputs are collected and logged along
                                 with LightGBM model artifacts during training. If ``False``,
                                 signatures are not logged.
                                 Note: Model signatures are MLflow model attributes
                                 and are only collected if ``log_models`` is also ``True``.
    :param log_models: If ``True``, trained models are logged as MLflow model artifacts.
                       If ``False``, trained models are not logged.
                       Input examples and model signatures, which are attributes of MLflow models,
                       are also omitted when ``log_models`` is ``False``.
    :param disable: If ``True``, disables the LightGBM autologging integration. If ``False``,
                    enables the LightGBM autologging integration.
    :param exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
                      If ``False``, autologged content is logged to the active fluent run,
                      which may be user-created.
    :param disable_for_unsupported_versions: If ``True``, disable autologging for versions of
                      lightgbm that have not been tested against this version of the MLflow client
                      or are incompatible.
    :param silent: If ``True``, suppress all event logs and warnings from MLflow during LightGBM
                   autologging. If ``False``, show all events and warnings during LightGBM
                   autologging.
    """
    import lightgbm
    import numpy as np

    # Patching this function so we can get a copy of the data given to Dataset.__init__
    #   to use as an input example and for inferring the model signature.
    #   (there is no way to get the data back from a Dataset object once it is consumed by train)
    # We store it on the Dataset object so the train function is able to read it.
    def __init__(original, self, *args, **kwargs):
        data = args[0] if len(args) > 0 else kwargs.get("data")

        if data is not None:
            try:
                if isinstance(data, str):
                    raise Exception(
                        "cannot gather example input when dataset is loaded from a file."
                    )

                input_example_info = InputExampleInfo(
                    input_example=deepcopy(data[:INPUT_EXAMPLE_SAMPLE_ROWS])
                )
            except Exception as e:
                input_example_info = InputExampleInfo(error_msg=str(e))

            setattr(self, "input_example_info", input_example_info)

        original(self, *args, **kwargs)

    def train(original, *args, **kwargs):
        def record_eval_results(eval_results, metrics_logger):
            """
            Create a callback function that records evaluation results.
            """
            return picklable_exception_safe_function(
                functools.partial(
                    _autolog_callback, metrics_logger=metrics_logger, eval_results=eval_results
                )
            )

        def log_feature_importance_plot(features, importance, importance_type):
            """
            Log feature importance plot.
            """
            import matplotlib.pyplot as plt

            indices = np.argsort(importance)
            features = np.array(features)[indices]
            importance = importance[indices]
            num_features = len(features)

            # If num_features > 10, increase the figure height to prevent the plot
            # from being too dense.
            w, h = [6.4, 4.8]  # matplotlib's default figure size
            h = h + 0.1 * num_features if num_features > 10 else h
            fig, ax = plt.subplots(figsize=(w, h))

            yloc = np.arange(num_features)
            ax.barh(yloc, importance, align="center", height=0.5)
            ax.set_yticks(yloc)
            ax.set_yticklabels(features)
            ax.set_xlabel("Importance")
            ax.set_title("Feature Importance ({})".format(importance_type))
            fig.tight_layout()

            tmpdir = tempfile.mkdtemp()
            try:
                # pylint: disable=undefined-loop-variable
                filepath = os.path.join(tmpdir, "feature_importance_{}.png".format(imp_type))
                fig.savefig(filepath)
                mlflow.log_artifact(filepath)
            finally:
                plt.close(fig)
                shutil.rmtree(tmpdir)

        autologging_client = MlflowAutologgingQueueingClient()

        # logging booster params separately via mlflow.log_params to extract key/value pairs
        # and make it easier to compare them across runs.
        booster_params = args[0] if len(args) > 0 else kwargs["params"]
        autologging_client.log_params(run_id=mlflow.active_run().info.run_id, params=booster_params)

        unlogged_params = [
            "params",
            "train_set",
            "valid_sets",
            "valid_names",
            "fobj",
            "feval",
            "init_model",
            "learning_rates",
            "callbacks",
        ]
        if Version(lightgbm.__version__) <= Version("3.3.1"):
            # The parameter `evals_result` in `lightgbm.train` is removed in this PR:
            # https://github.com/microsoft/LightGBM/pull/4882
            unlogged_params.append("evals_result")

        params_to_log_for_fn = get_mlflow_run_params_for_fn_args(
            original, args, kwargs, unlogged_params
        )
        autologging_client.log_params(
            run_id=mlflow.active_run().info.run_id, params=params_to_log_for_fn
        )

        param_logging_operations = autologging_client.flush(synchronous=False)

        all_arg_names = _get_arg_names(original)
        num_pos_args = len(args)

        # adding a callback that records evaluation results.
        eval_results = []
        callbacks_index = all_arg_names.index("callbacks")
        run_id = mlflow.active_run().info.run_id
        with batch_metrics_logger(run_id) as metrics_logger:
            callback = record_eval_results(eval_results, metrics_logger)
            if num_pos_args >= callbacks_index + 1:
                tmp_list = list(args)
                tmp_list[callbacks_index] += [callback]
                args = tuple(tmp_list)
            elif "callbacks" in kwargs and kwargs["callbacks"] is not None:
                kwargs["callbacks"] += [callback]
            else:
                kwargs["callbacks"] = [callback]

            # training model
            model = original(*args, **kwargs)

            # If early stopping is activated, logging metrics at the best iteration
            # as extra metrics with the max step + 1.
            early_stopping = model.best_iteration > 0
            if early_stopping:
                extra_step = len(eval_results)
                autologging_client.log_metrics(
                    run_id=mlflow.active_run().info.run_id,
                    metrics={
                        "stopped_iteration": extra_step,
                        # best_iteration is set even if training does not stop early.
                        "best_iteration": model.best_iteration,
                    },
                )
                # iteration starts from 1 in LightGBM.
                last_iter_results = eval_results[model.best_iteration - 1]
                autologging_client.log_metrics(
                    run_id=mlflow.active_run().info.run_id,
                    metrics=last_iter_results,
                    step=extra_step,
                )
                early_stopping_logging_operations = autologging_client.flush(synchronous=False)

        # logging feature importance as artifacts.
        for imp_type in ["split", "gain"]:
            features = model.feature_name()
            importance = model.feature_importance(importance_type=imp_type)
            try:
                log_feature_importance_plot(features, importance, imp_type)
            except Exception:
                _logger.exception(
                    "Failed to log feature importance plot. LightGBM autologging "
                    "will ignore the failure and continue. Exception: "
                )

            imp = {ft: imp for ft, imp in zip(features, importance.tolist())}
            tmpdir = tempfile.mkdtemp()
            try:
                filepath = os.path.join(tmpdir, "feature_importance_{}.json".format(imp_type))
                with open(filepath, "w") as f:
                    json.dump(imp, f, indent=2)
                mlflow.log_artifact(filepath)
            finally:
                shutil.rmtree(tmpdir)

        # train_set must exist as the original train function already ran successfully
        train_set = args[1] if len(args) > 1 else kwargs.get("train_set")

        # it is possible that the dataset was constructed before the patched
        #   constructor was applied, so we cannot assume the input_example_info exists
        input_example_info = getattr(train_set, "input_example_info", None)

        def get_input_example():
            if input_example_info is None:
                raise Exception(ENSURE_AUTOLOGGING_ENABLED_TEXT)
            if input_example_info.error_msg is not None:
                raise Exception(input_example_info.error_msg)
            return input_example_info.input_example

        def infer_model_signature(input_example):
            model_output = model.predict(input_example)
            model_signature = infer_signature(input_example, model_output)
            return model_signature

        # Whether to automatically log the trained model based on boolean flag.
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
                model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
            )

        param_logging_operations.await_completion()
        if early_stopping:
            early_stopping_logging_operations.await_completion()

        return model

    safe_patch(FLAVOR_NAME, lightgbm, "train", train, manage_run=True)
    safe_patch(FLAVOR_NAME, lightgbm.Dataset, "__init__", __init__)
