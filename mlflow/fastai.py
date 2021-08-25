"""
The ``mlflow.fastai`` module provides an API for logging and loading fast.ai models. This module
exports fast.ai models with the following flavors:

fastai (native) format
    This is the main flavor that can be loaded back into fastai.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.

.. _fastai.Learner:
    https://docs.fast.ai/basic_train.html#Learner
.. _fastai.Learner.export:
    https://docs.fast.ai/basic_train.html#Learner.export
"""
import os
import yaml
import tempfile
import shutil
import pandas as pd
import numpy as np

from mlflow import pyfunc
from mlflow.models import Model, ModelSignature, ModelInputExample
import mlflow.tracking
from mlflow.exceptions import MlflowException
from mlflow.models.utils import _save_example
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
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
from mlflow.utils.annotations import experimental, deprecated
from mlflow.utils.autologging_utils import (
    try_mlflow_log,
    log_fn_args_as_params,
    safe_patch,
    batch_metrics_logger,
    autologging_integration,
    ExceptionSafeClass,
)
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS


FLAVOR_NAME = "fastai"


@deprecated(
    alternative="fast.ai V2 support, which will be available in MLflow soon",
    since="MLflow version 1.20.0",
    impact="This method will no longer support fast.ai V1 in a near future release.",
)
def get_default_pip_requirements(include_cloudpickle=False):
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    pip_deps = [_get_pinned_requirement("fastai")]
    if include_cloudpickle:
        pip_deps.append(_get_pinned_requirement("cloudpickle"))

    return pip_deps


@deprecated(
    alternative="fast.ai V2 support, which will be available in MLflow soon",
    since="MLflow version 1.20.0",
    impact="This method will no longer support fast.ai V1 in a near future release.",
)
def get_default_conda_env(include_cloudpickle=False):
    """
    :return: The default Conda environment as a dictionary for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.


    .. code-block:: python
        :caption: Example

        import mlflow.fastai

        # Start MLflow session and log the fastai learner model
        with mlflow.start_run():
           model.fit(epochs, learning_rate)
           mlflow.fastai.log_model(model, "model")

        # Fetch the default conda environment
        env = mlflow.fastai.get_default_conda_env()
        print("conda environment: {}".format(env))

    .. code-block:: text
        :caption: Output

        conda environment: {'name': 'mlflow-env',
                            'channels': ['defaults', 'conda-forge'],
                            'dependencies': ['python=3.7.5', 'fastai=1.0.61',
                                             'pip', {'pip': ['mlflow']}]}
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements(include_cloudpickle))


@deprecated(
    alternative="fast.ai V2 support, which will be available in MLflow soon",
    since="MLflow version 1.20.0",
)
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    fastai_learner,
    path,
    conda_env=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    **kwargs
):
    """
    Save a fastai Learner to a path on the local file system.

    :param fastai_learner: fastai Learner to be saved.
    :param path: Local path where the model is to be saved.
    :param conda_env: {{ conda_env }}
    :param mlflow_model: MLflow model config this flavor is being added to.

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

    :param kwargs: kwargs to pass to ``Learner.save`` method.

    .. code-block:: python
        :caption: Example

        import os

        import mlflow.fastai

        # Create a fastai Learner model
        model = ...

        # Start MLflow session and save model to current working directory
        with mlflow.start_run():
            model.fit(epochs, learning_rate)
            mlflow.fastai.save_model(model, 'model')

        # Load saved model for inference
        model_uri = "{}/{}".format(os.getcwd(), 'model')
        loaded_model = mlflow.fastai.load_model(model_uri)
        results = loaded_model.predict(predict_data)
    """
    import fastai
    from pathlib import Path

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException("Path '{}' already exists".format(path))
    model_data_subpath = "model.fastai"
    model_data_path = os.path.join(path, model_data_subpath)
    model_data_path = Path(model_data_path)
    os.makedirs(path)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    # Save an Learner
    fastai_learner.export(model_data_path, **kwargs)

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.fastai",
        data=model_data_subpath,
        env=_CONDA_ENV_FILE_NAME,
    )
    mlflow_model.add_flavor(FLAVOR_NAME, fastai_version=fastai.__version__, data=model_data_subpath)
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
            # To ensure `_load_pyfunc` can successfully load the model during the dependency
            # inference, `mlflow_model.save` must be called beforehand to save an MLmodel file.
            inferred_reqs = mlflow.models.infer_pip_requirements(
                path, FLAVOR_NAME, fallback=default_reqs,
            )
            default_reqs = sorted(set(inferred_reqs).union(default_reqs))
        else:
            default_reqs = None
        conda_env, pip_requirements, pip_constraints = _process_pip_requirements(
            default_reqs, pip_requirements, extra_pip_requirements,
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


@deprecated(
    alternative="fast.ai V2 support, which will be available in MLflow soon",
    since="MLflow version 1.20.0",
)
@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    fastai_learner,
    artifact_path,
    conda_env=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    **kwargs
):
    """
    Log a fastai model as an MLflow artifact for the current run.

    :param fastai_learner: Fastai model (an instance of `fastai.Learner`_) to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: {{ conda_env }}
    :param registered_model_name: This argument may change or be removed in a
                                  future release without warning. If given, create a model
                                  version under ``registered_model_name``, also creating a
                                  registered model if one with the given name does not exist.

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

    :param kwargs: kwargs to pass to `fastai.Learner.export`_ method.
    :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}

    .. code-block:: python
        :caption: Example

        import fastai.vision as vis
        import mlflow.fastai
        from mlflow.tracking import MlflowClient

        def main(epochs=5, learning_rate=0.01):
            # Download and untar the MNIST data set
            path = vis.untar_data(vis.URLs.MNIST_SAMPLE)

           # Prepare, transform, and normalize the data
           data = vis.ImageDataBunch.from_folder(path, ds_tfms=(vis.rand_pad(2, 28), []), bs=64)
           data.normalize(vis.imagenet_stats)

           # Create the CNN Learner model
           model = vis.cnn_learner(data, vis.models.resnet18, metrics=vis.accuracy)

           # Start MLflow session and log model
           with mlflow.start_run() as run:
                model.fit(epochs, learning_rate)
                mlflow.fastai.log_model(model, 'model')

           # fetch the logged model artifacts
           artifacts = [f.path for f in MlflowClient().list_artifacts(run.info.run_id, 'model')]
           print("artifacts: {}".format(artifacts))

        main()

    .. code-block:: text
        :caption: Output

        artifacts: ['model/MLmodel', 'model/conda.yaml', 'model/model.fastai']
    """
    Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.fastai,
        registered_model_name=registered_model_name,
        fastai_learner=fastai_learner,
        conda_env=conda_env,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        **kwargs,
    )


def _load_model(path):
    from fastai.basic_train import load_learner

    abspath = os.path.abspath(path)
    path, file = os.path.split(abspath)
    return load_learner(path, file)


class _FastaiModelWrapper:
    def __init__(self, learner):
        self.learner = learner

    def predict(self, dataframe):
        from fastai.tabular import TabularList
        from fastai.basic_data import DatasetType

        test_data = TabularList.from_df(dataframe, cont_names=self.learner.data.cont_names)
        self.learner.data.add_test(test_data)
        preds, target = self.learner.get_preds(DatasetType.Test)
        preds = pd.Series(map(np.array, preds.numpy()), name="predictions")
        target = pd.Series(target.numpy(), name="target")
        return pd.concat([preds, target], axis="columns")


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.

    :param path: Local filesystem path to the MLflow Model with the ``fastai`` flavor.
    """
    return _FastaiModelWrapper(_load_model(path))


@deprecated(
    alternative="fast.ai V2 support, which will be available in MLflow soon",
    since="MLflow version 1.20.0",
)
def load_model(model_uri):
    """
    Load a fastai model from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
                      artifact-locations>`_.

    :return: A fastai model (an instance of `fastai.Learner`_).

    .. code-block:: python
        :caption: Example

        import mlflow.fastai

        # Define the Learner model
        model = ...

        # log the fastai Leaner model
        with mlflow.start_run() as run:
            model.fit(epochs, learning_rate)
            mlflow.fastai.log_model(model, "model")

        # Load the model for scoring
        model_uri = "runs:/{}/model".format(run.info.run_id)
        loaded_model = mlflow.fastai.load_model(model_uri)
        results = loaded_model.predict(predict_data)
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    model_file_path = os.path.join(local_model_path, flavor_conf.get("data", "model.fastai"))
    return _load_model(path=model_file_path)


@deprecated(
    alternative="fast.ai V2 support, which will be available in MLflow soon",
    since="MLflow version 1.20.0",
    impact="This method will no longer support fast.ai V1 in a near future release.",
)
@experimental
@autologging_integration(FLAVOR_NAME)
def autolog(
    log_models=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
):  # pylint: disable=unused-argument
    """
    Enable automatic logging from Fastai to MLflow.
    Logs loss and any other metrics specified in the fit
    function, and optimizer data as parameters. Model checkpoints
    are logged as artifacts to a 'models' directory.

    MLflow will also log the parameters of the
    `EarlyStoppingCallback <https://docs.fast.ai/callbacks.html#EarlyStoppingCallback>`_
    and `OneCycleScheduler <https://docs.fast.ai/callbacks.html#OneCycleScheduler>`_ callbacks

    :param log_models: If ``True``, trained models are logged as MLflow model artifacts.
                       If ``False``, trained models are not logged.
    :param disable: If ``True``, disables the Fastai autologging integration. If ``False``,
                    enables the Fastai autologging integration.
    :param exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
                      If ``False``, autologged content is logged to the active fluent run,
                      which may be user-created.
    :param disable_for_unsupported_versions: If ``True``, disable autologging for versions of
                      fastai that have not been tested against this version of the MLflow client
                      or are incompatible.
    :param silent: If ``True``, suppress all event logs and warnings from MLflow during Fastai
                   autologging. If ``False``, show all events and warnings during Fastai
                   autologging.

    .. code-block:: python
        :caption: Example

        # This is a modified example from
        # https://github.com/mlflow/mlflow/tree/master/examples/fastai
        # demonstrating autolog capabilites.

        import fastai.vision as vis
        import mlflow.fastai
        from mlflow.tracking import MlflowClient

        def print_auto_logged_info(r):
            tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
            artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
            print("run_id: {}".format(r.info.run_id))
            print("artifacts: {}".format(artifacts))
            print("params: {}".format(r.data.params))
            print("metrics: {}".format(r.data.metrics))
            print("tags: {}".format(tags))

        def main(epochs=5, learning_rate=0.01):
            # Download and untar the MNIST data set
            path = vis.untar_data(vis.URLs.MNIST_SAMPLE)

            # Prepare, transform, and normalize the data
            data = vis.ImageDataBunch.from_folder(path, ds_tfms=(vis.rand_pad(2, 28), []), bs=64)
            data.normalize(vis.imagenet_stats)

            # Create CNN the Learner model
            model = vis.cnn_learner(data, vis.models.resnet18, metrics=vis.accuracy)

            # Enable auto logging
            mlflow.fastai.autolog()

            # Start MLflow session
            with mlflow.start_run() as run:
                model.fit(epochs, learning_rate)

            # fetch the auto logged parameters, metrics, and artifacts
            print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

        main()

    .. code-block:: text
        :caption: output

        run_id: 5a23dcbcaa334637814dbce7a00b2f6a
        artifacts: ['model/MLmodel', 'model/conda.yaml', 'model/model.fastai']
        params: {'wd': 'None',
                 'bn_wd': 'True',
                 'opt_func': 'Adam',
                 'epochs': '5', '
                 train_bn': 'True',
                 'num_layers': '60',
                 'lr': '0.01',
                 'true_wd': 'True'}
        metrics: {'train_loss': 0.024,
                  'accuracy': 0.99214,
                  'valid_loss': 0.021}
        # Tags model summary omitted too long
        tags: {...}

    .. figure:: ../_static/images/fastai_autolog.png

        Fastai autologged MLflow entities
    """
    from fastai.basic_train import LearnerCallback, Learner
    from fastai.callbacks.hooks import model_summary, layers_info
    from fastai.callbacks import EarlyStoppingCallback, OneCycleScheduler

    def getFastaiCallback(metrics_logger, learner):
        class __MLflowFastaiCallback(LearnerCallback, metaclass=ExceptionSafeClass):
            """
            Callback for auto-logging metrics and parameters.
            Records model structural information as params when training begins
            """

            def __init__(self, learner):
                super().__init__(learner)
                self.learner = learner
                self.opt = self.learn.opt
                self.metrics_names = ["train_loss", "valid_loss"] + [
                    o.__name__ for o in learner.metrics
                ]

            def on_epoch_end(self, **kwargs):
                """
                Log loss and other metrics values after each epoch
                """
                if kwargs["smooth_loss"] is None or kwargs["last_metrics"] is None:
                    return
                epoch = kwargs["epoch"]
                metrics = [kwargs["smooth_loss"]] + kwargs["last_metrics"]
                metrics = map(float, metrics)
                metrics = dict(zip(self.metrics_names, metrics))
                metrics_logger.record_metrics(metrics, epoch)

            def on_train_begin(self, **kwargs):
                info = layers_info(self.learner)
                try_mlflow_log(mlflow.log_param, "num_layers", len(info))
                try_mlflow_log(mlflow.log_param, "opt_func", self.opt_func.func.__name__)

                if hasattr(self.opt, "true_wd"):
                    try_mlflow_log(mlflow.log_param, "true_wd", self.opt.true_wd)

                if hasattr(self.opt, "bn_wd"):
                    try_mlflow_log(mlflow.log_param, "bn_wd", self.opt.bn_wd)

                if hasattr(self.opt, "train_bn"):
                    try_mlflow_log(mlflow.log_param, "train_bn", self.train_bn)

                summary = model_summary(self.learner)

                tempdir = tempfile.mkdtemp()
                try:
                    summary_file = os.path.join(tempdir, "model_summary.txt")
                    with open(summary_file, "w") as f:
                        f.write(summary)
                    try_mlflow_log(mlflow.log_artifact, local_path=summary_file)
                finally:
                    shutil.rmtree(tempdir)

            def on_train_end(self, **kwargs):
                if log_models:
                    try_mlflow_log(log_model, self.learner, artifact_path="model")

        return __MLflowFastaiCallback(learner)

    def _find_callback_of_type(callback_type, callbacks):
        for callback in callbacks:
            if isinstance(callback, callback_type):
                return callback
        return None

    def _log_early_stop_callback_params(callback):
        if callback:
            try:
                earlystopping_params = {
                    "early_stop_monitor": callback.monitor,
                    "early_stop_min_delta": callback.min_delta,
                    "early_stop_patience": callback.patience,
                    "early_stop_mode": callback.mode,
                }
                try_mlflow_log(mlflow.log_params, earlystopping_params)
            except Exception:  # pylint: disable=W0703
                return

    def _log_one_cycle_callback_params(callback):
        if callback:
            try:
                params = {
                    "lr_max": callback.lr_max,
                    "div_factor": callback.div_factor,
                    "pct_start": callback.pct_start,
                    "final_div": callback.final_div,
                    "tot_epochs": callback.tot_epochs,
                    "start_epoch": callback.start_epoch,
                    "moms": callback.moms,
                }
                try_mlflow_log(mlflow.log_params, params)
            except Exception:  # pylint: disable=W0703
                return

    def _run_and_log_function(self, original, args, kwargs, unlogged_params, callback_arg_index):
        log_fn_args_as_params(original, list(args), kwargs, unlogged_params)

        callbacks = [cb(self) for cb in self.callback_fns] + (self.callbacks or [])

        run_id = mlflow.active_run().info.run_id
        with batch_metrics_logger(run_id) as metrics_logger:
            mlflowFastaiCallback = getFastaiCallback(metrics_logger, self)

            # Checking if the 'callback' argument of the function is set
            if len(args) > callback_arg_index:
                tmp_list = list(args)
                callbacks += list(args[callback_arg_index])
                tmp_list[callback_arg_index] += [mlflowFastaiCallback]
                args = tuple(tmp_list)
            elif kwargs.get("callbacks"):
                callbacks += list(kwargs["callbacks"])
                kwargs["callbacks"] += [mlflowFastaiCallback]
            else:
                kwargs["callbacks"] = [mlflowFastaiCallback]

            early_stop_callback = _find_callback_of_type(EarlyStoppingCallback, callbacks)
            one_cycle_callback = _find_callback_of_type(OneCycleScheduler, callbacks)

            _log_early_stop_callback_params(early_stop_callback)
            _log_one_cycle_callback_params(one_cycle_callback)

            result = original(self, *args, **kwargs)

        return result

    def fit(original, self, *args, **kwargs):
        unlogged_params = ["self", "callbacks", "learner"]
        return _run_and_log_function(self, original, args, kwargs, unlogged_params, 3)

    safe_patch(FLAVOR_NAME, Learner, "fit", fit, manage_run=True)
