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
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml

import mlflow.tracking
from mlflow import pyfunc
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.autologging_utils import (
    autologging_integration,
    batch_metrics_logger,
    log_fn_args_as_params,
    safe_patch,
)
from mlflow.utils.docstring_utils import LOG_MODEL_PARAM_DOCS, format_docstring
from mlflow.utils.environment import (
    _CONDA_ENV_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _mlflow_conda_env,
    _process_conda_env,
    _process_pip_requirements,
    _PythonEnv,
    _validate_env_arguments,
)
from mlflow.utils.file_utils import get_total_file_size, write_to
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "fastai"

_MODEL_DATA_SUBPATH = "model.fastai"

_logger = logging.getLogger(__name__)


def get_default_pip_requirements(include_cloudpickle=False):
    """
    Returns:
        A list of default pip requirements for MLflow Models produced by this flavor.
        Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
        that, at minimum, contains these requirements.
    """
    pip_deps = [_get_pinned_requirement("fastai")]
    if include_cloudpickle:
        pip_deps.append(_get_pinned_requirement("cloudpickle"))

    return pip_deps


def get_default_conda_env(include_cloudpickle=False):
    """
    Returns:
        The default Conda environment for MLflow Models produced by calls to
        :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    fastai_learner,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    **kwargs,
):
    """Save a fastai Learner to a path on the local file system.

    Args:
        fastai_learner: fastai Learner to be saved.
        path: Local path where the model is to be saved.
        conda_env: {{ conda_env }}
        code_paths: A list of local filesystem paths to Python file dependencies (or directories
            containing file dependencies). These files are *prepended* to the system
            path when the model is loaded.
        mlflow_model: MLflow model config this flavor is being added to.
        signature: Describes model input and output Schema. The model signature can be inferred
            from datasets with valid model input (e.g. the training dataset with target
            column omitted) and valid model output (e.g. model predictions generated on
            the training dataset), for example:

            .. code-block:: python

                from mlflow.models import infer_signature

                train = df.drop_column("target_label")
                predictions = ...  # compute model predictions
                signature = infer_signature(train, predictions)

        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}
        kwargs: kwargs to pass to ``Learner.save`` method.

    .. code-block:: python
        :caption: Example

        import os

        import mlflow.fastai

        # Create a fastai Learner model
        model = ...

        # Start MLflow session and save model to current working directory
        with mlflow.start_run():
            model.fit(epochs, learning_rate)
            mlflow.fastai.save_model(model, "model")

        # Load saved model for inference
        model_uri = "{}/{}".format(os.getcwd(), "model")
        loaded_model = mlflow.fastai.load_model(model_uri)
        results = loaded_model.predict(predict_data)
    """
    import fastai
    from fastai.callback.all import ParamScheduler

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
    model_data_subpath = _MODEL_DATA_SUBPATH
    model_data_path = os.path.join(path, model_data_subpath)
    model_data_path = Path(model_data_path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    saved_example = _save_example(mlflow_model, input_example, path)
    if signature is None and saved_example is not None:
        wrapped_model = _FastaiModelWrapper(fastai_learner)
        signature = _infer_signature_from_input_example(saved_example, wrapped_model)

    if signature is not None:
        mlflow_model.signature = signature
    if metadata is not None:
        mlflow_model.metadata = metadata

    # ParamScheduler currently is not pickable
    # hence it is been removed before export and added again after export
    cbs = [c for c in fastai_learner.cbs if isinstance(c, ParamScheduler)]
    fastai_learner.remove_cbs(cbs)
    fastai_learner.export(model_data_path, **kwargs)
    fastai_learner.add_cbs(cbs)

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.fastai",
        data=model_data_subpath,
        code=code_dir_subpath,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
    )
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        fastai_version=fastai.__version__,
        data=model_data_subpath,
        code=code_dir_subpath,
    )
    if size := get_total_file_size(path):
        mlflow_model.model_size_bytes = size
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
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

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    fastai_learner,
    artifact_path,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    **kwargs,
):
    """Log a fastai model as an MLflow artifact for the current run.

    Args:
        fastai_learner: Fastai model (an instance of `fastai.Learner`_) to be saved.
        artifact_path: Run-relative artifact path.
        conda_env: {{ conda_env }}
        code_paths: A list of local filesystem paths to Python file dependencies (or directories
            containing file dependencies). These files are *prepended* to the system
            path when the model is loaded.
        registered_model_name: This argument may change or be removed in a
            future release without warning. If given, create a model
            version under ``registered_model_name``, also creating a
            registered model if one with the given name does not exist.
        signature: Describes model input and output Schema. The model signature can be inferred
            from datasets with valid model input (e.g. the training dataset with target
            column omitted) and valid model output (e.g. model predictions generated on
            the training dataset), for example:

            .. code-block:: python

              from mlflow.models import infer_signature

              train = df.drop_column("target_label")
              predictions = ...  # compute model predictions
              signature = infer_signature(train, predictions)

        input_example: {{ input_example }}
        kwargs: kwargs to pass to `fastai.Learner.export`_ method.
        await_registration_for: Number of seconds to wait for the model version to finish
            being created and is in ``READY`` status. By default, the function
            waits for five minutes. Specify 0 or None to skip waiting.
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}

    Returns:
        A ModelInfo instance that contains the metadata of the logged model.

    .. code-block:: python
        :caption: Example

        import fastai.vision as vis
        import mlflow.fastai
        from mlflow import MlflowClient


        def main(epochs=5, learning_rate=0.01):
            # Download and untar the MNIST data set
            path = vis.untar_data(vis.URLs.MNIST_SAMPLE)
            # Prepare, transform, and normalize the data
            data = vis.ImageDataBunch.from_folder(
                path, ds_tfms=(vis.rand_pad(2, 28), []), bs=64
            )
            data.normalize(vis.imagenet_stats)
            # Create the CNN Learner model
            model = vis.cnn_learner(data, vis.models.resnet18, metrics=vis.accuracy)
            # Start MLflow session and log model
            with mlflow.start_run() as run:
                model.fit(epochs, learning_rate)
                mlflow.fastai.log_model(model, "model")
            # fetch the logged model artifacts
            artifacts = [
                f.path for f in MlflowClient().list_artifacts(run.info.run_id, "model")
            ]
            print(f"artifacts: {artifacts}")


        main()

    .. code-block:: text
        :caption: Output

        artifacts: ['model/MLmodel', 'model/conda.yaml', 'model/model.fastai']
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.fastai,
        registered_model_name=registered_model_name,
        fastai_learner=fastai_learner,
        conda_env=conda_env,
        code_paths=code_paths,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
        **kwargs,
    )


def _load_model(path):
    from fastai.learner import load_learner

    return load_learner(os.path.abspath(path))


class _FastaiModelWrapper:
    def __init__(self, learner):
        self.learner = learner

    def get_raw_model(self):
        """
        Returns the underlying model.
        """
        return self.learner

    def predict(
        self,
        dataframe,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            dataframe: Model input data.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Model predictions.
        """
        dl = self.learner.dls.test_dl(dataframe)
        preds, _ = self.learner.get_preds(dl=dl)
        return pd.Series(map(np.array, preds.numpy())).to_frame("predictions")


def _load_pyfunc(path):
    """Load PyFunc implementation. Called by ``pyfunc.load_model``.

    Args:
        path: Local filesystem path to the MLflow Model with the ``fastai`` flavor.

    """
    return _FastaiModelWrapper(_load_model(path))


def load_model(model_uri, dst_path=None):
    """Load a fastai model from a local file or a run.

    Args:
        model_uri: The location, in URI format, of the MLflow model. For example:

            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``

            For more information about supported URI schemes, see
            `Referencing Artifacts <https://www.mlflow.org/docs/latest/tracking.html#
            artifact-locations>`_.
        dst_path: The local filesystem path to which to download the model artifact.
            This directory must already exist. If unspecified, a local output
            path will be created.

    Returns:
        A fastai model (an instance of `fastai.Learner`_).

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
        model_uri = f"runs:/{run.info.run_id}/model"
        loaded_model = mlflow.fastai.load_model(model_uri)
        results = loaded_model.predict(predict_data)
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)

    model_file_path = os.path.join(local_model_path, flavor_conf.get("data", "model.fastai"))
    return _load_model(path=model_file_path)


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_models=True,
    log_datasets=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    registered_model_name=None,
    extra_tags=None,
):
    """
    Enable automatic logging from Fastai to MLflow.

    Logs loss and any other metrics specified in the fit function, and optimizer data as parameters.
    Model checkpoints are logged as artifacts to a 'models' directory.

    MLflow will also log the parameters of the
    `EarlyStoppingCallback <https://docs.fast.ai/callback.tracker.html#EarlyStoppingCallback>`_
    and `OneCycleScheduler <https://docs.fast.ai/callback.schedule.html#ParamScheduler>`_ callbacks

    Args:
        log_models: If ``True``, trained models are logged as MLflow model artifacts. If ``False``,
            trained models are not logged.
        log_datasets: If ``True``, dataset information is logged to MLflow Tracking. If ``False``,
            dataset information is not logged.
        disable: If ``True``, disables the Fastai autologging integration. If ``False``, enables
            the Fastai autologging integration.
        exclusive: If ``True``, autologged content is not logged to user-created fluent runs. If
            ``False``, autologged content is logged to the active fluent run, which may
            be user-created.
        disable_for_unsupported_versions: If ``True``, disable autologging for versions of fastai
            that have not been tested against this version of the MLflow client or are
            incompatible.
        silent: If ``True``, suppress all event logs and warnings from MLflow during Fastai
            autologging. If ``False``, show all events and warnings during Fastai autologging.
        registered_model_name: If given, each time a model is trained, it is registered as a new
            model version of the registered model with this name. The registered model is created
            if it does not already exist.
        extra_tags: A dictionary of extra tags to set on each managed run created by
            autologging.


    .. code-block:: python
        :caption: Example

        # This is a modified example from
        # https://github.com/mlflow/mlflow/tree/master/examples/fastai
        # demonstrating autolog capabilities.

        import fastai.vision as vis
        import mlflow.fastai
        from mlflow import MlflowClient


        def print_auto_logged_info(r):
            tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
            artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
            print(f"run_id: {r.info.run_id}")
            print(f"artifacts: {artifacts}")
            print(f"params: {r.data.params}")
            print(f"metrics: {r.data.metrics}")
            print(f"tags: {tags}")


        def main(epochs=5, learning_rate=0.01):
            # Download and untar the MNIST data set
            path = vis.untar_data(vis.URLs.MNIST_SAMPLE)

            # Prepare, transform, and normalize the data
            data = vis.ImageDataBunch.from_folder(
                path, ds_tfms=(vis.rand_pad(2, 28), []), bs=64
            )
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
    from fastai.callback.all import EarlyStoppingCallback, TrackerCallback
    from fastai.callback.hook import _print_shapes, find_bs, layer_info, module_summary
    from fastai.learner import Learner

    def getFastaiCallback(metrics_logger, is_fine_tune=False):
        from mlflow.fastai.callback import __MlflowFastaiCallback

        return __MlflowFastaiCallback(
            metrics_logger=metrics_logger,
            log_models=log_models,
            is_fine_tune=is_fine_tune,
        )

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
                    "early_stop_comp": callback.comp.__name__,
                }
                mlflow.log_params(earlystopping_params)
            except Exception:
                return

    def _log_model_info(learner):
        # The process executed here, are incompatible with TrackerCallback
        # Hence it is removed and add again after the execution
        remove_cbs = [cb for cb in learner.cbs if isinstance(cb, TrackerCallback)]
        if remove_cbs:
            learner.remove_cbs(remove_cbs)

        xb = learner.dls.train.one_batch()[: learner.dls.train.n_inp]
        infos = layer_info(learner, *xb)
        bs = find_bs(xb)
        inp_sz = _print_shapes((x.shape for x in xb), bs)
        mlflow.log_param("input_size", inp_sz)
        mlflow.log_param("num_layers", len(infos))

        summary = module_summary(learner, *xb)

        # Add again TrackerCallback
        if remove_cbs:
            learner.add_cbs(remove_cbs)

        with tempfile.TemporaryDirectory() as tempdir:
            summary_file = os.path.join(tempdir, "module_summary.txt")
            with open(summary_file, "w") as f:
                f.write(summary)
            mlflow.log_artifact(local_path=summary_file)

    def _run_and_log_function(self, original, args, kwargs, unlogged_params, is_fine_tune=False):
        # Check if is trying to fit while fine tuning or not
        mlflow_cbs = [cb for cb in self.cbs if cb.name == "___mlflow_fastai"]
        fit_in_fine_tune = (
            original.__name__ == "fit" and len(mlflow_cbs) > 0 and mlflow_cbs[0].is_fine_tune
        )

        if not fit_in_fine_tune:
            log_fn_args_as_params(original, list(args), kwargs, unlogged_params)

        run_id = mlflow.active_run().info.run_id
        with batch_metrics_logger(run_id) as metrics_logger:
            if not fit_in_fine_tune:
                early_stop_callback = _find_callback_of_type(EarlyStoppingCallback, self.cbs)
                _log_early_stop_callback_params(early_stop_callback)

                # First try to remove if any already registered callback
                self.remove_cbs(mlflow_cbs)

                # Log information regarding model and data without bar and print-out
                with self.no_bar(), self.no_logging():
                    _log_model_info(learner=self)

                mlflowFastaiCallback = getFastaiCallback(
                    metrics_logger=metrics_logger, is_fine_tune=is_fine_tune
                )

                # Add the new callback
                self.add_cb(mlflowFastaiCallback)

            return original(self, *args, **kwargs)

    def fit(original, self, *args, **kwargs):
        unlogged_params = ["self", "cbs", "learner", "lr", "lr_max", "wd"]
        return _run_and_log_function(
            self, original, args, kwargs, unlogged_params, is_fine_tune=False
        )

    safe_patch(FLAVOR_NAME, Learner, "fit", fit, manage_run=True, extra_tags=extra_tags)

    def fine_tune(original, self, *args, **kwargs):
        unlogged_params = ["self", "cbs", "learner", "lr", "lr_max", "wd"]
        return _run_and_log_function(
            self, original, args, kwargs, unlogged_params, is_fine_tune=True
        )

    safe_patch(FLAVOR_NAME, Learner, "fine_tune", fine_tune, manage_run=True, extra_tags=extra_tags)
