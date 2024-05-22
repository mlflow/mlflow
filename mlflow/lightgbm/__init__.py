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
import functools
import json
import logging
import os
import tempfile
from copy import deepcopy
from typing import Any, Dict, Optional

import yaml
from packaging.version import Version

import mlflow
from mlflow import pyfunc
from mlflow.data.code_dataset_source import CodeDatasetSource
from mlflow.data.numpy_dataset import from_numpy
from mlflow.data.pandas_dataset import from_pandas
from mlflow.entities.dataset_input import DatasetInput
from mlflow.entities.input_tag import InputTag
from mlflow.models import Model, ModelInputExample, ModelSignature, infer_signature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.sklearn import _SklearnTrainingSession
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.tracking.context import registry as context_registry
from mlflow.utils import _get_fully_qualified_class_name
from mlflow.utils.arguments_utils import _get_arg_names
from mlflow.utils.autologging_utils import (
    ENSURE_AUTOLOGGING_ENABLED_TEXT,
    INPUT_EXAMPLE_SAMPLE_ROWS,
    InputExampleInfo,
    MlflowAutologgingQueueingClient,
    autologging_integration,
    batch_metrics_logger,
    get_mlflow_run_params_for_fn_args,
    picklable_exception_safe_function,
    resolve_input_example_and_signature,
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
from mlflow.utils.mlflow_tags import (
    MLFLOW_DATASET_CONTEXT,
)
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "lightgbm"


_logger = logging.getLogger(__name__)


def get_default_pip_requirements(include_cloudpickle=False):
    """
    Returns:
        A list of default pip requirements for MLflow Models produced by this flavor.
        Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
        that, at minimum, contains these requirements.
    """
    pip_deps = [_get_pinned_requirement("lightgbm")]
    if include_cloudpickle:
        pip_deps.append(_get_pinned_requirement("cloudpickle"))
    return pip_deps


def get_default_conda_env(include_cloudpickle=False):
    """
    Returns:
        The default Conda environment for MLflow Models produced by calls to
        :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements(include_cloudpickle))


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    lgb_model,
    path,
    conda_env=None,
    code_paths=None,
    mlflow_model=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
):
    """
    Save a LightGBM model to a path on the local file system.

    Args:
        lgb_model: LightGBM model (an instance of `lightgbm.Booster`_) or
            models that implement the `scikit-learn API`_  to be saved.
        path: Local path where the model is to be saved.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.
        signature: {{ signature }}
        input_example: {{ input_example }}
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}

    .. code-block:: python
        :caption: Example

        from pathlib import Path
        from lightgbm import LGBMClassifier
        from sklearn import datasets
        import mlflow

        # Load iris dataset
        X, y = datasets.load_iris(return_X_y=True, as_frame=True)

        # Initialize our model
        model = LGBMClassifier(objective="multiclass", random_state=42)

        # Train the model
        model.fit(X, y)

        # Save the model
        path = "model"
        mlflow.lightgbm.save_model(model, path)

        # Load model for inference
        loaded_model = mlflow.lightgbm.load_model(Path.cwd() / path)
        print(loaded_model.predict(X[:5]))

    .. code-block:: text
        :caption: Output

        [0 0 0 0 0]
    """
    import lightgbm as lgb

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
    model_data_subpath = "model.lgb" if isinstance(lgb_model, lgb.Booster) else "model.pkl"
    model_data_path = os.path.join(path, model_data_subpath)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if signature is None and input_example is not None:
        wrapped_model = _LGBModelWrapper(lgb_model)
        signature = _infer_signature_from_input_example(input_example, wrapped_model)
    elif signature is False:
        signature = None

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    if metadata is not None:
        mlflow_model.metadata = metadata

    # Save a LightGBM model
    _save_model(lgb_model, model_data_path)

    lgb_model_class = _get_fully_qualified_class_name(lgb_model)
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.lightgbm",
        data=model_data_subpath,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
    )
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        lgb_version=lgb.__version__,
        data=model_data_subpath,
        model_class=lgb_model_class,
        code=code_dir_subpath,
    )
    if size := get_total_file_size(path):
        mlflow_model.model_size_bytes = size
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

    _PythonEnv.current().to_yaml(os.path.join(path, _PYTHON_ENV_FILE_NAME))


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
    """
    Log a LightGBM model as an MLflow artifact for the current run.

    Args:
        lgb_model: LightGBM model (an instance of `lightgbm.Booster`_) or
            models that implement the `scikit-learn API`_  to be saved.
        artifact_path: Run-relative artifact path.
        conda_env: {{ conda_env }}
        code_paths: {{ code_paths }}
        registered_model_name: If given, create a model version under
            ``registered_model_name``, also creating a registered model if one
            with the given name does not exist.

        signature: {{ signature }}
        input_example: {{ input_example }}
        await_registration_for: Number of seconds to wait for the model version to finish
            being created and is in ``READY`` status. By default, the function
            waits for five minutes. Specify 0 or None to skip waiting.
        pip_requirements: {{ pip_requirements }}
        extra_pip_requirements: {{ extra_pip_requirements }}
        metadata: {{ metadata }}
        kwargs: kwargs to pass to `lightgbm.Booster.save_model`_ method.

    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
        metadata of the logged model.

    .. code-block:: python
        :caption: Example

        from lightgbm import LGBMClassifier
        from sklearn import datasets
        import mlflow
        from mlflow.models import infer_signature

        # Load iris dataset
        X, y = datasets.load_iris(return_X_y=True, as_frame=True)

        # Initialize our model
        model = LGBMClassifier(objective="multiclass", random_state=42)

        # Train the model
        model.fit(X, y)

        # Create model signature
        predictions = model.predict(X)
        signature = infer_signature(X, predictions)

        # Log the model
        artifact_path = "model"
        with mlflow.start_run():
            model_info = mlflow.lightgbm.log_model(model, artifact_path, signature=signature)

        # Fetch the logged model artifacts
        print(f"run_id: {run.info.run_id}")
        client = mlflow.MlflowClient()
        artifacts = [f.path for f in client.list_artifacts(run.info.run_id, artifact_path)]
        print(f"artifacts: {artifacts}")

    .. code-block:: text
        :caption: Output

        artifacts: ['model/MLmodel',
                    'model/conda.yaml',
                    'model/model.pkl',
                    'model/python_env.yaml',
                    'model/requirements.txt']
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.lightgbm,
        registered_model_name=registered_model_name,
        lgb_model=lgb_model,
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
    """
    Load Model Implementation.

    Args:
        path: Local filesystem path to
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
    Load PyFunc implementation. Called by ``pyfunc.load_model``.

    Args:
        path: Local filesystem path to the MLflow Model with the ``lightgbm`` flavor.
    """
    return _LGBModelWrapper(_load_model(path))


def load_model(model_uri, dst_path=None):
    """
    Load a LightGBM model from a local file or a run.

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
            A LightGBM model (an instance of `lightgbm.Booster`_) or a LightGBM scikit-learn
            model, depending on the saved model class specification.

    .. code-block:: python
        :caption: Example

        from lightgbm import LGBMClassifier
        from sklearn import datasets
        import mlflow

        # Auto log all MLflow entities
        mlflow.lightgbm.autolog()

        # Load iris dataset
        X, y = datasets.load_iris(return_X_y=True, as_frame=True)

        # Initialize our model
        model = LGBMClassifier(objective="multiclass", random_state=42)

        # Train the model
        model.fit(X, y)

        # Load model for inference
        model_uri = f"runs:/{mlflow.last_active_run().info.run_id}/model"
        loaded_model = mlflow.lightgbm.load_model(model_uri)
        print(loaded_model.predict(X[:5]))

    .. code-block:: text
        :caption: Output

        [0 0 0 0 0]
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(local_model_path, FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    return _load_model(path=local_model_path)


class _LGBModelWrapper:
    def __init__(self, lgb_model):
        self.lgb_model = lgb_model

    def predict(self, dataframe, params: Optional[Dict[str, Any]] = None):
        """
        Args:
            dataframe: Model input data.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Model predictions.
        """
        return self.lgb_model.predict(dataframe)


def _patch_metric_names(metric_dict):
    # lightgbm provides some metrics with "@", e.g. "ndcg@3" that are not valid MLflow metric names
    patched_metrics = {
        metric_name.replace("@", "_at_"): value for metric_name, value in metric_dict.items()
    }
    changed_keys = set(patched_metrics.keys()) - set(metric_dict.keys())
    if changed_keys:
        _logger.info(
            "Identified one or more metrics with names containing the invalid character `@`."
            " These metric names have been sanitized by replacing `@` with `_at_`, as follows: %s",
            ", ".join(changed_keys),
        )

    return patched_metrics


def _autolog_callback(env, metrics_logger, eval_results):
    res = {}
    for data_name, eval_name, value, _ in env.evaluation_result_list:
        key = data_name + "-" + eval_name
        res[key] = value
    res = _patch_metric_names(res)
    metrics_logger.record_metrics(res, env.iteration)
    eval_results.append(res)


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_input_examples=False,
    log_model_signatures=True,
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
    Enables (or disables) and configures autologging from LightGBM to MLflow. Logs the following:

        - parameters specified in `lightgbm.train`_.
        - metrics on each iteration (if ``valid_sets`` specified).
        - metrics at the best iteration (if ``early_stopping_rounds`` specified or
          ``early_stopping`` callback is set).
        - feature importance (both "split" and "gain") as JSON files and plots.
        - trained model, including:
            - an example of valid input.
            - inferred signature of the inputs and outputs of the model.

    Note that the `scikit-learn API`_ is now supported.

    Args:
        log_input_examples: If ``True``, input examples from training datasets are collected and
            logged along with LightGBM model artifacts during training. If
            ``False``, input examples are not logged.
            Note: Input examples are MLflow model attributes
            and are only collected if ``log_models`` is also ``True``.
        log_model_signatures: If ``True``,
            :py:class:`ModelSignatures <mlflow.models.ModelSignature>`
            describing model inputs and outputs are collected and logged along
            with LightGBM model artifacts during training. If ``False``,
            signatures are not logged.
            Note: Model signatures are MLflow model attributes
            and are only collected if ``log_models`` is also ``True``.
        log_models: If ``True``, trained models are logged as MLflow model artifacts.
            If ``False``, trained models are not logged.
            Input examples and model signatures, which are attributes of MLflow models,
            are also omitted when ``log_models`` is ``False``.
        log_datasets: If ``True``, train and validation dataset information is logged to MLflow
            Tracking if applicable. If ``False``, dataset information is not logged.
        disable: If ``True``, disables the LightGBM autologging integration. If ``False``,
            enables the LightGBM autologging integration.
        exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
            If ``False``, autologged content is logged to the active fluent run,
            which may be user-created.
        disable_for_unsupported_versions: If ``True``, disable autologging for versions of
            lightgbm that have not been tested against this version of the MLflow client
            or are incompatible.
        silent: If ``True``, suppress all event logs and warnings from MLflow during LightGBM
            autologging. If ``False``, show all events and warnings during LightGBM
            autologging.
        registered_model_name: If given, each time a model is trained, it is registered as a
            new model version of the registered model with this name.
            The registered model is created if it does not already exist.
        extra_tags: A dictionary of extra tags to set on each managed run created by autologging.

    .. code-block:: python
        :caption: Example

        import mlflow
        from lightgbm import LGBMClassifier
        from sklearn import datasets


        def print_auto_logged_info(run):
            tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
            artifacts = [
                f.path for f in mlflow.MlflowClient().list_artifacts(run.info.run_id, "model")
            ]
            feature_importances = [
                f.path
                for f in mlflow.MlflowClient().list_artifacts(run.info.run_id)
                if f.path != "model"
            ]
            print(f"run_id: {run.info.run_id}")
            print(f"artifacts: {artifacts}")
            print(f"feature_importances: {feature_importances}")
            print(f"params: {run.data.params}")
            print(f"metrics: {run.data.metrics}")
            print(f"tags: {tags}")


        # Load iris dataset
        X, y = datasets.load_iris(return_X_y=True, as_frame=True)

        # Initialize our model
        model = LGBMClassifier(objective="multiclass", random_state=42)

        # Auto log all MLflow entities
        mlflow.lightgbm.autolog()

        # Train the model
        with mlflow.start_run() as run:
            model.fit(X, y)

        # fetch the auto logged parameters and metrics
        print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

    .. code-block:: text
        :caption: Output

        run_id: e08dd59d57a74971b68cf78a724dfaf6
        artifacts: ['model/MLmodel',
                    'model/conda.yaml',
                    'model/model.pkl',
                    'model/python_env.yaml',
                    'model/requirements.txt']
        feature_importances: ['feature_importance_gain.json',
                              'feature_importance_gain.png',
                              'feature_importance_split.json',
                              'feature_importance_split.png']
        params: {'boosting_type': 'gbdt',
                 'categorical_feature': 'auto',
                 'colsample_bytree': '1.0',
                 ...
                 'verbose_eval': 'warn'}
        metrics: {}
        tags: {}
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

            self.input_example_info = input_example_info

        original(self, *args, **kwargs)

    def train_impl(_log_models, _log_datasets, original, *args, **kwargs):
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
            ax.set_title(f"Feature Importance ({importance_type})")
            fig.tight_layout()

            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    filepath = os.path.join(tmpdir, f"feature_importance_{imp_type}.png")
                    fig.savefig(filepath)
                    mlflow.log_artifact(filepath)
                finally:
                    plt.close(fig)

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

        train_set = args[1] if len(args) > 1 else kwargs.get("train_set")

        # Whether to automatically log the training dataset as a dataset artifact.
        if _log_datasets and train_set:
            try:
                context_tags = context_registry.resolve_tags()
                source = CodeDatasetSource(tags=context_tags)

                _log_lightgbm_dataset(train_set, source, "train", autologging_client)

                valid_sets = kwargs.get("valid_sets")
                if valid_sets is not None:
                    valid_names = kwargs.get("valid_names")
                    if valid_names is None:
                        for valid_set in valid_sets:
                            _log_lightgbm_dataset(valid_set, source, "eval", autologging_client)
                    else:
                        for valid_set, valid_name in zip(valid_sets, valid_names):
                            _log_lightgbm_dataset(
                                valid_set, source, "eval", autologging_client, name=valid_name
                            )

                dataset_logging_operations = autologging_client.flush(synchronous=False)
                dataset_logging_operations.await_completion()
            except Exception as e:
                _logger.warning(
                    "Failed to log dataset information to MLflow Tracking. Reason: %s", e
                )

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

            imp = dict(zip(features, importance.tolist()))
            with tempfile.TemporaryDirectory() as tmpdir:
                filepath = os.path.join(tmpdir, f"feature_importance_{imp_type}.json")
                with open(filepath, "w") as f:
                    json.dump(imp, f, indent=2)
                mlflow.log_artifact(filepath)

        # train_set must exist as the original train function already ran successfully
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
            return infer_signature(input_example, model_output)

        # Whether to automatically log the trained model based on boolean flag.
        if _log_models:
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
                registered_model_name=registered_model_name,
            )

        param_logging_operations.await_completion()
        if early_stopping:
            early_stopping_logging_operations.await_completion()

        return model

    def train(_log_models, _log_datasets, original, *args, **kwargs):
        with _SklearnTrainingSession(estimator=lightgbm.train, allow_children=False) as t:
            if t.should_log():
                return train_impl(_log_models, _log_datasets, original, *args, **kwargs)
            else:
                return original(*args, **kwargs)

    safe_patch(FLAVOR_NAME, lightgbm.Dataset, "__init__", __init__)
    safe_patch(
        FLAVOR_NAME,
        lightgbm,
        "train",
        functools.partial(train, log_models, log_datasets),
        manage_run=True,
        extra_tags=extra_tags,
    )
    # The `train()` method logs LightGBM models as Booster objects. When using LightGBM
    # scikit-learn models, we want to save / log models as their model classes. So we turn
    # off the log_models functionality in the `train()` method patched to `lightgbm.sklearn`.
    # Instead the model logging is handled in `fit_mlflow_xgboost_and_lightgbm()`
    # in `mlflow.sklearn._autolog()`, where models are logged as LightGBM scikit-learn models
    # after the `fit()` method returns.
    safe_patch(
        FLAVOR_NAME,
        lightgbm.sklearn,
        "train",
        functools.partial(train, False, log_datasets),
        manage_run=True,
        extra_tags=extra_tags,
    )

    # enable LightGBM scikit-learn estimators autologging
    import mlflow.sklearn

    mlflow.sklearn._autolog(
        flavor_name=FLAVOR_NAME,
        log_input_examples=log_input_examples,
        log_model_signatures=log_model_signatures,
        log_models=log_models,
        log_datasets=log_datasets,
        disable=disable,
        exclusive=exclusive,
        disable_for_unsupported_versions=disable_for_unsupported_versions,
        silent=silent,
        max_tuning_runs=None,
        log_post_training_metrics=True,
        extra_tags=extra_tags,
    )


def _log_lightgbm_dataset(lgb_dataset, source, context, autologging_client, name=None):
    import numpy as np
    import pandas as pd
    from scipy.sparse import issparse

    data = lgb_dataset.data
    label = lgb_dataset.label
    if isinstance(data, pd.DataFrame):
        dataset = from_pandas(df=data, source=source, name=name)
    elif issparse(data):
        arr_data = data.toarray() if issparse(data) else data
        dataset = from_numpy(features=arr_data, targets=label, source=source, name=name)
    elif isinstance(data, np.ndarray):
        dataset = from_numpy(features=data, targets=label, source=source, name=name)
    else:
        _logger.warning("Unrecognized dataset type %s. Dataset logging skipped.", type(data))
        return
    tags = [InputTag(key=MLFLOW_DATASET_CONTEXT, value=context)]
    dataset_input = DatasetInput(dataset=dataset._to_mlflow_entity(), tags=tags)

    # log the dataset
    autologging_client.log_inputs(run_id=mlflow.active_run().info.run_id, datasets=[dataset_input])
