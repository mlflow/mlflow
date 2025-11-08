"""
The ``mlflow.paddle`` module provides an API for logging and loading paddle models.
This module exports paddle models with the following flavors:

Paddle (native) format
    This is the main flavor that can be loaded back into paddle.

:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
    NOTE: The `mlflow.pyfunc` flavor is only added for paddle models that define `predict()`,
    since `predict()` is required for pyfunc model inference.
"""

import logging
import os
from typing import Any

import yaml

import mlflow
from mlflow import pyfunc
from mlflow.models import Model, ModelInputExample, ModelSignature
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import _infer_signature_from_input_example
from mlflow.models.utils import _save_example
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.autologging_utils import autologging_integration, safe_patch
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
from mlflow.utils.file_utils import write_to
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _validate_and_prepare_target_save_path,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement

FLAVOR_NAME = "paddle"

_MODEL_DATA_SUBPATH = "model"

_logger = logging.getLogger(__name__)


def get_default_pip_requirements():
    """
    Returns:
        A list of default pip requirements for MLflow Models produced by this flavor.
        Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
        that, at minimum, contains these requirements.
    """
    return [_get_pinned_requirement("paddlepaddle", module="paddle")]


def get_default_conda_env():
    """
    Returns:
        The default Conda environment for MLflow Models produced by calls to
        :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def save_model(
    pd_model,
    path,
    training=False,
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
    Save a paddle model to a path on the local file system. Produces an MLflow Model
    containing the following flavors:

        - :py:mod:`mlflow.paddle`
        - :py:mod:`mlflow.pyfunc`. NOTE: This flavor is only included for paddle models
          that define `predict()`, since `predict()` is required for pyfunc model inference.

    Args:
        pd_model: paddle model to be saved.
        path: Local path where the model is to be saved.
        training: Only valid when saving a model trained using the PaddlePaddle high level API.
            If set to True, the saved model supports both re-training and
            inference. If set to False, it only supports inference.
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

        import mlflow.paddle
        import paddle
        from paddle.nn import Linear
        import paddle.nn.functional as F
        import numpy as np
        import os
        import random
        from sklearn.datasets import load_diabetes
        from sklearn.model_selection import train_test_split
        from sklearn import preprocessing


        def load_data():
            # dataset on boston housing prediction
            X, y = load_diabetes(return_X_y=True, as_frame=True)
            min_max_scaler = preprocessing.MinMaxScaler()
            X_min_max = min_max_scaler.fit_transform(X)
            X_normalized = preprocessing.scale(X_min_max, with_std=False)
            X_train, X_test, y_train, y_test = train_test_split(
                X_normalized, y, test_size=0.2, random_state=42
            )
            y_train = y_train.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)
            return np.concatenate((X_train, y_train), axis=1), np.concatenate(
                (X_test, y_test), axis=1
            )


        class Regressor(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.fc = Linear(in_features=13, out_features=1)

            @paddle.jit.to_static
            def forward(self, inputs):
                x = self.fc(inputs)
                return x


        model = Regressor()
        model.train()
        training_data, test_data = load_data()
        opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
        EPOCH_NUM = 10
        BATCH_SIZE = 10
        for epoch_id in range(EPOCH_NUM):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + BATCH_SIZE]
                for k in range(0, len(training_data), BATCH_SIZE)
            ]
            for iter_id, mini_batch in enumerate(mini_batches):
                x = np.array(mini_batch[:, :-1]).astype("float32")
                y = np.array(mini_batch[:, -1:]).astype("float32")
                house_features = paddle.to_tensor(x)
                prices = paddle.to_tensor(y)
                predicts = model(house_features)
                loss = F.square_error_cost(predicts, label=prices)
                avg_loss = paddle.mean(loss)
                if iter_id % 20 == 0:
                    print(f"epoch: {epoch_id}, iter: {iter_id}, loss is: {avg_loss.numpy()}")
                avg_loss.backward()
                opt.step()
                opt.clear_grad()
        mlflow.log_param("learning_rate", 0.01)
        mlflow.paddle.log_model(model, name="model")
        sk_path_dir = "./test-out"
        mlflow.paddle.save_model(model, sk_path_dir)
        print("Model saved in run %s" % mlflow.active_run().info.run_id)
    """
    import paddle

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)

    _validate_and_prepare_target_save_path(path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)

    if mlflow_model is None:
        mlflow_model = Model()
    saved_example = _save_example(mlflow_model, input_example, path)

    if signature is None and saved_example is not None:
        wrapped_model = _PaddleWrapper(pd_model)
        signature = _infer_signature_from_input_example(saved_example, wrapped_model)
    elif signature is False:
        signature = None

    if signature is not None:
        mlflow_model.signature = signature
    if metadata is not None:
        mlflow_model.metadata = metadata

    model_data_subpath = _MODEL_DATA_SUBPATH
    output_path = os.path.join(path, model_data_subpath)

    if isinstance(pd_model, paddle.Model):
        pd_model.save(output_path, training=training)
    else:
        paddle.jit.save(pd_model, output_path)

    # `PyFuncModel` only works for paddle models that define `predict()`.
    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.paddle",
        model_path=model_data_subpath,
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
    )
    mlflow_model.add_flavor(
        FLAVOR_NAME,
        pickled_model=model_data_subpath,
        paddle_version=paddle.__version__,
        code=code_dir_subpath,
    )
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


def load_model(model_uri, model=None, dst_path=None, **kwargs):
    """
    Load a paddle model from a local file or a run.

    Args:
        model_uri: The location, in URI format, of the MLflow model, for example:
            - ``/Users/me/path/to/local/model``
            - ``relative/path/to/local/model``
            - ``s3://my_bucket/path/to/model``
            - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
            - ``models:/<model_name>/<model_version>``
            - ``models:/<model_name>/<stage>``
        model: Required when loading a `paddle.Model` model saved with `training=True`.
        dst_path: The local filesystem path to which to download the model artifact.
            This directory must already exist. If unspecified, a local output
            path will be created.
        kwargs: The keyword arguments to pass to `paddle.jit.load`
            or `model.load`.

    For more information about supported URI schemes, see
    `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
    artifact-locations>`_.

    Returns:
        A paddle model.

    .. code-block:: python
        :caption: Example

        import mlflow.paddle

        pd_model = mlflow.paddle.load_model("runs:/96771d893a5e46159d9f3b49bf9013e2/pd_models")
        # use Pandas DataFrame to make predictions
        np_array = ...
        predictions = pd_model(np_array)
    """
    import paddle

    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    pd_model_artifacts_path = os.path.join(local_model_path, flavor_conf["pickled_model"])
    if model is None:
        return paddle.jit.load(pd_model_artifacts_path, **kwargs)
    elif not isinstance(model, paddle.Model):
        raise TypeError(f"Invalid object type `{type(model)}` for `model`, must be `paddle.Model`")
    else:
        contains_pdparams = _contains_pdparams(local_model_path)
        if not contains_pdparams:
            raise TypeError(
                "This model can't be loaded via `model.load` because a '.pdparams' file "
                "doesn't exist. Please leave `model` unspecified to load the model via "
                "`paddle.jit.load` or set `training` to True when saving a model."
            )

        model.load(pd_model_artifacts_path, **kwargs)
        return model


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name=FLAVOR_NAME))
def log_model(
    pd_model,
    artifact_path: str | None = None,
    training=False,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
    name: str | None = None,
    params: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
    model_type: str | None = None,
    step: int = 0,
    model_id: str | None = None,
):
    """
    Log a paddle model as an MLflow artifact for the current run. Produces an MLflow Model
    containing the following flavors:

        - :py:mod:`mlflow.paddle`
        - :py:mod:`mlflow.pyfunc`. NOTE: This flavor is only included for paddle models
          that define `predict()`, since `predict()` is required for pyfunc model inference.

    Args:
        pd_model: paddle model to be saved.
        artifact_path: Deprecated. Use `name` instead.
        training: Only valid when saving a model trained using the PaddlePaddle high level API.
            If set to True, the saved model supports both re-training and
            inference. If set to False, it only supports inference.
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
        name: {{ name }}
        params: {{ params }}
        tags: {{ tags }}
        model_type: {{ model_type }}
        step: {{ step }}
        model_id: {{ model_id }}

    Returns:
        A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
        metadata of the logged model.

    .. code-block:: python
        :caption: Example

        import mlflow.paddle


        def load_data(): ...


        class Regressor: ...


        model = Regressor()
        model.train()
        training_data, test_data = load_data()
        opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
        EPOCH_NUM = 10
        BATCH_SIZE = 10
        for epoch_id in range(EPOCH_NUM):
            ...
        mlflow.log_param("learning_rate", 0.01)
        mlflow.paddle.log_model(model, name="model")
        sk_path_dir = ...
        mlflow.paddle.save_model(model, sk_path_dir)
    """
    return Model.log(
        artifact_path=artifact_path,
        name=name,
        flavor=mlflow.paddle,
        pd_model=pd_model,
        conda_env=conda_env,
        code_paths=code_paths,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        await_registration_for=await_registration_for,
        training=training,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
        params=params,
        tags=tags,
        model_type=model_type,
        step=step,
        model_id=model_id,
    )


def _load_pyfunc(path):
    """
    Loads PyFunc implementation. Called by ``pyfunc.load_model``.

    Args:
        path: Local filesystem path to the MLflow Model with the ``paddle`` flavor.
    """
    return _PaddleWrapper(load_model(path))


class _PaddleWrapper:
    """
    Wrapper class that creates a predict function such that
    predict(data: pd.DataFrame) -> model's output as pd.DataFrame (pandas DataFrame)
    """

    def __init__(self, pd_model):
        self.pd_model = pd_model

    def get_raw_model(self):
        """
        Returns the underlying model.
        """
        return self.pd_model

    def predict(
        self,
        data,
        params: dict[str, Any] | None = None,
    ):
        """
        Args:
            data: Model input data.
            params: Additional parameters to pass to the model for inference.

        Returns:
            Model predictions.
        """
        import numpy as np
        import paddle
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            inp_data = data.values.astype(np.float32)
        elif isinstance(data, np.ndarray):
            inp_data = data
        elif isinstance(data, (list, dict)):
            raise TypeError(
                "The paddle flavor does not support List or Dict input types. "
                "Please use a pandas.DataFrame or a numpy.ndarray"
            )
        else:
            raise TypeError("Input data should be pandas.DataFrame or numpy.ndarray")
        inp_data = np.squeeze(inp_data)

        self.pd_model.eval()

        predicted = self.pd_model(paddle.to_tensor(inp_data))
        return pd.DataFrame(predicted.numpy())


def _contains_pdparams(path):
    file_list = os.listdir(path)
    return any(".pdparams" in file for file in file_list)


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_every_n_epoch=1,
    log_models=True,
    disable=False,
    exclusive=False,
    silent=False,
    registered_model_name=None,
    extra_tags=None,
):
    """
    Enables (or disables) and configures autologging from PaddlePaddle to MLflow.

    Autologging is performed when the `fit` method of `paddle.Model`_ is called.

    .. _paddle.Model:
        https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/Model_en.html

    Args:
        log_every_n_epoch: If specified, logs metrics once every `n` epochs. By default, metrics
            are logged after every epoch.
        log_models: If ``True``, trained models are logged as MLflow model artifacts.
            If ``False``, trained models are not logged.
        disable: If ``True``, disables the PaddlePaddle autologging integration.
            If ``False``, enables the PaddlePaddle autologging integration.
        exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
            If ``False``, autologged content is logged to the active fluent run,
            which may be user-created.
        silent: If ``True``, suppress all event logs and warnings from MLflow during PyTorch
            Lightning autologging. If ``False``, show all events and warnings during
            PaddlePaddle autologging.
        registered_model_name: If given, each time a model is trained, it is registered as a
            new model version of the registered model with this name.
            The registered model is created if it does not already exist.
        extra_tags: A dictionary of extra tags to set on each managed run created by autologging.

    .. code-block:: python
        :caption: Example

        import paddle
        import mlflow
        from mlflow import MlflowClient


        def show_run_data(run_id):
            run = mlflow.get_run(run_id)
            print(f"params: {run.data.params}")
            print(f"metrics: {run.data.metrics}")
            client = MlflowClient()
            artifacts = [f.path for f in client.list_artifacts(run.info.run_id, "model")]
            print(f"artifacts: {artifacts}")


        class LinearRegression(paddle.nn.Layer):
            def __init__(self):
                super().__init__()
                self.fc = paddle.nn.Linear(13, 1)

            def forward(self, feature):
                return self.fc(feature)


        train_dataset = paddle.text.datasets.UCIHousing(mode="train")
        eval_dataset = paddle.text.datasets.UCIHousing(mode="test")
        model = paddle.Model(LinearRegression())
        optim = paddle.optimizer.SGD(learning_rate=1e-2, parameters=model.parameters())
        model.prepare(optim, paddle.nn.MSELoss(), paddle.metric.Accuracy())
        mlflow.paddle.autolog()
        with mlflow.start_run() as run:
            model.fit(train_dataset, eval_dataset, batch_size=16, epochs=10)
        show_run_data(run.info.run_id)

    .. code-block:: text
        :caption: Output

        params: {
            "learning_rate": "0.01",
            "optimizer_name": "SGD",
        }
        metrics: {
            "loss": 17.482044,
            "step": 25.0,
            "acc": 0.0,
            "eval_step": 6.0,
            "eval_acc": 0.0,
            "eval_batch_size": 6.0,
            "batch_size": 4.0,
            "eval_loss": 24.717455,
        }
        artifacts: [
            "model/MLmodel",
            "model/conda.yaml",
            "model/model.pdiparams",
            "model/model.pdiparams.info",
            "model/model.pdmodel",
            "model/requirements.txt",
        ]
    """
    import paddle

    from mlflow.paddle._paddle_autolog import patched_fit

    safe_patch(
        FLAVOR_NAME, paddle.Model, "fit", patched_fit, manage_run=True, extra_tags=extra_tags
    )
