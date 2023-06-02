from packaging.version import Version
import os

import numpy as np
import pandas as pd
import yaml

import mlflow
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.utils.model_utils import (
    _get_flavor_configuration,
    _validate_and_copy_code_paths,
    _add_code_from_conf_to_system_path,
    _validate_and_prepare_target_save_path,
)
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import (
    _mlflow_conda_env,
    _validate_env_arguments,
    _process_pip_requirements,
    _process_conda_env,
    _CONDA_ENV_FILE_NAME,
    _REQUIREMENTS_FILE_NAME,
    _CONSTRAINTS_FILE_NAME,
    _PYTHON_ENV_FILE_NAME,
    _PythonEnv,
)
from mlflow.utils.requirements_utils import _get_pinned_requirement
from mlflow.utils.docstring_utils import format_docstring, LOG_MODEL_PARAM_DOCS
from mlflow.utils.file_utils import write_to
from mlflow.utils.autologging_utils import (
    autologging_integration,
    safe_patch,
    batch_metrics_logger,
)

FLAVOR_NAME = "gluon"
_MODEL_SAVE_PATH = "net"


def load_model(model_uri, ctx, dst_path=None):
    """
    Load a Gluon model from a local file or a run.

    :param model_uri: The location, in URI format, of the MLflow model. For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``s3://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                      artifact-locations>`_.
    :param ctx: Either CPU or GPU.
    :param dst_path: The local filesystem path to which to download the model artifact.
                     This directory must already exist. If unspecified, a local output
                     path will be created.

    :return: A Gluon model instance.

    .. code-block:: python
        :caption: Example

        # Load persisted model as a Gluon model, make inferences against an NDArray
        model = mlflow.gluon.load_model("runs:/" + gluon_random_data_run.info.run_id + "/model")
        model(nd.array(np.random.rand(1000, 1, 32)))
    """
    import mxnet as mx
    from mxnet import gluon
    from mxnet import sym

    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)

    model_arch_path = os.path.join(local_model_path, "data", _MODEL_SAVE_PATH) + "-symbol.json"
    model_params_path = os.path.join(local_model_path, "data", _MODEL_SAVE_PATH) + "-0000.params"

    if Version(mx.__version__) >= Version("2.0.0"):
        return gluon.SymbolBlock.imports(
            model_arch_path, input_names=["data"], param_file=model_params_path, ctx=ctx
        )
    else:
        symbol = sym.load(model_arch_path)
        inputs = sym.var("data", dtype="float32")
        net = gluon.SymbolBlock(symbol, inputs)
        net.collect_params().load(model_params_path, ctx)
    return net


class _GluonModelWrapper:
    def __init__(self, gluon_model):
        self.gluon_model = gluon_model

    def predict(self, data):
        """
        :param data: Either a pandas DataFrame or a numpy array containing input array values.
                     If the input is a DataFrame, it will be converted to an array first by a
                     `ndarray = df.values`.
        :return: Model predictions. If the input is a pandas.DataFrame, the predictions are returned
                 in a pandas.DataFrame. If the input is a numpy array, the predictions are returned
                 as either a numpy.ndarray or a plain list for hybrid models.


        """
        import mxnet as mx

        if isinstance(data, pd.DataFrame):
            ndarray = mx.nd.array(data.values)
            preds = self.gluon_model(ndarray)
            if isinstance(preds, mx.ndarray.ndarray.NDArray):
                preds = preds.asnumpy()
            return pd.DataFrame(preds)
        elif isinstance(data, np.ndarray):
            ndarray = mx.nd.array(data)
            preds = self.gluon_model(ndarray)
            if isinstance(preds, mx.ndarray.ndarray.NDArray):
                preds = preds.asnumpy()
            return preds
        else:
            raise TypeError("Input data should be pandas.DataFrame or numpy.ndarray")


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_model``.

    :param path: Local filesystem path to the MLflow Model with the ``gluon`` flavor.
    """
    import mxnet as mx

    m = load_model(path, mx.current_context())
    return _GluonModelWrapper(m)


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name="mxnet"))
def save_model(
    gluon_model,
    path,
    mlflow_model=None,
    conda_env=None,
    code_paths=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
):
    """
    Save a Gluon model to a path on the local file system.

    :param gluon_model: Gluon model to be saved. Must be already hybridized.
    :param path: Local path where the model is to be saved.
    :param mlflow_model: MLflow model config this flavor is being added to.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
    :param signature: :py:class:`ModelSignature <mlflow.models.ModelSignature>`
                      describes model input and output :py:class:`Schema <mlflow.types.Schema>`.
                      The model signature can be :py:func:`inferred <mlflow.models.infer_signature>`
                      from datasets with valid model input (e.g. the training dataset with target
                      column omitted) and valid model output (e.g. model predictions generated on
                      the training dataset), for example:

                      .. code-block:: python

                        from mlflow.models.signature import infer_signature

                        train = df.drop_column("target_label")
                        predictions = ...  # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example can be a Pandas DataFrame where the given
                          example will be serialized to json using the Pandas split-oriented
                          format, or a numpy array where the example will be serialized to json
                          by converting it to a list. Bytes are base64-encoded.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.

    .. code-block:: python
        :caption: Example

        from mxnet.gluon import Trainer
        from mxnet.gluon.contrib import estimator
        from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
        from mxnet.gluon.nn import HybridSequential
        from mxnet.metric import Accuracy
        import mlflow

        # Build, compile, and train your model
        gluon_model_path = ...
        net = HybridSequential()
        with net.name_scope():
            ...
        net.hybridize()
        net.collect_params().initialize()
        softmax_loss = SoftmaxCrossEntropyLoss()
        trainer = Trainer(net.collect_params())
        est = estimator.Estimator(
            net=net, loss=softmax_loss, metrics=Accuracy(), trainer=trainer
        )
        est.fit(train_data=train_data, epochs=100, val_data=validation_data)
        # Save the model as an MLflow Model
        mlflow.gluon.save_model(net, gluon_model_path)
    """
    import mxnet as mx

    _validate_env_arguments(conda_env, pip_requirements, extra_pip_requirements)
    path = os.path.abspath(path)
    _validate_and_prepare_target_save_path(path)
    data_subpath = "data"
    data_path = os.path.join(path, data_subpath)
    os.makedirs(data_path)
    code_dir_subpath = _validate_and_copy_code_paths(code_paths, path)
    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)
    if metadata is not None:
        mlflow_model.metadata = metadata

    # The epoch argument of the export method does not play any role in selecting
    # a specific epoch's parameters, and is there only for display purposes.
    gluon_model.export(os.path.join(data_path, _MODEL_SAVE_PATH))

    pyfunc.add_to_model(
        mlflow_model,
        loader_module="mlflow.gluon",
        conda_env=_CONDA_ENV_FILE_NAME,
        python_env=_PYTHON_ENV_FILE_NAME,
        code=code_dir_subpath,
    )
    mlflow_model.add_flavor(FLAVOR_NAME, mxnet_version=mx.__version__, code=code_dir_subpath)
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))

    if conda_env is None:
        if pip_requirements is None:
            default_reqs = get_default_pip_requirements()
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


def get_default_pip_requirements():
    """
    :return: A list of default pip requirements for MLflow Models produced by this flavor.
             Calls to :func:`save_model()` and :func:`log_model()` produce a pip environment
             that, at minimum, contains these requirements.
    """
    return [_get_pinned_requirement("mxnet")]


def get_default_conda_env():
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    return _mlflow_conda_env(additional_pip_deps=get_default_pip_requirements())


@format_docstring(LOG_MODEL_PARAM_DOCS.format(package_name="mxnet"))
def log_model(
    gluon_model,
    artifact_path,
    conda_env=None,
    code_paths=None,
    registered_model_name=None,
    signature: ModelSignature = None,
    input_example: ModelInputExample = None,
    pip_requirements=None,
    extra_pip_requirements=None,
    metadata=None,
):
    """
    Log a Gluon model as an MLflow artifact for the current run.

    :param gluon_model: Gluon model to be saved. Must be already hybridized.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: {{ conda_env }}
    :param code_paths: A list of local filesystem paths to Python file dependencies (or directories
                       containing file dependencies). These files are *prepended* to the system
                       path when the model is loaded.
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
                        predictions = ...  # compute model predictions
                        signature = infer_signature(train, predictions)
    :param input_example: Input example provides one or several instances of valid
                          model input. The example can be used as a hint of what data to feed the
                          model. The given example can be a Pandas DataFrame where the given
                          example will be serialized to json using the Pandas split-oriented
                          format, or a numpy array where the example will be serialized to json
                          by converting it to a list. Bytes are base64-encoded.
    :param pip_requirements: {{ pip_requirements }}
    :param extra_pip_requirements: {{ extra_pip_requirements }}
    :param metadata: Custom metadata dictionary passed to the model and stored in the MLmodel file.

                     .. Note:: Experimental: This parameter may change or be removed in a future
                                             release without warning.
    :return: A :py:class:`ModelInfo <mlflow.models.model.ModelInfo>` instance that contains the
             metadata of the logged model.

    .. code-block:: python
        :caption: Example

        from mxnet.gluon import Trainer
        from mxnet.gluon.contrib import estimator
        from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
        from mxnet.gluon.nn import HybridSequential
        from mxnet.metric import Accuracy
        import mlflow

        # Build, compile, and train your model
        net = HybridSequential()
        with net.name_scope():
            ...
        net.hybridize()
        net.collect_params().initialize()
        softmax_loss = SoftmaxCrossEntropyLoss()
        trainer = Trainer(net.collect_params())
        est = estimator.Estimator(
            net=net, loss=softmax_loss, metrics=Accuracy(), trainer=trainer
        )
        # Log metrics and log the model
        with mlflow.start_run():
            est.fit(train_data=train_data, epochs=100, val_data=validation_data)
            mlflow.gluon.log_model(net, "model")
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow.gluon,
        gluon_model=gluon_model,
        conda_env=conda_env,
        code_paths=code_paths,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        metadata=metadata,
    )


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_models=True,
    log_datasets=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    registered_model_name=None,
):  # pylint: disable=unused-argument
    """
    Enables (or disables) and configures autologging from Gluon to MLflow.
    Logs loss and any other metrics specified in the fit
    function, and optimizer data as parameters. Model checkpoints
    are logged as artifacts to a 'models' directory.

    :param log_models: If ``True``, trained models are logged as MLflow model artifacts.
                       If ``False``, trained models are not logged.
    :param log_datasets: If ``True``, dataset information is logged to MLflow Tracking.
                         If ``False``, dataset information is not logged.
    :param disable: If ``True``, disables the MXNet Gluon autologging integration. If ``False``,
                    enables the MXNet Gluon autologging integration.
    :param exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
                      If ``False``, autologged content is logged to the active fluent run,
                      which may be user-created.
    :param disable_for_unsupported_versions: If ``True``, disable autologging for versions of
                      gluon that have not been tested against this version of the MLflow client
                      or are incompatible.
    :param silent: If ``True``, suppress all event logs and warnings from MLflow during MXNet Gluon
                   autologging. If ``False``, show all events and warnings during MXNet Gluon
                   autologging.
    :param registered_model_name: If given, each time a model is trained, it is registered as a
                                  new model version of the registered model with this name.
                                  The registered model is created if it does not already exist.
    """

    from mxnet.gluon.contrib.estimator import Estimator
    from mlflow.gluon._autolog import __MLflowGluonCallback

    def getGluonCallback(metrics_logger):
        return __MLflowGluonCallback(log_models, metrics_logger)

    def fit(original, self, *args, **kwargs):
        # Wrap `fit` execution within a batch metrics logger context.
        run_id = mlflow.active_run().info.run_id
        with batch_metrics_logger(run_id) as metrics_logger:
            mlflowGluonCallback = getGluonCallback(metrics_logger)
            if len(args) >= 4:
                args = (*args[:3], args[3] + [mlflowGluonCallback], *args[4:])
            elif "event_handlers" in kwargs:
                kwargs["event_handlers"] += [mlflowGluonCallback]
            else:
                kwargs["event_handlers"] = [mlflowGluonCallback]
            result = original(self, *args, **kwargs)

        return result

    safe_patch(FLAVOR_NAME, Estimator, "fit", fit, manage_run=True)
