"""
The ``mlflow.keras`` module provides an API for logging and loading Keras models. This module
exports Keras models with the following flavors:

Keras (native) format
    This is the main flavor that can be loaded back into Keras.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""
import importlib
import os
import yaml
import gorilla
import tempfile
import shutil

import pandas as pd

from distutils.version import LooseVersion
from mlflow import pyfunc
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
import mlflow.tracking
from mlflow.exceptions import MlflowException
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample, _save_example
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.environment import _mlflow_conda_env
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.utils.annotations import experimental
from mlflow.utils.autologging_utils import try_mlflow_log, log_fn_args_as_params


FLAVOR_NAME = "keras"
# File name to which custom objects cloudpickle is saved - used during save and load
_CUSTOM_OBJECTS_SAVE_PATH = "custom_objects.cloudpickle"
_KERAS_MODULE_SPEC_PATH = "keras_module.txt"
# File name to which keras model is saved
_MODEL_SAVE_PATH = "model.h5"
# Conda env subpath when saving/loading model
_CONDA_ENV_SUBPATH = "conda.yaml"


def get_default_conda_env(include_cloudpickle=False, keras_module=None):
    """
    :return: The default Conda environment for MLflow Models produced by calls to
             :func:`save_model()` and :func:`log_model()`.
    """
    import tensorflow as tf
    conda_deps = []  # if we use tf.keras we only need to declare dependency on tensorflow
    pip_deps = []
    if keras_module is None:
        import keras
        keras_module = keras
    if keras_module.__name__ == "keras":
        # Temporary fix: the created conda environment has issues installing keras >= 2.3.1
        if LooseVersion(keras_module.__version__) < LooseVersion('2.3.1'):
            conda_deps.append("keras=={}".format(keras_module.__version__))
        else:
            pip_deps.append("keras=={}".format(keras_module.__version__))
    if include_cloudpickle:
        import cloudpickle
        pip_deps.append("cloudpickle=={}".format(cloudpickle.__version__))
    # Temporary fix: conda-forge currently does not have tensorflow > 1.14
    # The Keras pyfunc representation requires the TensorFlow
    # backend for Keras. Therefore, the conda environment must
    # include TensorFlow
    if LooseVersion(tf.__version__) <= LooseVersion('1.13.2'):
        conda_deps.append("tensorflow=={}".format(tf.__version__))
    else:
        pip_deps.append("tensorflow=={}".format(tf.__version__))

    return _mlflow_conda_env(
        additional_conda_deps=conda_deps,
        additional_pip_deps=pip_deps,
        additional_conda_channels=None)


def save_model(keras_model, path, conda_env=None, mlflow_model=None, custom_objects=None,
               keras_module=None,
               signature: ModelSignature = None, input_example: ModelInputExample = None,
               **kwargs):
    """
    Save a Keras model to a path on the local file system.

    :param keras_model: Keras model to be saved.
    :param path: Local path where the model is to be saved.
    :param conda_env: Either a dictionary representation of a Conda environment or the path to a
                      Conda environment yaml file. If provided, this decsribes the environment
                      this model should be run in. At minimum, it should specify the
                      dependencies contained in :func:`get_default_conda_env()`. If
                      ``None``, the default :func:`get_default_conda_env()` environment is
                      added to the model. The following is an *example* dictionary
                      representation of a Conda environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'keras=2.2.4',
                                'tensorflow=1.8.0'
                            ]
                        }
    :param mlflow_model: MLflow model config this flavor is being added to.
    :param custom_objects: A Keras ``custom_objects`` dictionary mapping names (strings) to
                           custom classes or functions associated with the Keras model. MLflow saves
                           these custom layers using CloudPickle and restores them automatically
                           when the model is loaded with :py:func:`mlflow.keras.load_model` and
                           :py:func:`mlflow.pyfunc.load_model`.
    :param keras_module: Keras module to be used to save / load the model
                         (``keras`` or ``tf.keras``). If not provided, MLflow will
                         attempt to infer the Keras module based on the given model.
    :param kwargs: kwargs to pass to ``keras_model.save`` method.

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

        import mlflow
        # Build, compile, and train your model
        keras_model = ...
        keras_model_path = ...
        keras_model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])
        results = keras_model.fit(
            x_train, y_train, epochs=20, batch_size = 128, validation_data=(x_val, y_val))
        # Save the model as an MLflow Model
        mlflow.keras.save_model(keras_model, keras_model_path)
    """
    if keras_module is None:
        def _is_plain_keras(model):
            try:
                # NB: Network is the first parent with save method
                import keras.engine.network
                return isinstance(model, keras.engine.network.Network)
            except ImportError:
                return False

        def _is_tf_keras(model):
            try:
                # NB: Network is not exposed in tf.keras, we check for Model instead.
                import tensorflow.keras.models
                return isinstance(model, tensorflow.keras.models.Model)
            except ImportError:
                return False

        if _is_plain_keras(keras_model):
            keras_module = importlib.import_module("keras")
        elif _is_tf_keras(keras_model):
            keras_module = importlib.import_module("tensorflow.keras")
        else:
            raise MlflowException("Unable to infer keras module from the model, please specify "
                                  "which keras module ('keras' or 'tensorflow.keras') is to be "
                                  "used to save and load the model.")
    elif type(keras_module) == str:
        keras_module = importlib.import_module(keras_module)

    # check if path exists
    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException("Path '{}' already exists".format(path))

    # construct new data folder in existing path
    data_subpath = "data"
    data_path = os.path.join(path, data_subpath)
    os.makedirs(data_path)

    if mlflow_model is None:
        mlflow_model = Model()
    if signature is not None:
        mlflow_model.signature = signature
    if input_example is not None:
        _save_example(mlflow_model, input_example, path)

    # save custom objects if there are custom objects
    if custom_objects is not None:
        _save_custom_objects(data_path, custom_objects)

    # save keras module spec to path/data/keras_module.txt
    with open(os.path.join(data_path, _KERAS_MODULE_SPEC_PATH), "w") as f:
        f.write(keras_module.__name__)

    # save keras model to path/data/model.h5
    model_subpath = os.path.join(data_subpath, _MODEL_SAVE_PATH)
    model_path = os.path.join(path, model_subpath)
    if path.startswith('/dbfs/'):
        # The Databricks Filesystem uses a FUSE implementation that does not support
        # random writes. It causes an error.
        with tempfile.NamedTemporaryFile(suffix='.h5') as f:
            keras_model.save(f.name, **kwargs)
            f.flush()  # force flush the data
            shutil.copyfile(src=f.name, dst=model_path)
    else:
        keras_model.save(model_path, **kwargs)

    # update flavor info to mlflow_model
    mlflow_model.add_flavor(FLAVOR_NAME,
                            keras_module=keras_module.__name__,
                            keras_version=keras_module.__version__,
                            data=data_subpath)

    # save conda.yaml info to path/conda.yml
    if conda_env is None:
        conda_env = get_default_conda_env(include_cloudpickle=custom_objects is not None,
                                          keras_module=keras_module)
    elif not isinstance(conda_env, dict):
        with open(conda_env, "r") as f:
            conda_env = yaml.safe_load(f)
    with open(os.path.join(path, _CONDA_ENV_SUBPATH), "w") as f:
        yaml.safe_dump(conda_env, stream=f, default_flow_style=False)

    # append loader_module, data and env data to mlflow_model
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.keras",
                        data=data_subpath, env=_CONDA_ENV_SUBPATH)

    # save mlflow_model to path/MLmodel
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


def log_model(keras_model, artifact_path, conda_env=None, custom_objects=None, keras_module=None,
              registered_model_name=None, signature: ModelSignature=None,
              input_example: ModelInputExample=None, **kwargs):
    """
    Log a Keras model as an MLflow artifact for the current run.

    :param keras_model: Keras model to be saved.
    :param artifact_path: Run-relative artifact path.
    :param conda_env: Either a dictionary representation of a Conda environment or
                      the path to a Conda environment yaml file.
                      If provided, this describes the environment this model should be
                      run in. At minimum, it should specify the dependencies
                      contained in :func:`get_default_conda_env()`. If ``None``, the default
                      :func:`mlflow.keras.get_default_conda_env()` environment is added to
                      the model. The following is an *example* dictionary representation of a
                      Conda environment::

                        {
                            'name': 'mlflow-env',
                            'channels': ['defaults'],
                            'dependencies': [
                                'python=3.7.0',
                                'keras=2.2.4',
                                'tensorflow=1.8.0'
                            ]
                        }

    :param custom_objects: A Keras ``custom_objects`` dictionary mapping names (strings) to
                           custom classes or functions associated with the Keras model. MLflow saves
                           these custom layers using CloudPickle and restores them automatically
                           when the model is loaded with :py:func:`mlflow.keras.load_model` and
                           :py:func:`mlflow.pyfunc.load_model`.
    :param keras_module: Keras module to be used to save / load the model
                         (``keras`` or ``tf.keras``). If not provided, MLflow will
                         attempt to infer the Keras module based on the given model.
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

    :param kwargs: kwargs to pass to ``keras_model.save`` method.

    .. code-block:: python
        :caption: Example

        from keras import Dense, layers
        import mlflow
        # Build, compile, and train your model
        keras_model = ...
        keras_model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])
        results = keras_model.fit(
            x_train, y_train, epochs=20, batch_size = 128, validation_data=(x_val, y_val))
        # Log metrics and log the model
        with mlflow.start_run() as run:
            mlflow.keras.log_model(keras_model, "models")
    """
    Model.log(artifact_path=artifact_path, flavor=mlflow.keras,
              keras_model=keras_model, conda_env=conda_env, custom_objects=custom_objects,
              keras_module=keras_module, registered_model_name=registered_model_name,
              signature=signature, input_example=input_example,
              **kwargs)


def _save_custom_objects(path, custom_objects):
    """
    Save custom objects dictionary to a cloudpickle file so a model can be easily loaded later.

    :param path: An absolute path that points to the data directory within /path/to/model.
    :param custom_objects: Keras ``custom_objects`` is a dictionary mapping
                           names (strings) to custom classes or functions to be considered
                           during deserialization. MLflow saves these custom layers using
                           CloudPickle and restores them automatically when the model is
                           loaded with :py:func:`mlflow.keras.load_model` and
                           :py:func:`mlflow.pyfunc.load_model`.
    """
    import cloudpickle
    custom_objects_path = os.path.join(path, _CUSTOM_OBJECTS_SAVE_PATH)
    with open(custom_objects_path, "wb") as out_f:
        cloudpickle.dump(custom_objects, out_f)


def _load_model(model_path, keras_module, **kwargs):
    keras_models = importlib.import_module(keras_module.__name__ + ".models")
    custom_objects = kwargs.pop("custom_objects", {})
    custom_objects_path = None
    if os.path.isdir(model_path):
        if os.path.isfile(os.path.join(model_path, _CUSTOM_OBJECTS_SAVE_PATH)):
            custom_objects_path = os.path.join(model_path, _CUSTOM_OBJECTS_SAVE_PATH)
        model_path = os.path.join(model_path, _MODEL_SAVE_PATH)
    if custom_objects_path is not None:
        import cloudpickle
        with open(custom_objects_path, "rb") as in_f:
            pickled_custom_objects = cloudpickle.load(in_f)
            pickled_custom_objects.update(custom_objects)
            custom_objects = pickled_custom_objects
    from distutils.version import StrictVersion
    if StrictVersion(keras_module.__version__.split('-')[0]) >= StrictVersion("2.2.3"):
        # NOTE: Keras 2.2.3 does not work with unicode paths in python2. Pass in h5py.File instead
        # of string to avoid issues.
        import h5py
        with h5py.File(os.path.abspath(model_path), "r") as model_path:
            return keras_models.load_model(model_path, custom_objects=custom_objects, **kwargs)
    else:
        # NOTE: Older versions of Keras only handle filepath.
        return keras_models.load_model(model_path, custom_objects=custom_objects, **kwargs)


class _KerasModelWrapper:
    def __init__(self, keras_model, graph, sess):
        self.keras_model = keras_model
        self._graph = graph
        self._sess = sess

    def predict(self, dataframe):
        # In TensorFlow < 2.0, we use a graph and session to predict
        if self._graph is not None:
            with self._graph.as_default():
                with self._sess.as_default():
                    predicted = pd.DataFrame(self.keras_model.predict(dataframe.values))
        # In TensorFlow >= 2.0, we do not use a graph and session to predict
        else:
            predicted = pd.DataFrame(self.keras_model.predict(dataframe.values))
        predicted.index = dataframe.index
        return predicted


def _load_pyfunc(path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.

    :param path: Local filesystem path to the MLflow Model with the ``keras`` flavor.
    """
    import tensorflow as tf
    if os.path.isfile(os.path.join(path, _KERAS_MODULE_SPEC_PATH)):
        with open(os.path.join(path, _KERAS_MODULE_SPEC_PATH), "r") as f:
            keras_module = importlib.import_module(f.read())
    else:
        import keras
        keras_module = keras

    K = importlib.import_module(keras_module.__name__ + ".backend")
    if keras_module.__name__ == "tensorflow.keras" or K.backend() == 'tensorflow':
        if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
            graph = tf.Graph()
            sess = tf.Session(graph=graph)
            # By default tf backed models depend on the global graph and session.
            # We create an use new Graph and Session and store them with the model
            # This way the model is independent on the global state.
            with graph.as_default():
                with sess.as_default():  # pylint:disable=not-context-manager
                    K.set_learning_phase(0)
                    m = _load_model(path, keras_module=keras_module, compile=False)
                    return _KerasModelWrapper(m, graph, sess)
        else:
            K.set_learning_phase(0)
            m = _load_model(path, keras_module=keras_module, compile=False)
            return _KerasModelWrapper(m, None, None)

    else:
        raise MlflowException("Unsupported backend '%s'" % K._BACKEND)


def load_model(model_uri, **kwargs):
    """
    Load a Keras model from a local file or a run.

    Extra arguments are passed through to keras.load_model.

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

    :return: A Keras model instance.

    .. code-block:: python
        :caption: Example

        # Load persisted model as a Keras model or as a PyFunc, call predict() on a pandas DataFrame
        keras_model = mlflow.keras.load_model("runs:/96771d893a5e46159d9f3b49bf9013e2" + "/models")
        predictions = keras_model.predict(x_test)
    """
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri)
    flavor_conf = _get_flavor_configuration(model_path=local_model_path, flavor_name=FLAVOR_NAME)
    keras_module = importlib.import_module(flavor_conf.get("keras_module", "keras"))
    keras_model_artifacts_path = os.path.join(
        local_model_path,
        flavor_conf.get("data", _MODEL_SAVE_PATH))
    return _load_model(model_path=keras_model_artifacts_path, keras_module=keras_module, **kwargs)


@experimental
def autolog():
    # pylint: disable=E0611
    """
    Enables automatic logging from Keras to MLflow. Autologging captures the following information:

    **Metrics** and **Parameters**
     - Training loss; validation loss; user-specified metrics
     - Metrics associated with the ``EarlyStopping`` callbacks: ``stopped_epoch``,
       ``restored_epoch``, ``restore_best_weight``, ``last_epoch``, etc
     - ``fit()`` or ``fit_generator()`` parameters; optimizer name; learning rate; epsilon
     - ``fit()`` or ``fit_generator()`` parameters associated with ``EarlyStopping``: ``min_delta``,
       ``patience``, ``baseline``, ``restore_best_weights``, etc
    **Artifacts**
     - Model summary on training start
     - `MLflow Model <https://mlflow.org/docs/latest/models.html>`_ (Keras model) on training end

    .. code-block:: python
        :caption: Example

        import mlflow
        import mlflow.keras
        # Build, compile, enable autologging, and train your model
        keras_model = ...
        keras_model.compile(optimizer="rmsprop", loss="mse", metrics=["accuracy"])
        # autolog your metrics, parameters, and model
        mlflow.keras.autolog()
        results = keras_model.fit(
            x_train, y_train, epochs=20, batch_size=128, validation_data=(x_val, y_val))

    ``EarlyStopping Integration with Keras AutoLogging``

    MLflow will detect if an ``EarlyStopping`` callback is used in a ``fit()`` or
    ``fit_generator()`` call, and if the ``restore_best_weights`` parameter is set to be ``True``,
    then MLflow will log the metrics associated with the restored model as a final, extra step.
    The epoch of the restored model will also be logged as the metric ``restored_epoch``.
    This allows for easy comparison between the actual metrics of the restored model and
    the metrics of other models.

    If ``restore_best_weights`` is set to be ``False``, then MLflow will not log an additional step.

    Regardless of ``restore_best_weights``, MLflow will also log ``stopped_epoch``,
    which indicates the epoch at which training stopped due to early stopping.

    If training does not end due to early stopping, then ``stopped_epoch`` will be logged as ``0``.

    MLflow will also log the parameters of the ``EarlyStopping`` callback,
    excluding ``mode`` and ``verbose``.
    """
    import keras

    class __MLflowKerasCallback(keras.callbacks.Callback):
        """
        Callback for auto-logging metrics and parameters.
        Records available logs after each epoch.
        Records model structural information as params when training begins
        """
        def on_train_begin(self, logs=None):  # pylint: disable=unused-argument
            try_mlflow_log(mlflow.log_param, 'num_layers', len(self.model.layers))
            try_mlflow_log(mlflow.log_param, 'optimizer_name', type(self.model.optimizer).__name__)
            if hasattr(self.model.optimizer, 'lr'):
                lr = self.model.optimizer.lr if \
                    type(self.model.optimizer.lr) is float \
                    else keras.backend.eval(self.model.optimizer.lr)
                try_mlflow_log(mlflow.log_param, 'learning_rate', lr)
            if hasattr(self.model.optimizer, 'epsilon'):
                epsilon = self.model.optimizer.epsilon if \
                    type(self.model.optimizer.epsilon) is float \
                    else keras.backend.eval(self.model.optimizer.epsilon)
                try_mlflow_log(mlflow.log_param, 'epsilon', epsilon)

            sum_list = []
            self.model.summary(print_fn=sum_list.append)
            summary = '\n'.join(sum_list)
            tempdir = tempfile.mkdtemp()
            try:
                summary_file = os.path.join(tempdir, "model_summary.txt")
                with open(summary_file, 'w') as f:
                    f.write(summary)
                try_mlflow_log(mlflow.log_artifact, local_path=summary_file)
            finally:
                shutil.rmtree(tempdir)

        def on_epoch_end(self, epoch, logs=None):
            if not logs:
                return
            try_mlflow_log(mlflow.log_metrics, logs, step=epoch)

        def on_train_end(self, logs=None):
            try_mlflow_log(log_model, self.model, artifact_path='model')

        # As of Keras 2.4.0, Keras Callback implementations must define the following
        # methods indicating whether or not the callback overrides functions for
        # batch training/testing/inference
        def _implements_train_batch_hooks(self): return False

        def _implements_test_batch_hooks(self): return False

        def _implements_predict_batch_hooks(self): return False

    def _early_stop_check(callbacks):
        if LooseVersion(keras.__version__) < LooseVersion('2.3.0'):
            es_callback = keras.callbacks.EarlyStopping
        else:
            es_callback = keras.callbacks.callbacks.EarlyStopping
        for callback in callbacks:
            if isinstance(callback, es_callback):
                return callback
        return None

    def _log_early_stop_callback_params(callback):
        if callback:
            try:
                earlystopping_params = {'monitor': callback.monitor,
                                        'min_delta': callback.min_delta,
                                        'patience': callback.patience,
                                        'baseline': callback.baseline,
                                        'restore_best_weights': callback.restore_best_weights}
                try_mlflow_log(mlflow.log_params, earlystopping_params)
            except Exception:  # pylint: disable=W0703
                return

    def _get_early_stop_callback_attrs(callback):
        try:
            return callback.stopped_epoch, callback.restore_best_weights, callback.patience
        except Exception:  # pylint: disable=W0703
            return None

    def _log_early_stop_callback_metrics(callback, history):
        if callback:
            callback_attrs = _get_early_stop_callback_attrs(callback)
            if callback_attrs is None:
                return
            stopped_epoch, restore_best_weights, patience = callback_attrs
            try_mlflow_log(mlflow.log_metric, 'stopped_epoch', stopped_epoch)
            # Weights are restored only if early stopping occurs
            if stopped_epoch != 0 and restore_best_weights:
                restored_epoch = stopped_epoch - max(1, patience)
                try_mlflow_log(mlflow.log_metric, 'restored_epoch', restored_epoch)
                restored_metrics = {key: history.history[key][restored_epoch]
                                    for key in history.history.keys()}
                # Checking that a metric history exists
                metric_key = next(iter(history.history), None)
                if metric_key is not None:
                    last_epoch = len(history.history[metric_key])
                    try_mlflow_log(mlflow.log_metrics, restored_metrics, step=last_epoch)

    def _run_and_log_function(self, original, args, kwargs, unlogged_params, callback_arg_index):
        if not mlflow.active_run():
            try_mlflow_log(mlflow.start_run)
            auto_end_run = True
        else:
            auto_end_run = False

        log_fn_args_as_params(original, args, kwargs, unlogged_params)
        early_stop_callback = None

        # Checking if the 'callback' argument of the function is set
        if len(args) > callback_arg_index:
            tmp_list = list(args)
            early_stop_callback = _early_stop_check(tmp_list[callback_arg_index])
            tmp_list[callback_arg_index] += [__MLflowKerasCallback()]
            args = tuple(tmp_list)
        elif 'callbacks' in kwargs:
            early_stop_callback = _early_stop_check(kwargs['callbacks'])
            kwargs['callbacks'] += [__MLflowKerasCallback()]
        else:
            kwargs['callbacks'] = [__MLflowKerasCallback()]

        _log_early_stop_callback_params(early_stop_callback)

        history = original(self, *args, **kwargs)

        _log_early_stop_callback_metrics(early_stop_callback, history)

        if auto_end_run:
            try_mlflow_log(mlflow.end_run)

        return history

    @gorilla.patch(keras.Model)
    def fit(self, *args, **kwargs):
        original = gorilla.get_original_attribute(keras.Model, 'fit')
        unlogged_params = ['self', 'x', 'y', 'callbacks', 'validation_data', 'verbose']
        return _run_and_log_function(self, original, args, kwargs, unlogged_params, 5)

    @gorilla.patch(keras.Model)
    def fit_generator(self, *args, **kwargs):
        original = gorilla.get_original_attribute(keras.Model, 'fit_generator')
        unlogged_params = ['self', 'generator', 'callbacks', 'validation_data', 'verbose']
        return _run_and_log_function(self, original, args, kwargs, unlogged_params, 4)

    settings = gorilla.Settings(allow_hit=True, store_hit=True)
    gorilla.apply(gorilla.Patch(keras.Model, 'fit', fit, settings=settings))
    gorilla.apply(gorilla.Patch(keras.Model, 'fit_generator', fit_generator, settings=settings))
