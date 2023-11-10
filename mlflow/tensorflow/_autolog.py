import warnings
from typing import Dict, Union

import numpy as np
import tensorflow
from tensorflow.keras.callbacks import Callback, TensorBoard

import mlflow
from mlflow.utils.autologging_utils import (
    INPUT_EXAMPLE_SAMPLE_ROWS,
    ExceptionSafeClass,
)


class _TensorBoard(TensorBoard, metaclass=ExceptionSafeClass):
    pass


class __MLflowTfKeras2Callback(Callback, metaclass=ExceptionSafeClass):
    """
    Callback for auto-logging parameters and metrics in TensorFlow >= 2.0.0.
    Records model structural information as params when training starts.
    """

    def __init__(self, metrics_logger, log_every_n_steps):
        super().__init__()
        self.metrics_logger = metrics_logger
        self.log_every_n_steps = log_every_n_steps

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def on_train_begin(self, logs=None):  # pylint: disable=unused-argument
        config = self.model.optimizer.get_config()
        for attribute in config:
            mlflow.log_param("opt_" + attribute, config[attribute])

        sum_list = []
        try:
            self.model.summary(print_fn=sum_list.append)
            summary = "\n".join(sum_list)
            mlflow.log_text(summary, artifact_file="model_summary.txt")
        except ValueError as ex:
            if "This model has not yet been built" in str(ex):
                warnings.warn(str(ex))
            else:
                raise ex

    def on_epoch_end(self, epoch, logs=None):
        # NB: tf.Keras uses zero-indexing for epochs, while other TensorFlow Estimator
        # APIs (e.g., tf.Estimator) use one-indexing. Accordingly, the modular arithmetic
        # used here is slightly different from the arithmetic used in `_log_event`, which
        # provides  metric logging hooks for TensorFlow Estimator & other TensorFlow APIs
        if epoch % self.log_every_n_steps == 0:
            self.metrics_logger.record_metrics(logs, epoch)


def _extract_input_example_from_tensor_or_ndarray(
    input_features: Union[tensorflow.Tensor, np.ndarray]
) -> np.ndarray:
    """
    Extracts first `INPUT_EXAMPLE_SAMPLE_ROWS` from the next_input, which can either be of
    numpy array or tensor type.

    :param input_features: an input of type `np.ndarray` or `tensorflow.Tensor`
    :return: A slice (of limit `INPUT_EXAMPLE_SAMPLE_ROWS`)  of the input of type `np.ndarray`.
             Returns `None` if the type of `input_features` is unsupported.

    Examples
    --------
    when next_input is nd.array:
    >>> input_data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> _extract_input_example_from_tensor_or_ndarray(input_data)
    array([1, 2, 3, 4, 5])


    when next_input is tensorflow.Tensor:
    >>> input_data = tensorflow.convert_to_tensor([1, 2, 3, 4, 5, 6])
    >>> _extract_input_example_from_tensor_or_ndarray(input_data)
    array([1, 2, 3, 4, 5])
    """

    input_feature_slice = None
    if isinstance(input_features, tensorflow.Tensor):
        input_feature_slice = input_features.numpy()[0:INPUT_EXAMPLE_SAMPLE_ROWS]
    elif isinstance(input_features, np.ndarray):
        input_feature_slice = input_features[0:INPUT_EXAMPLE_SAMPLE_ROWS]
    return input_feature_slice


def _extract_sample_numpy_dict(
    input_numpy_features_dict: Dict[str, np.ndarray]
) -> Union[Dict[str, np.ndarray], np.ndarray]:
    """
    Extracts `INPUT_EXAMPLE_SAMPLE_ROWS` sample from next_input
    as numpy array of dict(str -> ndarray) type.

    :param input_numpy_features_dict: A tensor or numpy array
    :return:a slice (limit `INPUT_EXAMPLE_SAMPLE_ROWS`)  of the input of same type as next_input.
            Returns `None` if the type of `input_numpy_features_dict` is unsupported.

    Examples
    --------
    when next_input is dict:
    >>> input_data = {"a": np.array([1, 2, 3, 4, 5, 6, 7, 8])}
    >>> _extract_sample_numpy_dict(input_data)
    {'a': array([1, 2, 3, 4, 5])}

    """
    sliced_data_as_numpy = None
    if isinstance(input_numpy_features_dict, dict):
        sliced_data_as_numpy = {
            k: _extract_input_example_from_tensor_or_ndarray(v)
            for k, v in input_numpy_features_dict.items()
        }
    return sliced_data_as_numpy


def _extract_input_example_from_batched_tf_dataset(
    dataset: tensorflow.data.Dataset,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Extracts sample feature tensors from the input dataset as numpy array.
    Input Dataset's tensors must contain tuple of (features, labels) that are
    used for tensorflow/keras train or fit methods


    :param dataset: a tensorflow batched/unbatched dataset representing tuple of (features, labels)
    :return: a numpy array of length `INPUT_EXAMPLE_SAMPLE_ROWS`
             Returns `None` if the type of `dataset` slices are unsupported.

    Examples
    --------
    >>> input_dataset = tensorflow.data.Dataset.from_tensor_slices(
    ...                 ({'SepalLength': np.array(list(range(0, 20))),
    ...                 'SepalWidth': np.array(list(range(0, 20))),
    ...                 'PetalLength': np.array(list(range(0, 20))),
    ...                 'PetalWidth': np.array(list(range(0, 20)))},
    ...                 np.array(list(range(0, 20))))).batch(10)
    >>> _extract_input_example_from_batched_tf_dataset(input_dataset)
    {'SepalLength': array([0, 1, 2, 3, 4]),
    'SepalWidth': array([0, 1, 2, 3, 4]),
    'PetalLength': array([0, 1, 2, 3, 4]),
    'PetalWidth': array([0, 1, 2, 3, 4])}

    """
    limited_df_iter = list(dataset.take(INPUT_EXAMPLE_SAMPLE_ROWS))
    first_batch = limited_df_iter[0]
    input_example_slice = None
    if isinstance(first_batch, tuple):
        features = first_batch[0]
        if isinstance(features, dict):
            input_example_slice = _extract_sample_numpy_dict(features)
        elif isinstance(features, (np.ndarray, tensorflow.Tensor)):
            input_example_slice = _extract_input_example_from_tensor_or_ndarray(features)
    return input_example_slice


def extract_input_example_from_tf_input_fn(input_fn):
    """
    Extracts sample data from dict (str -> ndarray),
    ``tensorflow.Tensor`` or ``tensorflow.data.Dataset`` type.

    :param input_fn: Tensorflow's input function used for train method
    :return: a slice (of limit ``mlflow.utils.autologging_utils.INPUT_EXAMPLE_SAMPLE_ROWS``)
             of the input of type `np.ndarray`.
             Returns `None` if the return type of ``input_fn`` is unsupported.
    """

    input_training_data = input_fn()
    input_features = None
    if isinstance(input_training_data, tuple):
        features = input_training_data[0]
        if isinstance(features, dict):
            input_features = _extract_sample_numpy_dict(features)
        elif isinstance(features, (np.ndarray, tensorflow.Tensor)):
            input_features = _extract_input_example_from_tensor_or_ndarray(features)
    elif isinstance(input_training_data, tensorflow.data.Dataset):
        input_features = _extract_input_example_from_batched_tf_dataset(input_training_data)
    return input_features


def extract_tf_keras_input_example(input_training_data):
    """
    Generates a sample ndarray or dict (str -> ndarray)
    from the input type 'x' for keras ``fit`` or ``fit_generator``

    :param input_training_data: Keras input function used for ``fit`` or
                                ``fit_generator`` methods
    :return: a slice of type ndarray or
             dict (str -> ndarray) limited to
             ``mlflow.utils.autologging_utils.INPUT_EXAMPLE_SAMPLE_ROWS``.
             Throws ``MlflowException`` exception, if input_training_data is unsupported.
             Returns `None` if the type of input_training_data is unsupported.
    """
    input_data_slice = None
    if isinstance(input_training_data, tensorflow.keras.utils.Sequence):
        input_training_data = input_training_data[:][0]

    if isinstance(input_training_data, (np.ndarray, tensorflow.Tensor)):
        input_data_slice = _extract_input_example_from_tensor_or_ndarray(input_training_data)
    elif isinstance(input_training_data, dict):
        input_data_slice = _extract_sample_numpy_dict(input_training_data)
    elif isinstance(input_training_data, tensorflow.data.Dataset):
        input_data_slice = _extract_input_example_from_batched_tf_dataset(input_training_data)
    return input_data_slice
