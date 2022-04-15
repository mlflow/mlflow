import warnings

import tensorflow
import numpy as np
from tensorflow.keras.callbacks import Callback, TensorBoard
from typing import Union, Dict

import mlflow
from mlflow.utils.autologging_utils import ExceptionSafeClass
from mlflow.utils.autologging_utils import (
    INPUT_EXAMPLE_SAMPLE_ROWS,
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


def extract_sample_tensor_as_numpy_input_example_slice(
    next_input: Union[tensorflow.Tensor, np.ndarray]
) -> np.ndarray:
    """
    Extracts first `INPUT_EXAMPLE_SAMPLE_ROWS` from the next_input, which can either be of
    numpy array or tensor type.

    :param next_input: an input of type `np.ndarray` or `tensorflow.Tensor`
    :return: a slice (of limit `INPUT_EXAMPLE_SAMPLE_ROWS`)  of the input of type `np.ndarray`

    Examples
    --------
    when next_input is nd.array:
    >>> input_data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    >>> extract_sample_tensor_as_numpy_input_example_slice(input_data)
    array([1, 2, 3, 4, 5])


    when next_input is tensorflow.Tensor:
    >>> input_data = tensorflow.convert_to_tensor([1, 2, 3, 4, 5, 6])
    >>> extract_sample_tensor_as_numpy_input_example_slice(input_data)
    array([1, 2, 3, 4, 5])
    """

    input_feature_slice = None
    if isinstance(next_input, tensorflow.Tensor):
        input_feature_slice = next_input.numpy()[0:INPUT_EXAMPLE_SAMPLE_ROWS]
    elif isinstance(next_input, np.ndarray):
        input_feature_slice = next_input[0:INPUT_EXAMPLE_SAMPLE_ROWS]
    else:
        warnings.warn(
            f"Tensorflow estimator doesn't support '{type(next_input)}' type for features"
        )
    return input_feature_slice


def extract_sample_dict_or_tensor(
    next_input: Union[Dict[str, Union[tensorflow.Tensor, np.ndarray]], tensorflow.Tensor]
) -> Union[Dict[str, np.ndarray], np.ndarray]:
    """
    Extracts `INPUT_EXAMPLE_SAMPLE_ROWS` sample from next_input
    as numpy array of dict(str -> ndarray) type.

    :param next_input: A tensor or numpy array
    :return:a slice (limit `INPUT_EXAMPLE_SAMPLE_ROWS`)  of the input of same type as next_input

    Examples
    --------
    when next_input is dict:
    >>> input_data = {"a": np.array([1, 2, 3, 4, 5, 6, 7, 8])}
    >>> extract_sample_dict_or_tensor(input_data)
    {'a': array([1, 2, 3, 4, 5])}


    when next_input is tensorflow.Tensor:
    >>> input_data = tensorflow.convert_to_tensor([1, 2, 3, 4, 5, 6])
    >>> extract_sample_dict_or_tensor(input_data)
    array([1, 2, 3, 4, 5])
    """
    sliced_data_as_numpy = None
    if isinstance(next_input, dict):
        sliced_data_as_numpy = {
            k: extract_sample_tensor_as_numpy_input_example_slice(v) for k, v in next_input.items()
        }
    elif isinstance(next_input, tensorflow.Tensor):
        sliced_data_as_numpy = extract_sample_tensor_as_numpy_input_example_slice(next_input)
    return sliced_data_as_numpy


def extract_sample_features_from_batched_tf_dataset(
    dataset: tensorflow.data.Dataset,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Extracts sample feature tensors from the input dataset as numpy array.
    Input Dataset's tensors must contain tuple of (features, labels) that are
    used for tensorflow/keras train or fit methods


    :param dataset: a tensorflow batched/unbatched dataset representing tuple of (features, labels)
    :return: a numpy array of length `INPUT_EXAMPLE_SAMPLE_ROWS`

    Examples
    --------::
    >>> input_dataset = tensorflow.data.Dataset.from_tensor_slices(
    >>>    ({'SepalLength': np.array(list(range(0, 20))),
    >>>      'SepalWidth': np.array(list(range(0, 20))),
    >>>      'PetalLength': np.array(list(range(0, 20))),
    >>>      'PetalWidth': np.array(list(range(0, 20)))}, np.array(list(range(0, 20))))).batch(10)
    >>> extract_sample_features_from_batched_tf_dataset(input_dataset)
    {'SepalLength': array([0, 1, 2, 3, 4]),
    'SepalWidth': array([0, 1, 2, 3, 4]),
    'PetalLength': array([0, 1, 2, 3, 4]),
    'PetalWidth': array([0, 1, 2, 3, 4])}

    """
    limited_df_iter = list(dataset.take(INPUT_EXAMPLE_SAMPLE_ROWS))
    first_batch = limited_df_iter[0]
    if isinstance(first_batch, tuple):
        features = first_batch[0]
        if isinstance(features, (dict, tensorflow.Tensor)):
            return extract_sample_dict_or_tensor(features)
        elif isinstance(features, np.ndarray):
            return extract_sample_tensor_as_numpy_input_example_slice(features)
        raise TypeError(f"Unsupported type for features of type: {type(features)}")
    raise TypeError(f"Unsupported type for dataset batch slice of type: {type(first_batch)}")
