import numpy as np
import tensorflow
from tensorflow.keras.callbacks import TensorBoard

from mlflow.utils.autologging_utils import (
    INPUT_EXAMPLE_SAMPLE_ROWS,
    ExceptionSafeClass,
)


class _TensorBoard(TensorBoard, metaclass=ExceptionSafeClass):
    pass


def _extract_input_example_from_tensor_or_ndarray(
    input_features: tensorflow.Tensor | np.ndarray,
) -> np.ndarray:
    """
    Extracts first `INPUT_EXAMPLE_SAMPLE_ROWS` from the next_input, which can either be of
    numpy array or tensor type.

    Args:
        input_features: an input of type `np.ndarray` or `tensorflow.Tensor`

    Returns:
        A slice (of limit `INPUT_EXAMPLE_SAMPLE_ROWS`)  of the input of type `np.ndarray`.
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
    input_numpy_features_dict: dict[str, np.ndarray],
) -> dict[str, np.ndarray] | np.ndarray:
    """
    Extracts `INPUT_EXAMPLE_SAMPLE_ROWS` sample from next_input
    as numpy array of dict(str -> ndarray) type.

    Args:
        input_numpy_features_dict: A tensor or numpy array

    Returns:
        A slice (limit `INPUT_EXAMPLE_SAMPLE_ROWS`)  of the input of same type as next_input.
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
) -> np.ndarray | dict[str, np.ndarray]:
    """
    Extracts sample feature tensors from the input dataset as numpy array.
    Input Dataset's tensors must contain tuple of (features, labels) that are
    used for tensorflow/keras train or fit methods


    Args:
        dataset: a tensorflow batched/unbatched dataset representing tuple of (features, labels)

    Returns:
        a numpy array of length `INPUT_EXAMPLE_SAMPLE_ROWS`
        Returns `None` if the type of `dataset` slices are unsupported.

    Examples
    --------
    >>> input_dataset = tensorflow.data.Dataset.from_tensor_slices(
    ...     (
    ...         {
    ...             "SepalLength": np.array(list(range(0, 20))),
    ...             "SepalWidth": np.array(list(range(0, 20))),
    ...             "PetalLength": np.array(list(range(0, 20))),
    ...             "PetalWidth": np.array(list(range(0, 20))),
    ...         },
    ...         np.array(list(range(0, 20))),
    ...     )
    ... ).batch(10)
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

    Args:
        input_fn: Tensorflow's input function used for train method

    Returns:
        A slice (of limit ``mlflow.utils.autologging_utils.INPUT_EXAMPLE_SAMPLE_ROWS``)
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

    Args:
        input_training_data: Keras input function used for ``fit`` or ``fit_generator`` methods.

    Returns:
        a slice of type ndarray or
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
