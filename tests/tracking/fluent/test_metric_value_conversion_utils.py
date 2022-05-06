import pytest

from mlflow.exceptions import MlflowException, INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.tracking.metric_value_conversion_utils import *
from tests.helper_functions import random_int

import numpy as np
import tensorflow as tf
import torch


def test_reraised_value_errors():
    multi_item_array = np.random.rand(2, 2)
    multi_item_tf_tensor = tf.random.uniform([2, 2])
    multi_item_torch_tensor = torch.rand((2, 2))

    with pytest.raises(MlflowException) as e:
        convert_metric_value_to_float_if_possible(multi_item_array)

    assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    with pytest.raises(MlflowException) as e:
        convert_metric_value_to_float_if_possible(multi_item_tf_tensor)

    assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)

    with pytest.raises(MlflowException) as e:
        convert_metric_value_to_float_if_possible(multi_item_torch_tensor)

    assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_convert_metric_value_to_float():
    float_metric_val = float(random_int(10, 50))

    assert float_metric_val == convert_metric_value_to_float_if_possible(float_metric_val)

    ndarray_val = np.random.rand(1)
    assert float(ndarray_val[0]) == convert_metric_value_to_float_if_possible(ndarray_val)

    tf_tensor_val = tf.random.uniform([])
    assert float(tf_tensor_val.numpy()) == convert_metric_value_to_float_if_possible(tf_tensor_val)

    torch_tensor_val = torch.rand(1)
    assert float(torch_tensor_val[0]) == convert_metric_value_to_float_if_possible(torch_tensor_val)
