import pytest

import mlflow
from mlflow import tracking
from mlflow.tracking.fluent import start_run

from mlflow.exceptions import MlflowException, INVALID_PARAMETER_VALUE, ErrorCode
from mlflow.tracking.metric_value_conversion_utils import convert_metric_value_to_float_if_possible
from tests.helper_functions import random_int

import numpy as np


def test_reraised_value_errors():
    multi_item_array = np.random.rand(2, 2)

    with pytest.raises(MlflowException, match=r"Failed to convert metric value to float") as e:
        convert_metric_value_to_float_if_possible(multi_item_array)

    assert e.value.error_code == ErrorCode.Name(INVALID_PARAMETER_VALUE)


def test_convert_metric_value_to_float():
    float_metric_val = float(random_int(10, 50))

    assert convert_metric_value_to_float_if_possible(float_metric_val) == float_metric_val

    ndarray_val = np.random.rand(1)
    assert convert_metric_value_to_float_if_possible(ndarray_val) == float(ndarray_val[0])


def test_log_np_array_as_metric():
    ndarray_val = np.random.rand(1)
    ndarray_float_val = float(ndarray_val[0])

    with start_run() as run:
        mlflow.log_metric("name_numpy", ndarray_val)

    finished_run = tracking.MlflowClient().get_run(run.info.run_id)
    assert finished_run.data.metrics == {"name_numpy": ndarray_float_val}
