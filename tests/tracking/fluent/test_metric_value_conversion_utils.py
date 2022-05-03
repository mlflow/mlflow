import unittest

from mlflow.exceptions import MlflowException
from mlflow.tracking.metic_value_conversion_utils import *
from tests.helper_functions import random_str, random_int

import numpy as np
import tensorflow as tf
import torch

import pyspark.ml.param
import pyspark.ml.util


class TestMetricValueConversionUtils(unittest.TestCase):
    def test_reraised_value_errors(self):
        multi_item_array = np.random.rand(2, 2)
        multi_item_tf_tensor = tf.random.uniform([2, 2])
        multi_item_torch_tensor = torch.rand((2, 2))

        with self.assertRaises(MlflowException):
            convert_metric_value_to_float_if_possible(multi_item_array)

        with self.assertRaises(MlflowException):
            convert_metric_value_to_float_if_possible(multi_item_tf_tensor)

        with self.assertRaises(MlflowException):
            convert_metric_value_to_float_if_possible(multi_item_torch_tensor)

    def test_convert_metric_value_to_str(self):
        str_mertic_val = random_str(random_int(10, 50))

        self.assertEqual(str_mertic_val, convert_metric_value_to_str_if_possible(str_mertic_val))

        random_parent = pyspark.ml.util.Identifiable()
        pyspark_ml_param = pyspark.ml.param.Param(random_parent, str_mertic_val, "doc")

        self.assertEqual(str_mertic_val, convert_metric_value_to_str_if_possible(pyspark_ml_param))

        ndarray_val = np.random.rand(1)
        self.assertEqual(
            f"[{float(ndarray_val[0])}]", convert_metric_value_to_str_if_possible(ndarray_val)
        )

        tf_tensor_val = tf.random.uniform([])
        self.assertEqual(
            f"[{float(tf_tensor_val.numpy())}]",
            convert_metric_value_to_str_if_possible(tf_tensor_val),
        )

        torch_tensor_val = torch.rand(1)
        self.assertEqual(
            f"[{float(torch_tensor_val[0])}]",
            convert_metric_value_to_str_if_possible(torch_tensor_val),
        )

    def test_convert_metric_value_to_float(self):
        float_metric_val = float(random_int(10, 50))

        self.assertEqual(
            float_metric_val, convert_metric_value_to_float_if_possible(float_metric_val)
        )

        ndarray_val = np.random.rand(1)
        self.assertEqual(
            float(ndarray_val[0]), convert_metric_value_to_float_if_possible(ndarray_val)
        )

        tf_tensor_val = tf.random.uniform([])
        self.assertEqual(
            float(tf_tensor_val.numpy()), convert_metric_value_to_float_if_possible(tf_tensor_val)
        )

        torch_tensor_val = torch.rand(1)
        self.assertEqual(
            float(torch_tensor_val[0]), convert_metric_value_to_float_if_possible(torch_tensor_val)
        )
