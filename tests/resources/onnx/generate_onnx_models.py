# pylint: disable=import-error

"""
Generates the following test resources:

    - tf_model_multiple_inputs_float32.onnx
    - tf_model_multiple_inputs_float64.onnx
    - sklearn_model.onnx

Usage: python generate_onnx_models.py
"""


import numpy as np
import onnx
import onnxmltools
import pandas as pd
import sklearn.datasets as datasets
import tensorflow.compat.v1 as tf
import tf2onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression

tf.disable_v2_behavior()


def generate_tf_onnx_model_multiple_inputs_float64():
    graph = tf.Graph()
    with graph.as_default():
        t_in1 = tf.placeholder(tf.float64, 10, name="first_input")
        t_in2 = tf.placeholder(tf.float64, 10, name="second_input")
        t_out = tf.multiply(t_in1, t_in2)
        tf.identity(t_out, name="output")

    sess = tf.Session(graph=graph)

    onnx_graph = tf2onnx.tfonnx.process_tf_graph(
        sess.graph, input_names=["first_input:0", "second_input:0"], output_names=["output:0"]
    )
    model_proto = onnx_graph.make_model("test")

    onnx.save_model(model_proto, "tf_model_multiple_inputs_float64.onnx")


def generate_tf_onnx_model_multiple_inputs_float32():
    graph = tf.Graph()
    with graph.as_default():
        t_in1 = tf.placeholder(tf.float32, 10, name="first_input")
        t_in2 = tf.placeholder(tf.float32, 10, name="second_input")
        t_out = tf.multiply(t_in1, t_in2)
        tf.identity(t_out, name="output")

    sess = tf.Session(graph=graph)

    onnx_graph = tf2onnx.tfonnx.process_tf_graph(
        sess.graph, input_names=["first_input:0", "second_input:0"], output_names=["output:0"]
    )
    model_proto = onnx_graph.make_model("test")

    onnx.save_model(model_proto, "tf_model_multiple_inputs_float32.onnx")


def generate_sklearn_onnx_model():
    iris = datasets.load_iris()
    data = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
    )
    y = data["target"]
    x = data.drop("target", axis=1)

    model = LogisticRegression()
    model.fit(x, y)

    initial_type = [("float_input", FloatTensorType([None, 4]))]
    onx = onnxmltools.convert_sklearn(model, initial_types=initial_type)
    onnx.save_model(onx, "sklearn_model.onnx")


generate_tf_onnx_model_multiple_inputs_float32()
generate_tf_onnx_model_multiple_inputs_float64()
generate_sklearn_onnx_model()
