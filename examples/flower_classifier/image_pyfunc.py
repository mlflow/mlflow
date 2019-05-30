"""
Example of a custom python function implementing image classifier with image preprocessing embedded
in the model.
"""
import base64
from io import BytesIO
import keras
import numpy as np
import os
import pandas as pd
import PIL
from PIL import Image
import yaml
import tensorflow as tf

import mlflow
import mlflow.keras
from mlflow.utils import PYTHON_VERSION
from mlflow.utils.file_utils import TempDir
from mlflow.utils.environment import _mlflow_conda_env


def decode_and_resize_image(raw_bytes, size):
    """
    Read, decode and resize raw image bytes (e.g. raw content of a jpeg file).

    :param raw_bytes: Image bits, e.g. jpeg image.
    :param size: requested output dimensions
    :return: Multidimensional numpy array representing the resized image.
    """
    return np.asarray(Image.open(BytesIO(raw_bytes)).resize(size), dtype=np.float32)


class KerasImageClassifierPyfunc(object):
    """
    Image classification model with embedded pre-processing.

    This class is essentially an MLflow custom python function wrapper around a Keras model.
    The wrapper provides image preprocessing so that the model can be applied to images directly.
    The input to the model is base64 encoded image binary data (e.g. contents of a jpeg file).
    The output is the predicted class label, predicted class id followed by probabilities for each
    class.

    The model declares current local versions of Keras, Tensorlow and pillow as dependencies in its
    conda environment file.
    """

    def __init__(self, graph, session, model, image_dims, domain):
        self._graph = graph
        self._session = session
        self._model = model
        self._image_dims = image_dims
        self._domain = domain
        probs_names = ["p({})".format(x) for x in domain]
        self._column_names = ["predicted_label", "predicted_label_id"] + probs_names

    def predict(self, input):
        """
        Generate predictions for the data.

        :param input: pandas.DataFrame with one column containing images to be scored. The image
                     column must contain base64 encoded binary content of the image files. The image
                     format must be supported by PIL (e.g. jpeg or png).

        :return: pandas.DataFrame containing predictions with the following schema:
                     Predicted class: string,
                     Predicted class index: int,
                     Probability(class==0): float,
                     ...,
                     Probability(class==N): float,
        """

        # decode image bytes from base64 encoding
        def decode_img(x):
            return pd.Series(base64.decodebytes(bytearray(x[0], encoding="utf8")))

        images = input.apply(axis=1, func=decode_img)
        probs = self._predict_images(images)
        m, n = probs.shape
        label_idx = np.argmax(probs, axis=1)
        labels = np.array([self._domain[i] for i in label_idx], dtype=np.str).reshape(m, 1)
        output_data = np.concatenate((labels, label_idx.reshape(m, 1), probs), axis=1)
        res = pd.DataFrame(columns=self._column_names, data=output_data)
        res.index = input.index
        return res

    def _predict_images(self, images):
        """
        Generate predictions for input images.
        :param images: binary image data
        :return: predicted probabilities for each class
        """

        def preprocess_f(z):
            return decode_and_resize_image(z, self._image_dims[:2])

        x = np.array(
            images[images.columns[0]].apply(preprocess_f).tolist())
        with self._graph.as_default():
            with self._session.as_default():
                return self._model.predict(x)


def log_model(keras_model, artifact_path, image_dims, domain):
    """
    Log a KerasImageClassifierPyfunc model as an MLflow artifact for the current run.

    :param keras_model: Keras model to be saved.
    :param artifact_path: Run-relative artifact path this model is to be saved to.
    :param image_dims: Image dimensions the Keras model expects.
    :param domain: Labels for the classes this model can predict.
    """

    with TempDir() as tmp:
        data_path = tmp.path("image_model")
        os.mkdir(data_path)
        conf = {
            "image_dims": "/".join(map(str, image_dims)),
            "domain": "/".join(map(str, domain))
        }
        with open(os.path.join(data_path, "conf.yaml"), "w") as f:
            yaml.safe_dump(conf, stream=f)
        keras_path = os.path.join(data_path, "keras_model")
        mlflow.keras.save_model(keras_model, path=keras_path)
        conda_env = tmp.path("conda_env.yaml")
        with open(conda_env, "w") as f:
            f.write(conda_env_template.format(python_version=PYTHON_VERSION,
                                              keras_version=keras.__version__,
                                              tf_name=tf.__name__,  # can have optional -gpu suffix
                                              tf_version=tf.__version__,
                                              pillow_version=PIL.__version__))

        mlflow.pyfunc.log_model(artifact_path=artifact_path,
                                loader_module=__name__,
                                code_path=[__file__],
                                data_path=data_path,
                                conda_env=conda_env)


def _load_pyfunc(path):
    """
    Load the KerasImageClassifierPyfunc model.
    """
    with open(os.path.join(path, "conf.yaml"), "r") as f:
        conf = yaml.safe_load(f)
    keras_model_path = os.path.join(path, "keras_model")
    domain = conf["domain"].split("/")
    image_dims = np.array([int(x) for x in conf["image_dims"].split("/")], dtype=np.int32)
    # NOTE: TensorFlow based models depend on global state (Graph and Session) given by the context.
    # To make sure we score the model in the same session as we loaded it in, we create a new
    # session and a new graph here and store them with the model.
    with tf.Graph().as_default() as g:
        with tf.Session().as_default() as sess:
            keras.backend.set_session(sess)
            keras_model = mlflow.keras.load_model(keras_model_path)
    return KerasImageClassifierPyfunc(g, sess, keras_model, image_dims, domain=domain)


conda_env_template = """        
name: flower_classifier
channels:
  - defaults
  - anaconda
dependencies:
  - python=={python_version}
  - keras=={keras_version}  
  - {tf_name}=={tf_version} 
  - pip:    
    - pillow=={pillow_version}
"""
