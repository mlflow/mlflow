from abc import abstractmethod
import base64
from io import BytesIO
import math
import numpy as np
import os
import pandas as pd
import yaml

from PIL import Image
from sklearn.model_selection import train_test_split

import keras
from keras.utils import np_utils
import tensorflow as tf
from keras.models import Model
from keras.callbacks import Callback
from keras.applications import *
from keras.applications import vgg16
from keras.layers import Input, Dense, Flatten, Lambda

import mlflow
import mlflow.pyfunc
import mlflow.keras
from mlflow.utils.file_utils import TempDir
from mlflow.utils.environment import _mlflow_conda_env


def read_image(filename):
    with open(filename, "rb") as f:
        return f.read()


def decode_and_resize_image(raw_bytes, size):
    return np.asarray(Image.open(BytesIO(raw_bytes)).resize(size), dtype=np.float32)


class MLflowLogger(Callback):
    """
    Logs training metrics and final model with MLflow.

    We log metrics provided by Keras during training and keep track of the best model (best loss
    on validation dataset). At the end of the training, we log the model with weights that produced
    the best result on the validation dataset during training.

    The output model accepts raw images on the output and produces class label on the output when
    scored. It is a composition of the actual DL model with following pre and post processing:
       - preprocessing: decode and resize the image.
       - post-processing: select output class id
    """

    def __init__(self, model, x_train, y_train, x_valid, y_valid,
                 **kwargs):
        self._model = model
        self._best_val_loss = math.inf
        self._train = (x_train, y_train)
        self._valid = (x_valid, y_valid)
        self._pyfunc_params = kwargs
        self._best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        """
        Log Keras metrics with MLflow. If model improved on the validation data, evaluate it on
        a test set and store it as the best model.
        """
        if not logs:
            return
        for name, value in logs.items():
            if name.startswith("val_"):
                name = "valid_" + name[4:]
            else:
                name = "train_" + name
            mlflow.log_metric(name, value)
        val_loss = logs["val_loss"]
        if val_loss < self._best_val_loss:
            # Save the "best" weights
            self._best_val_loss = val_loss
            self._best_weights = [x.copy() for x in self._model.get_weights()]

    def on_train_end(self, logs=None):
        self._model.set_weights(self._best_weights)
        x, y = self._train
        train_res = self._model.evaluate(x=x, y=y)
        for name, value in zip(self._model.metrics_names, train_res):
            mlflow.log_metric("train_{}".format(name), value)
        x, y = self._valid
        valid_res = self._model.evaluate(x=x, y=y)
        for name, value in zip(self._model.metrics_names, valid_res):
            mlflow.log_metric("valid_{}".format(name), value)
        KerasImageClassifierPyfunc.log_model(keras_model=self._model, **self._pyfunc_params)


class KerasImageClassifier(object):
    """
        Base class for image classification with MLflow and Keras.

        Provides implementation of the image classification workflow. The images are read, decoded
        and resized on the fly and the same code is packaged with the resulting model.

        You can create your own KerasImageClassifier by extending this class and overriding the
        abstract methods.

    """

    def __init__(self, image_dims):
        self._input_shape = image_dims

    @abstractmethod
    def params(self):
        """
        Override this method to provide any additional parameters belonging to your model.
        :return: dictionary of additional parameters to be logged with MLflow.
        """
        return {}

    @abstractmethod
    def create_model(self, classes):
        """
        Create the Keras model to be trained.

        The model is expected to take decoded image (shape=self._input_shape) on the input and to
        produce vector of class probabilities on the output.

        :param classes: Number of output classes.
        :return: Keras model; input is decoded image, output is vector of class probabilities.
        """
        pass

    def train(self,
              image_files,
              labels,
              domain,
              epochs=1,
              batch_size=16,
              test_ratio=0.2,
              custom_preprocessor=None,
              optimizer=lambda: keras.optimizers.SGD(decay=1e-5, nesterov=True, momentum=.9),
              seed=None):
        """
        Train the model on provided image files. The images are expected to be raw and will be read,
        decoded and resized prior to model training.

        The train will create a new MLflow run, log all parameters and train the model. The model
        training and resulting models are logged with MlflowLogger.


        :param image_files: List of image files to be used for training.
        :param labels: List of labels for the image files.
        :param domain: Dictionary representing the domain of the reponse.
                       Provides mapping label-name -> label-id.
        :param epochs: Number of epochs to train the model for.
        :param batch_size: Batch size used during training.
        :param test_ratio: Fraction of dataset to be used for validation. This data will not be used
                           during training.
        :param custom_preprocessor: Custom preprocessor to use for reading and resizing images.
        :param optimizer: Keras optimizer to be used to train the model.
        :param seed: Random seed. Used e.g. when spliting the dataset into train / validation.
        """
        assert len(set(labels)) == len(domain)
        with mlflow.start_run() as run:
            for param, value in self.params().items():
                mlflow.log_param(str(param), str(value))

            mlflow.log_param("epochs", str(epochs))
            mlflow.log_param("batch_size", str(batch_size))
            mlflow.log_param("validation_ratio", str(test_ratio))
            mlflow.log_param("optimizer", optimizer.__name__)
            if seed:
                mlflow.log_param("seed", str(seed))

            with tf.Graph().as_default() as graph:
                import keras.backend as K
                K.set_session(tf.Session(graph=graph))
                dims = self._input_shape[:2]
                x = np.array([decode_and_resize_image(read_image(x), dims) for x in image_files])
                y = np_utils.to_categorical(np.array(labels), num_classes=len(domain))
                x_train, x_valid, y_train, y_valid = train_test_split(x, y, random_state=seed,
                                                                      train_size=(1 - test_ratio))
                model = self.create_model(classes=len(domain))
                model.compile(
                    optimizer=optimizer(),
                    loss=tf.keras.losses.categorical_crossentropy,
                    metrics=["accuracy"])
                model.fit(
                    x=x_train,
                    y=y_train,
                    validation_data=(x_valid, y_valid),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[MLflowLogger(model=model,
                                            x_train=x_train,
                                            y_train=y_train,
                                            x_valid=x_valid,
                                            y_valid=y_valid,
                                            artifact_path="model",
                                            domain=sorted(domain.keys(), key=lambda x: domain[x]),
                                            image_dims=self._input_shape)])


class KerasImageClassifierPyfunc(object):
    """
    PyFunc for Keras Image Classifier.

    Input is expected to be raw base64 encoded images. The images are decoded and resized prior to
    passing them to the model.

    The prediction output is the predicted class id.
    """

    def __init__(self, model, image_dims, domain):
        self._model = model
        self._image_dims = image_dims
        self._domain = domain
        probs_names = ["p({})".format(x) for x in domain]
        self._column_names = ["predicted_label_id", "predicted_label"] + probs_names

    def predict_images(self, images):
        def preprocess_f(z):
            return decode_and_resize_image(z, self._image_dims[:2])

        x = np.array(
            images[images.columns[0]].apply(preprocess_f).tolist())
        return self._model.predict(x)

    def predict(self, data):
        # decode image bytes from base64 encoding
        def decode_img(x):
            return pd.Series(base64.decodebytes(bytearray(x[0], encoding="utf8")))

        images = data.apply(axis=1, func=decode_img)
        probs = self.predict_images(images)
        m, n = probs.shape
        label_idx = np.argmax(probs, axis=1)
        labels = np.array([self._domain[i] for i in label_idx], dtype=np.str).reshape(m, 1)
        data = np.concatenate((label_idx.reshape(m, 1), labels, probs), axis=1)
        return pd.DataFrame(columns=self._column_names, data=data)

    @staticmethod
    def save_model(path, mlflow_model, keras_model, image_dims, domain):
        with TempDir() as tmp:
            conf = {
                "image_dims": "/".join(map(str, image_dims)),
                "domain": "/".join(map(str, domain))
            }
            with open(tmp.path("conf.yaml"), "w") as f:
                yaml.safe_dump(conf, stream=f)

            keras_path = tmp.path("keras_model")
            mlflow.keras.save_model(keras_model, keras_path)
            conda_env = tmp.path("conda_env.yaml")
            conda_env = _mlflow_conda_env(
                path=conda_env,
                additional_conda_deps=[
                    "keras={}".format(keras.__version__),
                    "{tf}=={version}".format(tf=tf.__name__, version=tf.__version__)
                ],
                additional_pip_deps=["pillow"],
                additional_conda_channels=None,
            )
            mlflow.pyfunc.save_model(path,
                                     __name__,
                                     code_path=[__file__],
                                     data_path=tmp.path(""),
                                     model=mlflow_model,
                                     conda_env=conda_env)

    @classmethod
    def log_model(cls, keras_model, artifact_path, image_dims, domain):
        from mlflow.models import Model
        Model.log(artifact_path=artifact_path,
                  flavor=cls,
                  keras_model=keras_model,
                  image_dims=image_dims,
                  domain=domain)


def _load_pyfunc(path):
    with open(os.path.join(path, "conf.yaml"), "r") as f:
        conf = yaml.safe_load(f)
    pyfunc_model = mlflow.pyfunc.load_pyfunc(os.path.join(path, "keras_model"))
    domain = conf["domain"].split("/")
    image_dims = np.array([int(x) for x in conf["image_dims"].split("/")], dtype=np.int32)
    return KerasImageClassifierPyfunc(pyfunc_model, image_dims, domain=domain)


class MLflowInceptionV3(KerasImageClassifier):
    """
    Image classifier based on Keras InceptionV3 model.

    Can be trained from scratch or use pre-trained model with weights trained on imagenet dataset.
    When using pretrained weights, the last softmax layer of the model is dropped and the
    model will train a new one. Note that  in this mode, the pre-initialized layers are fixed and
    only the last output layer is trained.
    """

    def __init__(self, weights='imagenet'):
        super(MLflowInceptionV3, self).__init__(image_dims=(299, 299, 3))
        self.weights = weights

    def params(self):
        return {"weights": self.weights}

    def create_model(self, classes=None):
        include_top = (classes == 1000)
        image = Input(shape=self._input_shape)
        lambda_layer = Lambda(lambda x: (x / 127.5) - 1)
        preprocessed_image = lambda_layer(image)
        preprocessed_image.trainable = False
        model = inception_v3.InceptionV3(classes=classes,
                                         weights=self.weights,
                                         include_top=include_top,
                                         input_tensor=preprocessed_image,
                                         pooling='avg')
        if not include_top:
            if self._weights == 'imagenet':
                for l in model.layers:
                    l.trainable = False
            model = Model(inputs=model.input,
                          outputs=Dense(classes,
                                        activation='softmax',
                                        name='predictions')(model.output))
        return model


def _imagenet_preprocess_tf(x):
    return (x / 127.5) - 1


class MLflow_VGG16(KerasImageClassifier):
    """
    Image classifier based on Keras VGG16 model.

    Can be trained from scratch or use pre-trained model with weights trained on imagenet dataset.
    When using pretrained weights, the fully connected layers are dropped and replaced with new
    ones. Note that in this mode, the pre-initialized layers are fixed and only the fully connected
    layers are trained.
    """

    def __init__(self, weights="imagenet"):
        super(MLflow_VGG16, self).__init__(image_dims=(224, 224, 3))
        self._weights = weights

    def params(self):
        return {"weights": self._weights}

    def create_model(self, classes):
        include_top = (classes == 1000)
        image = Input(shape=self._input_shape)
        lambda_layer = Lambda(_imagenet_preprocess_tf)
        preprocessed_image = lambda_layer(image)
        preprocessed_image.trainable = False
        model = vgg16.VGG16(classes=classes,
                            input_tensor=preprocessed_image,
                            weights=self._weights, include_top=include_top)
        if not include_top:
            if self._weights == 'imagenet':
                for l in model.layers:
                    l.trainable = False
            x = Flatten(name='flatten')(model.output)
            x = Dense(4096, activation='relu', name='fc1')(x)
            x = Dense(4096, activation='relu', name='fc2')(x)
            x = Dense(classes, activation='softmax', name='predictions')(x)
            model = Model(inputs=model.input, outputs=x)
        return model
