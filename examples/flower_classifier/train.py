"""
Example of image classification with MLflow using Keras to classify flowers from photos. The data is
taken from ``http://download.tensorflow.org/example_images/flower_photos.tgz`` and may be
downloaded during running this project if it is missing.
"""
import math
import os

import click
import keras
from keras.utils import np_utils
from keras.models import Model
from keras.callbacks import Callback
from keras.applications import vgg16
from keras.layers import Input, Dense, Flatten, Lambda
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

import mlflow

from image_pyfunc import decode_and_resize_image, log_model, KerasImageClassifierPyfunc


def download_input():
    import requests

    url = "http://download.tensorflow.org/example_images/flower_photos.tgz"
    print("downloading '{}' into '{}'".format(url, os.path.abspath("flower_photos.tgz")))
    r = requests.get(url)
    with open("flower_photos.tgz", "wb") as f:
        f.write(r.content)
    import tarfile

    print("decompressing flower_photos.tgz to '{}'".format(os.path.abspath("flower_photos")))
    with tarfile.open("flower_photos.tgz") as tar:
        tar.extractall(path="./")


@click.command(
    help="Trains an Keras model on flower_photos dataset."
    "The input is expected as a directory tree with pictures for each category in a"
    " folder named by the category."
    "The model and its metrics are logged with mlflow."
)
@click.option("--epochs", type=click.INT, default=1, help="Maximum number of epochs to evaluate.")
@click.option(
    "--batch-size", type=click.INT, default=16, help="Batch size passed to the learning algo."
)
@click.option("--image-width", type=click.INT, default=224, help="Input image width in pixels.")
@click.option("--image-height", type=click.INT, default=224, help="Input image height in pixels.")
@click.option("--seed", type=click.INT, default=97531, help="Seed for the random generator.")
@click.option("--training-data", type=click.STRING, default="./flower_photos")
@click.option("--test-ratio", type=click.FLOAT, default=0.2)
def run(training_data, test_ratio, epochs, batch_size, image_width, image_height, seed):
    image_files = []
    labels = []
    domain = {}
    print("Training model with the following parameters:")
    for param, value in locals().items():
        print("  ", param, "=", value)

    if training_data == "./flower_photos" and not os.path.exists(training_data):
        print("Input data not found, attempting to download the data from the web.")
        download_input()

    for (dirname, _, files) in os.walk(training_data):
        for filename in files:
            if filename.endswith("jpg"):
                image_files.append(os.path.join(dirname, filename))
                clazz = os.path.basename(dirname)
                if clazz not in domain:
                    domain[clazz] = len(domain)
                labels.append(domain[clazz])

    train(
        image_files,
        labels,
        domain,
        epochs=epochs,
        test_ratio=test_ratio,
        batch_size=batch_size,
        image_width=image_width,
        image_height=image_height,
        seed=seed,
    )


class MLflowLogger(Callback):
    """
    Keras callback for logging metrics and final model with MLflow.

    Metrics are logged after every epoch. The logger keeps track of the best model based on the
    validation metric. At the end of the training, the best model is logged with MLflow.
    """

    def __init__(self, model, x_train, y_train, x_valid, y_valid, **kwargs):
        self._model = model
        self._best_val_loss = math.inf
        self._train = (x_train, y_train)
        self._valid = (x_valid, y_valid)
        self._pyfunc_params = kwargs
        self._best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        """
        Log Keras metrics with MLflow. Update the best model if the model improved on the validation
        data.
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

    def on_train_end(self, *args, **kwargs):
        """
        Log the best model with MLflow and evaluate it on the train and validation data so that the
        metrics stored with MLflow reflect the logged model.
        """
        self._model.set_weights(self._best_weights)
        x, y = self._train
        train_res = self._model.evaluate(x=x, y=y)
        for name, value in zip(self._model.metrics_names, train_res):
            mlflow.log_metric("train_{}".format(name), value)
        x, y = self._valid
        valid_res = self._model.evaluate(x=x, y=y)
        for name, value in zip(self._model.metrics_names, valid_res):
            mlflow.log_metric("valid_{}".format(name), value)
        log_model(keras_model=self._model, **self._pyfunc_params)


def _imagenet_preprocess_tf(x):
    return (x / 127.5) - 1


def _create_model(input_shape, classes):
    image = Input(input_shape)
    lambda_layer = Lambda(_imagenet_preprocess_tf)
    preprocessed_image = lambda_layer(image)
    model = vgg16.VGG16(
        classes=classes, input_tensor=preprocessed_image, weights=None, include_top=False
    )

    x = Flatten(name="flatten")(model.output)
    x = Dense(4096, activation="relu", name="fc1")(x)
    x = Dense(4096, activation="relu", name="fc2")(x)
    x = Dense(classes, activation="softmax", name="predictions")(x)
    return Model(inputs=model.input, outputs=x)


def train(
    image_files,
    labels,
    domain,
    image_width=224,
    image_height=224,
    epochs=1,
    batch_size=16,
    test_ratio=0.2,
    seed=None,
):
    """
    Train VGG16 model on provided image files. This will create a new MLflow run and log all
    parameters, metrics and the resulting model with MLflow. The resulting model is an instance
    of KerasImageClassifierPyfunc - a custom python function model that embeds all necessary
    preprocessing together with the VGG16 Keras model. The resulting model can be applied
    directly to image base64 encoded image data.

    :param image_height: Height of the input image in pixels.
    :param image_width: Width of the input image in pixels.
    :param image_files: List of image files to be used for training.
    :param labels: List of labels for the image files.
    :param domain: Dictionary representing the domain of the response.
                   Provides mapping label-name -> label-id.
    :param epochs: Number of epochs to train the model for.
    :param batch_size: Batch size used during training.
    :param test_ratio: Fraction of dataset to be used for validation. This data will not be used
                       during training.
    :param seed: Random seed. Used e.g. when splitting the dataset into train / validation.
    """
    assert len(set(labels)) == len(domain)

    input_shape = (image_width, image_height, 3)

    with mlflow.start_run() as run:
        mlflow.log_param("epochs", str(epochs))
        mlflow.log_param("batch_size", str(batch_size))
        mlflow.log_param("validation_ratio", str(test_ratio))
        if seed:
            mlflow.log_param("seed", str(seed))

        def _read_image(filename):
            with open(filename, "rb") as f:
                return f.read()

        with tf.Graph().as_default() as g:
            with tf.compat.v1.Session(graph=g).as_default():
                dims = input_shape[:2]
                x = np.array([decode_and_resize_image(_read_image(x), dims) for x in image_files])
                y = np_utils.to_categorical(np.array(labels), num_classes=len(domain))
                train_size = 1 - test_ratio
                x_train, x_valid, y_train, y_valid = train_test_split(
                    x, y, random_state=seed, train_size=train_size
                )
                model = _create_model(input_shape=input_shape, classes=len(domain))
                model.compile(
                    optimizer=keras.optimizers.SGD(decay=1e-5, nesterov=True, momentum=0.9),
                    loss=keras.losses.categorical_crossentropy,
                    metrics=["accuracy"],
                )
                sorted_domain = sorted(domain.keys(), key=lambda x: domain[x])
                model.fit(
                    x=x_train,
                    y=y_train,
                    validation_data=(x_valid, y_valid),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[
                        MLflowLogger(
                            model=model,
                            x_train=x_train,
                            y_train=y_train,
                            x_valid=x_valid,
                            y_valid=y_valid,
                            artifact_path="model",
                            domain=sorted_domain,
                            image_dims=input_shape,
                        )
                    ],
                )


if __name__ == "__main__":
    run()
