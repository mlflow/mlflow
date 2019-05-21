"""
Train a simple Keras DL model on the dataset used in MLflow tutorial (wine-quality.csv).

Dataset is split into train (~ 0.56), validation(~ 0.19) and test (0.25).
Validation data is used to select the best hyperparameters, test set performance is evaluated only
at epochs which improved performance on the validation dataset. The model with best validation set
performance is logged with MLflow.
"""
from __future__ import print_function

import warnings

import math

import keras
import numpy as np
import pandas as pd

import click

from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, Lambda
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.keras


def eval_and_log_metrics(prefix, actual, pred, epoch):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mlflow.log_metric("{}_rmse".format(prefix), rmse, step=epoch)
    return rmse


def get_standardize_f(train):
    mu = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    return lambda x: (x - mu) / std


class MLflowCheckpoint(Callback):
    """
    Example of Keras MLflow logger.
    Logs training metrics and final model with MLflow.

    We log metrics provided by Keras during training and keep track of the best model (best loss
    on validation dataset). Every improvement of the best model is also evaluated on the test set.

    At the end of the training, log the best model with MLflow.
    """

    def __init__(self, test_x, test_y, loss="rmse"):
        self._test_x = test_x
        self._test_y = test_y
        self.train_loss = "train_{}".format(loss)
        self.val_loss = "val_{}".format(loss)
        self.test_loss = "test_{}".format(loss)
        self._best_train_loss = math.inf
        self._best_val_loss = math.inf
        self._best_model = None
        self._next_step = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Log the best model at the end of the training run.
        """
        if not self._best_model:
            raise Exception("Failed to build any model")
        mlflow.log_metric(self.train_loss, self._best_train_loss, step=self._next_step)
        mlflow.log_metric(self.val_loss, self._best_val_loss, step=self._next_step)
        mlflow.keras.log_model(self._best_model, "model")

    def on_epoch_end(self, epoch, logs=None):
        """
        Log Keras metrics with MLflow. If model improved on the validation data, evaluate it on
        a test set and store it as the best model.
        """
        if not logs:
            return
        self._next_step = epoch + 1
        train_loss = logs["loss"]
        val_loss = logs["val_loss"]
        mlflow.log_metrics({
            self.train_loss: train_loss,
            self.val_loss: val_loss
        }, step=epoch)

        if val_loss < self._best_val_loss:
            # The result improved in the validation set.
            # Log the model with mlflow and also evaluate and log on test set.
            self._best_train_loss = train_loss
            self._best_val_loss = val_loss
            self._best_model = keras.models.clone_model(self.model)
            self._best_model.set_weights([x.copy() for x in self.model.get_weights()])
            preds = self._best_model.predict(self._test_x)
            eval_and_log_metrics("test", self._test_y, preds, epoch)


@click.command(help="Trains an Keras model on wine-quality dataset."
                    "The input is expected in csv format."
                    "The model and its metrics are logged with mlflow.")
@click.option("--epochs", type=click.INT, default=100, help="Maximum number of epochs to evaluate.")
@click.option("--batch-size", type=click.INT, default=16,
              help="Batch size passed to the learning algo.")
@click.option("--learning-rate", type=click.FLOAT, default=1e-2, help="Learning rate.")
@click.option("--momentum", type=click.FLOAT, default=.9, help="SGD momentum.")
@click.option("--seed", type=click.INT, default=97531, help="Seed for the random generator.")
@click.argument("training_data")
def run(training_data, epochs, batch_size, learning_rate, momentum, seed):
    warnings.filterwarnings("ignore")
    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    data = pd.read_csv(training_data)
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data, random_state=seed)
    train, valid = train_test_split(train, random_state=seed)
    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1).as_matrix()
    train_x = (train_x).astype('float32')
    train_y = train[["quality"]].as_matrix().astype('float32')
    valid_x = (valid.drop(["quality"], axis=1).as_matrix()).astype('float32')

    valid_y = valid[["quality"]].as_matrix().astype('float32')

    test_x = (test.drop(["quality"], axis=1).as_matrix()).astype("float32")
    test_y = test[["quality"]].as_matrix().astype("float32")

    with mlflow.start_run():
        if epochs == 0:  # score null model
            eval_and_log_metrics("train", train_y, np.ones(len(train_y)) * np.mean(train_y),
                                 epoch=-1)
            eval_and_log_metrics("val", valid_y, np.ones(len(valid_y)) * np.mean(valid_y), epoch=-1)
            eval_and_log_metrics("test", test_y, np.ones(len(test_y)) * np.mean(test_y), epoch=-1)
        else:
            with MLflowCheckpoint(test_x, test_y) as mlflow_logger:
                model = Sequential()
                model.add(Lambda(get_standardize_f(train_x)))
                model.add(Dense(train_x.shape[1],
                                activation='relu',
                                kernel_initializer='normal',
                                input_shape=(train_x.shape[1],)))
                model.add(Dense(16,
                                activation='relu',
                                kernel_initializer='normal'))
                model.add(Dense(16,
                                activation='relu',
                                kernel_initializer='normal'))
                model.add(Dense(1,
                                kernel_initializer='normal',
                                activation='linear'))
                model.compile(loss='mean_squared_error',
                              optimizer=SGD(
                                  lr=learning_rate,
                                  momentum=momentum
                              ),
                              metrics=[])

                model.fit(train_x, train_y,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=1,
                          validation_data=(valid_x, valid_y),
                          callbacks=[mlflow_logger])


if __name__ == '__main__':
    run()
