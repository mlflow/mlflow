from __future__ import print_function

import warnings

import math
import numpy as np
import pandas as pd

import click
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Lambda
from keras.optimizers import Adamax

import mlflow
import mlflow.keras



def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def get_standardize_f(train):
    mu = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    return lambda x: (x - mu) / std


@click.group()
@click.version_option()
def cli():
    pass


from keras.callbacks import Callback


class MLflowLogger(Callback):
    def __init__(self, score_and_save_period=100):
        super(MLflowLogger, self).__init__()
        self._score_f = None

    def on_epoch_end(self, epoch, logs=None):
        # log standard keras metrics
        if logs:
            for k, v in logs.items():
                mlflow.log_metric(k, v)


class MLflowCheckpoint(Callback):
    def __init__(self, test_x, test_y):
        # self._epoch_since_last_save = 0
        # self._checkpoint_period = checkpoint_period
        self._test_x = test_x
        self._test_y = test_y
        self._best = math.inf

    def on_epoch_end(self, epoch, logs=None):
        if not logs:
            return
        v = logs["val_loss"]
        if v < self._best:
            self._best = v
            preds = self.model.predict((self._test_x))
            rmse, mae, r2 = eval_metrics(self._test_y, preds)
            print("  RMSE: %s" % rmse)
            print("  MAE: %s" % mae)
            print("  R2: %s" % r2)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.keras.log_model(self.model, "model_checkpoint_{}".format(epoch + 1))
            mlflow.keras.log_model(self.model, "model")


@cli.command()
@click.option("--epochs", type=click.INT, default=100, help="Maximum number of epochs to evaluate.")
@click.option("--batch-size", type=click.INT, default=16,
              help="Batch size passed to the learning algo.")
@click.option("--learning-rate", type=click.FLOAT, default=1e-2, help="Learning rate.")
@click.option("--beta1", type=click.FLOAT, default=.9, help="Learning rate decay.")
@click.option("--beta2", type=click.FLOAT, default=.9, help="Gradient moving average decay.")
@click.option("--seed", type=click.INT, default=97531, help="Seed for the random generator.")
@click.argument("training_data")
def run(training_data, epochs, batch_size, learning_rate, beta1, beta2, seed):
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

    from keras import metrics
    metrics.mae
    with mlflow.start_run():
        rmse, mae, r2 = eval_metrics(test_y, np.ones(len(test_y)) * np.mean(test_y))
        mlflow.log_metric("rmse_null", rmse)
        mlflow.log_metric("mae_null", mae)
        mlflow.log_metric("r2_null", r2)
        print("  RMSE (null): %s" % rmse)
        print("  MAE (null) : %s" % mae)
        print("  R2 (null): %s" % r2)
        model = Sequential()
        model.add(Lambda(get_standardize_f(train_x)))
        model.add(Dense(train_x.shape[1],
                        activation='tanh',
                        kernel_initializer='normal',
                        input_shape=(train_x.shape[1],)))
        model.add(Dense(128,
                        activation='tanh',
                        kernel_initializer='normal',
                        input_shape=(train_x.shape[1],)))
        model.add(Dense(64,
                        activation='relu',
                        kernel_initializer='normal',
                        input_shape=(train_x.shape[1],)))
        model.add(Dense(32,
                        activation='relu',
                        kernel_initializer='normal',
                        input_shape=(train_x.shape[1],)))
        model.add(Dense(1,
                        kernel_initializer='normal',
                        activation='linear'))
        model.compile(loss='mean_squared_error',
                      optimizer=Adamax(
                          lr=learning_rate,
                          beta_1=beta1,
                          beta_2=beta2
                      ),
                      metrics=["mae"])
        model.fit(train_x, train_y,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(valid_x, valid_y),
                  callbacks=[MLflowLogger(), MLflowCheckpoint(test_x, test_y)])


if __name__ == '__main__':
    run()
