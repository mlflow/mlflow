from __future__ import print_function

import warnings

import click
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow.keras

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

import mlflow


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


@click.group()
@click.version_option()
def cli():
    pass


@cli.command()
@click.option("--epochs", type=click.INT, default=100, help="Maximum number of epochs to evaluate.")
@click.option("--batch-size", type=click.INT, default=16,
              help="Batch size passed to the learning algo.")
@click.option("--learning-rate", type=click.FLOAT, default=1e-8, help="Learning rate")
@click.option("--dropout", type=click.FLOAT, default=0.0, help="Dropout applied across all layers")
@click.option("--seed", type=click.INT, default=97531, help="Seed for the random generator")
@click.argument("training_data")
def run(training_data, epochs, batch_size, learning_rate, dropout, seed):
    warnings.filterwarnings("ignore")
    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    data = pd.read_csv(training_data)
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data, random_state=seed)
    train, valid = train_test_split(train, random_state=seed)
    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1).as_matrix().astype('float32')
    train_y = train[["quality"]].as_matrix().astype('float32')
    valid_x = valid.drop(["quality"], axis=1).as_matrix().astype('float32')
    train_x /= 255.0
    valid_x /= 255.0
    valid_y = valid[["quality"]].as_matrix().astype('float32')

    test_x = test.drop(["quality"], axis=1)
    test_y = test[["quality"]]

    print(train_x.shape[0], 'train samples')
    print(test_x.shape[0], 'test samples')
    with mlflow.start_run():
        model = Sequential()
        model.add(Dense(train_x.shape[1], activation='relu', input_shape=(train_x.shape[1],)))
        model.add(Dropout(dropout))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(1, activation='linear'))
        model.summary()
        model.compile(loss='mean_absolute_error',
                      optimizer=RMSprop(lr=learning_rate) if learning_rate else RMSprop(),
                      metrics=['mse'])

        model.fit(train_x, train_y,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(valid_x, valid_y))
        score = model.evaluate(valid_x, valid_y, verbose=0)
        print('Valid loss:', score[0])
        print('Valid rmse:', score[1])
        preds = model.predict((test_x.as_matrix() / 255).astype('float32'))
        rmse, mae, r2 = eval_metrics(test_y, preds)
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.keras.log_model(model, "model")


if __name__ == '__main__':
    run()
