from __future__ import print_function

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow.sklearn


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10



def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else None
    learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else None
    drop_out_1 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    drop_out_2 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
    warnings.filterwarnings("ignore")
    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wine-quality.csv")
    data = pd.read_csv(wine_path)
    import keras
    keras.metrics
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    train, valid = train_test_split(train)

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
        # TODO: not clear if I should log params here, depends on how this was launched
        # mlflow.log_param("epochs", epochs)
        # mlflow.log_param("learning_rate", learning_rate)
        # mlflow.log_param("drop_out_1", drop_out_1)
        # mlflow.log_param("drop_out_2", drop_out_2)

        model = Sequential()
        model.add(Dense(train_x.shape[1], activation='relu', input_shape=(train_x.shape[1],)))
        model.add(Dropout(drop_out_1))
        model.add(Dense(4, activation='relu'))
        model.add(Dropout(0))
        model.add(Dense(1, activation='linear'))
        model.summary()
        model.compile(loss='mean_absolute_error',
                      optimizer=RMSprop(lr=learning_rate) if learning_rate else RMSprop(),
                      metrics=['mse'])

        history = model.fit(train_x, train_y,
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
