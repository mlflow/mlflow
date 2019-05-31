"""
Trains a Keras model for user/movie ratings. The input is a Parquet
ratings dataset (see etl_data.py) and an ALS model (see als.py), which we
will use to supplement our input and train using.
"""
import click

import mlflow
import mlflow.keras
import mlflow.spark

from itertools import chain
import pyspark
from pyspark.sql.functions import *
from pyspark.sql.types import *

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import pandas as pd


@click.command()
@click.option("--ratings-data", help="Path readable by Spark to the ratings Parquet file")
@click.option("--als-model-uri", help="Path readable by load_model to ALS MLmodel")
@click.option("--hidden-units", default=20, type=int)
def train_keras(ratings_data, als_model_uri, hidden_units):
    np.random.seed(0)
    tf.set_random_seed(42)  # For reproducibility

    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    als_model = mlflow.spark.load_model(als_model_uri).stages[0]

    ratings_df = spark.read.parquet(ratings_data)

    (training_df, test_df) = ratings_df.randomSplit([0.8, 0.2], seed=42)
    training_df.cache()
    test_df.cache()

    mlflow.log_metric("training_nrows", training_df.count())
    mlflow.log_metric("test_nrows", test_df.count())

    print('Training: {0}, test: {1}'.format(training_df.count(), test_df.count()))

    user_factors = als_model.userFactors.selectExpr("id as userId", "features as uFeatures")
    item_factors = als_model.itemFactors.selectExpr("id as movieId", "features as iFeatures")
    joined_train_df = training_df.join(item_factors, on="movieId").join(user_factors, on="userId")
    joined_test_df = test_df.join(item_factors, on="movieId").join(user_factors, on="userId")

    # We'll combine the movies and ratings vectors into a single vector of length 24.
    # We will then explode this features vector into a set of columns.
    def concat_arrays(*args):
        return list(chain(*args))

    concat_arrays_udf = udf(concat_arrays, ArrayType(FloatType()))

    concat_train_df = (joined_train_df
                       .select('userId', 'movieId', concat_arrays_udf(col("iFeatures"), col("uFeatures")).alias("features"),
                               col('rating').cast("float")))
    concat_test_df = (joined_test_df
                      .select('userId', 'movieId', concat_arrays_udf(col("iFeatures"), col("uFeatures")).alias("features"),
                              col('rating').cast("float")))

    pandas_df = concat_train_df.toPandas()
    pandas_test_df = concat_test_df.toPandas()

    # This syntax will create a new DataFrame where elements of the 'features' vector
    # are each in their own column. This is what we'll train our neural network on.
    x_test = pd.DataFrame(pandas_test_df.features.values.tolist(), index=pandas_test_df.index)
    x_train = pd.DataFrame(pandas_df.features.values.tolist(), index=pandas_df.index)

    # Show matrix for example.
    print("Training matrix:")
    print(x_train)

    # Create our Keras model with two fully connected hidden layers.
    model = Sequential()
    model.add(Dense(30, input_dim=24, activation='relu'))
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=.0001))

    filepath = '/tmp/ALS_checkpoint_weights.hdf5'
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, mode='auto')

    model.fit(x_train, pandas_df["rating"], validation_split=.2, verbose=2, epochs=3,
              batch_size=128, shuffle=False, callbacks=[early_stopping])

    train_mse = model.evaluate(x_train, pandas_df["rating"], verbose=2)
    test_mse = model.evaluate(x_test, pandas_test_df["rating"], verbose=2)
    mlflow.log_metric("test_mse", test_mse)
    mlflow.log_metric("train_mse", train_mse)

    print('The model had a MSE on the test set of {0}'.format(test_mse))
    mlflow.keras.log_model(model, "keras-model")


if __name__ == '__main__':
    train_keras()
