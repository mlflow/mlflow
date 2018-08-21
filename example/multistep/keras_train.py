import click
import tempfile

import mlflow
import mlflow.spark
import time

from itertools import chain
import pyspark
from pyspark.sql.functions import *
from pyspark.sql.types import *

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.regularizers import l2
import numpy as np
import pandas as pd


@click.group()
def cli():
    pass

@cli.command(help="Trains a Keras model for user/movie ratings. The input is a Parquet "
                  "ratings dataset (see etl_data.py) and an ALS model (see als.py).")
@click.option("--ratings-data", help="Path readable by Spark to the ratings Parquet file")
@click.option("--als-model-uri", help="Path readable by load_model to ALS MLmodel")
def keras_train(ratings_data, als_model_uri):
    np.random.seed(0)
    tf.set_random_seed(42)  # For reproducibility

    alsModel = mlflow.spark.load_model(als_model_uri).stages[0]

    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    ratingsDF = spark.read.parquet(ratings_data)

    (trainingDF, testDF) = ratingsDF.randomSplit([0.8, 0.2], seed=42)
    trainingDF.cache()
    testDF.cache()

    mlflow.log_metric("training_nrows", trainingDF.count())
    mlflow.log_metric("test_nrows", testDF.count())

    print('Training: {0}, test: {1}'.format(trainingDF.count(), testDF.count()))

    userFactors = alsModel.userFactors.selectExpr("id as userId", "features as uFeatures")
    itemFactors = alsModel.itemFactors.selectExpr("id as movieId", "features as iFeatures")
    joinedTrainDF = trainingDF.join(itemFactors, on="movieId").join(userFactors, on="userId")
    joinedTestDF = testDF.join(itemFactors, on="movieId").join(userFactors, on="userId")

    def concat_arrays(*args):
        return list(chain(*args))

    concat_arrays_udf = udf(concat_arrays, ArrayType(FloatType()))

    concatTrainDF = (joinedTrainDF
                     .select('userId', 'movieId', concat_arrays_udf(col("iFeatures"), col("uFeatures")).alias("features"),
                             col('rating').cast("float")))
    concatTestDF = (joinedTestDF
                    .select('userId', 'movieId', concat_arrays_udf(col("iFeatures"), col("uFeatures")).alias("features"),
                            col('rating').cast("float")))

    pandasDF = concatTrainDF.toPandas()
    pandasTestDF = concatTestDF.toPandas()

    pandasTestDF[['f0','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13','f14','f15',
              'f16','f17','f18','f19','f20','f21','f22','f23']] = pd.DataFrame(pandasTestDF.features.values.tolist(), index= pandasTestDF.index)

    x_test = pandasTestDF.drop(["features", "userId", "movieId", "rating"], axis=1)

    pandasDF[['f0','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13','f14','f15',
              'f16','f17','f18','f19','f20','f21','f22','f23']] = pd.DataFrame(pandasDF.features.values.tolist(), index= pandasDF.index)

    x_train = pandasDF.drop(["features", "userId", "movieId", "rating"], axis=1)

    model = Sequential()
    model.add(Dense(30, input_dim=24, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=.0001))

    filepath = '/tmp/ALS_checkpoint_weights.hdf5'
    checkpointer = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)
    earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, mode='auto')

    history = model.fit(x_train, pandasDF["rating"], validation_split=.2, verbose=2, epochs=3, batch_size=128, shuffle=False, callbacks=[earlyStopping, checkpointer])

    trainMse = model.evaluate(x_train, pandasDF["rating"], verbose=2)
    testMse = model.evaluate(x_test, pandasTestDF["rating"], verbose=2)
    mlflow.log_metric("test_mse", testMse)
    mlflow.log_metric("train_mse", trainMse)

    print('The model had a MSE on the test set of {0}'.format(testMse))


if __name__ == '__main__':
    cli()
