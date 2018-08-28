"""
Trains an Alternating Least Squares (ALS) model for user/movie ratings.
The input is a Parquet ratings dataset (see etl_data.py), and we output
an mlflow artifact called 'als-model'.
"""
import click

import mlflow
import mlflow.spark

import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


@click.command()
@click.option("--ratings-data")
@click.option("--split-prop", default=0.8, type=float)
@click.option("--max-iter", default=10, type=int)
@click.option("--reg-param", default=0.1, type=float)
@click.option("--rank", default=12, type=int)
@click.option("--cold-start-strategy", default="drop")
def train_als(ratings_data, split_prop, max_iter, reg_param, rank, cold_start_strategy):
    seed = 42

    spark = pyspark.sql.SparkSession.builder.getOrCreate()

    ratings_df = spark.read.parquet(ratings_data)
    (training_df, test_df) = ratings_df.randomSplit([split_prop, 1 - split_prop], seed=seed)
    training_df.cache()
    test_df.cache()

    mlflow.log_metric("training_nrows", training_df.count())
    mlflow.log_metric("test_nrows", test_df.count())

    print('Training: {0}, test: {1}'.format(training_df.count(), test_df.count()))

    als = (ALS()
           .setUserCol("userId")
           .setItemCol("movieId")
           .setRatingCol("rating")
           .setPredictionCol("predictions")
           .setMaxIter(max_iter)
           .setSeed(seed)
           .setRegParam(reg_param)
           .setColdStartStrategy(cold_start_strategy)
           .setRank(rank))

    als_model = Pipeline(stages=[als]).fit(training_df)

    reg_eval = RegressionEvaluator(predictionCol="predictions", labelCol="rating", metricName="mse")

    predicted_test_dF = als_model.transform(test_df)

    test_mse = reg_eval.evaluate(predicted_test_dF)
    train_mse = reg_eval.evaluate(als_model.transform(training_df))

    print('The model had a MSE on the test set of {0}'.format(test_mse))
    print('The model had a MSE on the (train) set of {0}'.format(train_mse))
    mlflow.log_metric("test_mse", test_mse)
    mlflow.log_metric("train_mse", train_mse)
    mlflow.spark.log_model(als_model, "als-model")


if __name__ == '__main__':
    train_als()
