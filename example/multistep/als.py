import click

import mlflow
import mlflow.spark

import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


@click.group()
def cli():
    pass


@cli.command(help="Trains an ALS model for user/movie ratings. The input is a Parquet "
                  "ratings dataset (see etl_data.py), and we output an mlflow artifact "
                  "called 'als-model'")
@click.option("--ratings-data")
@click.option("--split-prop", default=0.8, type=float)
@click.option("--max-iter", default=10, type=int)
@click.option("--reg-param", default=0.1, type=float)
@click.option("--rank", default=12, type=int)
@click.option("--cold-start-strategy", default="drop")
def train_als(ratings_data, split_prop, max_iter, reg_param, rank, cold_start_strategy):
    seed = 42

    spark = (pyspark.sql.SparkSession.builder \
             .config("spark.driver.memory", "2g") \
             .getOrCreate())

    ratingsDF = spark.read.parquet(ratings_data)
    (trainingDF, testDF) = ratingsDF.randomSplit([split_prop, 1 - split_prop], seed=seed)
    trainingDF.cache()
    testDF.cache()

    mlflow.log_metric("training_nrows", trainingDF.count())
    mlflow.log_metric("test_nrows", testDF.count())

    print('Training: {0}, test: {1}'.format(trainingDF.count(), testDF.count()))

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

    alsModel = Pipeline(stages=[als]).fit(trainingDF)

    regEval = RegressionEvaluator(predictionCol="predictions", labelCol="rating", metricName="mse")

    predictedTestDF = alsModel.transform(testDF)

    testMse = regEval.evaluate(predictedTestDF)
    trainMse = regEval.evaluate(alsModel.transform(trainingDF))

    print('The model had a MSE on the test set of {0}'.format(testMse))
    print('The model had a MSE on the (train) set of {0}'.format(trainMse))
    mlflow.log_metric("test_mse", testMse)
    mlflow.log_metric("train_mse", trainMse)
    mlflow.spark.log_model(alsModel, "als-model")


if __name__ == '__main__':
    cli()
