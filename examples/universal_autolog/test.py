import mlflow.spark
import os
import shutil

from pyspark.sql import SparkSession
# Create and persist some dummy data
spark = (SparkSession.builder
            .config("spark.jars.packages", "org.mlflow:mlflow-spark:1.11.0")
            .getOrCreate())

df = spark.createDataFrame([
        (4, "spark i j k"),
        (5, "l m n"),
        (6, "spark hadoop spark"),
        (7, "apache hadoop")], ["id", "text"])
import tempfile
tempdir = tempfile.mkdtemp()
df.write.format("csv").save(os.path.join(tempdir, "asdf"))
# Enable Spark datasource autologging.
mlflow.spark.autolog()
loaded_df = spark.read.format("csv").load(os.path.join(tempdir, "asdf"))
# Call collect() to trigger a read of the Spark datasource. Datasource info
# (path and format)is automatically logged to an MLflow run.
loaded_df.collect()
shutil.rmtree(tempdir) # clean up tempdir