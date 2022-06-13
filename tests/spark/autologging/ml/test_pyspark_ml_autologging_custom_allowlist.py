import os
import mlflow

from pyspark.sql import SparkSession


# Put this test in separate module because it require a spark context
# with a special conf and the conf is immutable in runtime.
def test_custom_log_model_allowlist(tmpdir):
    allowlist_file_path = os.path.join(tmpdir, "allowlist")
    with open(allowlist_file_path, "w") as f:
        f.write("pyspark.ml.regression.LinearRegressionModel\n")
        f.write("pyspark.ml.classification.NaiveBayesModel\n")

    spark_session = (
        SparkSession.builder.config(
            "spark.mlflow.pysparkml.autolog.logModelAllowlistFile", allowlist_file_path
        )
        .master("local[*]")
        .getOrCreate()
    )

    mlflow.pyspark.ml.autolog()
    assert mlflow.pyspark.ml._log_model_allowlist == {
        "pyspark.ml.regression.LinearRegressionModel",
        "pyspark.ml.classification.NaiveBayesModel",
    }

    spark_session.stop()


def test_log_model_allowlist_from_url():

    allowlist_file_path = "https://raw.githubusercontent.com/mlflow/mlflow/v1.26.0/mlflow/pyspark/ml/log_model_allowlist.txt"

    spark_session = (
        SparkSession.builder.config(
            "spark.mlflow.pysparkml.autolog.logModelAllowlistFile", allowlist_file_path
        )
        .master("local[*]")
        .getOrCreate()
    )

    mlflow.pyspark.ml.autolog()

    assert mlflow.pyspark.ml._log_model_allowlist == {
        "pyspark.ml.classification.LinearSVCModel",
        "pyspark.ml.classification.DecisionTreeClassificationModel",
        "pyspark.ml.classification.GBTClassificationModel",
        "pyspark.ml.classification.LogisticRegressionModel",
        "pyspark.ml.classification.RandomForestClassificationModel",
        "pyspark.ml.classification.NaiveBayesModel",
        "pyspark.ml.clustering.BisectingKMeansModel",
        "pyspark.ml.clustering.KMeansModel",
        "pyspark.ml.clustering.GaussianMixtureModel",
        "pyspark.ml.regression.AFTSurvivalRegressionModel",
        "pyspark.ml.regression.DecisionTreeRegressionModel",
        "pyspark.ml.regression.GBTRegressionModel",
        "pyspark.ml.regression.GeneralizedLinearRegressionModel",
        "pyspark.ml.regression.LinearRegressionModel",
        "pyspark.ml.regression.RandomForestRegressionModel",
        "pyspark.ml.feature.BucketedRandomProjectionLSHModel",
        "pyspark.ml.feature.ChiSqSelectorModel",
        "pyspark.ml.feature.CountVectorizerModel",
        "pyspark.ml.feature.IDFModel",
        "pyspark.ml.feature.ImputerModel",
        "pyspark.ml.feature.MaxAbsScalerModel",
        "pyspark.ml.feature.MinHashLSHModel",
        "pyspark.ml.feature.MinMaxScalerModel",
        "pyspark.ml.feature.OneHotEncoderModel",
        "pyspark.ml.feature.RobustScalerModel",
        "pyspark.ml.feature.RFormulaModel",
        "pyspark.ml.feature.StandardScalerModel",
        "pyspark.ml.feature.StringIndexerModel",
        "pyspark.ml.feature.VarianceThresholdSelectorModel",
        "pyspark.ml.feature.VectorIndexerModel",
        "pyspark.ml.feature.UnivariateFeatureSelectorModel",
        "pyspark.ml.classification.OneVsRestModel",
        "pyspark.ml.pipeline.PipelineModel",
        "pyspark.ml.tuning.CrossValidatorModel",
        "pyspark.ml.tuning.TrainValidationSplitModel",
    }

    spark_session.stop()
