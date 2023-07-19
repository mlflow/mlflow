import os
import mlflow
import pytest
import sklearn
import pandas as pd
import numpy as np

"""
In this test, we check that the extra_tags parameter is passed to the autologging for 
mlflow.autolog(), mlflow.sklearn.autolog(), mlflow.fastai.autolog(), and 
mlflow.pyspark.ml.autolog() correctly.

All the other autologging functions are tested in their corresponding test files.
"""

mlflow.autolog(silent=True, log_models=False, extra_tags={"test_tag": "autolog"})


def get_iris():
    iris = sklearn.datasets.load_iris()
    return iris.data[:, :2], iris.target


def get_run():
    return mlflow.search_runs(
        order_by=["start_time desc"],
        max_results=1,
        output_format="list",
    )[0]


@pytest.mark.parametrize("autolog_variant", ["all", "self"])
def test_extra_tags_sklearn_autolog(autolog_variant):
    if autolog_variant == "self":
        mlflow.sklearn.autolog(
            silent=True, log_models=False, extra_tags={"test_tag": "sklearn_autolog"}
        )
    model = sklearn.linear_model.LogisticRegression()
    X, y = get_iris()
    model.fit(X, y)

    run = get_run()
    if autolog_variant == "self":
        assert run.data.tags["test_tag"] == "sklearn_autolog"
    else:
        assert run.data.tags["test_tag"] == "autolog"


@pytest.mark.parametrize("autolog_variant", ["all", "self"])
def test_extra_tags_fastai_autolog(autolog_variant):
    from torch import nn
    from fastai.learner import Learner
    from fastai.tabular.all import TabularDataLoaders

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(2, 3)
            self.linear2 = nn.Linear(3, 1)

        def forward(self, _, x_cont):
            x = self.linear1(x_cont)
            return self.linear2(x)

    def splitter(model):
        """
        Splits model parameters into multiple groups to allow fine-tuning
        """
        params = list(model.parameters())
        return [
            # weights and biases of the first linear layer
            params[:2],
            # weights and biases of the second linear layer
            params[2:],
        ]

    def fastai_tabular_model(data, **kwargs):
        # Create a fine-tunable learner
        return Learner(data, Model(), loss_func=nn.MSELoss(), splitter=splitter, **kwargs)

    X, y = get_iris()
    X = pd.DataFrame(X, columns=["x1", "x2"])
    y = pd.Series(y, name="label", dtype=np.float32)
    iris_data = TabularDataLoaders.from_df(
        df=pd.concat([X, y], axis=1), cont_names=list(X.columns), y_names="label"
    )

    if autolog_variant == "self":
        mlflow.fastai.autolog(
            silent=True, log_models=False, extra_tags={"test_tag": "fastai_autolog"}
        )
    model = fastai_tabular_model(iris_data)
    model.fit_one_cycle(1)

    run = get_run()
    if autolog_variant == "self":
        assert run.data.tags["test_tag"] == "fastai_autolog"
    else:
        assert run.data.tags["test_tag"] == "autolog"


@pytest.mark.parametrize("autolog_variant", ["all", "self"])
def test_extra_tags_spark_autolog(autolog_variant):
    from pyspark.sql import SparkSession
    from pyspark.ml.regression import LinearRegression
    from pyspark.ml.linalg import Vectors

    if autolog_variant == "self":
        mlflow.pyspark.ml.autolog(extra_tags={"test_tag": "spark_autolog"})

    with SparkSession.builder.config(
        "spark.jars.packages", "org.mlflow:mlflow-spark:1.11.0"
    ).master("local[*]").getOrCreate() as spark_session:
        dataset_binomial = spark_session.createDataFrame(
            [(1.0, Vectors.dense(1.0)), (0.0, Vectors.sparse(1, [], []))] * 100,
            ["label", "features"],
        ).cache()
        lr = LinearRegression()
        lr.fit(dataset_binomial)
        assert mlflow.active_run() is None

        run = mlflow.search_runs(
            order_by=["start_time desc"],
            max_results=1,
            output_format="list",
        )[0]
        if autolog_variant == "self":
            assert run.data.tags["test_tag"] == "spark_autolog"
        else:
            assert run.data.tags["test_tag"] == "autolog"
