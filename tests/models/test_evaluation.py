import mlflow

from mlflow.models.evaluation import (
    evaluate,
    EvaluationDataset,
    EvaluationResult,
    ModelEvaluator,
    EvaluationArtifact,
    EvaluationMetrics,
)
import hashlib
from mlflow.models.evaluation.base import _start_run_or_reuse_active_run
import sklearn
import os
import sklearn.datasets
import sklearn.linear_model
import pytest
import numpy as np
import pandas as pd
from unittest import mock
from mlflow.utils.file_utils import TempDir
from mlflow_test_plugin.dummy_evaluator import Array2DEvaluationArtifact
from mlflow.models.evaluation.evaluator_registry import _model_evaluation_registry

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)

from pyspark.sql import SparkSession

from mlflow.tracking.artifact_utils import get_artifact_uri
import json


def get_iris():
    iris = sklearn.datasets.load_iris()
    return iris.data[:, :2], iris.target


def get_diabetes_dataset():
    data = sklearn.datasets.load_diabetes()
    return data.data[:, :2], data.target


def get_run_data(run_id):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items()}
    artifacts = [f.path for f in client.list_artifacts(run_id)]
    return data.params, data.metrics, tags, artifacts


def get_local_artifact_path(run_id, artifact_path):
    return get_artifact_uri(run_id, artifact_path).replace("file://", "")


@pytest.fixture(scope="module")
def spark_session():
    session = SparkSession.builder.master("local[*]").getOrCreate()
    yield session
    session.stop()


@pytest.fixture(scope="module")
def regressor_model_uri():
    X, y = get_diabetes_dataset()
    reg = sklearn.linear_model.LinearRegression()
    reg.fit(X, y)

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(reg, "reg_model")
        regressor_model_uri = get_artifact_uri(run.info.run_id, "reg_model")

    return regressor_model_uri


@pytest.fixture(scope="module")
def classifier_model_uri():
    X, y = get_iris()
    clf = sklearn.linear_model.LogisticRegression()
    clf.fit(X, y)

    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(clf, "clf_model")
        classifier_model_uri = get_artifact_uri(run.info.run_id, "clf_model")

    return classifier_model_uri


@pytest.fixture(scope="module")
def iris_dataset():
    X, y = get_iris()
    eval_X, eval_y = X[0::3], y[0::3]
    return EvaluationDataset(data=eval_X, labels=eval_y, name="iris_dataset")


@pytest.fixture(scope="module")
def iris_pandas_df_dataset():
    X, y = get_iris()
    eval_X, eval_y = X[0::3], y[0::3]
    data = pd.DataFrame({"f1": eval_X[:, 0], "f2": eval_X[:, 1], "y": eval_y})
    labels = "y"
    return EvaluationDataset(data=data, labels=labels, name="iris_pandas_df_dataset")


def test_classifier_evaluate(classifier_model_uri, iris_dataset):
    y_true = iris_dataset.labels
    classifier_model = mlflow.pyfunc.load_model(classifier_model_uri)
    y_pred = classifier_model.predict(iris_dataset.data)
    expected_accuracy_score = accuracy_score(y_true, y_pred)
    expected_metrics = {
        "accuracy_score": expected_accuracy_score,
    }
    expected_saved_metrics = {
        "accuracy_score_on_iris_dataset": expected_accuracy_score,
    }

    expected_artifact = confusion_matrix(y_true, y_pred)

    with mlflow.start_run() as run:
        eval_result = evaluate(
            classifier_model,
            "classifier",
            iris_dataset,
            run_id=None,
            evaluators="dummy_evaluator",
        )

    artifact_name = "confusion_matrix_on_iris_dataset.csv"
    saved_artifact_path = get_local_artifact_path(run.info.run_id, artifact_name)

    _, saved_metrics, _, saved_artifacts = get_run_data(run.info.run_id)
    assert saved_metrics == expected_saved_metrics
    assert saved_artifacts == [artifact_name]

    assert eval_result.metrics == expected_metrics
    confusion_matrix_artifact = eval_result.artifacts[artifact_name]
    assert np.array_equal(confusion_matrix_artifact.content, expected_artifact)
    assert confusion_matrix_artifact.uri == get_artifact_uri(run.info.run_id, artifact_name)
    assert np.array_equal(confusion_matrix_artifact.load(saved_artifact_path), expected_artifact)

    with TempDir() as temp_dir:
        temp_dir_path = temp_dir.path()
        eval_result.save(temp_dir_path)

        with open(temp_dir.path("metrics.json"), "r") as fp:
            assert json.load(fp) == eval_result.metrics

        with open(temp_dir.path("artifacts_metadata.json"), "r") as fp:
            assert json.load(fp) == {
                "confusion_matrix_on_iris_dataset.csv": {
                    "uri": confusion_matrix_artifact.uri,
                    "class_name": "mlflow_test_plugin.dummy_evaluator.Array2DEvaluationArtifact",
                }
            }

        assert os.listdir(temp_dir.path("artifacts")) == ["confusion_matrix_on_iris_dataset.csv"]

        loaded_eval_result = EvaluationResult.load(temp_dir_path)
        assert loaded_eval_result.metrics == eval_result.metrics
        loaded_confusion_matrix_artifact = loaded_eval_result.artifacts[artifact_name]
        assert confusion_matrix_artifact.uri == loaded_confusion_matrix_artifact.uri
        assert np.array_equal(
            confusion_matrix_artifact.content,
            loaded_confusion_matrix_artifact.content,
        )

        new_confusion_matrix_artifact = Array2DEvaluationArtifact(uri=confusion_matrix_artifact.uri)
        new_confusion_matrix_artifact.load()
        assert np.array_equal(
            confusion_matrix_artifact.content,
            new_confusion_matrix_artifact.content,
        )


def test_regressor_evaluate(regressor_model_uri, iris_dataset):
    y_true = iris_dataset.labels
    regressor_model = mlflow.pyfunc.load_model(regressor_model_uri)
    y_pred = regressor_model.predict(iris_dataset.data)
    expected_mae = mean_absolute_error(y_true, y_pred)
    expected_mse = mean_squared_error(y_true, y_pred)
    expected_metrics = {
        "mean_absolute_error": expected_mae,
        "mean_squared_error": expected_mse,
    }
    expected_saved_metrics = {
        "mean_absolute_error_on_iris_dataset": expected_mae,
        "mean_squared_error_on_iris_dataset": expected_mse,
    }

    for model in [regressor_model, regressor_model_uri]:
        with mlflow.start_run() as run:
            eval_result = evaluate(
                model,
                "regressor",
                iris_dataset,
                run_id=None,
                evaluators="dummy_evaluator",
            )
        _, saved_metrics, _, _ = get_run_data(run.info.run_id)
        assert saved_metrics == expected_saved_metrics
        assert eval_result.metrics == expected_metrics


def test_dataset_name():
    X, y = get_iris()
    d1 = EvaluationDataset(data=X, labels=y, name="a1")
    assert d1.name == "a1"
    d2 = EvaluationDataset(data=X, labels=y)
    assert d2.name == d2.hash


def test_gen_md5_for_arraylike_obj():
    def get_md5(data):
        md5_gen = hashlib.md5()
        EvaluationDataset._gen_md5_for_arraylike_obj(md5_gen, data)
        return md5_gen.hexdigest()

    list0 = list(range(20))
    list1 = [100] + list0[1:]
    list2 = list0[:-1] + [100]
    list3 = list0[:10] + [100] + list0[10:]

    assert 4 == len({get_md5(list0), get_md5(list1), get_md5(list2), get_md5(list3)})

    list4 = list0[:10] + [99] + list0[10:]
    assert get_md5(list3) == get_md5(list4)


def test_dataset_hash(iris_dataset, iris_pandas_df_dataset):
    assert iris_dataset.hash == "c7417e63a9ce038a32f37ecd7fb829f6"
    assert iris_pandas_df_dataset.hash == "d06cfb6352dba29afe514d9be87021aa"


def test_datasset_extract_features_label(iris_dataset, iris_pandas_df_dataset):
    X1, y1 = iris_dataset._extract_features_and_labels()
    assert np.array_equal(X1, iris_dataset.data)
    assert np.array_equal(y1, iris_dataset.labels)

    X2, y2 = iris_pandas_df_dataset._extract_features_and_labels()
    assert list(X2.columns) == ["f1", "f2"]
    assert np.array_equal(X2["f1"], X1[:, 0])
    assert np.array_equal(X2["f2"], X1[:, 1])
    assert np.array_equal(y2, y1)


def test_spark_df_dataset(spark_session):
    spark_df = spark_session.createDataFrame([(1.0, 2.0, 3.0)] * 10, ["f1", "f2", "y"])
    with mock.patch.object(EvaluationDataset, "SPARK_DATAFRAME_LIMIT", 5):
        dataset = EvaluationDataset(spark_df, "y")
        assert list(dataset.data.columns) == ["f1", "f2", "y"]
        assert list(dataset.data["f1"]) == [1.0] * 5
        assert list(dataset.data["f2"]) == [2.0] * 5
        assert list(dataset.data["y"]) == [3.0] * 5


def test_log_dataset_tag(iris_dataset, iris_pandas_df_dataset):
    with mlflow.start_run() as run:
        client = mlflow.tracking.MlflowClient()
        iris_dataset._log_dataset_tag(client, run.info.run_id)
        _, _, tags, _ = get_run_data(run.info.run_id)
        assert json.loads(tags["mlflow.datasets"]) == [iris_dataset._metadata]

        # Test appending dataset tag
        iris_pandas_df_dataset._log_dataset_tag(client, run.info.run_id)
        _, _, tags, _ = get_run_data(run.info.run_id)
        assert json.loads(tags["mlflow.datasets"]) == [
            iris_dataset._metadata,
            iris_pandas_df_dataset._metadata,
        ]

        # Test log repetitive dataset
        iris_dataset._log_dataset_tag(client, run.info.run_id)
        _, _, tags, _ = get_run_data(run.info.run_id)
        assert json.loads(tags["mlflow.datasets"]) == [
            iris_dataset._metadata,
            iris_pandas_df_dataset._metadata,
        ]


class FakeEvauator1(ModelEvaluator):
    def can_evaluate(self, model_type, evaluator_config=None, **kwargs):
        raise RuntimeError()

    def evaluate(self, model, model_type, dataset, run_id, evaluator_config, **kwargs):
        raise RuntimeError()


class FakeEvauator2(ModelEvaluator):
    def can_evaluate(self, model_type, evaluator_config=None, **kwargs):
        raise RuntimeError()

    def evaluate(self, model, model_type, dataset, run_id, evaluator_config, **kwargs):
        raise RuntimeError()


class FakeArtifact1(EvaluationArtifact):
    def save(self, output_artifact_path):
        raise RuntimeError()

    def _load_content_from_file(self, local_artifact_path):
        raise RuntimeError()


class FakeArtifact2(EvaluationArtifact):
    def save(self, output_artifact_path):
        raise RuntimeError()

    def _load_content_from_file(self, local_artifact_path):
        raise RuntimeError()


def test_evaluator_interface(classifier_model_uri, iris_dataset):
    with mock.patch.object(
        _model_evaluation_registry, "_registry", {"test_evaluator1": FakeEvauator1}
    ):
        evaluator1_config = {"eval1_confg_a": 3, "eval1_confg_b": 4}
        evaluator1_return_value = EvaluationResult(
            metrics=EvaluationMetrics({"m1": 5, "m2": 6}),
            artifacts={"a1": FakeArtifact1(uri="uri1"), "a2": FakeArtifact2(uri="uri2")},
        )
        with mock.patch.object(
            FakeEvauator1, "can_evaluate", return_value=False
        ) as mock_can_evaluate, mock.patch.object(
            FakeEvauator1, "evaluate", return_value=evaluator1_return_value
        ) as mock_evaluate:
            with mlflow.start_run():
                evaluate(
                    classifier_model_uri,
                    "classifier",
                    iris_dataset,
                    run_id=None,
                    evaluators="test_evaluator1",
                    evaluator_config=evaluator1_config,
                )
                mock_can_evaluate.assert_called_once_with("classifier", evaluator1_config)
                mock_evaluate.assert_not_called()
        with mock.patch.object(
            FakeEvauator1, "can_evaluate", return_value=True
        ) as mock_can_evaluate, mock.patch.object(
            FakeEvauator1, "evaluate", return_value=evaluator1_return_value
        ) as mock_evaluate:
            classifier_model = mlflow.pyfunc.load_model(classifier_model_uri)
            with mlflow.start_run() as run:
                eval1_result = evaluate(
                    classifier_model,
                    "classifier",
                    iris_dataset,
                    run_id=None,
                    evaluators="test_evaluator1",
                    evaluator_config=evaluator1_config,
                )
                assert eval1_result.metrics == evaluator1_return_value.metrics
                assert eval1_result.artifacts == evaluator1_return_value.artifacts

                mock_can_evaluate.assert_called_once_with("classifier", evaluator1_config)
                mock_evaluate.assert_called_once_with(
                    classifier_model, "classifier", iris_dataset, run.info.run_id, evaluator1_config
                )


def test_evaluate_with_multi_evaluators(classifier_model_uri, iris_dataset):
    with mock.patch.object(
        _model_evaluation_registry,
        "_registry",
        {"test_evaluator1": FakeEvauator1, "test_evaluator2": FakeEvauator2},
    ):
        evaluator1_config = {"eval1_confg": 3}
        evaluator2_config = {"eval2_confg": 4}
        evaluator1_return_value = EvaluationResult(
            metrics=EvaluationMetrics({"m1": 5}), artifacts={"a1": FakeArtifact1(uri="uri1")}
        )
        evaluator2_return_value = EvaluationResult(
            metrics=EvaluationMetrics({"m2": 6}), artifacts={"a2": FakeArtifact2(uri="uri2")}
        )
        with mock.patch.object(
            FakeEvauator1, "can_evaluate", return_value=True
        ) as mock_can_evaluate1, mock.patch.object(
            FakeEvauator1, "evaluate", return_value=evaluator1_return_value
        ) as mock_evaluate1, mock.patch.object(
            FakeEvauator2, "can_evaluate", return_value=True
        ) as mock_can_evaluate2, mock.patch.object(
            FakeEvauator2, "evaluate", return_value=evaluator2_return_value
        ) as mock_evaluate2:
            classifier_model = mlflow.pyfunc.load_model(classifier_model_uri)
            with mlflow.start_run() as run:
                eval_result = evaluate(
                    classifier_model,
                    "classifier",
                    iris_dataset,
                    run_id=None,
                    evaluators=["test_evaluator1", "test_evaluator2"],
                    evaluator_config={
                        "test_evaluator1": evaluator1_config,
                        "test_evaluator2": evaluator2_config,
                    },
                )
                assert eval_result.metrics == {
                    **evaluator1_return_value.metrics,
                    **evaluator2_return_value.metrics,
                }
                assert eval_result.artifacts == {
                    **evaluator1_return_value.artifacts,
                    **evaluator2_return_value.artifacts,
                }
                mock_can_evaluate1.assert_called_once_with("classifier", evaluator1_config)
                mock_evaluate1.assert_called_once_with(
                    classifier_model, "classifier", iris_dataset, run.info.run_id, evaluator1_config
                )
                mock_can_evaluate2.assert_called_once_with("classifier", evaluator2_config)
                mock_evaluate2.assert_called_once_with(
                    classifier_model, "classifier", iris_dataset, run.info.run_id, evaluator2_config
                )


def test_start_run_or_reuse_active_run():
    with _start_run_or_reuse_active_run(run_id=None) as run_id:
        assert mlflow.active_run().info.run_id == run_id

    with mlflow.start_run() as run:
        pass
    previous_run_id = run.info.run_id

    with _start_run_or_reuse_active_run(run_id=previous_run_id) as run_id:
        assert previous_run_id == run_id
        assert mlflow.active_run().info.run_id == run_id

    with mlflow.start_run() as run:
        active_run_id = run.info.run_id

        with _start_run_or_reuse_active_run(run_id=None) as run_id:
            assert run_id == active_run_id

        with _start_run_or_reuse_active_run(run_id=active_run_id) as run_id:
            assert run_id == active_run_id

        with pytest.raises(ValueError, match="An active run exists"):
            with _start_run_or_reuse_active_run(run_id=previous_run_id):
                pass
