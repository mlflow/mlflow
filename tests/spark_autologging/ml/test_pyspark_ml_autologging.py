import importlib
import pytest
from collections import namedtuple
from distutils.version import LooseVersion
from unittest import mock

import mlflow
from mlflow.utils import _truncate_dict
from mlflow.utils.validation import (
    MAX_PARAM_VAL_LENGTH,
    MAX_ENTITY_KEY_LENGTH,
)

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.classification import (
    LinearSVC,
    LogisticRegression,
    MultilayerPerceptronClassifier,
    OneVsRest,
)
from mlflow.pyspark.ml import (
    _should_log_model,
    _get_instance_param_map,
    _get_warning_msg_for_skip_log_model,
    _get_warning_msg_for_fit_call_with_a_list_of_params,
)

pytestmark = pytest.mark.large

MODEL_DIR = "model"
MLFLOW_PARENT_RUN_ID = "mlflow.parentRunId"


@pytest.fixture(scope="module")
def spark_session():
    session = SparkSession.builder.master("local[*]").getOrCreate()
    yield session
    session.stop()


@pytest.fixture(scope="module")
def dataset_binomial(spark_session):
    return spark_session.createDataFrame(
        [(1.0, Vectors.dense(1.0)), (0.0, Vectors.sparse(1, [], []))] * 100, ["label", "features"]
    )


@pytest.fixture(scope="module")
def dataset_multinomial(spark_session):
    return spark_session.createDataFrame(
        [(1.0, Vectors.dense(1.0)), (0.0, Vectors.sparse(1, [], [])), (2.0, Vectors.dense(0.5))]
        * 100,
        ["label", "features"],
    )


def truncate_param_dict(d):
    return _truncate_dict(d, MAX_ENTITY_KEY_LENGTH, MAX_PARAM_VAL_LENGTH)


def stringify_dict_values(d):
    return {k: str(v) for k, v in d.items()}


def get_expected_class_tags(estimator):
    return {
        "estimator_name": estimator.__class__.__name__,
        "estimator_class": estimator.__class__.__module__ + "." + estimator.__class__.__name__,
    }


def get_run_data(run_id):
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    # Ignore tags mlflow logs by default (e.g. "mlflow.user")
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id)]

    RunData = namedtuple("RunData", ["params", "metrics", "tags", "artifacts"])
    return RunData(data.params, data.metrics, tags, artifacts)


def load_model_by_run_id(run_id):
    return mlflow.spark.load_model("runs:/{}/{}".format(run_id, MODEL_DIR))


def test_basic_estimator(dataset_binomial):
    mlflow.pyspark.ml.autolog()

    for estimator in [
        LinearRegression(),
        MultilayerPerceptronClassifier(layers=[2, 2, 2], seed=123, blockSize=1),
    ]:
        with mlflow.start_run() as run:
            model = estimator.fit(dataset_binomial)
        run_id = run.info.run_id
        run_data = get_run_data(run_id)
        assert run_data.params == truncate_param_dict(
            stringify_dict_values(_get_instance_param_map(estimator))
        )
        assert run_data.tags == get_expected_class_tags(estimator)
        if isinstance(estimator, MultilayerPerceptronClassifier):
            assert MODEL_DIR not in run_data.artifacts
        else:
            assert MODEL_DIR in run_data.artifacts
            loaded_model = load_model_by_run_id(run_id)
            assert loaded_model.stages[0].uid == model.uid


@pytest.mark.skipif(
    LooseVersion(pyspark.__version__) < LooseVersion("3.1"),
    reason="This test require spark version >= 3.1",
)
def test_models_in_allowlist_exist(spark_session):  # pylint: disable=unused-argument
    mlflow.pyspark.ml.autolog()  # initialize the variable `mlflow.pyspark.ml._log_model_allowlist`

    def model_does_not_exist(model_class):
        module_name, class_name = model_class.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
            return not hasattr(module, class_name)
        except ModuleNotFoundError:
            return True

    non_existent_classes = list(
        filter(model_does_not_exist, mlflow.pyspark.ml._log_model_allowlist)
    )
    assert len(non_existent_classes) == 0, "{} in log_model_allowlist don't exist".format(
        non_existent_classes
    )


def test_autolog_does_not_terminate_active_run(dataset_binomial):
    mlflow.pyspark.ml.autolog()
    mlflow.start_run()
    lr = LinearRegression()
    lr.fit(dataset_binomial)
    assert mlflow.active_run() is not None
    mlflow.end_run()


def test_meta_estimator_fit(dataset_binomial):
    mlflow.pyspark.ml.autolog()
    with mlflow.start_run() as run:
        svc = LinearSVC()
        ova = OneVsRest(classifier=svc)
        ova_model = ova.fit(dataset_binomial)

    run_id = run.info.run_id
    run_data = get_run_data(run_id)
    assert run_data.params == truncate_param_dict(
        stringify_dict_values(_get_instance_param_map(ova))
    )
    assert run_data.tags == get_expected_class_tags(ova)
    assert MODEL_DIR in run_data.artifacts
    loaded_model = load_model_by_run_id(run_id)
    assert loaded_model.stages[0].uid == ova_model.uid

    # assert no nested run spawned
    query = "tags.{} = '{}'".format(MLFLOW_PARENT_RUN_ID, run.info.run_id)
    assert len(mlflow.search_runs([run.info.experiment_id])) == 1
    assert len(mlflow.search_runs([run.info.experiment_id], query)) == 0


def test_fit_with_params(dataset_binomial):
    mlflow.pyspark.ml.autolog()
    lr = LinearRegression()
    extra_params = {lr.maxIter: 3, lr.standardization: False}
    with mlflow.start_run() as run:
        lr_model = lr.fit(dataset_binomial, params=extra_params)
    run_id = run.info.run_id
    run_data = get_run_data(run_id)
    assert run_data.params == truncate_param_dict(
        stringify_dict_values(_get_instance_param_map(lr.copy(extra_params)))
    )
    assert run_data.tags == get_expected_class_tags(lr)
    assert MODEL_DIR in run_data.artifacts
    loaded_model = load_model_by_run_id(run_id)
    assert loaded_model.stages[0].uid == lr_model.uid


def test_fit_with_a_list_of_params(dataset_binomial):
    mlflow.pyspark.ml.autolog()
    lr = LinearRegression()
    extra_params = {lr.maxIter: 3, lr.standardization: False}
    # Test calling fit with a list/tuple of paramMap
    for params in [[extra_params], (extra_params,)]:
        with mock.patch("mlflow.log_params") as mock_log_params, mock.patch(
            "mlflow.set_tags"
        ) as mock_set_tags:
            with mlflow.start_run():
                with mock.patch("mlflow.pyspark.ml._logger.warning") as mock_warning:
                    lr_model_iter = lr.fit(dataset_binomial, params=params)
                    mock_warning.called_once_with(
                        _get_warning_msg_for_fit_call_with_a_list_of_params(lr)
                    )
            assert isinstance(list(lr_model_iter)[0], LinearRegressionModel)
            mock_log_params.assert_not_called()
            mock_set_tags.assert_not_called()


def test_should_log_model(dataset_binomial, dataset_multinomial):
    mlflow.pyspark.ml.autolog(log_models=True)
    lor = LogisticRegression()

    ova1 = OneVsRest(classifier=lor)
    mlor_model = lor.fit(dataset_multinomial)
    assert _should_log_model(mlor_model)

    ova1_model = ova1.fit(dataset_multinomial)
    assert _should_log_model(ova1_model)

    with mock.patch(
        "mlflow.pyspark.ml._log_model_allowlist",
        {"pyspark.ml.regression.LinearRegressionModel", "pyspark.ml.classification.OneVsRestModel"},
    ), mock.patch("mlflow.pyspark.ml._logger.warning") as mock_warning:
        lr = LinearRegression()
        lr_model = lr.fit(dataset_binomial)
        assert _should_log_model(lr_model)
        lor_model = lor.fit(dataset_binomial)
        assert not _should_log_model(lor_model)
        mock_warning.called_once_with(_get_warning_msg_for_skip_log_model(lor_model))
        assert not _should_log_model(ova1_model)


def test_param_map_captures_wrapped_params(dataset_binomial):
    lor = LogisticRegression(maxIter=3, standardization=False)
    ova = OneVsRest(classifier=lor, labelCol="abcd")

    param_map = _get_instance_param_map(ova)
    assert param_map["labelCol"] == "abcd"
    assert param_map["classifier"] == lor.uid
    assert param_map[f"{lor.uid}.maxIter"] == 3
    assert not param_map[f"{lor.uid}.standardization"]
    assert param_map[f"{lor.uid}.tol"] == lor.getOrDefault(lor.tol)

    mlflow.pyspark.ml.autolog()
    with mlflow.start_run() as run:
        ova.fit(dataset_binomial.withColumn("abcd", dataset_binomial.label))
    run_id = run.info.run_id
    run_data = get_run_data(run_id)
    assert run_data.params == truncate_param_dict(
        stringify_dict_values(_get_instance_param_map(ova))
    )
