import importlib
import json
import math
import numpy as np
import pandas as pd
import pytest
from collections import namedtuple
from packaging.version import Version
from unittest import mock

import mlflow
from mlflow.entities import RunStatus
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_AUTOLOGGING
from mlflow.utils import _truncate_dict
from mlflow.utils.validation import (
    MAX_PARAM_VAL_LENGTH,
    MAX_ENTITY_KEY_LENGTH,
)
from mlflow.tracking.client import MlflowClient

import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
)
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.classification import (
    LinearSVC,
    LogisticRegression,
    MultilayerPerceptronClassifier,
    OneVsRest,
)
from pyspark.ml.feature import HashingTF, Tokenizer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from mlflow.pyspark.ml import (
    _should_log_model,
    _get_instance_param_map,
    _get_instance_param_map_recursively,
    _get_warning_msg_for_skip_log_model,
    _get_warning_msg_for_fit_call_with_a_list_of_params,
    _gen_estimator_metadata,
    _get_tuning_param_maps,
)
from pyspark.sql import SparkSession


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
    ).cache()


@pytest.fixture(scope="module")
def dataset_multinomial(spark_session):
    return spark_session.createDataFrame(
        [(1.0, Vectors.dense(1.0)), (0.0, Vectors.sparse(1, [], [])), (2.0, Vectors.dense(0.5))]
        * 100,
        ["label", "features"],
    ).cache()


@pytest.fixture(scope="module")
def dataset_text(spark_session):
    return spark_session.createDataFrame(
        [
            (0, "a b c d e spark", 1.0),
            (1, "b d", 0.0),
            (2, "spark f g h", 1.0),
            (3, "hadoop mapreduce", 0.0),
        ],
        ["id", "text", "label"],
    ).cache()


@pytest.fixture(scope="module")
def dataset_regression(spark_session):
    np.random.seed(1)
    num_features = 10
    coef = np.random.rand(num_features)

    rows = []
    for _ in range(300):
        features = (np.random.rand(num_features) * 2.0 - 1.0) * np.random.rand()
        err = (np.random.rand() * 2.0 - 1.0) * 0.05
        label = float(np.dot(coef, features)) + err
        rows.append((label, Vectors.dense(*features)))

    return spark_session.createDataFrame(rows, ["label", "features"]).cache()


@pytest.fixture(scope="module")
def dataset_iris_binomial(spark_session):
    from sklearn.datasets import load_iris

    df = load_iris(as_frame=True).frame.rename(columns={"target": "label"})
    df = spark_session.createDataFrame(df)
    df = VectorAssembler(inputCols=df.columns[:-1], outputCol="features").transform(df)
    df = df.filter(df.label < 2).select("features", "label")
    df.cache()
    return df


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


def get_params_to_log(estimator):
    metadata = _gen_estimator_metadata(estimator)
    return _get_instance_param_map(estimator, metadata.uid_to_indexed_name_map)


def load_json_artifact(artifact_path):
    fpath = mlflow.get_artifact_uri(artifact_path).replace("file://", "")
    with open(fpath, "r") as f:
        return json.load(f)


def load_json_csv(artifact_path):
    fpath = mlflow.get_artifact_uri(artifact_path).replace("file://", "")
    return pd.read_csv(fpath)


def load_model_by_run_id(run_id, model_dir=MODEL_DIR):
    return mlflow.spark.load_model("runs:/{}/{}".format(run_id, model_dir))


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
            stringify_dict_values(get_params_to_log(estimator))
        )
        assert run_data.tags == get_expected_class_tags(estimator)
        if isinstance(estimator, MultilayerPerceptronClassifier):
            assert MODEL_DIR not in run_data.artifacts
        else:
            assert MODEL_DIR in run_data.artifacts
            loaded_model = load_model_by_run_id(run_id)
            assert loaded_model.stages[0].uid == model.uid


@pytest.mark.skipif(
    Version(pyspark.__version__) < Version("3.1"),
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
    assert run_data.params == truncate_param_dict(stringify_dict_values(get_params_to_log(ova)))
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
        stringify_dict_values(get_params_to_log(lr.copy(extra_params)))
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


def test_should_log_model(dataset_binomial, dataset_multinomial, dataset_text):
    mlflow.pyspark.ml.autolog(log_models=True)
    lor = LogisticRegression()

    ova1 = OneVsRest(classifier=lor)
    with mlflow.start_run():
        mlor_model = lor.fit(dataset_multinomial)
    assert _should_log_model(mlor_model)

    with mlflow.start_run():
        ova1_model = ova1.fit(dataset_multinomial)
    assert _should_log_model(ova1_model)

    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
    lr = LogisticRegression(maxIter=2)
    pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
    with mlflow.start_run():
        pipeline_model = pipeline.fit(dataset_text)
    assert _should_log_model(pipeline_model)

    nested_pipeline = Pipeline(stages=[tokenizer, Pipeline(stages=[hashingTF, lr])])
    with mlflow.start_run():
        nested_pipeline_model = nested_pipeline.fit(dataset_text)
    assert _should_log_model(nested_pipeline_model)

    with mock.patch(
        "mlflow.pyspark.ml._log_model_allowlist",
        {
            "pyspark.ml.regression.LinearRegressionModel",
            "pyspark.ml.classification.OneVsRestModel",
            "pyspark.ml.pipeline.PipelineModel",
        },
    ), mock.patch("mlflow.pyspark.ml._logger.warning") as mock_warning:
        lr = LinearRegression()
        with mlflow.start_run():
            lr_model = lr.fit(dataset_binomial)
        assert _should_log_model(lr_model)
        with mlflow.start_run():
            lor_model = lor.fit(dataset_binomial)
        assert not _should_log_model(lor_model)
        mock_warning.called_once_with(_get_warning_msg_for_skip_log_model(lor_model))
        assert not _should_log_model(ova1_model)
        assert not _should_log_model(pipeline_model)
        assert not _should_log_model(nested_pipeline_model)


def test_should_log_model_with_wildcards_in_allowlist(dataset_binomial, dataset_multinomial):
    mlflow.pyspark.ml.autolog(log_models=True)
    lor = LogisticRegression()
    ova1 = OneVsRest(classifier=lor)
    ova1_model = ova1.fit(dataset_multinomial)

    with mock.patch(
        "mlflow.pyspark.ml._log_model_allowlist",
        {
            "pyspark.ml.regression.*",
            "pyspark.ml.classification.LogisticRegressionModel",
            "pyspark.ml.feature.*",
        },
    ):
        lr = LinearRegression()
        with mlflow.start_run():
            lr_model = lr.fit(dataset_binomial)
        assert _should_log_model(lr_model)
        with mlflow.start_run():
            lor_model = lor.fit(dataset_binomial)
        assert _should_log_model(lor_model)
        assert not _should_log_model(ova1_model)


def test_log_stage_type_params(spark_session):
    from pyspark.ml.base import Estimator, Transformer, Model
    from pyspark.ml.evaluation import Evaluator
    from pyspark.ml.param import Param, Params
    from pyspark.ml.feature import Binarizer, OneHotEncoder

    class TestingEstimator(Estimator):

        transformer = Param(Params._dummy(), "transformer", "a transformer param")
        model = Param(Params._dummy(), "model", "a model param")
        evaluator = Param(Params._dummy(), "evaluator", "an evaluator param")

        def setTransformer(self, transformer: Transformer):
            return self._set(transformer=transformer)

        def setModel(self, model: Model):
            return self._set(model=model)

        def setEvaluator(self, evaluator: Evaluator):
            return self._set(evaluator=evaluator)

        def _fit(self, dataset):  # pylint: disable=unused-argument
            return TestingModel()

    class TestingModel(Model):
        def _transform(self, dataset):
            return dataset

    binarizer = Binarizer(threshold=1.0, inputCol="values", outputCol="features")
    df = spark_session.createDataFrame([(0.0,), (1.0,), (2.0,)], ["input"])
    ohe = OneHotEncoder().setInputCols(["input"]).setOutputCols(["output"])
    ohemodel = ohe.fit(df)
    bcd = BinaryClassificationEvaluator(metricName="areaUnderROC")

    estimator = TestingEstimator().setTransformer(binarizer).setModel(ohemodel).setEvaluator(bcd)
    param_map = get_params_to_log(estimator)
    assert param_map["transformer"] == "Binarizer"
    assert param_map["model"] == "OneHotEncoderModel"
    assert param_map["evaluator"] == "BinaryClassificationEvaluator"

    mlflow.pyspark.ml.autolog()
    with mlflow.start_run() as run:
        estimator.fit(df)
        metadata = _gen_estimator_metadata(estimator)
        estimator_info = load_json_artifact("estimator_info.json")
        assert metadata.hierarchy == estimator_info["hierarchy"]
        assert isinstance(estimator_info["hierarchy"]["params"], dict)
        assert estimator_info["hierarchy"]["params"]["transformer"]["name"] == "Binarizer"
        assert estimator_info["hierarchy"]["params"]["model"]["name"] == "OneHotEncoderModel"
        assert (
            estimator_info["hierarchy"]["params"]["evaluator"]["name"]
            == "BinaryClassificationEvaluator"
        )
    run_id = run.info.run_id
    run_data = get_run_data(run_id)
    assert run_data.params == truncate_param_dict(
        stringify_dict_values(get_params_to_log(estimator))
    )


def test_param_map_captures_wrapped_params(dataset_binomial):
    lor = LogisticRegression(maxIter=3, standardization=False)
    ova = OneVsRest(classifier=lor, labelCol="abcd")

    param_map = get_params_to_log(ova)
    assert param_map["labelCol"] == "abcd"
    assert param_map["classifier"] == "LogisticRegression"
    assert param_map["LogisticRegression.maxIter"] == 3
    assert not param_map["LogisticRegression.standardization"]
    assert param_map["LogisticRegression.tol"] == lor.getOrDefault(lor.tol)

    mlflow.pyspark.ml.autolog()
    with mlflow.start_run() as run:
        ova.fit(dataset_binomial.withColumn("abcd", dataset_binomial.label))
        metadata = _gen_estimator_metadata(ova)
        estimator_info = load_json_artifact("estimator_info.json")
        assert metadata.hierarchy == estimator_info["hierarchy"]
    run_id = run.info.run_id
    run_data = get_run_data(run_id)
    assert run_data.params == truncate_param_dict(stringify_dict_values(get_params_to_log(ova)))


def test_pipeline(dataset_text):
    mlflow.pyspark.ml.autolog()

    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
    lr = LogisticRegression(maxIter=2, regParam=0.001)
    pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])
    inner_pipeline = Pipeline(stages=[hashingTF, lr])
    nested_pipeline = Pipeline(stages=[tokenizer, inner_pipeline])

    for estimator in [pipeline, nested_pipeline]:
        with mlflow.start_run() as run:
            model = estimator.fit(dataset_text)
            estimator_info = load_json_artifact("estimator_info.json")
            metadata = _gen_estimator_metadata(estimator)
            assert metadata.hierarchy == estimator_info["hierarchy"]

        uid_to_indexed_name_map = metadata.uid_to_indexed_name_map
        run_id = run.info.run_id
        run_data = get_run_data(run_id)
        assert run_data.params == truncate_param_dict(
            stringify_dict_values(_get_instance_param_map(estimator, uid_to_indexed_name_map))
        )
        assert run_data.tags == get_expected_class_tags(estimator)
        assert MODEL_DIR in run_data.artifacts
        loaded_model = load_model_by_run_id(run_id)
        assert loaded_model.uid == model.uid
        assert run_data.artifacts == ["estimator_info.json", "model"]


# Test on metric of rmse (smaller is better) and r2 (larger is better)
@pytest.mark.parametrize("metric_name", ["rmse", "r2"])
@pytest.mark.parametrize("param_search_estimator", [CrossValidator, TrainValidationSplit])
def test_param_search_estimator(  # pylint: disable=unused-argument
    metric_name, param_search_estimator, spark_session, dataset_regression
):
    mlflow.pyspark.ml.autolog()
    lr = LinearRegression(solver="l-bfgs", regParam=0.01)
    lrParamMaps = [
        {lr.maxIter: 1, lr.standardization: False},
        {lr.maxIter: 200, lr.standardization: True},
        {lr.maxIter: 2, lr.standardization: False},
    ]
    best_params = {"LinearRegression.maxIter": 200, "LinearRegression.standardization": True}
    eva = RegressionEvaluator(metricName=metric_name)
    estimator = param_search_estimator(estimator=lr, estimatorParamMaps=lrParamMaps, evaluator=eva)
    with mlflow.start_run() as run:
        model = estimator.fit(dataset_regression)
        estimator_info = load_json_artifact("estimator_info.json")
        metadata = _gen_estimator_metadata(estimator)
        assert metadata.hierarchy == estimator_info["hierarchy"]

        param_search_estiamtor_info = estimator_info[
            metadata.uid_to_indexed_name_map[estimator.uid]
        ]
        assert param_search_estiamtor_info[
            "tuned_estimator_parameter_map"
        ] == _get_instance_param_map_recursively(lr, 1, metadata.uid_to_indexed_name_map)
        assert param_search_estiamtor_info["tuning_parameter_map_list"] == _get_tuning_param_maps(
            estimator, metadata.uid_to_indexed_name_map
        )

        assert best_params == load_json_artifact("best_parameters.json")

        search_results = load_json_csv("search_results.csv")

    uid_to_indexed_name_map = metadata.uid_to_indexed_name_map
    run_id = run.info.run_id
    run_data = get_run_data(run_id)
    assert run_data.params == truncate_param_dict(
        stringify_dict_values(
            {
                **_get_instance_param_map(estimator, uid_to_indexed_name_map),
                **{f"best_{k}": v for k, v in best_params.items()},
            }
        )
    )
    assert run_data.tags == get_expected_class_tags(estimator)
    assert MODEL_DIR in run_data.artifacts
    loaded_model = load_model_by_run_id(run_id)
    assert loaded_model.stages[0].uid == model.uid
    loaded_best_model = load_model_by_run_id(run_id, "best_model")
    assert loaded_best_model.stages[0].uid == model.bestModel.uid
    assert run_data.artifacts == [
        "best_model",
        "best_parameters.json",
        "estimator_info.json",
        "model",
        "search_results.csv",
    ]

    client = mlflow.tracking.MlflowClient()
    child_runs = client.search_runs(
        run.info.experiment_id, "tags.`mlflow.parentRunId` = '{}'".format(run_id)
    )
    assert len(child_runs) == len(search_results)

    for row_index, row in search_results.iterrows():
        row_params = json.loads(row.get("params", "{}"))
        for param_name, param_value in row_params.items():
            assert param_value == row.get(f"param.{param_name}")

        params_search_clause = " and ".join(
            [
                "params.`{}` = '{}'".format(key.split(".")[1], value)
                for key, value in row_params.items()
            ]
        )
        search_filter = "tags.`mlflow.parentRunId` = '{}' and {}".format(
            run_id, params_search_clause
        )
        child_runs = client.search_runs(run.info.experiment_id, search_filter)
        assert len(child_runs) == 1
        child_run = child_runs[0]
        assert child_run.info.status == RunStatus.to_string(RunStatus.FINISHED)
        run_data = get_run_data(child_run.info.run_id)
        child_estimator = estimator.getEstimator().copy(
            estimator.getEstimatorParamMaps()[row_index]
        )
        assert run_data.tags == get_expected_class_tags(child_estimator)
        assert run_data.params == truncate_param_dict(
            stringify_dict_values(
                {**_get_instance_param_map(child_estimator, uid_to_indexed_name_map)}
            )
        )
        assert (
            child_run.data.tags.get(MLFLOW_AUTOLOGGING)
            == mlflow.pyspark.ml.AUTOLOGGING_INTEGRATION_NAME
        )

        metric_name = estimator.getEvaluator().getMetricName()
        if isinstance(estimator, CrossValidator):
            avg_metric_value = model.avgMetrics[row_index]
            avg_metric_name = f"avg_{metric_name}"
        else:
            avg_metric_value = model.validationMetrics[row_index]
            avg_metric_name = metric_name

        assert math.isclose(avg_metric_value, run_data.metrics[avg_metric_name], rel_tol=1e-6)
        assert math.isclose(avg_metric_value, float(row.get(avg_metric_name)), rel_tol=1e-6)

        if isinstance(estimator, CrossValidator) and Version(pyspark.__version__) >= Version("3.3"):
            std_metric_name = f"std_{metric_name}"
            std_metric_value = model.stdMetrics[row_index]
            assert math.isclose(std_metric_value, run_data.metrics[std_metric_name], rel_tol=1e-6)
            assert math.isclose(std_metric_value, float(row.get(std_metric_name)), rel_tol=1e-6)


def test_get_params_to_log(spark_session):  # pylint: disable=unused-argument
    lor = LogisticRegression(maxIter=3, standardization=False)
    lor_params = get_params_to_log(lor)
    assert (
        lor_params["maxIter"] == 3
        and not lor_params["standardization"]
        and lor_params["family"] == lor.getOrDefault(lor.family)
    )

    ova = OneVsRest(classifier=lor, labelCol="abcd")
    ova_params = get_params_to_log(ova)
    assert (
        ova_params["classifier"] == "LogisticRegression"
        and ova_params["labelCol"] == "abcd"
        and ova_params["LogisticRegression.maxIter"] == 3
        and ova_params["LogisticRegression.family"] == lor.getOrDefault(lor.family)
    )

    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
    pipeline = Pipeline(stages=[tokenizer, hashingTF, ova])
    inner_pipeline = Pipeline(stages=[hashingTF, ova])
    nested_pipeline = Pipeline(stages=[tokenizer, inner_pipeline])

    pipeline_params = get_params_to_log(pipeline)
    nested_pipeline_params = get_params_to_log(nested_pipeline)

    assert pipeline_params["stages"] == ["Tokenizer", "HashingTF", "OneVsRest"]
    assert nested_pipeline_params["stages"] == ["Tokenizer", "Pipeline_2"]
    assert nested_pipeline_params["Pipeline_2.stages"] == ["HashingTF", "OneVsRest"]
    assert nested_pipeline_params["OneVsRest.classifier"] == "LogisticRegression"

    for params_to_test in [pipeline_params, nested_pipeline_params]:
        assert (
            params_to_test["Tokenizer.inputCol"] == "text"
            and params_to_test["Tokenizer.outputCol"] == "words"
        )
        assert params_to_test["HashingTF.outputCol"] == "features"
        assert params_to_test["OneVsRest.classifier"] == "LogisticRegression"
        assert params_to_test["LogisticRegression.maxIter"] == 3


def test_gen_estimator_metadata(spark_session):  # pylint: disable=unused-argument
    tokenizer1 = Tokenizer(inputCol="text1", outputCol="words1")
    hashingTF1 = HashingTF(inputCol=tokenizer1.getOutputCol(), outputCol="features1")

    tokenizer2 = Tokenizer(inputCol="text2", outputCol="words2")
    hashingTF2 = HashingTF(inputCol=tokenizer2.getOutputCol(), outputCol="features2")

    vecAssembler = VectorAssembler(inputCols=["features1", "features2"], outputCol="features")

    lor = LogisticRegression(maxIter=10)
    ova = OneVsRest(classifier=lor)
    sub_pipeline1 = Pipeline(stages=[tokenizer1, hashingTF1])
    sub_pipeline2 = Pipeline(stages=[tokenizer2, hashingTF2])
    sub_pipeline3 = Pipeline(stages=[vecAssembler, ova])

    paramGrid = (
        ParamGridBuilder().addGrid(lor.maxIter, [10, 20]).addGrid(lor.regParam, [0.1, 0.01]).build()
    )
    eva = MulticlassClassificationEvaluator()
    crossval = CrossValidator(
        estimator=sub_pipeline3, estimatorParamMaps=paramGrid, evaluator=eva, numFolds=2
    )

    top_pipeline = Pipeline(stages=[sub_pipeline1, sub_pipeline2, crossval])

    metadata = _gen_estimator_metadata(top_pipeline)

    expected_hierarchy = {
        "name": "Pipeline_1",
        "stages": [
            {"name": "Pipeline_2", "stages": [{"name": "Tokenizer_1"}, {"name": "HashingTF_1"}]},
            {"name": "Pipeline_3", "stages": [{"name": "Tokenizer_2"}, {"name": "HashingTF_2"}]},
            {
                "name": "CrossValidator",
                "evaluator": {"name": "MulticlassClassificationEvaluator"},
                "tuned_estimator": {
                    "name": "Pipeline_4",
                    "stages": [
                        {"name": "VectorAssembler"},
                        {"name": "OneVsRest", "classifier": {"name": "LogisticRegression"}},
                    ],
                },
            },
        ],
    }
    assert metadata.hierarchy == expected_hierarchy
    assert metadata.uid_to_indexed_name_map == {
        top_pipeline.uid: "Pipeline_1",
        sub_pipeline1.uid: "Pipeline_2",
        tokenizer1.uid: "Tokenizer_1",
        hashingTF1.uid: "HashingTF_1",
        sub_pipeline2.uid: "Pipeline_3",
        tokenizer2.uid: "Tokenizer_2",
        hashingTF2.uid: "HashingTF_2",
        crossval.uid: "CrossValidator",
        sub_pipeline3.uid: "Pipeline_4",
        vecAssembler.uid: "VectorAssembler",
        ova.uid: "OneVsRest",
        lor.uid: "LogisticRegression",
        eva.uid: "MulticlassClassificationEvaluator",
    }
    assert (
        metadata.uid_to_indexed_name_map[metadata.param_search_estimators[0].uid]
        == "CrossValidator"
    )


def test_basic_post_training_metric_autologging(dataset_iris_binomial):
    mlflow.pyspark.ml.autolog()

    estimator = LogisticRegression(maxIter=1, family="binomial", regParam=5.0, fitIntercept=False)
    eval_dataset = dataset_iris_binomial.sample(fraction=0.3, seed=1)

    with mlflow.start_run() as run:
        model = estimator.fit(dataset_iris_binomial)
        mce = MulticlassClassificationEvaluator(metricName="logLoss")
        pred_result = model.transform(eval_dataset)
        logloss = mce.evaluate(pred_result)

        # test calling evaluate with extra params
        accuracy = mce.evaluate(pred_result, params={mce.metricName: "accuracy"})

        # generate a new validation dataset but reuse the variable name 'eval_dataset'
        # test the autologged metric use dataset name 'eval_dataset-2'
        eval_dataset = dataset_iris_binomial.sample(fraction=0.3, seed=1)
        bce = BinaryClassificationEvaluator(metricName="areaUnderROC")
        pred_result = model.transform(eval_dataset)
        areaUnderROC = bce.evaluate(pred_result)

        # test computing the same metric twice
        bce.evaluate(pred_result)

        metric_info = load_json_artifact("metric_info.json")

    run_data = get_run_data(run.info.run_id)

    assert np.isclose(logloss, run_data.metrics["logLoss_eval_dataset"])
    assert np.isclose(accuracy, run_data.metrics["accuracy_eval_dataset"])
    assert np.isclose(areaUnderROC, run_data.metrics["areaUnderROC_eval_dataset-2"])
    assert np.isclose(areaUnderROC, run_data.metrics["areaUnderROC-2_eval_dataset-2"])

    assert metric_info == {
        "accuracy_eval_dataset": {
            "evaluator_class": "pyspark.ml.evaluation.MulticlassClassificationEvaluator",
            "params": {
                "beta": 1.0,
                "eps": 1e-15,
                "labelCol": "label",
                "metricLabel": 0.0,
                "metricName": "accuracy",
                "predictionCol": "prediction",
                "probabilityCol": "probability",
            },
        },
        "areaUnderROC-2_eval_dataset-2": {
            "evaluator_class": "pyspark.ml.evaluation.BinaryClassificationEvaluator",
            "params": {
                "labelCol": "label",
                "metricName": "areaUnderROC",
                "numBins": 1000,
                "rawPredictionCol": "rawPrediction",
            },
        },
        "areaUnderROC_eval_dataset-2": {
            "evaluator_class": "pyspark.ml.evaluation.BinaryClassificationEvaluator",
            "params": {
                "labelCol": "label",
                "metricName": "areaUnderROC",
                "numBins": 1000,
                "rawPredictionCol": "rawPrediction",
            },
        },
        "logLoss_eval_dataset": {
            "evaluator_class": "pyspark.ml.evaluation.MulticlassClassificationEvaluator",
            "params": {
                "beta": 1.0,
                "eps": 1e-15,
                "labelCol": "label",
                "metricLabel": 0.0,
                "metricName": "logLoss",
                "predictionCol": "prediction",
                "probabilityCol": "probability",
            },
        },
    }

    mlflow.pyspark.ml.autolog(disable=True)
    recall_original = mce.evaluate(pred_result)
    assert np.isclose(logloss, recall_original)
    accuracy_original = mce.evaluate(pred_result, params={mce.metricName: "accuracy"})
    assert np.isclose(accuracy, accuracy_original)
    areaUnderROC_original = bce.evaluate(pred_result)
    assert np.isclose(areaUnderROC, areaUnderROC_original)


def test_multi_model_interleaved_fit_and_post_train_metric_call(dataset_iris_binomial):
    mlflow.pyspark.ml.autolog()

    estimator1 = LogisticRegression(maxIter=1, family="binomial", regParam=5.0, fitIntercept=False)
    estimator2 = LogisticRegression(maxIter=5, family="binomial", regParam=5.0, fitIntercept=False)
    eval_dataset1 = dataset_iris_binomial.sample(fraction=0.3, seed=1)
    eval_dataset2 = dataset_iris_binomial.sample(fraction=0.3, seed=2)
    mce = MulticlassClassificationEvaluator(metricName="logLoss")

    with mlflow.start_run() as run1:
        model1 = estimator1.fit(dataset_iris_binomial)

    with mlflow.start_run() as run2:
        model2 = estimator2.fit(dataset_iris_binomial)

    pred1_result = model1.transform(eval_dataset1)
    pred2_result = model2.transform(eval_dataset2)

    logloss1 = mce.evaluate(pred1_result)
    logloss2 = mce.evaluate(pred2_result)

    metrics1 = get_run_data(run1.info.run_id).metrics
    assert np.isclose(logloss1, metrics1["logLoss_eval_dataset1"])

    metrics2 = get_run_data(run2.info.run_id).metrics
    assert np.isclose(logloss2, metrics2["logLoss_eval_dataset2"])


def test_meta_estimator_disable_post_training_autologging(dataset_regression):
    mlflow.pyspark.ml.autolog()
    lr = LinearRegression(solver="l-bfgs", regParam=0.01)
    eval_dataset = dataset_regression.sample(fraction=0.3, seed=1)
    lrParamMaps = [
        {lr.maxIter: 1, lr.standardization: False},
        {lr.maxIter: 200, lr.standardization: True},
        {lr.maxIter: 2, lr.standardization: False},
    ]
    eva = RegressionEvaluator(metricName="rmse")
    estimator = TrainValidationSplit(estimator=lr, estimatorParamMaps=lrParamMaps, evaluator=eva)

    with mock.patch(
        "mlflow.pyspark.ml._AutologgingMetricsManager.register_model"
    ) as mock_register_model, mock.patch(
        "mlflow.sklearn._AutologgingMetricsManager.is_metric_value_loggable"
    ) as mock_is_metric_value_loggable, mock.patch(
        "mlflow.pyspark.ml._AutologgingMetricsManager.log_post_training_metric"
    ) as mock_log_post_training_metric, mock.patch(
        "mlflow.pyspark.ml._AutologgingMetricsManager.register_prediction_input_dataset"
    ) as mock_register_prediction_input_dataset:
        with mlflow.start_run():
            model = estimator.fit(dataset_regression)

        model.transform(eval_dataset)

        mock_register_model.assert_called_once()
        mock_is_metric_value_loggable.assert_not_called()
        mock_register_prediction_input_dataset.assert_not_called()
        mock_log_post_training_metric.assert_not_called()


def test_is_metrics_value_loggable():
    is_metric_value_loggable = mlflow.pyspark.ml._AutologgingMetricsManager.is_metric_value_loggable
    assert is_metric_value_loggable(3)
    assert is_metric_value_loggable(3.5)
    assert is_metric_value_loggable(np.float32(3.5))
    assert not is_metric_value_loggable(True)
    assert not is_metric_value_loggable([1, 2])
    assert not is_metric_value_loggable(np.array([1, 2]))


def test_log_post_training_metrics_configuration(dataset_iris_binomial):
    estimator = LogisticRegression(maxIter=1)
    mce = MulticlassClassificationEvaluator()
    metric_name = mce.getMetricName()

    # Ensure post-traning metrics autologging can be toggled on / off
    for log_post_training_metrics in [True, False, True]:
        mlflow.pyspark.ml.autolog(log_post_training_metrics=log_post_training_metrics)

        with mlflow.start_run() as run:
            model = estimator.fit(dataset_iris_binomial)
            pred_result = model.transform(dataset_iris_binomial)
            mce.evaluate(pred_result)

        metrics = get_run_data(run.info.run_id)[1]
        assert any(k.startswith(metric_name) for k in metrics.keys()) is log_post_training_metrics


def test_autologging_handle_wrong_pipeline_stage(dataset_regression):
    mlflow.pyspark.ml.autolog()

    lr = LinearRegression(maxIter=1)
    pipeline = Pipeline(stages=lr)
    with pytest.raises(TypeError, match="Pipeline stages should be iterable"):
        pipeline.fit(dataset_regression)


def test_autologging_handle_wrong_tuning_params(dataset_regression):
    mlflow.pyspark.ml.autolog()

    lr = LinearRegression(maxIter=1)
    lr2 = LinearRegression(maxIter=2)

    grid = ParamGridBuilder().addGrid(lr2.maxIter, [1, 2]).build()
    evaluator = RegressionEvaluator()
    cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator)

    pipeline = Pipeline(stages=[cv])

    with pytest.raises(
        ValueError, match="Tuning params should not include params not owned by the tuned estimator"
    ):
        pipeline.fit(dataset_regression)


# pylint: disable=unused-argument
@pytest.mark.large
def test_autolog_registering_model(spark_session, dataset_binomial):
    registered_model_name = "test_autolog_registered_model"
    mlflow.pyspark.ml.autolog(registered_model_name=registered_model_name)
    with mlflow.start_run():
        lr = LinearRegression()
        lr.fit(dataset_binomial)

        registered_model = MlflowClient().get_registered_model(registered_model_name)
        assert registered_model.name == registered_model_name
