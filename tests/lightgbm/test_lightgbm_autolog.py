import functools
import json
import os
import pickle
from unittest import mock
from unittest.mock import patch

import lightgbm as lgb
import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
from sklearn import datasets

import mlflow
import mlflow.lightgbm
from mlflow import MlflowClient
from mlflow.lightgbm import _autolog_callback
from mlflow.models import Model
from mlflow.types.utils import _infer_schema
from mlflow.utils.autologging_utils import BatchMetricsLogger, picklable_exception_safe_function

mpl.use("Agg")


def get_latest_run():
    client = MlflowClient()
    return client.get_run(client.search_runs(["0"])[0].info.run_id)


def get_model_conf(artifact_uri, model_subpath="model"):
    model_conf_path = os.path.join(artifact_uri, model_subpath, "MLmodel")
    return Model.load(model_conf_path)


@pytest.fixture(scope="module")
def bst_params():
    return {
        "objective": "multiclass",
        "num_class": 3,
    }


@pytest.fixture(scope="module")
def train_set():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
    y = iris.target
    # set free_raw_data False to use raw data later.
    return lgb.Dataset(X, y, free_raw_data=False)


@pytest.fixture(scope="module")
def valid_set():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data[:, :2] * 2, columns=iris.feature_names[:2])
    y = iris.target
    # set free_raw_data False to use raw data later.
    return lgb.Dataset(X, y, free_raw_data=False)


def test_lgb_autolog_ends_auto_created_run(bst_params, train_set):
    mlflow.lightgbm.autolog()
    lgb.train(bst_params, train_set, num_boost_round=1)
    assert mlflow.active_run() is None


def test_extra_tags_lightgbm_autolog(bst_params, train_set):
    mlflow.lightgbm.autolog(extra_tags={"test_tag": "lgb_autolog"})
    lgb.train(bst_params, train_set, num_boost_round=1)

    run = mlflow.last_active_run()
    assert run.data.tags["test_tag"] == "lgb_autolog"
    assert run.data.tags[mlflow.utils.mlflow_tags.MLFLOW_AUTOLOGGING] == "lightgbm"


def test_lgb_autolog_persists_manually_created_run(bst_params, train_set):
    mlflow.lightgbm.autolog()
    with mlflow.start_run() as run:
        lgb.train(bst_params, train_set, num_boost_round=1)
        assert mlflow.active_run()
        assert mlflow.active_run().info.run_id == run.info.run_id


def test_lgb_autolog_logs_default_params(bst_params, train_set):
    mlflow.lightgbm.autolog()
    lgb.train(bst_params, train_set)
    run = get_latest_run()
    params = run.data.params

    expected_params = {
        "num_boost_round": 100,
        "keep_training_booster": False,
    }
    if Version(lgb.__version__) <= Version("3.3.1"):
        # The parameter `verbose_eval` in `lightgbm.train` is removed in this PR:
        # https://github.com/microsoft/LightGBM/pull/4878
        expected_params["verbose_eval"] = (
            # The default value of `verbose_eval` in `lightgbm.train` has been changed to 'warn'
            # in this PR: https://github.com/microsoft/LightGBM/pull/4577
            "warn" if Version(lgb.__version__) > Version("3.2.1") else True
        )
    if Version(lgb.__version__) <= Version("4.5.0"):
        # these params were removed in this PR:
        # https://github.com/microsoft/LightGBM/pull/6706
        expected_params["feature_name"] = "auto"
        expected_params["categorical_feature"] = "auto"

    expected_params.update(bst_params)

    for key, val in expected_params.items():
        assert key in params
        assert params[key] == str(val)

    unlogged_params = [
        "params",
        "train_set",
        "valid_sets",
        "valid_names",
        "fobj",
        "feval",
        "init_model",
        "learning_rates",
        "callbacks",
    ]
    if Version(lgb.__version__) <= Version("3.3.1"):
        # The parameter `evals_result` in `lightgbm.train` is removed in this PR:
        # https://github.com/microsoft/LightGBM/pull/4882
        unlogged_params.append("evals_result")

    for param in unlogged_params:
        assert param not in params


def test_lgb_autolog_logs_specified_params(bst_params, train_set):
    mlflow.lightgbm.autolog()
    expected_params = {
        "num_boost_round": 10,
    }
    if Version(lgb.__version__) <= Version("3.3.1"):
        # The parameter `verbose_eval` in `lightgbm.train` is removed in this PR:
        # https://github.com/microsoft/LightGBM/pull/4878
        # The parameter `early_stopping_rounds` in `lightgbm.train` is removed in this PR:
        # https://github.com/microsoft/LightGBM/pull/4908
        expected_params["verbose_eval"] = False
        expected_params["early_stopping_rounds"] = 5
    lgb.train(bst_params, train_set, valid_sets=[train_set], **expected_params)
    run = get_latest_run()
    params = run.data.params

    expected_params.update(bst_params)

    for key, val in expected_params.items():
        assert key in params
        assert params[key] == str(val)

    unlogged_params = [
        "params",
        "train_set",
        "valid_sets",
        "valid_names",
        "fobj",
        "feval",
        "init_model",
        "learning_rates",
        "callbacks",
    ]
    if Version(lgb.__version__) <= Version("3.3.1"):
        # The parameter `evals_result` in `lightgbm.train` is removed in this PR:
        # https://github.com/microsoft/LightGBM/pull/4882
        unlogged_params.append("evals_result")

    for param in unlogged_params:
        assert param not in params


def test_lgb_autolog_sklearn():
    mlflow.lightgbm.autolog()

    X, y = datasets.load_iris(return_X_y=True)
    params = {"n_estimators": 10, "reg_lambda": 1}
    model = lgb.LGBMClassifier(**params)

    with mlflow.start_run() as run:
        model.fit(X, y)

    client = MlflowClient()
    run = client.get_run(run.info.run_id)
    assert run.data.metrics.items() <= params.items()
    artifacts = {x.path for x in client.list_artifacts(run.info.run_id)}
    assert artifacts >= {
        "feature_importance_gain.png",
        "feature_importance_gain.json",
        "feature_importance_split.png",
        "feature_importance_split.json",
    }
    logged_model = mlflow.last_logged_model()
    loaded_model = mlflow.lightgbm.load_model(logged_model.model_uri)
    np.testing.assert_allclose(loaded_model.predict(X), model.predict(X))


def test_lgb_autolog_sklearn_nested_in_pipeline():
    from sklearn.pipeline import make_pipeline

    mlflow.lightgbm.autolog()
    mlflow.sklearn.autolog()

    X, y = datasets.load_iris(return_X_y=True)
    params = {"n_estimators": 10, "reg_lambda": 1}
    model = lgb.LGBMClassifier(**params)

    model = make_pipeline(model)

    with mlflow.start_run() as run:
        model.fit(X, y)

    client = MlflowClient()
    run = client.get_run(run.info.run_id)
    # assert pipeline logged
    assert run.data.params["lgbmclassifier__reg_lambda"] == "1"
    # assert nested lgb classifier not logged
    assert "reg_lambda" not in run.data.params


def test_lgb_autolog_with_sklearn_outputs_do_not_reflect_training_dataset_mutations():
    original_lgb_classifier_fit = lgb.LGBMClassifier.fit
    original_lgb_classifier_predict = lgb.LGBMClassifier.predict

    def patched_lgb_classifier_fit(self, *args, **kwargs):
        X = args[0]
        X["TESTCOL"] = 5
        return original_lgb_classifier_fit(self, *args, **kwargs)

    def patched_lgb_classifier_predict(self, *args, **kwargs):
        X = args[0]
        X["TESTCOL"] = 5
        return original_lgb_classifier_predict(self, *args, **kwargs)

    with (
        mock.patch("lightgbm.LGBMClassifier.fit", patched_lgb_classifier_fit),
        mock.patch("lightgbm.LGBMClassifier.predict", patched_lgb_classifier_predict),
    ):
        mlflow.lightgbm.autolog(log_models=True, log_model_signatures=True, log_input_examples=True)

        X = pd.DataFrame(
            {
                "Total Volume": [64236.62, 54876.98, 118220.22],
                "Total Bags": [8696.87, 9505.56, 8145.35],
                "Small Bags": [8603.62, 9408.07, 8042.21],
                "Large Bags": [93.25, 97.49, 103.14],
                "XLarge Bags": [0.0, 0.0, 0.0],
            }
        )
        y = pd.Series([1, 0, 1])

        params = {"n_estimators": 10, "reg_lambda": 1}
        model = lgb.LGBMClassifier(**params)
        model.fit(X, y)

        logged_model = mlflow.last_logged_model()
        mlflow_model = Model.load(logged_model.model_uri)
        input_example = mlflow_model.load_input_example()
        model_signature_input_names = [inp.name for inp in mlflow_model.signature.inputs.inputs]
        assert "XLarge Bags" in model_signature_input_names
        assert "XLarge Bags" in input_example.columns
        assert "TESTCOL" not in model_signature_input_names
        assert "TESTCOL" not in input_example.columns


def test_lgb_autolog_logs_metrics_with_validation_data(bst_params, train_set):
    mlflow.lightgbm.autolog()
    evals_result = {}
    if Version(lgb.__version__) <= Version("3.3.1"):
        lgb.train(
            bst_params,
            train_set,
            num_boost_round=10,
            valid_sets=[train_set],
            valid_names=["train"],
            evals_result=evals_result,
        )
    else:
        lgb.train(
            bst_params,
            train_set,
            num_boost_round=10,
            valid_sets=[train_set],
            valid_names=["train"],
            callbacks=[lgb.record_evaluation(evals_result)],
        )
    run = get_latest_run()
    data = run.data
    client = MlflowClient()
    metric_key = "train-multi_logloss"
    metric_history = [x.value for x in client.get_metric_history(run.info.run_id, metric_key)]
    assert metric_key in data.metrics
    assert len(metric_history) == 10
    assert metric_history == evals_result["train"]["multi_logloss"]
    logged_model = mlflow.last_logged_model()
    assert {m.key: m.value for m in logged_model.metrics} == data.metrics


def test_lgb_autolog_logs_metrics_with_multi_validation_data(bst_params, train_set):
    mlflow.lightgbm.autolog()
    evals_result = {}
    # If we use [train_set, train_set] here, LightGBM ignores the first dataset.
    # To avoid that, create a new Dataset object.
    valid_sets = [train_set, lgb.Dataset(train_set.data)]
    valid_names = ["train", "valid"]
    if Version(lgb.__version__) <= Version("3.3.1"):
        lgb.train(
            bst_params,
            train_set,
            num_boost_round=10,
            valid_sets=valid_sets,
            valid_names=valid_names,
            evals_result=evals_result,
        )
    else:
        lgb.train(
            bst_params,
            train_set,
            num_boost_round=10,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[lgb.record_evaluation(evals_result)],
        )
    run = get_latest_run()
    data = run.data
    client = MlflowClient()
    for valid_name in valid_names:
        metric_key = f"{valid_name}-multi_logloss"
        metric_history = [x.value for x in client.get_metric_history(run.info.run_id, metric_key)]
        assert metric_key in data.metrics
        assert len(metric_history) == 10
        assert metric_history == evals_result[valid_name]["multi_logloss"]
    logged_model = mlflow.last_logged_model()
    assert {m.key: m.value for m in logged_model.metrics} == data.metrics


def test_lgb_autolog_logs_metrics_with_multi_metrics(bst_params, train_set):
    mlflow.lightgbm.autolog()
    evals_result = {}
    params = {"metric": ["multi_error", "multi_logloss"]}
    params.update(bst_params)
    valid_sets = [train_set]
    valid_names = ["train"]
    if Version(lgb.__version__) <= Version("3.3.1"):
        lgb.train(
            params,
            train_set,
            num_boost_round=10,
            valid_sets=valid_sets,
            valid_names=valid_names,
            evals_result=evals_result,
        )
    else:
        lgb.train(
            params,
            train_set,
            num_boost_round=10,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[lgb.record_evaluation(evals_result)],
        )
    run = get_latest_run()
    data = run.data
    client = MlflowClient()
    for metric_name in params["metric"]:
        metric_key = f"{valid_names[0]}-{metric_name}"
        metric_history = [x.value for x in client.get_metric_history(run.info.run_id, metric_key)]
        assert metric_key in data.metrics
        assert len(metric_history) == 10
        assert metric_history == evals_result["train"][metric_name]
    logged_model = mlflow.last_logged_model()
    assert {m.key: m.value for m in logged_model.metrics} == data.metrics


def test_lgb_autolog_logs_metrics_with_multi_validation_data_and_metrics(bst_params, train_set):
    mlflow.lightgbm.autolog()
    evals_result = {}
    params = {"metric": ["multi_error", "multi_logloss"]}
    params.update(bst_params)
    valid_sets = [train_set, lgb.Dataset(train_set.data)]
    valid_names = ["train", "valid"]
    if Version(lgb.__version__) <= Version("3.3.1"):
        lgb.train(
            params,
            train_set,
            num_boost_round=10,
            valid_sets=valid_sets,
            valid_names=valid_names,
            evals_result=evals_result,
        )
    else:
        lgb.train(
            params,
            train_set,
            num_boost_round=10,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[lgb.record_evaluation(evals_result)],
        )
    run = get_latest_run()
    data = run.data
    client = MlflowClient()
    for valid_name in valid_names:
        for metric_name in params["metric"]:
            metric_key = f"{valid_name}-{metric_name}"
            metric_history = [
                x.value for x in client.get_metric_history(run.info.run_id, metric_key)
            ]
            assert metric_key in data.metrics
            assert len(metric_history) == 10
            assert metric_history == evals_result[valid_name][metric_name]
    logged_model = mlflow.last_logged_model()
    assert {m.key: m.value for m in logged_model.metrics} == data.metrics


def test_lgb_autolog_batch_metrics_logger_logs_expected_metrics(bst_params, train_set):
    patched_metrics_data = []

    # Mock patching BatchMetricsLogger.record_metrics()
    # to ensure that expected metrics are being logged.
    original = BatchMetricsLogger.record_metrics

    with patch(
        "mlflow.utils.autologging_utils.BatchMetricsLogger.record_metrics", autospec=True
    ) as record_metrics_mock:

        def record_metrics_side_effect(self, metrics, step=None):
            patched_metrics_data.extend(metrics.items())
            original(self, metrics, step)

        record_metrics_mock.side_effect = record_metrics_side_effect

        mlflow.lightgbm.autolog()
        evals_result = {}
        params = {"metric": ["multi_error", "multi_logloss"]}
        params.update(bst_params)
        valid_sets = [train_set, lgb.Dataset(train_set.data)]
        valid_names = ["train", "valid"]
        if Version(lgb.__version__) <= Version("3.3.1"):
            lgb.train(
                params,
                train_set,
                num_boost_round=10,
                valid_sets=valid_sets,
                valid_names=valid_names,
                evals_result=evals_result,
            )
        else:
            lgb.train(
                params,
                train_set,
                num_boost_round=10,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=[lgb.record_evaluation(evals_result)],
            )

    run = get_latest_run()
    original_metrics = run.data.metrics
    patched_metrics_data = dict(patched_metrics_data)
    for metric_name in original_metrics:
        assert metric_name in patched_metrics_data
        assert original_metrics[metric_name] == patched_metrics_data[metric_name]

    assert "train-multi_logloss" in original_metrics
    assert "train-multi_logloss" in patched_metrics_data


def ranking_dataset(num_rows=100, num_queries=10):
    # https://stackoverflow.com/a/67621253
    num_rows_per_query = num_rows // num_queries
    df = pd.DataFrame(
        {
            "query_id": [i for i in range(num_queries) for _ in range(num_rows_per_query)],
            "f1": np.random.random(size=(num_rows,)),
            "f2": np.random.random(size=(num_rows,)),
            "relevance": np.random.randint(2, size=(num_rows,)),
        }
    )
    train_size = int(num_rows * 0.75)
    df_train = df[:train_size]
    df_valid = df[train_size:]
    # Train
    group_train = df_train.groupby("query_id")["query_id"].count().to_numpy()
    X_train = df_train.drop(["query_id", "relevance"], axis=1)
    y_train = df_train["relevance"]
    train_set = lgb.Dataset(X_train, y_train, group=group_train)
    # Validation
    group_val = df_valid.groupby("query_id")["query_id"].count().to_numpy()
    X_valid = df_valid.drop(["query_id", "relevance"], axis=1)
    y_valid = df_valid["relevance"]
    valid_set = lgb.Dataset(X_valid, y_valid, group=group_val)
    return train_set, valid_set


def test_lgb_autolog_atsign_metrics(train_set):
    train_set, valid_set = ranking_dataset()
    mlflow.lightgbm.autolog()
    params = {"objective": "regression", "metric": ["map"], "eval_at": [1]}
    lgb.train(
        params,
        train_set,
        valid_sets=[valid_set],
        valid_names=["valid"],
        num_boost_round=5,
    )
    run = get_latest_run()
    assert set(run.data.metrics) == {"valid-map_at_1"}


@pytest.mark.parametrize("log_models", [True, False])
def test_lgb_autolog_logs_metrics_with_early_stopping(bst_params, train_set, log_models):
    mlflow.lightgbm.autolog(log_models=log_models)
    evals_result = {}
    params = {"metric": ["multi_error", "multi_logloss"]}
    params.update(bst_params)
    valid_sets = [train_set, lgb.Dataset(train_set.data)]
    valid_names = ["train", "valid"]
    if Version(lgb.__version__) <= Version("3.3.1"):
        model = lgb.train(
            params,
            train_set,
            num_boost_round=10,
            early_stopping_rounds=5,
            valid_sets=valid_sets,
            valid_names=valid_names,
            evals_result=evals_result,
        )
    else:
        model = lgb.train(
            params,
            train_set,
            num_boost_round=10,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.record_evaluation(evals_result),
                lgb.early_stopping(5),
            ],
        )
    run = get_latest_run()
    data = run.data
    client = MlflowClient()
    assert "best_iteration" in data.metrics
    assert int(data.metrics["best_iteration"]) == model.best_iteration
    assert "stopped_iteration" in data.metrics
    assert int(data.metrics["stopped_iteration"]) == len(evals_result["train"]["multi_logloss"])

    logged_model = mlflow.last_logged_model()
    logged_model_metrics = {m.key: m.value for m in logged_model.metrics} if log_models else {}
    if log_models:
        assert data.metrics.items() <= logged_model_metrics.items()
    for valid_name in valid_names:
        for metric_name in params["metric"]:
            metric_key = f"{valid_name}-{metric_name}"
            metric_history = [
                x.value for x in client.get_metric_history(run.info.run_id, metric_key)
            ]
            assert metric_key in data.metrics

            best_metrics = evals_result[valid_name][metric_name][model.best_iteration - 1]
            assert metric_history == evals_result[valid_name][metric_name] + [best_metrics]
            if log_models:
                assert logged_model_metrics[metric_key] == best_metrics


def test_lgb_autolog_logs_feature_importance(bst_params, train_set):
    mlflow.lightgbm.autolog()
    model = lgb.train(bst_params, train_set, num_boost_round=10)
    run = get_latest_run()
    run_id = run.info.run_id
    artifacts_dir = run.info.artifact_uri.replace("file://", "")
    client = MlflowClient()
    artifacts = [x.path for x in client.list_artifacts(run_id)]

    for imp_type in ["split", "gain"]:
        plot_name = f"feature_importance_{imp_type}.png"
        assert plot_name in artifacts

        json_name = f"feature_importance_{imp_type}.json"
        assert json_name in artifacts

        json_path = os.path.join(artifacts_dir, json_name)
        with open(json_path) as f:
            loaded_imp = json.load(f)

        features = model.feature_name()
        importance = model.feature_importance(importance_type=imp_type)
        imp = dict(zip(features, importance.tolist()))

        assert loaded_imp == imp


def test_no_figure_is_opened_after_logging(bst_params, train_set):
    mlflow.lightgbm.autolog()
    lgb.train(bst_params, train_set, num_boost_round=10)
    assert mpl.pyplot.get_fignums() == []


def test_lgb_autolog_loads_model_from_artifact(bst_params, train_set):
    mlflow.lightgbm.autolog()
    model = lgb.train(bst_params, train_set, num_boost_round=10)
    run = get_latest_run()
    run_id = run.info.run_id

    loaded_model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
    np.testing.assert_array_almost_equal(
        model.predict(train_set.data), loaded_model.predict(train_set.data)
    )


def test_lgb_autolog_gets_input_example(bst_params):
    # we need to check the example input against the initial input given to train function.
    # we can't use the train_set fixture for this as it defines free_raw_data=False but this
    # feature should work even if it is True
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
    y = iris.target
    mlflow.lightgbm.autolog(log_input_examples=True)
    dataset = lgb.Dataset(X, y, free_raw_data=True)
    lgb.train(bst_params, dataset)

    logged_model = mlflow.last_logged_model()
    mlflow_model = Model.load(logged_model.model_uri)
    input_example = mlflow_model.load_input_example()

    pd.testing.assert_frame_equal(input_example, (X[:5]))

    pyfunc_model = mlflow.pyfunc.load_model(logged_model.model_uri)
    # make sure reloading the input_example and predicting on it does not error
    pyfunc_model.predict(input_example)


def test_lgb_autolog_infers_model_signature_correctly(bst_params):
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
    y = iris.target
    mlflow.lightgbm.autolog(log_model_signatures=True)
    dataset = lgb.Dataset(X, y, free_raw_data=True)
    lgb.train(bst_params, dataset)

    logged_model = mlflow.last_logged_model()
    model_info = mlflow.models.get_model_info(logged_model.model_uri)
    signature = model_info.signature
    assert signature.inputs == _infer_schema(X)
    assert signature.outputs.to_dict() == [
        {"type": "tensor", "tensor-spec": {"dtype": "float64", "shape": (-1, 3)}},
    ]


def test_lgb_autolog_continues_logging_even_if_signature_inference_fails(tmp_path):
    tmp_csv = tmp_path.joinpath("data.csv")
    with tmp_csv.open("w") as f:
        f.write("2,6.4,2.8,5.6,2.2\n")
        f.write("1,5.0,2.3,3.3,1.0\n")
        f.write("2,4.9,2.5,4.5,1.7\n")
        f.write("0,4.9,3.1,1.5,0.1\n")
        f.write("0,5.7,3.8,1.7,0.3\n")

    # signature and input example inference should fail here since the dataset is given
    #   as a file path
    dataset = lgb.Dataset(str(tmp_csv))

    bst_params = {
        "objective": "multiclass",
        "num_class": 3,
    }

    mlflow.lightgbm.autolog(log_model_signatures=True)
    lgb.train(bst_params, dataset)
    logged_model = mlflow.last_logged_model()
    assert logged_model is not None
    model_info = mlflow.models.get_model_info(logged_model.model_uri)
    assert model_info.run_id is not None
    assert model_info.signature is None


@pytest.mark.parametrize("log_input_examples", [True, False])
@pytest.mark.parametrize("log_model_signatures", [True, False])
def test_lgb_autolog_configuration_options(bst_params, log_input_examples, log_model_signatures):
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
    y = iris.target

    with mlflow.start_run():
        mlflow.lightgbm.autolog(
            log_input_examples=log_input_examples, log_model_signatures=log_model_signatures
        )
        dataset = lgb.Dataset(X, y)
        lgb.train(bst_params, dataset)
    logged_model = mlflow.last_logged_model()
    mlflow_model = Model.load(logged_model.model_uri)
    assert (mlflow_model.load_input_example() is not None) == log_input_examples
    assert (mlflow_model.signature is not None) == log_model_signatures


@pytest.mark.parametrize("log_models", [True, False])
def test_lgb_autolog_log_models_configuration(bst_params, log_models):
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
    y = iris.target

    with mlflow.start_run():
        mlflow.lightgbm.autolog(log_models=log_models)
        dataset = lgb.Dataset(X, y)
        lgb.train(bst_params, dataset)

    logged_model = mlflow.last_logged_model()
    assert (logged_model is not None) == log_models


def test_lgb_autolog_does_not_break_dataset_instantiation_with_data_none():
    """
    This test verifies that `lightgbm.Dataset(None)` doesn't fail after patching.
    LightGBM internally calls `lightgbm.Dataset(None)` to create a subset of `Dataset`:
    https://github.com/microsoft/LightGBM/blob/v3.0.0/python-package/lightgbm/basic.py#L1381
    """
    mlflow.lightgbm.autolog()
    lgb.Dataset(None)


def test_callback_func_is_pickable():
    cb = picklable_exception_safe_function(
        functools.partial(_autolog_callback, BatchMetricsLogger(run_id="1234"), eval_results={})
    )
    pickle.dumps(cb)


def test_sklearn_api_autolog_registering_model():
    registered_model_name = "test_autolog_registered_model"
    mlflow.lightgbm.autolog(registered_model_name=registered_model_name)

    X, y = datasets.load_iris(return_X_y=True)
    params = {"n_estimators": 10, "reg_lambda": 1}
    model = lgb.LGBMClassifier(**params)

    with mlflow.start_run():
        model.fit(X, y)

        registered_model = MlflowClient().get_registered_model(registered_model_name)
        assert registered_model.name == registered_model_name


def test_lgb_api_autolog_registering_model(bst_params, train_set):
    registered_model_name = "test_autolog_registered_model"
    mlflow.lightgbm.autolog(registered_model_name=registered_model_name)

    with mlflow.start_run():
        lgb.train(bst_params, train_set, num_boost_round=1)

        registered_model = MlflowClient().get_registered_model(registered_model_name)
        assert registered_model.name == registered_model_name


@pytest.mark.parametrize("log_datasets", [True, False])
def test_lgb_log_datasets(bst_params, train_set, log_datasets):
    with mlflow.start_run() as run:
        mlflow.lightgbm.autolog(log_datasets=log_datasets)
        lgb.train(bst_params, train_set, num_boost_round=1)

    run_id = run.info.run_id
    client = MlflowClient()
    dataset_inputs = client.get_run(run_id).inputs.dataset_inputs
    if log_datasets:
        assert len(dataset_inputs) == 1
        assert dataset_inputs[0].dataset.schema == json.dumps(
            {"mlflow_colspec": _infer_schema(train_set.data).to_dict()}
        )
    else:
        assert len(dataset_inputs) == 0


def test_lgb_log_datasets_with_valid_set(bst_params, train_set, valid_set):
    with mlflow.start_run() as run:
        mlflow.lightgbm.autolog(log_datasets=True)
        lgb.train(bst_params, train_set, valid_sets=[valid_set], num_boost_round=1)

    run_id = run.info.run_id
    client = MlflowClient()
    dataset_inputs = client.get_run(run_id).inputs.dataset_inputs
    assert len(dataset_inputs) == 2
    assert dataset_inputs[0].tags[0].value == "train"
    assert dataset_inputs[0].dataset.schema == json.dumps(
        {"mlflow_colspec": _infer_schema(train_set.data).to_dict()}
    )
    assert dataset_inputs[1].tags[0].value == "eval"
    assert dataset_inputs[1].dataset.schema == json.dumps(
        {"mlflow_colspec": _infer_schema(valid_set.data).to_dict()}
    )


def test_lgb_log_datasets_with_valid_set_with_name(bst_params, train_set, valid_set):
    with mlflow.start_run() as run:
        mlflow.lightgbm.autolog(log_datasets=True)
        lgb.train(
            bst_params,
            train_set,
            valid_sets=[valid_set],
            valid_names=["my_valid_set"],
            num_boost_round=1,
        )

    run_id = run.info.run_id
    client = MlflowClient()
    dataset_inputs = client.get_run(run_id).inputs.dataset_inputs
    assert len(dataset_inputs) == 2
    assert dataset_inputs[0].tags[0].value == "train"
    assert dataset_inputs[1].tags[0].value == "eval"
    assert dataset_inputs[1].dataset.name == "my_valid_set"
