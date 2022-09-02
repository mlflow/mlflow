from packaging.version import Version
import os
import json
import functools
import pickle
import pytest
import numpy as np
import pandas as pd
from sklearn import datasets
import xgboost as xgb
import matplotlib as mpl
import yaml
from unittest import mock

import mlflow
import mlflow.xgboost
from mlflow.xgboost._autolog import IS_TRAINING_CALLBACK_SUPPORTED, autolog_callback
from mlflow.models import Model
from mlflow.models.utils import _read_example
from mlflow import MlflowClient
from mlflow.utils.autologging_utils import BatchMetricsLogger, picklable_exception_safe_function

mpl.use("Agg")


def get_latest_run():
    client = MlflowClient()
    return client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)


def get_model_conf(artifact_uri, model_subpath="model"):
    model_conf_path = os.path.join(artifact_uri, model_subpath, "MLmodel")
    return Model.load(model_conf_path)


@pytest.fixture(scope="module")
def bst_params():
    return {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
    }


@pytest.fixture(scope="module")
def dtrain():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
    y = iris.target
    return xgb.DMatrix(X, y)


def test_xgb_autolog_ends_auto_created_run(bst_params, dtrain):
    mlflow.xgboost.autolog()
    xgb.train(bst_params, dtrain)
    assert mlflow.active_run() is None


def test_xgb_autolog_persists_manually_created_run(bst_params, dtrain):
    mlflow.xgboost.autolog()
    with mlflow.start_run() as run:
        xgb.train(bst_params, dtrain)
        assert mlflow.active_run()
        assert mlflow.active_run().info.run_id == run.info.run_id


def test_xgb_autolog_logs_default_params(bst_params, dtrain):
    mlflow.xgboost.autolog()
    xgb.train(bst_params, dtrain)
    run = get_latest_run()
    params = run.data.params

    expected_params = {
        "num_boost_round": 10,
        # In xgboost >= 1.3.0, the default value for `maximize` in `xgboost.train` is None:
        #   https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.train
        # In < 1.3.0, it's False:
        #   https://xgboost.readthedocs.io/en/release_1.2.0/python/python_api.html#xgboost.train
        # TODO: Remove `replace("SNAPSHOT", "dev")` once the following issue is addressed:
        #       https://github.com/dmlc/xgboost/issues/6984
        "maximize": (
            None
            if Version(xgb.__version__.replace("SNAPSHOT", "dev")) >= Version("1.3.0")
            else False
        ),
        "early_stopping_rounds": None,
        "verbose_eval": True,
    }
    expected_params.update(bst_params)

    for key, val in expected_params.items():
        assert key in params
        assert params[key] == str(val)

    unlogged_params = [
        "dtrain",
        "evals",
        "obj",
        "feval",
        "evals_result",
        "xgb_model",
        "callbacks",
        "learning_rates",
    ]

    for param in unlogged_params:
        assert param not in params


def test_xgb_autolog_logs_specified_params(bst_params, dtrain):
    mlflow.xgboost.autolog()
    expected_params = {
        "num_boost_round": 20,
        "early_stopping_rounds": 5,
        "verbose_eval": False,
    }
    xgb.train(bst_params, dtrain, evals=[(dtrain, "train")], **expected_params)
    run = get_latest_run()
    params = run.data.params

    expected_params.update(bst_params)

    for key, val in expected_params.items():
        assert key in params
        assert params[key] == str(val)

    unlogged_params = [
        "dtrain",
        "evals",
        "obj",
        "feval",
        "evals_result",
        "xgb_model",
        "callbacks",
        "learning_rates",
    ]

    for param in unlogged_params:
        assert param not in params


def test_xgb_autolog_atsign_metrics():
    mlflow.xgboost.autolog()
    xgb_metrics = ["ndcg@2", "map@3-", "error@0.4"]
    expected_metrics = {"train-ndcg_at_2", "train-map_at_3-", "train-error_at_0.4"}

    params = {"objective": "rank:pairwise", "eval_metric": xgb_metrics}
    dtrain = xgb.DMatrix(np.array([[0], [1]]), label=[1, 0])
    xgb.train(params, dtrain, evals=[(dtrain, "train")], num_boost_round=1)
    run = get_latest_run()
    assert set(run.data.metrics) == expected_metrics


@pytest.mark.parametrize("xgb_metric", ["ndcg@2", "error"])
def test_xgb_autolog_atsign_metrics_info_log(xgb_metric):
    mlflow.xgboost.autolog()

    with mock.patch("mlflow.xgboost._autolog._logger.info") as mock_info_log:
        params = {"objective": "rank:pairwise", "eval_metric": [xgb_metric, "map"]}
        dtrain = xgb.DMatrix(np.array([[0], [1]]), label=[1, 0])
        xgb.train(params, dtrain, evals=[(dtrain, "train")], num_boost_round=1)

    if "@" in xgb_metric:
        mock_info_log.assert_called_once()
        (
            first_pos_arg,
            second_pos_arg,
        ) = mock_info_log.call_args[0]
        assert "metric names have been sanitized" in first_pos_arg
        assert xgb_metric.replace("@", "_at_") in second_pos_arg
        assert "map" not in second_pos_arg
    else:
        mock_info_log.assert_not_called()


def test_xgb_autolog_sklearn():

    mlflow.xgboost.autolog()

    X, y = datasets.load_iris(return_X_y=True)
    params = {"n_estimators": 10, "reg_lambda": 1}
    model = xgb.XGBRegressor(**params)

    with mlflow.start_run() as run:
        model.fit(X, y)
        model_uri = mlflow.get_artifact_uri("model")

    client = MlflowClient()
    run = client.get_run(run.info.run_id)
    assert run.data.metrics.items() <= params.items()
    artifacts = {x.path for x in client.list_artifacts(run.info.run_id)}
    assert artifacts >= {"feature_importance_weight.png", "feature_importance_weight.json"}
    loaded_model = mlflow.xgboost.load_model(model_uri)
    np.testing.assert_allclose(loaded_model.predict(X), model.predict(X))


def test_xgb_autolog_with_sklearn_outputs_do_not_reflect_training_dataset_mutations():
    original_xgb_regressor_fit = xgb.XGBRegressor.fit
    original_xgb_regressor_predict = xgb.XGBRegressor.predict

    def patched_xgb_regressor_fit(self, *args, **kwargs):
        X = args[0]
        X["TESTCOL"] = 5
        return original_xgb_regressor_fit(self, *args, **kwargs)

    def patched_xgb_regressor_predict(self, *args, **kwargs):
        X = args[0]
        X["TESTCOL"] = 5
        return original_xgb_regressor_predict(self, *args, **kwargs)

    with mock.patch("xgboost.XGBRegressor.fit", patched_xgb_regressor_fit), mock.patch(
        "xgboost.XGBRegressor.predict", patched_xgb_regressor_predict
    ):
        xgb.XGBRegressor.fit = patched_xgb_regressor_fit
        xgb.XGBRegressor.predict = patched_xgb_regressor_predict

        mlflow.xgboost.autolog(log_models=True, log_model_signatures=True, log_input_examples=True)

        X = pd.DataFrame(
            {
                "Total Volume": [64236.62, 54876.98, 118220.22],
                "Total Bags": [8696.87, 9505.56, 8145.35],
                "Small Bags": [8603.62, 9408.07, 8042.21],
                "Large Bags": [93.25, 97.49, 103.14],
                "XLarge Bags": [0.0, 0.0, 0.0],
            }
        )
        y = pd.Series([1.33, 1.35, 0.93])

        params = {"n_estimators": 10, "reg_lambda": 1}
        model = xgb.XGBRegressor(**params)
        model.fit(X, y)

        run_artifact_uri = mlflow.last_active_run().info.artifact_uri
        model_conf = get_model_conf(run_artifact_uri)
        input_example = pd.read_json(
            os.path.join(run_artifact_uri, "model", "input_example.json"), orient="split"
        )
        model_signature_input_names = [inp.name for inp in model_conf.signature.inputs.inputs]
        assert "XLarge Bags" in model_signature_input_names
        assert "XLarge Bags" in input_example.columns
        assert "TESTCOL" not in model_signature_input_names
        assert "TESTCOL" not in input_example.columns


def test_xgb_autolog_logs_metrics_with_validation_data(bst_params, dtrain):
    mlflow.xgboost.autolog()
    evals_result = {}
    xgb.train(
        bst_params, dtrain, num_boost_round=20, evals=[(dtrain, "train")], evals_result=evals_result
    )
    run = get_latest_run()
    data = run.data
    metric_key = "train-mlogloss"
    client = MlflowClient()
    metric_history = [x.value for x in client.get_metric_history(run.info.run_id, metric_key)]
    assert metric_key in data.metrics
    assert len(metric_history) == 20
    assert metric_history == evals_result["train"]["mlogloss"]


def test_xgb_autolog_logs_metrics_with_multi_validation_data(bst_params, dtrain):
    mlflow.xgboost.autolog()
    evals_result = {}
    evals = [(dtrain, "train"), (dtrain, "valid")]
    xgb.train(bst_params, dtrain, num_boost_round=20, evals=evals, evals_result=evals_result)
    run = get_latest_run()
    data = run.data
    client = MlflowClient()
    for eval_name in [e[1] for e in evals]:
        metric_key = "{}-mlogloss".format(eval_name)
        metric_history = [x.value for x in client.get_metric_history(run.info.run_id, metric_key)]
        assert metric_key in data.metrics
        assert len(metric_history) == 20
        assert metric_history == evals_result[eval_name]["mlogloss"]


def test_xgb_autolog_logs_metrics_with_multi_metrics(bst_params, dtrain):
    mlflow.xgboost.autolog()
    evals_result = {}
    params = {**bst_params, "eval_metric": ["merror", "mlogloss"]}
    xgb.train(
        params, dtrain, num_boost_round=20, evals=[(dtrain, "train")], evals_result=evals_result
    )
    run = get_latest_run()
    data = run.data
    client = MlflowClient()
    for metric_name in params["eval_metric"]:
        metric_key = "train-{}".format(metric_name)
        metric_history = [x.value for x in client.get_metric_history(run.info.run_id, metric_key)]
        assert metric_key in data.metrics
        assert len(metric_history) == 20
        assert metric_history == evals_result["train"][metric_name]


def test_xgb_autolog_logs_metrics_with_multi_validation_data_and_metrics(bst_params, dtrain):
    mlflow.xgboost.autolog()
    evals_result = {}
    params = {**bst_params, "eval_metric": ["merror", "mlogloss"]}
    evals = [(dtrain, "train"), (dtrain, "valid")]
    xgb.train(params, dtrain, num_boost_round=20, evals=evals, evals_result=evals_result)
    run = get_latest_run()
    data = run.data
    client = MlflowClient()
    for eval_name in [e[1] for e in evals]:
        for metric_name in params["eval_metric"]:
            metric_key = "{}-{}".format(eval_name, metric_name)
            metric_history = [
                x.value for x in client.get_metric_history(run.info.run_id, metric_key)
            ]
            assert metric_key in data.metrics
            assert len(metric_history) == 20
            assert metric_history == evals_result[eval_name][metric_name]


def test_xgb_autolog_logs_metrics_with_early_stopping(bst_params, dtrain):
    mlflow.xgboost.autolog()
    evals_result = {}
    params = {**bst_params, "eval_metric": ["merror", "mlogloss"]}
    evals = [(dtrain, "train"), (dtrain, "valid")]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=20,
        early_stopping_rounds=5,
        evals=evals,
        evals_result=evals_result,
    )
    run = get_latest_run()
    data = run.data

    assert "best_iteration" in data.metrics
    assert int(data.metrics["best_iteration"]) == model.best_iteration
    assert "stopped_iteration" in data.metrics
    assert int(data.metrics["stopped_iteration"]) == len(evals_result["train"]["merror"]) - 1
    client = MlflowClient()

    for eval_name in [e[1] for e in evals]:
        for metric_name in params["eval_metric"]:
            metric_key = "{}-{}".format(eval_name, metric_name)
            metric_history = [
                x.value for x in client.get_metric_history(run.info.run_id, metric_key)
            ]
            assert metric_key in data.metrics
            assert len(metric_history) == 20 + 1

            best_metrics = evals_result[eval_name][metric_name][model.best_iteration]
            assert metric_history == evals_result[eval_name][metric_name] + [best_metrics]


def test_xgb_autolog_logs_feature_importance(bst_params, dtrain):
    mlflow.xgboost.autolog()
    model = xgb.train(bst_params, dtrain)
    run = get_latest_run()
    run_id = run.info.run_id
    artifacts_dir = run.info.artifact_uri.replace("file://", "")
    client = MlflowClient()
    artifacts = [x.path for x in client.list_artifacts(run_id)]

    importance_type = "weight"
    plot_name = "feature_importance_{}.png".format(importance_type)
    assert plot_name in artifacts

    json_name = "feature_importance_{}.json".format(importance_type)
    assert json_name in artifacts

    json_path = os.path.join(artifacts_dir, json_name)
    with open(json_path, "r") as f:
        loaded_imp = json.load(f)

    assert loaded_imp == model.get_score(importance_type=importance_type)


def test_xgb_autolog_logs_specified_feature_importance(bst_params, dtrain):
    importance_types = ["weight", "total_gain"]
    mlflow.xgboost.autolog(importance_types=importance_types)
    model = xgb.train(bst_params, dtrain)
    run = get_latest_run()
    run_id = run.info.run_id
    artifacts_dir = run.info.artifact_uri.replace("file://", "")
    client = MlflowClient()
    artifacts = [x.path for x in client.list_artifacts(run_id)]

    for imp_type in importance_types:
        plot_name = "feature_importance_{}.png".format(imp_type)
        assert plot_name in artifacts

        json_name = "feature_importance_{}.json".format(imp_type)
        assert json_name in artifacts

        json_path = os.path.join(artifacts_dir, json_name)
        with open(json_path, "r") as f:
            loaded_imp = json.load(f)

        assert loaded_imp == model.get_score(importance_type=imp_type)


@pytest.mark.skipif(
    Version(xgb.__version__) <= Version("1.4.2"),
    reason=(
        "In XGBoost <= 1.4.2, linear boosters do not support `get_score()` for importance value"
        " creation."
    ),
)
def test_xgb_autolog_logs_feature_importance_for_linear_boosters(dtrain):
    mlflow.xgboost.autolog()

    bst_params = {"objective": "multi:softprob", "num_class": 3, "booster": "gblinear"}
    model = xgb.train(bst_params, dtrain)

    run = get_latest_run()
    run_id = run.info.run_id
    artifacts_dir = run.info.artifact_uri.replace("file://", "")
    client = MlflowClient()
    artifacts = [x.path for x in client.list_artifacts(run_id)]

    importance_type = "weight"
    plot_name = "feature_importance_{}.png".format(importance_type)
    assert plot_name in artifacts

    json_name = "feature_importance_{}.json".format(importance_type)
    assert json_name in artifacts

    json_path = os.path.join(artifacts_dir, json_name)
    with open(json_path, "r") as f:
        loaded_imp = json.load(f)

    assert loaded_imp == model.get_score(importance_type=importance_type)


def test_no_figure_is_opened_after_logging(bst_params, dtrain):
    mlflow.xgboost.autolog()
    xgb.train(bst_params, dtrain)
    assert mpl.pyplot.get_fignums() == []


def test_xgb_autolog_loads_model_from_artifact(bst_params, dtrain):
    mlflow.xgboost.autolog()
    model = xgb.train(bst_params, dtrain)
    run = get_latest_run()
    run_id = run.info.run_id

    loaded_model = mlflow.xgboost.load_model("runs:/{}/model".format(run_id))
    np.testing.assert_array_almost_equal(model.predict(dtrain), loaded_model.predict(dtrain))


@pytest.mark.skipif(
    Version(xgb.__version__) > Version("1.4.2"),
    reason=(
        "In XGBoost <= 1.4.2, linear boosters do not support `get_score()` for importance value"
        " creation. In XGBoost > 1.4.2, all boosters support `get_score()`."
    ),
)
def test_xgb_autolog_does_not_throw_if_importance_values_not_supported(dtrain):
    # the gblinear booster does not support calling get_score on it,
    #   where get_score is used to create the importance values plot.
    bst_params = {"objective": "multi:softprob", "num_class": 3, "booster": "gblinear"}

    mlflow.xgboost.autolog()

    # we make sure here that we do not throw while attempting to plot
    #   importance values on a model with a linear booster.
    model = xgb.train(bst_params, dtrain)

    with pytest.raises(ValueError, match="Feature importance is not defined"):
        model.get_score(importance_type="weight")


def test_xgb_autolog_gets_input_example(bst_params):
    mlflow.xgboost.autolog(log_input_examples=True)

    # we cannot use dtrain fixture, as the dataset must be constructed
    #   after the call to autolog() in order to get the input example
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
    y = iris.target
    dataset = xgb.DMatrix(X, y)

    xgb.train(bst_params, dataset)
    run = get_latest_run()

    model_path = os.path.join(run.info.artifact_uri, "model")
    model_conf = Model.load(os.path.join(model_path, "MLmodel"))

    input_example = _read_example(model_conf, model_path)

    pd.testing.assert_frame_equal(input_example, X[:5])

    pyfunc_model = mlflow.pyfunc.load_model(os.path.join(run.info.artifact_uri, "model"))

    # make sure reloading the input_example and predicting on it does not error
    pyfunc_model.predict(input_example)


def test_xgb_autolog_infers_model_signature_correctly(bst_params):
    mlflow.xgboost.autolog(log_model_signatures=True)

    # we cannot use dtrain fixture, as the dataset must be constructed
    #   after the call to autolog() in order to get the input example
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
    y = iris.target
    dataset = xgb.DMatrix(X, y)

    xgb.train(bst_params, dataset)
    run = get_latest_run()
    run_id = run.info.run_id
    artifacts_dir = run.info.artifact_uri.replace("file://", "")
    client = MlflowClient()
    artifacts = [x.path for x in client.list_artifacts(run_id, "model")]

    ml_model_filename = "MLmodel"
    assert str(os.path.join("model", ml_model_filename)) in artifacts
    ml_model_path = os.path.join(artifacts_dir, "model", ml_model_filename)

    data = None
    with open(ml_model_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    assert data is not None
    assert "signature" in data
    signature = data["signature"]
    assert signature is not None

    assert "inputs" in signature
    assert json.loads(signature["inputs"]) == [
        {"name": "sepal length (cm)", "type": "double"},
        {"name": "sepal width (cm)", "type": "double"},
    ]

    assert "outputs" in signature
    assert json.loads(signature["outputs"]) == [
        {"type": "tensor", "tensor-spec": {"dtype": "float32", "shape": [-1, 3]}},
    ]


def test_xgb_autolog_does_not_throw_if_importance_values_are_empty(bst_params, tmpdir):
    tmp_csv = tmpdir.join("data.csv")
    tmp_csv.write("1,0.3,1.2\n")
    tmp_csv.write("0,2.4,5.2\n")
    tmp_csv.write("1,0.3,-1.2\n")

    mlflow.xgboost.autolog()

    dataset = xgb.DMatrix(tmp_csv.strpath + "?format=csv&label_column=0")

    # we make sure here that we do not throw while attempting to plot
    #   importance values on a dataset that returns no importance values.
    model = xgb.train(bst_params, dataset)

    assert model.get_score(importance_type="weight") == {}


def test_xgb_autolog_continues_logging_even_if_signature_inference_fails(bst_params, tmpdir):
    tmp_csv = tmpdir.join("data.csv")
    tmp_csv.write("1,0.3,1.2\n")
    tmp_csv.write("0,2.4,5.2\n")
    tmp_csv.write("1,0.3,-1.2\n")

    mlflow.xgboost.autolog(importance_types=[], log_model_signatures=True)

    # signature and input example inference should fail here since the dataset is given
    #   as a file path
    dataset = xgb.DMatrix(tmp_csv.strpath + "?format=csv&label_column=0")

    xgb.train(bst_params, dataset)
    run = get_latest_run()
    run_id = run.info.run_id
    artifacts_dir = run.info.artifact_uri.replace("file://", "")
    client = MlflowClient()
    artifacts = [x.path for x in client.list_artifacts(run_id, "model")]

    ml_model_filename = "MLmodel"
    assert os.path.join("model", ml_model_filename) in artifacts
    ml_model_path = os.path.join(artifacts_dir, "model", ml_model_filename)

    data = None
    with open(ml_model_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    assert data is not None
    assert "run_id" in data
    assert "signature" not in data


def test_xgb_autolog_does_not_break_dmatrix_serialization(bst_params, tmpdir):
    mlflow.xgboost.autolog()

    # we cannot use dtrain fixture, as the dataset must be constructed
    #   after the call to autolog() in order to test the serialization
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
    y = iris.target
    dataset = xgb.DMatrix(X, y)

    xgb.train(bst_params, dataset)
    save_path = tmpdir.join("dataset_serialization_test").strpath
    dataset.save_binary(save_path)  # serialization should not throw
    xgb.DMatrix(save_path)  # deserialization also should not throw


@pytest.mark.parametrize("log_input_examples", [True, False])
@pytest.mark.parametrize("log_model_signatures", [True, False])
def test_xgb_autolog_configuration_options(bst_params, log_input_examples, log_model_signatures):
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
    y = iris.target

    with mlflow.start_run() as run:
        mlflow.xgboost.autolog(
            log_input_examples=log_input_examples, log_model_signatures=log_model_signatures
        )
        dataset = xgb.DMatrix(X, y)
        xgb.train(bst_params, dataset)
    model_conf = get_model_conf(run.info.artifact_uri)
    assert ("saved_input_example_info" in model_conf.to_dict()) == log_input_examples
    assert ("signature" in model_conf.to_dict()) == log_model_signatures


@pytest.mark.parametrize("log_models", [True, False])
def test_xgb_autolog_log_models_configuration(bst_params, log_models):
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
    y = iris.target

    with mlflow.start_run() as run:
        mlflow.xgboost.autolog(log_models=log_models)
        dataset = xgb.DMatrix(X, y)
        xgb.train(bst_params, dataset)

    run_id = run.info.run_id
    client = MlflowClient()
    artifacts = [f.path for f in client.list_artifacts(run_id)]
    assert ("model" in artifacts) == log_models


def test_xgb_autolog_does_not_break_dmatrix_instantiation_with_data_none():
    """
    This test verifies that `xgboost.DMatrix(None)` doesn't fail after patching.
    XGBoost internally calls `xgboost.DMatrix(None)` to create a blank `DMatrix` object.
    Example: https://github.com/dmlc/xgboost/blob/v1.2.1/python-package/xgboost/core.py#L701
    """
    mlflow.xgboost.autolog()
    xgb.DMatrix(None)


def test_callback_func_is_pickable():
    cb = picklable_exception_safe_function(
        functools.partial(autolog_callback, BatchMetricsLogger(run_id="1234"), eval_results={})
    )
    pickle.dumps(cb)


@pytest.mark.skipif(
    not IS_TRAINING_CALLBACK_SUPPORTED,
    reason="`xgboost.callback.TrainingCallback` is not supported",
)
def test_callback_class_is_pickable():
    from mlflow.xgboost._autolog import AutologCallback

    cb = AutologCallback(BatchMetricsLogger(run_id="1234"), eval_results={})
    pickle.dumps(cb)


def test_sklearn_api_autolog_registering_model():
    registered_model_name = "test_autolog_registered_model"
    mlflow.xgboost.autolog(registered_model_name=registered_model_name)

    X, y = datasets.load_iris(return_X_y=True)
    params = {"n_estimators": 10, "reg_lambda": 1}
    model = xgb.XGBRegressor(**params)

    with mlflow.start_run():
        model.fit(X, y)

        registered_model = MlflowClient().get_registered_model(registered_model_name)
        assert registered_model.name == registered_model_name


def test_xgb_api_autolog_registering_model(bst_params, dtrain):
    registered_model_name = "test_autolog_registered_model"
    mlflow.xgboost.autolog(registered_model_name=registered_model_name)

    with mlflow.start_run():
        xgb.train(bst_params, dtrain)

        registered_model = MlflowClient().get_registered_model(registered_model_name)
        assert registered_model.name == registered_model_name
