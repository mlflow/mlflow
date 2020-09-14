import os
import json
import pytest
import numpy as np
import pandas as pd
from sklearn import datasets
import xgboost as xgb
import matplotlib as mpl
import yaml

import mlflow
import mlflow.xgboost
from mlflow.models import Model
from mlflow.models.utils import _read_example

mpl.use("Agg")


def get_latest_run():
    client = mlflow.tracking.MlflowClient()
    return client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)


@pytest.fixture(scope="session")
def bst_params():
    return {
        "objective": "multi:softprob",
        "num_class": 3,
    }


@pytest.fixture(scope="session")
def dtrain():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
    y = iris.target
    return xgb.DMatrix(X, y)


@pytest.mark.large
def test_xgb_autolog_ends_auto_created_run(bst_params, dtrain):
    mlflow.xgboost.autolog()
    xgb.train(bst_params, dtrain)
    assert mlflow.active_run() is None


@pytest.mark.large
def test_xgb_autolog_persists_manually_created_run(bst_params, dtrain):
    mlflow.xgboost.autolog()
    with mlflow.start_run() as run:
        xgb.train(bst_params, dtrain)
        assert mlflow.active_run()
        assert mlflow.active_run().info.run_id == run.info.run_id


@pytest.mark.large
def test_xgb_autolog_logs_default_params(bst_params, dtrain):
    mlflow.xgboost.autolog()
    xgb.train(bst_params, dtrain)
    run = get_latest_run()
    params = run.data.params

    expected_params = {
        "num_boost_round": 10,
        "maximize": False,
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


@pytest.mark.large
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


@pytest.mark.large
def test_xgb_autolog_logs_metrics_with_validation_data(bst_params, dtrain):
    mlflow.xgboost.autolog()
    evals_result = {}
    xgb.train(
        bst_params, dtrain, num_boost_round=20, evals=[(dtrain, "train")], evals_result=evals_result
    )
    run = get_latest_run()
    data = run.data
    metric_key = "train-merror"
    client = mlflow.tracking.MlflowClient()
    metric_history = [x.value for x in client.get_metric_history(run.info.run_id, metric_key)]
    assert metric_key in data.metrics
    assert len(metric_history) == 20
    assert metric_history == evals_result["train"]["merror"]


@pytest.mark.large
def test_xgb_autolog_logs_metrics_with_multi_validation_data(bst_params, dtrain):
    mlflow.xgboost.autolog()
    evals_result = {}
    evals = [(dtrain, "train"), (dtrain, "valid")]
    xgb.train(bst_params, dtrain, num_boost_round=20, evals=evals, evals_result=evals_result)
    run = get_latest_run()
    data = run.data
    client = mlflow.tracking.MlflowClient()
    for eval_name in [e[1] for e in evals]:
        metric_key = "{}-merror".format(eval_name)
        metric_history = [x.value for x in client.get_metric_history(run.info.run_id, metric_key)]
        assert metric_key in data.metrics
        assert len(metric_history) == 20
        assert metric_history == evals_result[eval_name]["merror"]


@pytest.mark.large
def test_xgb_autolog_logs_metrics_with_multi_metrics(bst_params, dtrain):
    mlflow.xgboost.autolog()
    evals_result = {}
    params = {"eval_metric": ["merror", "mlogloss"]}
    params.update(bst_params)
    xgb.train(
        params, dtrain, num_boost_round=20, evals=[(dtrain, "train")], evals_result=evals_result
    )
    run = get_latest_run()
    data = run.data
    client = mlflow.tracking.MlflowClient()
    for metric_name in params["eval_metric"]:
        metric_key = "train-{}".format(metric_name)
        metric_history = [x.value for x in client.get_metric_history(run.info.run_id, metric_key)]
        assert metric_key in data.metrics
        assert len(metric_history) == 20
        assert metric_history == evals_result["train"][metric_name]


@pytest.mark.large
def test_xgb_autolog_logs_metrics_with_multi_validation_data_and_metrics(bst_params, dtrain):
    mlflow.xgboost.autolog()
    evals_result = {}
    params = {"eval_metric": ["merror", "mlogloss"]}
    params.update(bst_params)
    evals = [(dtrain, "train"), (dtrain, "valid")]
    xgb.train(params, dtrain, num_boost_round=20, evals=evals, evals_result=evals_result)
    run = get_latest_run()
    data = run.data
    client = mlflow.tracking.MlflowClient()
    for eval_name in [e[1] for e in evals]:
        for metric_name in params["eval_metric"]:
            metric_key = "{}-{}".format(eval_name, metric_name)
            metric_history = [
                x.value for x in client.get_metric_history(run.info.run_id, metric_key)
            ]
            assert metric_key in data.metrics
            assert len(metric_history) == 20
            assert metric_history == evals_result[eval_name][metric_name]


@pytest.mark.large
def test_xgb_autolog_logs_metrics_with_early_stopping(bst_params, dtrain):
    mlflow.xgboost.autolog()
    evals_result = {}
    params = {"eval_metric": ["merror", "mlogloss"]}
    params.update(bst_params)
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
    client = mlflow.tracking.MlflowClient()

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


@pytest.mark.large
def test_xgb_autolog_logs_feature_importance(bst_params, dtrain):
    mlflow.xgboost.autolog()
    model = xgb.train(bst_params, dtrain)
    run = get_latest_run()
    run_id = run.info.run_id
    artifacts_dir = run.info.artifact_uri.replace("file://", "")
    client = mlflow.tracking.MlflowClient()
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


@pytest.mark.large
def test_xgb_autolog_logs_specified_feature_importance(bst_params, dtrain):
    importance_types = ["weight", "total_gain"]
    mlflow.xgboost.autolog(importance_types)
    model = xgb.train(bst_params, dtrain)
    run = get_latest_run()
    run_id = run.info.run_id
    artifacts_dir = run.info.artifact_uri.replace("file://", "")
    client = mlflow.tracking.MlflowClient()
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


@pytest.mark.large
def test_no_figure_is_opened_after_logging(bst_params, dtrain):
    mlflow.xgboost.autolog()
    xgb.train(bst_params, dtrain)
    assert mpl.pyplot.get_fignums() == []


@pytest.mark.large
def test_xgb_autolog_loads_model_from_artifact(bst_params, dtrain):
    mlflow.xgboost.autolog()
    model = xgb.train(bst_params, dtrain)
    run = get_latest_run()
    run_id = run.info.run_id

    loaded_model = mlflow.xgboost.load_model("runs:/{}/model".format(run_id))
    np.testing.assert_array_almost_equal(model.predict(dtrain), loaded_model.predict(dtrain))


@pytest.mark.large
def test_xgb_autolog_does_not_throw_if_importance_values_not_supported(dtrain):
    # the gblinear booster does not support calling get_score on it,
    #   where get_score is used to create the importance values plot.
    bst_params = {"objective": "multi:softprob", "num_class": 3, "booster": "gblinear"}

    mlflow.xgboost.autolog()

    # we make sure here that we do not throw while attempting to plot
    #   importance values on a model with a linear booster.
    model = xgb.train(bst_params, dtrain)

    with pytest.raises(Exception):
        model.get_score(importance_type="weight")


@pytest.mark.large
def test_xgb_autolog_gets_input_example(bst_params):
    mlflow.xgboost.autolog()

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

    assert input_example.equals(X[:5])

    pyfunc_model = mlflow.pyfunc.load_model(os.path.join(run.info.artifact_uri, "model"))

    # make sure reloading the input_example and predicting on it does not error
    pyfunc_model.predict(input_example)


@pytest.mark.large
def test_xgb_autolog_infers_model_signature_correctly(bst_params):
    mlflow.xgboost.autolog()

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
    client = mlflow.tracking.MlflowClient()
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
        {"type": "float"},
        {"type": "float"},
        {"type": "float"},
    ]


@pytest.mark.large
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


@pytest.mark.large
def test_xgb_autolog_continues_logging_even_if_signature_inference_fails(bst_params, tmpdir):
    tmp_csv = tmpdir.join("data.csv")
    tmp_csv.write("1,0.3,1.2\n")
    tmp_csv.write("0,2.4,5.2\n")
    tmp_csv.write("1,0.3,-1.2\n")

    mlflow.xgboost.autolog(importance_types=[])

    # signature and input example inference should fail here since the dataset is given
    #   as a file path
    dataset = xgb.DMatrix(tmp_csv.strpath + "?format=csv&label_column=0")

    xgb.train(bst_params, dataset)
    run = get_latest_run()
    run_id = run.info.run_id
    artifacts_dir = run.info.artifact_uri.replace("file://", "")
    client = mlflow.tracking.MlflowClient()
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


@pytest.mark.large
def test_xgb_autolog_does_not_break_dmatrix_serialization(bst_params):
    mlflow.xgboost.autolog()

    # we cannot use dtrain fixture, as the dataset must be constructed
    #   after the call to autolog() in order to get the input example
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
    y = iris.target
    dataset = xgb.DMatrix(X, y)

    xgb.train(bst_params, dataset)

    dataset.save_binary("dataset_serialization_test")  # serialization should not throw
    xgb.DMatrix("dataset_serialization_test")  # deserialization also should not throw
    os.remove("dataset_serialization_test")