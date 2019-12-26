import os
import json
import pytest
import numpy as np
import pandas as pd
from sklearn import datasets
import xgboost as xgb

import mlflow
import mlflow.xgboost

client = mlflow.tracking.MlflowClient()


def get_latest_run():
    return client.get_run(client.list_run_infos(experiment_id='0')[0].run_id)


@pytest.fixture(scope="session")
def bst_params():
    return {
        'objective': 'multi:softprob',
        'num_class': 3,
        'eval_metric': 'mlogloss',
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
        'params': bst_params,
        'num_boost_round': 10,
        'maximize': False,
        'early_stopping_rounds': None,
        'verbose_eval': True,
    }

    for key, val in expected_params.items():
        assert key in params
        assert params[key] == str(val)

    unlogged_params = ['dtrain', 'evals', 'obj', 'feval', 'evals_result',
                       'xgb_model', 'callbacks', 'learning_rates']

    for param in unlogged_params:
        assert param not in params


@pytest.mark.large
def test_xgb_autolog_logs_specified_params(bst_params, dtrain):
    mlflow.xgboost.autolog()
    expected_params = {
        'num_boost_round': 20,
        'early_stopping_rounds': 5,
        'verbose_eval': False,
    }
    xgb.train(bst_params, dtrain, evals=[(dtrain, 'train')], **expected_params)
    run = get_latest_run()
    params = run.data.params

    expected_params.update({'params': bst_params})

    for key, val in expected_params.items():
        assert key in params
        assert params[key] == str(val)

    unlogged_params = ['dtrain', 'evals', 'obj', 'feval', 'evals_result',
                       'xgb_model', 'callbacks', 'learning_rates']

    for param in unlogged_params:
        assert param not in params


@pytest.mark.large
def test_xgb_autolog_logs_metrics_with_validation_data(bst_params, dtrain):
    mlflow.xgboost.autolog()
    evals_result = {}
    xgb.train(bst_params, dtrain, num_boost_round=20,
              evals=[(dtrain, 'train')], evals_result=evals_result)
    run = get_latest_run()
    data = run.data

    metric_key = 'train-mlogloss'
    metric_history = [x.value for x in client.get_metric_history(run.info.run_id, metric_key)]
    assert metric_key in data.metrics
    assert len(metric_history) == 20
    assert metric_history == evals_result['train']['mlogloss']

    # these metrics should not be logged because early_stopping_rounds wasn't specified.
    attrs = ['best_iteration', 'best_ntree_limit', 'best_score']

    for attr in attrs:
        assert attr not in data.metrics


@pytest.mark.large
def test_xgb_autolog_logs_metrics_with_early_stopping(bst_params, dtrain):
    mlflow.xgboost.autolog()
    evals_result = {}
    model = xgb.train(bst_params, dtrain, num_boost_round=20, early_stopping_rounds=5,
                      evals=[(dtrain, 'train')], evals_result=evals_result)
    run = get_latest_run()
    data = run.data

    metric_key = 'train-mlogloss'
    metric_history = [x.value for x in client.get_metric_history(run.info.run_id, metric_key)]
    assert metric_key in data.metrics
    assert len(metric_history) == 20
    assert metric_history == evals_result['train']['mlogloss']

    attrs = ['best_iteration', 'best_ntree_limit', 'best_score']

    for attr in attrs:
        assert attr in data.metrics
        assert data.metrics[attr] == getattr(model, attr)


@pytest.mark.large
def test_xgb_autolog_logs_feature_importance(bst_params, dtrain):
    mlflow.xgboost.autolog()
    model = xgb.train(bst_params, dtrain)
    run = get_latest_run()
    run_id = run.info.run_id
    artifacts_dir = run.info.artifact_uri.replace('file://', '')
    artifacts = [x.path for x in client.list_artifacts(run_id)]

    importance_type = 'weight'
    filename = 'feature_importance_{}.json'.format(importance_type)
    filepath = os.path.join(artifacts_dir, filename)
    with open(filepath, 'r') as f:
        loaded_imp = json.load(f)

    assert filename in artifacts
    assert loaded_imp == model.get_score(importance_type=importance_type)


@pytest.mark.large
def test_xgb_autolog_logs_specified_feature_importance(bst_params, dtrain):
    importance_types = ['weight', 'total_gain']
    mlflow.xgboost.autolog(importance_types)
    model = xgb.train(bst_params, dtrain)
    run = get_latest_run()
    run_id = run.info.run_id
    artifacts_dir = run.info.artifact_uri.replace('file://', '')
    artifacts = [x.path for x in client.list_artifacts(run_id)]

    for imp_type in importance_types:
        filename = 'feature_importance_{}.json'.format(imp_type)
        filepath = os.path.join(artifacts_dir, filename)
        with open(filepath, 'r') as f:
            loaded_imp = json.load(f)

        assert filename in artifacts
        assert loaded_imp == model.get_score(importance_type=imp_type)


@pytest.mark.large
def test_xgb_autolog_loads_model_from_artifact(bst_params, dtrain):
    mlflow.xgboost.autolog()
    model = xgb.train(bst_params, dtrain)
    run = get_latest_run()
    run_id = run.info.run_id

    loaded_model = mlflow.xgboost.load_model('runs:/{}/model'.format(run_id))
    np.testing.assert_array_almost_equal(model.predict(dtrain), loaded_model.predict(dtrain))
