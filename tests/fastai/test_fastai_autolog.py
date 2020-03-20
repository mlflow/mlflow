import pytest
import numpy as np
from tests.projects.utils import tracking_uri_mock  # pylint: disable=W0611

import pandas as pd
import sklearn.datasets as datasets
from fastai.tabular import tabular_learner, TabularList
from fastai.metrics import accuracy
import mlflow  # noqa
import mlflow.fastai  # noqa
from fastai.callbacks import EarlyStoppingCallback, SaveModelCallback

np.random.seed(1337)

LARGE_EPOCHS = 5


@pytest.fixture(params=[True, False])
def manual_run(request, tracking_uri_mock):
    if request.param:
        mlflow.start_run()
    yield
    mlflow.end_run()


@pytest.fixture(scope="session")
def iris_data():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data[:, :2], columns=iris.feature_names[:2])
    y = pd.Series(iris.target, name='label')
    return (TabularList.from_df(pd.concat([X, y], axis=1), cont_names=X.columns)
            .split_by_rand_pct(valid_pct=0.1, seed=42)
            .label_from_df(cols='label')
            .databunch())


def fastai_model(data, **kwargs):
    return tabular_learner(data, metrics=accuracy, layers=[5, 3, 2], **kwargs)


@pytest.mark.large
@pytest.mark.parametrize('fit_variant', ['fit', 'fit_one_cycle'])
def test_fastai_autolog_ends_auto_created_run(iris_data, fit_variant):
    mlflow.fastai.autolog()
    model = fastai_model(iris_data)
    if fit_variant == 'fit_one_cycle':
        model.fit_one_cycle(1)
    else:
        model.fit(1)
    assert mlflow.active_run() is None


@pytest.mark.large
@pytest.mark.parametrize('fit_variant', ['fit', 'fit_one_cycle'])
def test_fastai_autolog_persists_manually_created_run(iris_data, fit_variant):
    mlflow.fastai.autolog()

    with mlflow.start_run() as run:
        model = fastai_model(iris_data)

        if fit_variant == 'fit_one_cycle':
            model.fit_one_cycle(LARGE_EPOCHS)
        else:
            model.fit(LARGE_EPOCHS)

        assert mlflow.active_run()
        assert mlflow.active_run().info.run_id == run.info.run_id


@pytest.fixture
def fastai_random_data_run(iris_data, fit_variant, manual_run):
    mlflow.fastai.autolog()

    model = fastai_model(iris_data)

    if fit_variant == 'fit_one_cycle':
        model.fit_one_cycle(LARGE_EPOCHS)
    else:
        model.fit(LARGE_EPOCHS)

    client = mlflow.tracking.MlflowClient()
    return model, client.get_run(client.list_run_infos(experiment_id='0')[0].run_id)


@pytest.mark.large
@pytest.mark.parametrize('fit_variant', ['fit', 'fit_one_cycle'])
def test_fastai_autolog_logs_expected_data(fastai_random_data_run, fit_variant):
    model, run = fastai_random_data_run
    data = run.data

    # Testing metrics are logged
    assert 'train_loss' in data.metrics
    assert 'valid_loss' in data.metrics
    for o in model.metrics:
        assert o.__name__ in data.metrics

    # Testing explicitly passed parameters are logged correctly
    assert 'epochs' in data.params
    assert data.params['epochs'] == str(LARGE_EPOCHS)

    if fit_variant == 'fit_one_cycle':
        assert 'cyc_len' in data.params
        assert data.params['cyc_len'] == str(LARGE_EPOCHS)

    # Testing unwanted parameters are not logged
    assert 'callbacks' not in data.params

    # Testing optimizer parameters are logged
    assert 'optimizer_name' in data.params
    assert data.params['optimizer_name'] == model.opt.__name__
    assert 'model_summary' in data.tags

    # Testing model_summary.txt is saved
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert 'model_summary.txt' in artifacts


@pytest.mark.large
@pytest.mark.parametrize('fit_variant', ['fit', 'fit_one_cycle'])
def test_fastai_autolog_logs_default_params(fastai_random_data_run, fit_variant):
    _, run = fastai_random_data_run
    if fit_variant == 'fit':
        assert 'lr' in run.data.params
        assert run.data.params['lr'] == 'slice(None, 0.003, None)'
    else:
        assert 'pct_start' in run.data.params
        assert run.data.params['pct_start'] == '0.3'


@pytest.mark.large
@pytest.mark.parametrize('fit_variant', ['fit', 'fit_one_cycle'])
def test_fastai_autolog_model_can_load_from_artifact(fastai_random_data_run, random_train_data):
    run_id = fastai_random_data_run[1].info.run_id
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert 'model' in artifacts
    model = mlflow.fastai.load_model("runs:/" + run_id + "/model")
    model.predict(random_train_data)


@pytest.fixture
def fastai_random_data_run_with_callback(iris_data, fit_variant, manual_run, callback, patience):
    mlflow.fastai.autolog()
    callbacks = []

    if callback == 'early':
        # min_delta is set as such to guarantee early stopping
        callbacks.append(lambda learn: EarlyStoppingCallback(
            learn, patience=patience, min_delta=99999999))

    model = fastai_model(iris_data, callback_fns=[callback])

    if fit_variant == 'fit_one_cycle':
        model.fit_one_cycle(1)
    else:
        model.fit(1)

    client = mlflow.tracking.MlflowClient()
    return client.get_run(client.list_run_infos(experiment_id='0')[0].run_id)


@pytest.mark.large
@pytest.mark.parametrize('fit_variant', ['fit', 'fit_one_cycle'])
@pytest.mark.parametrize('callback', ['early'])
@pytest.mark.parametrize('patience', [0, 1, 5])
def test_fastai_autolog_early_stop_logs(fastai_random_data_run_with_callback, patience):
    return
    run = fastai_random_data_run_with_callback
    metrics = run.data.metrics
    params = run.data.params
    assert 'patience' in params
    assert params['patience'] == str(patience)
    assert 'monitor' in params
    assert params['monitor'] == 'valid_loss'
    assert 'verbose' not in params
    assert 'mode' not in params
    assert 'stopped_epoch' in metrics
    assert 'restored_epoch' in metrics
    restored_epoch = int(metrics['restored_epoch'])
    assert int(metrics['stopped_epoch']) - max(1, callback.patience) == restored_epoch
    assert 'loss' in history.history
    num_of_epochs = len(history.history['loss'])
    client = mlflow.tracking.MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, 'loss')
    # Check the test epoch numbers are correct
    assert num_of_epochs == max(1, callback.patience) + 1
    # Check that MLflow has logged the metrics of the "best" model
    assert len(metric_history) == num_of_epochs + 1
    # Check that MLflow has logged the correct data
    assert history.history['loss'][restored_epoch] == metric_history[-1].value


@pytest.mark.large
@pytest.mark.parametrize('fit_variant', ['fit', 'fit_one_cycle'])
@pytest.mark.parametrize('callback', ['early'])
@pytest.mark.parametrize('patience', [11])
def test_fastai_autolog_early_stop_no_stop_does_not_log(fastai_random_data_run_with_callback):
    return
    run, history, callback = fastai_random_data_run_with_callback
    metrics = run.data.metrics
    params = run.data.params
    assert 'patience' in params
    assert params['patience'] == str(callback.patience)
    assert 'monitor' in params
    assert params['monitor'] == 'loss'
    assert 'verbose' not in params
    assert 'mode' not in params
    assert 'stopped_epoch' in metrics
    assert metrics['stopped_epoch'] == 0
    assert 'restored_epoch' not in metrics
    assert 'loss' in history.history
    num_of_epochs = len(history.history['loss'])
    client = mlflow.tracking.MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, 'loss')
    # Check the test epoch numbers are correct
    assert num_of_epochs == 10
    assert len(metric_history) == num_of_epochs


@pytest.mark.large
@pytest.mark.parametrize('fit_variant', ['fit', 'fit_one_cycle'])
@pytest.mark.parametrize('callback', ['early'])
@pytest.mark.parametrize('patience', [5])
def test_fastai_autolog_early_stop_no_restore_does_not_log(fastai_random_data_run_with_callback):
    return
    run, history, callback = fastai_random_data_run_with_callback
    metrics = run.data.metrics
    params = run.data.params
    assert 'patience' in params
    assert params['patience'] == str(callback.patience)
    assert 'monitor' in params
    assert params['monitor'] == 'loss'
    assert 'verbose' not in params
    assert 'mode' not in params
    assert 'stopped_epoch' in metrics
    assert 'restored_epoch' not in metrics
    assert 'loss' in history.history
    num_of_epochs = len(history.history['loss'])
    client = mlflow.tracking.MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, 'loss')
    # Check the test epoch numbers are correct
    assert num_of_epochs == callback.patience + 1
    assert len(metric_history) == num_of_epochs


@pytest.mark.large
@pytest.mark.parametrize('fit_variant', ['fit', 'fit_one_cycle'])
@pytest.mark.parametrize('callback', ['not-early'])
@pytest.mark.parametrize('patience', [5])
def test_fastai_autolog_non_early_stop_callback_does_not_log(fastai_random_data_run_with_callback):
    return
    run = fastai_random_data_run_with_callback
    metrics = run.data.metrics
    params = run.data.params
    assert 'patience' not in params
    assert 'monitor' not in params
    assert 'verbose' not in params
    assert 'mode' not in params
    assert 'stopped_epoch' not in metrics
    assert 'restored_epoch' not in metrics
    assert 'loss' in history.history
    num_of_epochs = len(history.history['loss'])
    client = mlflow.tracking.MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, 'loss')
    # Check the test epoch numbers are correct
    assert num_of_epochs == 10
    assert len(metric_history) == num_of_epochs
