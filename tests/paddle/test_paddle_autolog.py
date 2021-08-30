import pytest
import paddle
import mlflow
import mlflow.paddle

NUM_EPOCHS = 6

pytestmark = pytest.mark.large


@pytest.fixture
def dataset():
    train_dataset = paddle.text.datasets.UCIHousing(mode="train")
    eval_dataset = paddle.text.datasets.UCIHousing(mode="test")
    return train_dataset, eval_dataset


@pytest.fixture
def test_model(dataset):
    class UCIHousing(paddle.nn.Layer):
        def __init__(self):
            super(UCIHousing, self).__init__()
            self.fc = paddle.nn.Linear(13, 1)

        def forward(self, feature):  # pylint: disable=arguments-differ
            pred = self.fc(feature)
            return pred

    model = paddle.Model(UCIHousing())
    optim = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())
    model.prepare(optim, paddle.nn.MSELoss())
    return model


@pytest.fixture
def paddle_model(dataset, test_model):
    mlflow.paddle.autolog()

    train_dataset, eval_dataset = dataset

    test_model.fit(train_dataset, eval_dataset, epochs=NUM_EPOCHS, batch_size=8, verbose=1)

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)
    return test_model, run


@pytest.mark.parametrize("log_models", [True, False])
def test_paddle_autolog_log_models_configuration(log_models, dataset, test_model):
    mlflow.paddle.autolog(log_models=log_models)

    train_dataset, eval_dataset = dataset

    test_model.fit(train_dataset, eval_dataset, epochs=NUM_EPOCHS, batch_size=8, verbose=1)

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)
    run_id = run.info.run_id
    artifacts = [f.path for f in client.list_artifacts(run_id)]
    assert ("model" in artifacts) == log_models


def test_paddle_autolog_logs_default_params(paddle_model):
    _, run = paddle_model
    data = run.data
    assert "optimizer_name" in data.params


def test_paddle_autolog_logs_expected_data(paddle_model):
    _, run = paddle_model
    data = run.data

    # Checking if metrics are logged
    client = mlflow.tracking.MlflowClient()
    for metric_key in ["batch_size", "loss", "step", "eval_batch_size", "eval_loss", "eval_step"]:
        assert metric_key in run.data.metrics
        metric_history = client.get_metric_history(run.info.run_id, metric_key)
        assert len(metric_history) == NUM_EPOCHS

    # Testing optimizer parameters are logged
    assert "optimizer_name" in data.params
    assert data.params["optimizer_name"] == "Adam"

    # Testing learning rate are logged
    assert "learning_rate" in data.params
    assert float(data.params["learning_rate"]) == 1e-2

    # Testing model_summary.txt is saved
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert "model_summary.txt" in artifacts


def test_paddle_autolog_persists_manually_created_run(dataset, test_model):
    with mlflow.start_run() as manual_run:
        mlflow.paddle.autolog()

        train_dataset, eval_dataset = dataset

        test_model.fit(train_dataset, eval_dataset, epochs=NUM_EPOCHS, batch_size=8, verbose=1)
        assert mlflow.active_run() is not None
        assert mlflow.active_run().info.run_id == manual_run.info.run_id


def test_paddle_autolog_ends_auto_created_run(paddle_model):  # pylint: disable=unused-argument
    assert mlflow.active_run() is None


@pytest.fixture
def paddle_model_with_early_stopping(patience, dataset, test_model):
    train_dataset, eval_dataset = dataset

    callbacks_earlystopping = paddle.callbacks.EarlyStopping(
        "loss",
        mode="min",
        patience=patience,
        verbose=2,
        min_delta=0,
        baseline=None,
        save_best_model=True,
    )

    mlflow.paddle.autolog()
    test_model.fit(
        train_dataset,
        eval_dataset,
        epochs=NUM_EPOCHS,
        batch_size=8,
        verbose=1,
        callbacks=[callbacks_earlystopping],
    )

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)
    return test_model, run


@pytest.mark.parametrize("patience", [1])
def test_paddle_early_stop_params_logged(paddle_model_with_early_stopping, patience):
    _, run = paddle_model_with_early_stopping
    data = run.data
    assert "monitor" in data.params
    assert "patience" in data.params
    assert float(data.params["patience"]) == patience
    assert "min_delta" in data.params
    assert "stopped_epoch" in data.params


def test_paddle_autolog_non_early_stop_callback_does_not_log(paddle_model):
    _, run = paddle_model
    client = mlflow.tracking.MlflowClient()
    loss_metric_history = client.get_metric_history(run.info.run_id, "loss")
    step_metric_history = client.get_metric_history(run.info.run_id, "step")
    assert len(loss_metric_history) == NUM_EPOCHS
    assert len(step_metric_history) == NUM_EPOCHS
    data = run.data
    assert "monitor" not in data.params
    assert "patience" not in data.params
    assert "min_delta" not in data.params
    assert "stopped_epoch" not in data.params
