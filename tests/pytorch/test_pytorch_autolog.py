import os
import pytest
import pytorch_lightning as pl
import torch
from iris import IrisClassification
import mlflow
import mlflow.pytorch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

NUM_EPOCHS = 20


@pytest.fixture
def pytorch_model():
    mlflow.pytorch.autolog()
    model = IrisClassification()
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS)
    trainer.fit(model)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)
    return trainer, run


def test_pytorch_autolog_logs_default_params(pytorch_model):
    _, run = pytorch_model
    data = run.data
    assert "lr" in data.params
    assert "eps" in data.params
    assert "optimizer_name" in data.params
    assert "weight_decay" in data.params
    assert "betas" in data.params


def test_pytorch_autolog_logs_expected_data(pytorch_model):
    _, run = pytorch_model
    data = run.data

    # Checking if metrics are logged
    assert "loss" in data.metrics
    assert "val_loss" in data.metrics

    # Testing optimizer parameters are logged
    assert "optimizer_name" in data.params
    assert data.params["optimizer_name"] == "Adam"

    # Testing model_summary.txt is saved
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert "model_summary.txt" in artifacts


# pylint: disable=unused-argument
def test_pytorch_autolog_persists_manually_created_run():
    with mlflow.start_run() as manual_run:
        mlflow.pytorch.autolog()
        model = IrisClassification()
        trainer = pl.Trainer(max_epochs=NUM_EPOCHS)
        trainer.fit(model)
        assert mlflow.active_run() is not None
        assert mlflow.active_run().info.run_id == manual_run.info.run_id


def test_pytorch_autolog_ends_auto_created_run(pytorch_model):
    assert mlflow.active_run() is None


@pytest.fixture
def pytorch_model_with_callback(patience):
    mlflow.pytorch.autolog()
    model = IrisClassification()
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=patience, verbose=True)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(), save_top_k=1, verbose=True, monitor="val_loss", mode="min", prefix=""
    )

    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS * 2,
        callbacks=[early_stopping],
        checkpoint_callback=checkpoint_callback,
    )
    trainer.fit(model)

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)

    return trainer, run


@pytest.mark.parametrize("patience", [3])
def test_pytorch_early_stop_artifacts_logged(pytorch_model_with_callback):
    _, run = pytorch_model_with_callback
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert "restored_model_checkpoint" in artifacts


@pytest.mark.parametrize("patience", [3])
def test_pytorch_autolog_model_can_load_from_artifact(pytorch_model_with_callback):
    _, run = pytorch_model_with_callback
    run_id = run.info.run_id
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert "model" in artifacts
    model = mlflow.pytorch.load_model("runs:/" + run_id + "/model")
    result = model(torch.Tensor([1.5, 2, 2.5, 3.5]).unsqueeze(0))
    assert result is not None


@pytest.mark.parametrize("patience", [0, 1, 5])
def test_pytorch_early_stop_params_logged(pytorch_model_with_callback, patience):
    _, run = pytorch_model_with_callback
    data = run.data
    assert "monitor" in data.params
    assert "mode" in data.params
    assert "patience" in data.params
    assert float(data.params["patience"]) == patience
    assert "min_delta" in data.params
    assert "stopped_epoch" in data.params


@pytest.mark.parametrize("patience", [3])
def test_pytorch_early_stop_metrics_logged(pytorch_model_with_callback):
    _, run = pytorch_model_with_callback
    data = run.data
    assert "stopped_epoch" in data.metrics
    assert "wait_count" in data.metrics
    assert "restored_epoch" in data.metrics


def test_pytorch_autolog_non_early_stop_callback_does_not_log(pytorch_model):
    trainer, run = pytorch_model
    client = mlflow.tracking.MlflowClient()
    metric_history = client.get_metric_history(run.info.run_id, "loss")
    assert trainer.max_epochs == NUM_EPOCHS
    assert len(metric_history) == NUM_EPOCHS


@pytest.fixture
def pytorch_model_tests():
    model = IrisClassification()

    trainer = pl.Trainer(max_epochs=NUM_EPOCHS)
    trainer.fit(model)
    trainer.test()
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)
    return trainer, run


def test_pytorch_test_metrics_logged(pytorch_model_tests):
    _, run = pytorch_model_tests
    data = run.data
    assert "test_loss" in data.metrics
    assert "test_acc" in data.metrics
