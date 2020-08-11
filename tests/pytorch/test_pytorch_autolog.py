import os
import pytest
import pytorch_lightning as pl
from iris import IrisClassification
from mlflow.pytorch.pytorch_autolog import __MLflowPLCallback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import MLFlowLogger

NUM_EPOCHS = 20


@pytest.fixture
def pytorch_model():
    mlflow_logger = MLFlowLogger(tracking_uri="http://localhost:5000")
    model = IrisClassification()
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS, callbacks=[__MLflowPLCallback()], logger=mlflow_logger
    )
    trainer.fit(model)
    client = trainer.logger.experiment
    return trainer, client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)


@pytest.mark.large
def test_pytorch_autolog_logs_default_params(pytorch_model):
    trainer, run = pytorch_model
    data = run.data
    assert "learning_rate" in data.params
    assert "epsilon" in data.params
    assert "optimizer_name" in data.params


@pytest.mark.large
def test_pytorch_autolog_logs_expected_data(pytorch_model):
    trainer, run = pytorch_model
    data = run.data

    # Testing metrics are logged
    assert "loss" in data.metrics
    assert "val_loss" in data.metrics

    assert "epoch" in data.metrics
    assert data.metrics["epoch"] == NUM_EPOCHS - 1

    # Testing unwanted parameters are not logged
    assert "callbacks" not in data.params

    # Testing optimizer parameters are logged
    assert "optimizer_name" in data.params
    assert data.params["optimizer_name"] == "Adam"

    # Testing model_summary.txt is saved
    artifacts = trainer.logger.experiment.list_artifacts(run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert "model_summary.txt" in artifacts


@pytest.fixture
def pytorch_model_with_callback(patience):
    mlflow_logger = MLFlowLogger(tracking_uri="http://localhost:5000")
    model = IrisClassification()
    early_stopping = EarlyStopping(
        monitor="val_loss", mode="min", patience=patience, verbose=True
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(),
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
        prefix="",
    )

    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        callbacks=[__MLflowPLCallback()],
        logger=mlflow_logger,
        early_stop_callback=early_stopping,
        checkpoint_callback=checkpoint_callback,
    )
    trainer.fit(model)
    trainer.test()
    client = trainer.logger.experiment
    return trainer, client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)


@pytest.mark.large
@pytest.mark.parametrize("patience", [3])
def test_pytorch_early_stop_metrics_logged(pytorch_model_with_callback, patience):
    trainer, run = pytorch_model_with_callback
    data = run.data
    artifacts = trainer.logger.experiment.list_artifacts(run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert "restored_model_checkpoint" in artifacts


@pytest.mark.large
@pytest.mark.parametrize("patience", [0, 1, 5])
def test_pytorch_early_stop_params_logged(pytorch_model_with_callback, patience):
    trainer, run = pytorch_model_with_callback
    data = run.data
    assert "monitor" in data.params
    assert "mode" in data.params
    assert "patience" in data.params
    assert float(data.params["patience"]) == patience
    assert "min_delta" in data.params
    assert "stopped_epoch" in data.params


@pytest.mark.large
@pytest.mark.parametrize("patience", [3])
def test_pytorch_early_stop_metrics_logged(pytorch_model_with_callback, patience):
    trainer, run = pytorch_model_with_callback
    data = run.data
    assert "Stopped_Epoch" in data.metrics
    assert "Best_Score" in data.metrics
    assert "Wait_Count" in data.metrics
    assert "Restored_Epoch" in data.metrics


@pytest.fixture
def pytorch_model_tests():
    mlflow_logger = MLFlowLogger(tracking_uri="http://localhost:5000")
    model = IrisClassification()

    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS, callbacks=[__MLflowPLCallback()], logger=mlflow_logger
    )
    trainer.fit(model)
    trainer.test()
    client = trainer.logger.experiment
    return trainer, client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)


@pytest.mark.large
def test_pytorch_test_metrics_logged(pytorch_model_tests):
    trainer, run = pytorch_model_tests
    data = run.data
    assert "test_loss" in data.metrics
    assert "test_acc" in data.metrics
