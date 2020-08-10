import pytest
import pytorch_lightning as pl
from iris import IrisClassification
from mlflow.pytorch.pytorch_autolog import __MLflowPLCallback
from pytorch_lightning.logging import MLFlowLogger

NUM_EPOCHS = 20


@pytest.fixture
def pytorch_model():
    mlflow_logger = MLFlowLogger(
        tracking_uri="http://localhost:5000/", experiment_name="Default"
    )
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
