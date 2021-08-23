from packaging.version import Version
import pytest
import pytorch_lightning as pl
import torch
from iris import IrisClassification, IrisClassificationWithoutValidation
import mlflow
import mlflow.pytorch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from mlflow.utils.file_utils import TempDir
from iris_data_module import IrisDataModule, IrisDataModuleWithoutValidation
from mlflow.pytorch._pytorch_autolog import _get_optimizer_name

NUM_EPOCHS = 20

pytestmark = pytest.mark.large


@pytest.fixture
def pytorch_model():
    mlflow.pytorch.autolog()
    model = IrisClassification()
    dm = IrisDataModule()
    dm.setup(stage="fit")
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS)
    trainer.fit(model, dm)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)
    return trainer, run


@pytest.fixture
def pytorch_model_without_validation():
    mlflow.pytorch.autolog()
    model = IrisClassificationWithoutValidation()
    dm = IrisDataModuleWithoutValidation()
    dm.setup(stage="fit")
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS)
    trainer.fit(model, dm)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)
    return trainer, run


@pytest.mark.parametrize("log_models", [True, False])
def test_pytorch_autolog_log_models_configuration(log_models):
    mlflow.pytorch.autolog(log_models=log_models)
    model = IrisClassification()
    dm = IrisDataModule()
    dm.setup(stage="fit")
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS)
    trainer.fit(model, dm)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)
    run_id = run.info.run_id
    client = mlflow.tracking.MlflowClient()
    artifacts = [f.path for f in client.list_artifacts(run_id)]
    assert ("model" in artifacts) == log_models


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
    client = mlflow.tracking.MlflowClient()
    for metric_key in ["loss", "train_acc", "val_loss", "val_acc"]:
        assert metric_key in run.data.metrics
        metric_history = client.get_metric_history(run.info.run_id, metric_key)
        assert len(metric_history) == NUM_EPOCHS

    # Testing optimizer parameters are logged
    assert "optimizer_name" in data.params
    assert data.params["optimizer_name"] == "Adam"

    # Testing model_summary.txt is saved
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert "model_summary.txt" in artifacts


def test_pytorch_autolog_logs_expected_metrics_without_validation(pytorch_model_without_validation):
    trainer, run = pytorch_model_without_validation
    assert trainer.disable_validation

    client = mlflow.tracking.MlflowClient()
    for metric_key in ["loss", "train_acc"]:
        assert metric_key in run.data.metrics
        metric_history = client.get_metric_history(run.info.run_id, metric_key)
        assert len(metric_history) == NUM_EPOCHS


# pylint: disable=unused-argument
def test_pytorch_autolog_persists_manually_created_run():
    with mlflow.start_run() as manual_run:
        mlflow.pytorch.autolog()
        model = IrisClassification()
        dm = IrisDataModule()
        dm.setup(stage="fit")
        trainer = pl.Trainer(max_epochs=NUM_EPOCHS)
        trainer.fit(model, dm)
        trainer.test()
        assert mlflow.active_run() is not None
        assert mlflow.active_run().info.run_id == manual_run.info.run_id


def test_pytorch_autolog_ends_auto_created_run(pytorch_model):
    assert mlflow.active_run() is None


@pytest.fixture
def pytorch_model_with_callback(patience):
    mlflow.pytorch.autolog()
    model = IrisClassification()
    dm = IrisDataModule()
    dm.setup(stage="fit")
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        min_delta=99999999,  # forces early stopping
        patience=patience,
        verbose=True,
    )

    with TempDir() as tmp:
        keyword = "dirpath" if Version(pl.__version__) >= Version("1.2.0") else "filepath"
        checkpoint_callback = ModelCheckpoint(
            **{keyword: tmp.path()}, save_top_k=1, verbose=True, monitor="val_loss", mode="min",
        )

        trainer = pl.Trainer(
            max_epochs=NUM_EPOCHS * 2, callbacks=[early_stopping, checkpoint_callback],
        )
        trainer.fit(model, dm)

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


@pytest.mark.parametrize("log_models", [True, False])
@pytest.mark.parametrize("patience", [3])
def test_pytorch_with_early_stopping_autolog_log_models_configuration_with(log_models, patience):
    mlflow.pytorch.autolog(log_models=log_models)
    model = IrisClassification()
    dm = IrisDataModule()
    dm.setup(stage="fit")
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=patience, verbose=True)

    with TempDir() as tmp:
        keyword = "dirpath" if Version(pl.__version__) >= Version("1.2.0") else "filepath"
        checkpoint_callback = ModelCheckpoint(
            **{keyword: tmp.path()}, save_top_k=1, verbose=True, monitor="val_loss", mode="min",
        )

        trainer = pl.Trainer(
            max_epochs=NUM_EPOCHS * 2, callbacks=[early_stopping, checkpoint_callback],
        )
        trainer.fit(model, dm)

        client = mlflow.tracking.MlflowClient()
        run = client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)
    run_id = run.info.run_id
    client = mlflow.tracking.MlflowClient()
    artifacts = [f.path for f in client.list_artifacts(run_id)]
    assert ("restored_model_checkpoint" in artifacts) == log_models


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


def test_pytorch_autolog_non_early_stop_callback_does_not_log(pytorch_model):
    trainer, run = pytorch_model
    client = mlflow.tracking.MlflowClient()
    loss_metric_history = client.get_metric_history(run.info.run_id, "loss")
    val_loss_metric_history = client.get_metric_history(run.info.run_id, "val_loss")
    assert trainer.max_epochs == NUM_EPOCHS
    assert len(loss_metric_history) == NUM_EPOCHS
    assert len(val_loss_metric_history) == NUM_EPOCHS


@pytest.fixture
def pytorch_model_tests():
    mlflow.pytorch.autolog()
    model = IrisClassification()
    dm = IrisDataModule()
    dm.setup(stage="fit")
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS)
    with mlflow.start_run() as run:
        trainer.fit(model, dm)
        trainer.test()
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run.info.run_id)
    return trainer, run


def test_pytorch_test_metrics_logged(pytorch_model_tests):
    _, run = pytorch_model_tests
    data = run.data
    assert "test_loss" in data.metrics
    assert "test_acc" in data.metrics


def test_get_optimizer_name():
    adam = torch.optim.Adam(torch.nn.Linear(1, 1).parameters())
    assert _get_optimizer_name(adam) == "Adam"


@pytest.mark.skipif(
    Version(pl.__version__) < Version("1.1.0"),
    reason="`LightningOptimizer` doesn't exist in pytorch-lightning < 1.1.0",
)
def test_get_optimizer_name_with_lightning_optimizer():
    from pytorch_lightning.core.optimizer import LightningOptimizer

    adam = torch.optim.Adam(torch.nn.Linear(1, 1).parameters())
    assert _get_optimizer_name(LightningOptimizer(adam)) == "Adam"


def test_pytorch_autologging_supports_data_parallel_execution():
    mlflow.pytorch.autolog()
    model = IrisClassification()
    dm = IrisDataModule()
    dm.setup(stage="fit")

    trainer = pl.Trainer(max_epochs=NUM_EPOCHS, accelerator="ddp_cpu", num_processes=4)

    with mlflow.start_run() as run:
        trainer.fit(model, dm)
        trainer.test()

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run.info.run_id)

    # Checking if metrics are logged
    client = mlflow.tracking.MlflowClient()
    for metric_key in ["loss", "train_acc", "val_loss", "val_acc"]:
        assert metric_key in run.data.metrics

    data = run.data
    assert "test_loss" in data.metrics
    assert "test_acc" in data.metrics

    # Testing optimizer parameters are logged
    assert "optimizer_name" in data.params
    assert data.params["optimizer_name"] == "Adam"

    # Testing model_summary.txt is saved
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run.info.run_id)
    artifacts = list(map(lambda x: x.path, artifacts))
    assert "model" in artifacts
    assert "model_summary.txt" in artifacts
