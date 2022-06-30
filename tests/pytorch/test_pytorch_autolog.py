from packaging.version import Version
import pytest
import pytorch_lightning as pl
import torch
from iris import IrisClassification, IrisClassificationWithoutValidation
import mlflow
from mlflow import MlflowClient
import mlflow.pytorch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from mlflow.utils.file_utils import TempDir
from iris_data_module import IrisDataModule, IrisDataModuleWithoutValidation
from mlflow.exceptions import MlflowException
from mlflow.pytorch._pytorch_autolog import _get_optimizer_name

NUM_EPOCHS = 20


@pytest.fixture
def pytorch_model():
    mlflow.pytorch.autolog()
    model = IrisClassification()
    dm = IrisDataModule()
    dm.setup(stage="fit")
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS)
    trainer.fit(model, dm)
    client = MlflowClient()
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
    client = MlflowClient()
    run = client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)
    return trainer, run


@pytest.fixture(params=[(1, 1), (1, 10), (2, 1)])
def pytorch_model_with_steps_logged(request):
    log_every_n_epoch, log_every_n_step = request.param
    mlflow.pytorch.autolog(log_every_n_epoch=log_every_n_epoch, log_every_n_step=log_every_n_step)
    model = IrisClassification()
    dm = IrisDataModule()
    dm.setup(stage="fit")
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS)
    trainer.fit(model, dm)
    client = MlflowClient()
    run = client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)
    return trainer, run, log_every_n_epoch, log_every_n_step


@pytest.mark.parametrize("log_models", [True, False])
def test_pytorch_autolog_log_models_configuration(log_models):
    mlflow.pytorch.autolog(log_models=log_models)
    model = IrisClassification()
    dm = IrisDataModule()
    dm.setup(stage="fit")
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS)
    trainer.fit(model, dm)
    client = MlflowClient()
    run = client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)
    run_id = run.info.run_id
    client = MlflowClient()
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

    # Checking if metrics are logged.
    # When autolog is configured with the default configuration to not log on steps,
    # then all metrics are logged per epoch, including step based metrics.
    client = MlflowClient()
    for metric_key in [
        "loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "loss_forked",
        "loss_forked_step",
        "loss_forked_epoch",
    ]:
        assert metric_key in run.data.metrics
        metric_history = client.get_metric_history(run.info.run_id, metric_key)
        assert len(metric_history) == NUM_EPOCHS

    # Testing optimizer parameters are logged
    assert "optimizer_name" in data.params
    assert data.params["optimizer_name"] == "Adam"

    # Testing model_summary.txt is saved
    client = MlflowClient()
    artifacts = client.list_artifacts(run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert "model_summary.txt" in artifacts


def test_pytorch_autolog_logs_expected_metrics_without_validation(pytorch_model_without_validation):
    trainer, run = pytorch_model_without_validation
    assert not trainer.enable_validation

    client = MlflowClient()
    for metric_key in ["loss", "train_acc"]:
        assert metric_key in run.data.metrics
        metric_history = client.get_metric_history(run.info.run_id, metric_key)
        assert len(metric_history) == NUM_EPOCHS


@pytest.mark.skipif(
    Version(pl.__version__) < Version("1.1.0"),
    reason="Access to metrics from the current step is only possible since PyTorch-lightning 1.1.0"
    "when LoggerConnector.cached_results was added",
)
def test_pytorch_autolog_logging_forked_metrics_on_step_and_epoch(
    pytorch_model_with_steps_logged,
):
    # When autolog is configured to log on steps as well as epochs,
    # then we only log step based metrics per step and not on epochs.
    trainer, run, log_every_n_epoch, log_every_n_step = pytorch_model_with_steps_logged
    num_logged_steps = trainer.global_step // log_every_n_step
    num_logged_epochs = NUM_EPOCHS // log_every_n_epoch

    client = MlflowClient()
    for (metric_key, expected_len) in [
        ("train_acc", num_logged_epochs),
        ("loss", num_logged_steps),
        ("loss_forked", num_logged_epochs),
        ("loss_forked_step", num_logged_steps),
        ("loss_forked_epoch", num_logged_epochs),
    ]:
        assert metric_key in run.data.metrics, f"Missing {metric_key} in metrics"
        metric_history = client.get_metric_history(run.info.run_id, metric_key)
        assert (
            len(metric_history) == expected_len
        ), f"Expected {expected_len} values for {metric_key}, got {len(metric_history)}"


@pytest.mark.skipif(
    Version(pl.__version__) >= Version("1.1.0"),
    reason="Logging step metrics is supported since PyTorch-Lightning 1.1.0",
)
def test_pytorch_autolog_raises_error_when_step_logging_unsupported():
    mlflow.pytorch.autolog(log_every_n_step=1)
    model = IrisClassification()
    dm = IrisDataModule()
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS)
    with pytest.raises(
        MlflowException, match="log_every_n_step is only supported for PyTorch-Lightning >= 1.1.0"
    ):
        trainer.fit(model, dm)


# pylint: disable=unused-argument
def test_pytorch_autolog_persists_manually_created_run():
    with mlflow.start_run() as manual_run:
        mlflow.pytorch.autolog()
        model = IrisClassification()
        dm = IrisDataModule()
        dm.setup(stage="fit")
        trainer = pl.Trainer(max_epochs=NUM_EPOCHS)
        trainer.fit(model, dm)
        trainer.test(datamodule=dm)
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
            **{keyword: tmp.path()},
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min",
        )

        trainer = pl.Trainer(
            max_epochs=NUM_EPOCHS * 2,
            callbacks=[early_stopping, checkpoint_callback],
        )
        trainer.fit(model, dm)

        client = MlflowClient()
        run = client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)

    return trainer, run


@pytest.mark.parametrize("patience", [3])
def test_pytorch_early_stop_artifacts_logged(pytorch_model_with_callback):
    _, run = pytorch_model_with_callback
    client = MlflowClient()
    artifacts = client.list_artifacts(run.info.run_id)
    artifacts = map(lambda x: x.path, artifacts)
    assert "restored_model_checkpoint" in artifacts


@pytest.mark.parametrize("patience", [3])
def test_pytorch_autolog_model_can_load_from_artifact(pytorch_model_with_callback):
    _, run = pytorch_model_with_callback
    run_id = run.info.run_id
    client = MlflowClient()
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
            **{keyword: tmp.path()},
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min",
        )

        trainer = pl.Trainer(
            max_epochs=NUM_EPOCHS * 2,
            callbacks=[early_stopping, checkpoint_callback],
        )
        trainer.fit(model, dm)

        client = MlflowClient()
        run = client.get_run(client.list_run_infos(experiment_id="0")[0].run_id)
    run_id = run.info.run_id
    client = MlflowClient()
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
    client = MlflowClient()
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
        trainer.fit(model, datamodule=dm)
        trainer.test(datamodule=dm)
    client = MlflowClient()
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

    accelerator = "cpu" if Version(pl.__version__) > Version("1.6.4") else "ddp_cpu"
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS, accelerator=accelerator, num_processes=4)

    with mlflow.start_run() as run:
        trainer.fit(model, datamodule=dm)
        trainer.test(datamodule=dm)

    client = MlflowClient()
    run = client.get_run(run.info.run_id)

    # Checking if metrics are logged
    client = MlflowClient()
    for metric_key in ["loss", "train_acc", "val_loss", "val_acc"]:
        assert metric_key in run.data.metrics

    data = run.data
    assert "test_loss" in data.metrics
    assert "test_acc" in data.metrics

    # Testing optimizer parameters are logged
    assert "optimizer_name" in data.params
    assert data.params["optimizer_name"] == "Adam"

    # Testing model_summary.txt is saved
    client = MlflowClient()
    artifacts = client.list_artifacts(run.info.run_id)
    artifacts = list(map(lambda x: x.path, artifacts))
    assert "model" in artifacts
    assert "model_summary.txt" in artifacts


def test_autolog_registering_model():
    registered_model_name = "test_autolog_registered_model"
    mlflow.pytorch.autolog(registered_model_name=registered_model_name)
    model = IrisClassification()
    dm = IrisDataModule()
    dm.setup(stage="fit")
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS)

    with mlflow.start_run():
        trainer.fit(model, dm)

        registered_model = MlflowClient().get_registered_model(registered_model_name)
        assert registered_model.name == registered_model_name
