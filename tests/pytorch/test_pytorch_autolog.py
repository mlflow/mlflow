import pytest
import pytorch_lightning as pl
import torch
from iris import (
    IrisClassification,
    IrisClassificationMultiOptimizer,
    IrisClassificationWithoutValidation,
)
from iris_data_module import IrisDataModule, IrisDataModuleWithoutValidation
from packaging.version import Version
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import mlflow
import mlflow.pytorch
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.pytorch._lightning_autolog import _get_optimizer_name
from mlflow.utils.file_utils import TempDir

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
    run = client.get_run(client.search_runs(["0"])[0].info.run_id)
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
    run = client.get_run(client.search_runs(["0"])[0].info.run_id)
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
    run = client.get_run(client.search_runs(["0"])[0].info.run_id)
    return trainer, run, log_every_n_epoch, log_every_n_step


@pytest.fixture(params=[(1, 1), (1, 10), (2, 1)])
def pytorch_multi_optimizer_model(request):
    log_every_n_epoch, log_every_n_step = request.param
    mlflow.pytorch.autolog(log_every_n_epoch=log_every_n_epoch, log_every_n_step=log_every_n_step)
    model = IrisClassificationMultiOptimizer()
    dm = IrisDataModule()
    dm.setup(stage="fit")
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS)
    trainer.fit(model, dm)
    client = MlflowClient()
    run = client.get_run(client.search_runs(["0"])[0].info.run_id)
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
    run = client.get_run(client.search_runs(["0"])[0].info.run_id)
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


def test_extra_tags_pytorch_autolog():
    mlflow.pytorch.autolog(extra_tags={"test_tag": "pytorch_autolog"})
    model = IrisClassification()
    dm = IrisDataModule()
    dm.setup(stage="fit")
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS)
    trainer.fit(model, dm)

    run = mlflow.last_active_run()
    assert run.data.tags["test_tag"] == "pytorch_autolog"
    assert run.data.tags[mlflow.utils.mlflow_tags.MLFLOW_AUTOLOGGING] == "pytorch"


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
    artifacts = (x.path for x in artifacts)
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
    reason="Access to metrics from the current step is only possible since PyTorch-lightning 1.1.0 "
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
    for metric_key, expected_len in [
        ("train_acc", num_logged_epochs),
        ("loss", num_logged_steps),
        ("loss_forked", num_logged_epochs),
        ("loss_forked_step", num_logged_steps),
        ("loss_forked_epoch", num_logged_epochs),
    ]:
        assert metric_key in run.data.metrics, f"Missing {metric_key} in metrics"
        metric_history = client.get_metric_history(run.info.run_id, metric_key)
        assert len(metric_history) == expected_len, (
            f"Expected {expected_len} values for {metric_key}, got {len(metric_history)}"
        )


@pytest.mark.skipif(
    Version(pl.__version__) < Version("1.1.0"),
    reason="Access to metrics from the current step is only possible since PyTorch-lightning 1.1.0 "
    "when LoggerConnector.cached_results was added",
)
def test_pytorch_autolog_log_on_step_with_multiple_optimizers(
    pytorch_multi_optimizer_model,
):
    trainer, run, log_every_n_epoch, log_every_n_step = pytorch_multi_optimizer_model
    num_logged_steps = NUM_EPOCHS * len(trainer.train_dataloader) // log_every_n_step
    num_logged_epochs = NUM_EPOCHS // log_every_n_epoch

    client = MlflowClient()
    for metric_key, expected_len in [
        ("loss", num_logged_epochs),
        ("loss_step", num_logged_steps),
    ]:
        assert metric_key in run.data.metrics, f"Missing {metric_key} in metrics"
        metric_history = client.get_metric_history(run.info.run_id, metric_key)
        assert len(metric_history) == expected_len, (
            f"Expected {expected_len} values for {metric_key}, got {len(metric_history)}"
        )


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
        run = client.get_run(client.search_runs(["0"])[0].info.run_id)

    return trainer, run


@pytest.mark.parametrize("patience", [3])
def test_pytorch_early_stop_artifacts_logged(pytorch_model_with_callback):
    _, run = pytorch_model_with_callback
    client = MlflowClient()
    artifacts = client.list_artifacts(run.info.run_id)
    artifacts = (x.path for x in artifacts)
    assert "restored_model_checkpoint" in artifacts


@pytest.mark.parametrize("patience", [3])
def test_pytorch_autolog_model_can_load_from_artifact(pytorch_model_with_callback):
    _, run = pytorch_model_with_callback
    run_id = run.info.run_id
    client = MlflowClient()
    artifacts = client.list_artifacts(run_id)
    artifacts = (x.path for x in artifacts)
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
        run = client.get_run(client.search_runs(["0"])[0].info.run_id)
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
    # this is logged through SummaryWriter
    assert "plain_loss" not in data.metrics


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
    devices_kwarg_name = (
        "devices" if Version(pl.__version__) > Version("1.6.4") else "num_processes"
    )
    extra_kwargs = {"strategy": "ddp_spawn"} if Version(pl.__version__) > Version("1.9.3") else {}
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator=accelerator,
        **{
            devices_kwarg_name: 4,
        },
        **extra_kwargs,
    )

    with mlflow.start_run() as run:
        trainer.fit(model, datamodule=dm)
        trainer.test(datamodule=dm)

    client = MlflowClient()
    run = client.get_run(run.info.run_id)

    # Checking if metrics are logged
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
    artifacts = [x.path for x in artifacts]
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


@pytest.mark.skipif(
    Version(pl.__version__) < Version("1.6.0"),
    reason="`Automatic model checkpointing doesn't exist in pytorch-lightning < 1.6.0",
)
def test_automatic_checkpoint_per_epoch_callback():
    mlflow.pytorch.autolog(
        checkpoint=True,
        checkpoint_monitor=None,
        checkpoint_mode=None,
        checkpoint_save_best_only=False,
        checkpoint_save_weights_only=False,
        checkpoint_save_freq="epoch",
    )

    model = IrisClassification()
    dm = IrisDataModule()
    dm.setup(stage="fit")
    trainer = pl.Trainer(max_epochs=1)

    with mlflow.start_run() as run:
        trainer.fit(model, dm)

    run_id = run.info.run_id

    logged_metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    logged_metrics.update({"epoch": 0, "global_step": 33})
    assert logged_metrics == mlflow.artifacts.load_dict(
        f"runs:/{run_id}/checkpoints/epoch_0/checkpoint_metrics.json"
    )

    IrisClassification.load_from_checkpoint(
        mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="checkpoints/epoch_0/checkpoint.pth"
        )
    )


@pytest.mark.skipif(
    Version(pl.__version__) < Version("1.6.0"),
    reason="`Automatic model checkpointing doesn't exist in pytorch-lightning < 1.6.0",
)
def test_automatic_checkpoint_per_epoch_save_weight_only_callback():
    mlflow.pytorch.autolog(
        checkpoint=True,
        checkpoint_monitor=None,
        checkpoint_mode=None,
        checkpoint_save_best_only=False,
        checkpoint_save_weights_only=True,
        checkpoint_save_freq="epoch",
    )

    model = IrisClassification()
    dm = IrisDataModule()
    dm.setup(stage="fit")
    trainer = pl.Trainer(max_epochs=1)

    with mlflow.start_run() as run:
        trainer.fit(model, dm)

    run_id = run.info.run_id

    logged_metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    logged_metrics.update({"epoch": 0, "global_step": 33})
    assert logged_metrics == mlflow.artifacts.load_dict(
        f"runs:/{run_id}/checkpoints/epoch_0/checkpoint_metrics.json"
    )

    IrisClassification.load_from_checkpoint(
        mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="checkpoints/epoch_0/checkpoint.weights.pth"
        )
    )


@pytest.mark.skipif(
    Version(pl.__version__) < Version("1.6.0"),
    reason="`Automatic model checkpointing doesn't exist in pytorch-lightning < 1.6.0",
)
def test_automatic_checkpoint_per_10_steps_callback():
    mlflow.pytorch.autolog(
        checkpoint=True,
        checkpoint_monitor=None,
        checkpoint_mode=None,
        checkpoint_save_best_only=False,
        checkpoint_save_weights_only=False,
        checkpoint_save_freq=10,
    )

    model = IrisClassification()
    dm = IrisDataModule()
    dm.setup(stage="fit")
    trainer = pl.Trainer(max_epochs=1)

    with mlflow.start_run() as run:
        trainer.fit(model, dm)

    run_id = run.info.run_id

    metric_keys = {"epoch", "loss_forked_step", "loss", "global_step", "loss_forked"}
    assert metric_keys == set(
        mlflow.artifacts.load_dict(
            f"runs:/{run_id}/checkpoints/global_step_10/checkpoint_metrics.json"
        )
    )
    IrisClassification.load_from_checkpoint(
        mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="checkpoints/global_step_10/checkpoint.pth"
        )
    )


@pytest.mark.skipif(
    Version(pl.__version__) < Version("1.6.0"),
    reason="`Automatic model checkpointing doesn't exist in pytorch-lightning < 1.6.0",
)
def test_automatic_checkpoint_per_30_steps_save_best_only_callback():
    mlflow.pytorch.autolog(
        checkpoint=True,
        checkpoint_monitor="loss_forked_step",
        checkpoint_mode="min",
        checkpoint_save_best_only=True,
        checkpoint_save_weights_only=False,
        checkpoint_save_freq=30,
    )

    model = IrisClassification()
    dm = IrisDataModule()
    dm.setup(stage="fit")
    trainer = pl.Trainer(max_epochs=1)

    with mlflow.start_run() as run:
        trainer.fit(model, dm)

    run_id = run.info.run_id

    metric_keys = {"epoch", "loss_forked_step", "loss", "global_step", "loss_forked"}
    logged_metrics = mlflow.artifacts.load_dict(
        f"runs:/{run_id}/checkpoints/latest_checkpoint_metrics.json"
    )
    assert logged_metrics["global_step"] == 30
    assert metric_keys == set(logged_metrics)

    IrisClassification.load_from_checkpoint(
        mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="checkpoints/latest_checkpoint.pth"
        )
    )


@pytest.mark.skipif(
    Version(pl.__version__) < Version("1.6.0"),
    reason="`Automatic model checkpointing doesn't exist in pytorch-lightning < 1.6.0",
)
def test_automatic_checkpoint_per_epoch_save_best_only_min_monitor_callback():
    mlflow.pytorch.autolog(
        checkpoint=True,
        checkpoint_monitor="custom_metric",
        checkpoint_mode="min",
        checkpoint_save_best_only=True,
        checkpoint_save_weights_only=False,
        checkpoint_save_freq="epoch",
    )

    class CustomIrisClassification(IrisClassification):
        def validation_step(self, batch, batch_idx):
            super().validation_step(batch, batch_idx)
            if self.current_epoch == 0:
                self.log("custom_metric", 0.8)
            elif self.current_epoch == 1:
                self.log("custom_metric", 0.9)
            elif self.current_epoch == 2:
                self.log("custom_metric", 0.85)  # better than the previous epoch, but not the best
            else:
                self.log("custom_metric", 0.7)

    model = CustomIrisClassification()
    dm = IrisDataModule()
    dm.setup(stage="fit")
    trainer = pl.Trainer(max_epochs=1)

    with mlflow.start_run() as run:
        trainer.fit(model, dm)

    run_id = run.info.run_id

    logged_metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    logged_metrics.update({"epoch": 0, "global_step": 33})
    assert logged_metrics == mlflow.artifacts.load_dict(
        f"runs:/{run_id}/checkpoints/latest_checkpoint_metrics.json"
    )

    IrisClassification.load_from_checkpoint(
        mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="checkpoints/latest_checkpoint.pth"
        )
    )

    trainer = pl.Trainer(max_epochs=2)
    with mlflow.start_run() as run:
        trainer.fit(model, dm)
    run_id = run.info.run_id
    assert (
        mlflow.artifacts.load_dict(f"runs:/{run_id}/checkpoints/latest_checkpoint_metrics.json")[
            "epoch"
        ]
        == 0
    )

    trainer = pl.Trainer(max_epochs=3)
    with mlflow.start_run() as run:
        trainer.fit(model, dm)
    run_id = run.info.run_id
    assert (
        mlflow.artifacts.load_dict(f"runs:/{run_id}/checkpoints/latest_checkpoint_metrics.json")[
            "epoch"
        ]
        == 0
    )

    trainer = pl.Trainer(max_epochs=4)
    with mlflow.start_run() as run:
        trainer.fit(model, dm)
    run_id = run.info.run_id
    assert (
        mlflow.artifacts.load_dict(f"runs:/{run_id}/checkpoints/latest_checkpoint_metrics.json")[
            "epoch"
        ]
        == 3
    )


@pytest.mark.skipif(
    Version(pl.__version__) < Version("1.6.0"),
    reason="`Automatic model checkpointing doesn't exist in pytorch-lightning < 1.6.0",
)
def test_automatic_checkpoint_per_epoch_save_best_only_max_monitor_callback():
    mlflow.pytorch.autolog(
        checkpoint=True,
        checkpoint_monitor="custom_metric",
        checkpoint_mode="max",
        checkpoint_save_best_only=True,
        checkpoint_save_weights_only=False,
        checkpoint_save_freq="epoch",
    )

    class CustomIrisClassification(IrisClassification):
        def validation_step(self, batch, batch_idx):
            super().validation_step(batch, batch_idx)
            if self.current_epoch == 0:
                self.log("custom_metric", 0.8)
            elif self.current_epoch == 1:
                self.log("custom_metric", 0.7)
            else:
                self.log("custom_metric", 0.9)

    model = CustomIrisClassification()
    dm = IrisDataModule()
    dm.setup(stage="fit")
    trainer = pl.Trainer(max_epochs=1)

    with mlflow.start_run() as run:
        trainer.fit(model, dm)

    run_id = run.info.run_id

    logged_metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    logged_metrics.update({"epoch": 0, "global_step": 33})
    assert logged_metrics == mlflow.artifacts.load_dict(
        f"runs:/{run_id}/checkpoints/latest_checkpoint_metrics.json"
    )

    IrisClassification.load_from_checkpoint(
        mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="checkpoints/latest_checkpoint.pth"
        )
    )

    trainer = pl.Trainer(max_epochs=2)
    with mlflow.start_run() as run:
        trainer.fit(model, dm)
    run_id = run.info.run_id
    assert (
        mlflow.artifacts.load_dict(f"runs:/{run_id}/checkpoints/latest_checkpoint_metrics.json")[
            "epoch"
        ]
        == 0
    )

    trainer = pl.Trainer(max_epochs=3)
    with mlflow.start_run() as run:
        trainer.fit(model, dm)
    run_id = run.info.run_id
    assert (
        mlflow.artifacts.load_dict(f"runs:/{run_id}/checkpoints/latest_checkpoint_metrics.json")[
            "epoch"
        ]
        == 2
    )
