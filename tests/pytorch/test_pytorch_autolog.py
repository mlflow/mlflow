from unittest import mock

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
from mlflow.pytorch import MLflowModelCheckpointCallback, load_checkpoint
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
        assert (
            len(metric_history) == expected_len
        ), f"Expected {expected_len} values for {metric_key}, got {len(metric_history)}"


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


class FakeTrainer:
    def __init__(self):
        self.callback_metrics = {}
        self.current_epoch = 0
        self.global_step = 0


@pytest.mark.skipif(
    Version(pl.__version__) < Version("1.6.0"),
    reason="`Automatic model checkpointing doesn't exist in pytorch-lightning < 1.6.0",
)
def test_model_checkpoint_per_epoch_callback():
    with mlflow.start_run() as run, mock.patch(
        "mlflow.client.MlflowClient.log_dict"
    ) as log_dict_mock, mock.patch(
        "mlflow.client.MlflowClient.log_artifact"
    ) as log_artifact_mock, mock.patch(
        "mlflow.pytorch.MLflowModelCheckpointCallback._save_checkpoint_rank_zero_only"
    ) as save_chekpoint_mock:
        model = object()

        client = MlflowClient()
        run_id = run.info.run_id

        # Test checkpoint per epoch, save_best_only = False, save_weights_only = False
        callback1 = MLflowModelCheckpointCallback(
            client=client,
            run_id=run_id,
            monitor=None,
            mode=None,
            save_best_only=False,
            save_weights_only=False,
            save_freq="epoch",
        )

        trainer1 = FakeTrainer()
        trainer1.current_epoch = 1
        trainer1.callback_metrics = {"loss": torch.tensor(1.5), "val_loss": torch.tensor(2.0)}
        trainer1.global_step = 100
        callback1.on_train_epoch_end(trainer1, model)

        log_dict_mock.assert_called_once_with(
            run_id,
            {"loss": 1.5, "val_loss": 2.0, "epoch": 1, "global_step": 100},
            "checkpoints/epoch_1/checkpoint_metrics.json",
        )
        log_artifact_mock.assert_called_once_with(
            run_id,
            mock.ANY,
            "checkpoints/epoch_1",
        )
        save_chekpoint_mock.assert_called_once()


@pytest.mark.skipif(
    Version(pl.__version__) < Version("1.6.0"),
    reason="`Automatic model checkpointing doesn't exist in pytorch-lightning < 1.6.0",
)
def test_model_checkpoint_per_epoch_save_weight_only_callback():
    with mlflow.start_run() as run, mock.patch(
        "mlflow.client.MlflowClient.log_dict"
    ) as log_dict_mock, mock.patch(
        "mlflow.client.MlflowClient.log_artifact"
    ) as log_artifact_mock, mock.patch(
        "mlflow.pytorch.MLflowModelCheckpointCallback._save_checkpoint_rank_zero_only"
    ) as save_chekpoint_mock:
        model = object()

        client = MlflowClient()
        run_id = run.info.run_id

        # Test save_weights_only = True
        callback2 = MLflowModelCheckpointCallback(
            client=client,
            run_id=run_id,
            monitor=None,
            mode=None,
            save_best_only=False,
            save_weights_only=True,
            save_freq="epoch",
        )

        trainer2 = FakeTrainer()
        trainer2.current_epoch = 1
        trainer2.global_step = 100
        trainer2.callback_metrics = {}
        callback2.on_train_epoch_end(trainer2, model)

        log_artifact_mock.assert_called_once_with(
            run_id,
            mock.ANY,
            "checkpoints/epoch_1",
        )
        save_chekpoint_mock.assert_called_once()


@pytest.mark.skipif(
    Version(pl.__version__) < Version("1.6.0"),
    reason="`Automatic model checkpointing doesn't exist in pytorch-lightning < 1.6.0",
)
def test_model_checkpoint_per_10_steps_callback():
    with mlflow.start_run() as run, mock.patch(
        "mlflow.client.MlflowClient.log_dict"
    ) as log_dict_mock, mock.patch(
        "mlflow.client.MlflowClient.log_artifact"
    ) as log_artifact_mock, mock.patch(
        "mlflow.pytorch.MLflowModelCheckpointCallback._save_checkpoint_rank_zero_only"
    ) as save_chekpoint_mock:
        model = object()

        client = MlflowClient()
        run_id = run.info.run_id

        # Test checkpoint per 10 steps, save_best_only = False, save_weights_only = False
        callback3 = MLflowModelCheckpointCallback(
            client=client,
            run_id=run_id,
            monitor=None,
            mode=None,
            save_best_only=False,
            save_weights_only=False,
            save_freq=10,
        )

        trainer3 = FakeTrainer()
        trainer3.current_epoch = 1
        trainer3.global_step = 5

        callback3.on_train_batch_end(trainer3, model, None, None, 5)
        log_dict_mock.assert_not_called()
        log_artifact_mock.assert_not_called()
        save_chekpoint_mock.reset_mock()

        trainer3.global_step = 10
        trainer3.callback_metrics = {"loss": 1.2, "val_loss": 1.3}
        trainer3.expect_weights_only_saving = False
        trainer3.expected_checkpoint_filename = "checkpoint.pth"
        callback3.on_train_batch_end(trainer3, model, None, None, 10)

        log_dict_mock.assert_called_once_with(
            run_id,
            {"loss": 1.2, "val_loss": 1.3, "epoch": 1, "global_step": 10},
            "checkpoints/global_step_10/checkpoint_metrics.json",
        )
        log_artifact_mock.assert_called_once_with(
            run_id,
            mock.ANY,
            "checkpoints/global_step_10",
        )


@pytest.mark.skipif(
    Version(pl.__version__) < Version("1.6.0"),
    reason="`Automatic model checkpointing doesn't exist in pytorch-lightning < 1.6.0",
)
def test_model_checkpoint_per_10_steps_save_best_only_callback():
    with mlflow.start_run() as run, mock.patch(
        "mlflow.client.MlflowClient.log_dict"
    ) as log_dict_mock, mock.patch(
        "mlflow.client.MlflowClient.log_artifact"
    ) as log_artifact_mock, mock.patch(
        "mlflow.pytorch.MLflowModelCheckpointCallback._save_checkpoint_rank_zero_only"
    ) as save_chekpoint_mock:
        model = object()

        client = MlflowClient()
        run_id = run.info.run_id

        # Test checkpoint every 10 steps, save_best_only = True, monitor = 'val_acc_step',
        # mode = "max"
        callback4 = MLflowModelCheckpointCallback(
            client=client,
            run_id=run_id,
            monitor="train_acc_step",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            save_freq=10,
        )
        trainer4 = FakeTrainer()
        trainer4.current_epoch = 1
        trainer4.global_step = 10
        trainer4.callback_metrics = {"train_acc_step": 0.7}
        trainer4.expect_weights_only_saving = False
        trainer4.expected_checkpoint_filename = "latest_checkpoint.pth"
        callback4.on_train_batch_end(trainer4, model, None, None, 10)

        log_dict_mock.assert_called_once_with(
            run_id,
            {"train_acc_step": 0.7, "epoch": 1, "global_step": 10},
            "checkpoints/latest_checkpoint_metrics.json",
        )
        log_artifact_mock.assert_called_once_with(
            run_id,
            mock.ANY,
            "checkpoints",
        )
        log_dict_mock.reset_mock()
        log_artifact_mock.reset_mock()
        save_chekpoint_mock.reset_mock()

        trainer4.current_epoch = 1
        trainer4.global_step = 20
        trainer4.callback_metrics = {"train_acc_step": 0.8}
        trainer4.expect_weights_only_saving = False
        trainer4.expected_checkpoint_filename = "latest_checkpoint.pth"
        callback4.on_train_batch_end(trainer4, model, None, None, 20)

        log_dict_mock.assert_called_once_with(
            run_id,
            {"train_acc_step": 0.8, "epoch": 1, "global_step": 20},
            "checkpoints/latest_checkpoint_metrics.json",
        )
        log_artifact_mock.assert_called_once_with(
            run_id,
            mock.ANY,
            "checkpoints",
        )


@pytest.mark.skipif(
    Version(pl.__version__) < Version("1.6.0"),
    reason="`Automatic model checkpointing doesn't exist in pytorch-lightning < 1.6.0",
)
def test_model_checkpoint_per_epoch_save_best_only_min_monitor_callback():
    with mlflow.start_run() as run, mock.patch(
        "mlflow.client.MlflowClient.log_dict"
    ) as log_dict_mock, mock.patch(
        "mlflow.client.MlflowClient.log_artifact"
    ) as log_artifact_mock, mock.patch(
        "mlflow.pytorch.MLflowModelCheckpointCallback._save_checkpoint_rank_zero_only"
    ) as save_chekpoint_mock:
        model = object()

        client = MlflowClient()
        run_id = run.info.run_id

        # Test checkpoint per epoch, save_best_only = True, monitor = 'val_loss', mode = "min"
        callback5 = MLflowModelCheckpointCallback(
            client=client,
            run_id=run_id,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            save_freq="epoch",
        )

        trainer5 = FakeTrainer()
        trainer5.current_epoch = 1
        trainer5.global_step = 100
        trainer5.callback_metrics = {"loss": 1.5, "val_loss": 1.6}
        trainer5.expect_weights_only_saving = False
        trainer5.expected_checkpoint_filename = "latest_checkpoint.pth"
        callback5.on_train_epoch_end(trainer5, model)

        log_dict_mock.assert_called_once_with(
            run_id,
            {"loss": 1.5, "val_loss": 1.6, "epoch": 1, "global_step": 100},
            "checkpoints/latest_checkpoint_metrics.json",
        )
        log_artifact_mock.assert_called_once_with(
            run_id,
            mock.ANY,
            "checkpoints",
        )
        log_dict_mock.reset_mock()
        log_artifact_mock.reset_mock()
        save_chekpoint_mock.reset_mock()

        trainer5.current_epoch = 2
        trainer5.callback_metrics = {"loss": 1.4, "val_loss": 1.65}
        callback5.on_train_epoch_end(trainer5, model)
        log_dict_mock.assert_not_called()
        log_artifact_mock.assert_not_called()
        save_chekpoint_mock.reset_mock()

        trainer5.current_epoch = 3
        trainer5.callback_metrics = {"loss": 1.3, "val_loss": 1.5}
        trainer5.expect_weights_only_saving = False
        trainer5.expected_checkpoint_filename = "latest_checkpoint.pth"
        callback5.on_train_epoch_end(trainer5, model)

        log_dict_mock.assert_called_once_with(
            run_id,
            {"loss": 1.3, "val_loss": 1.5, "epoch": 3, "global_step": 100},
            "checkpoints/latest_checkpoint_metrics.json",
        )
        log_artifact_mock.assert_called_once_with(
            run_id,
            mock.ANY,
            "checkpoints",
        )


@pytest.mark.skipif(
    Version(pl.__version__) < Version("1.6.0"),
    reason="`Automatic model checkpointing doesn't exist in pytorch-lightning < 1.6.0",
)
def test_model_checkpoint_per_epoch_save_best_only_max_monitor_callback():
    with mlflow.start_run() as run, mock.patch(
        "mlflow.client.MlflowClient.log_dict"
    ) as log_dict_mock, mock.patch(
        "mlflow.client.MlflowClient.log_artifact"
    ) as log_artifact_mock, mock.patch(
        "mlflow.pytorch.MLflowModelCheckpointCallback._save_checkpoint_rank_zero_only"
    ) as save_chekpoint_mock:
        model = object()

        client = MlflowClient()
        run_id = run.info.run_id

        # Test checkpoint per epoch, save_best_only = True, monitor = 'val_acc', mode = "max"
        callback6 = MLflowModelCheckpointCallback(
            client=client,
            run_id=run_id,
            monitor="val_acc",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            save_freq="epoch",
        )

        trainer6 = FakeTrainer()
        trainer6.current_epoch = 1
        trainer6.global_step = 100
        trainer6.callback_metrics = {"acc": 0.9, "val_acc": 0.8}
        trainer6.expect_weights_only_saving = False
        trainer6.expected_checkpoint_filename = "latest_checkpoint.pth"
        callback6.on_train_epoch_end(trainer6, model)

        log_dict_mock.assert_called_once_with(
            run_id,
            {"acc": 0.9, "val_acc": 0.8, "epoch": 1, "global_step": 100},
            "checkpoints/latest_checkpoint_metrics.json",
        )
        log_artifact_mock.assert_called_once_with(
            run_id,
            mock.ANY,
            "checkpoints",
        )
        log_dict_mock.reset_mock()
        log_artifact_mock.reset_mock()
        save_chekpoint_mock.reset_mock()

        trainer6.current_epoch = 2
        trainer6.callback_metrics = {"acc": 0.95, "val_acc": 0.85}
        trainer6.expect_weights_only_saving = False
        trainer6.expected_checkpoint_filename = "latest_checkpoint.pth"
        callback6.on_train_epoch_end(trainer6, model)

        log_dict_mock.assert_called_once_with(
            run_id,
            {"acc": 0.95, "val_acc": 0.85, "epoch": 2, "global_step": 100},
            "checkpoints/latest_checkpoint_metrics.json",
        )
        log_artifact_mock.assert_called_once_with(
            run_id,
            mock.ANY,
            "checkpoints",
        )


@pytest.mark.skipif(
    Version(pl.__version__) < Version("1.6.0"),
    reason="`Automatic model checkpointing doesn't exist in pytorch-lightning < 1.6.0",
)
def test_per_epoch_automatic_model_checkpoint():
    mlflow.pytorch.autolog(
        checkpoint=True,
        checkpoint_monitor="val_loss",
        checkpoint_mode="min",
        checkpoint_save_best_only=True,
        checkpoint_save_weights_only=True,
        checkpoint_save_freq="epoch",
    )

    model = IrisClassification()
    dm = IrisDataModule()
    dm.setup(stage="fit")
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS)

    with mlflow.start_run() as run:
        trainer.fit(model, dm)

    client = MlflowClient()
    artifacts = [artifact.path for artifact in client.list_artifacts(run.info.run_id)]

    assert "checkpoints" in artifacts

    result_metrics = mlflow.artifacts.load_dict(
        f"runs:/{run.info.run_id}/checkpoints/latest_checkpoint_metrics.json"
    )

    assert set(result_metrics.keys()) == {
        "loss",
        "global_step",
        "loss_forked",
        "loss_forked_step",
        "val_acc",
        "val_loss",
        "train_acc",
        "loss_forked_epoch",
        "epoch",
    }

    loaded_model = load_checkpoint(IrisClassification, run.info.run_id)
    assert isinstance(loaded_model, IrisClassification)

    # Test logging all history checkpoints
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
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS)

    with mlflow.start_run() as run:
        trainer.fit(model, dm)

    client = MlflowClient()
    artifacts = [artifact.path for artifact in client.list_artifacts(run.info.run_id)]

    assert "checkpoints" in artifacts

    result_metrics = mlflow.artifacts.load_dict(
        f"runs:/{run.info.run_id}/checkpoints/epoch_{NUM_EPOCHS - 1}/checkpoint_metrics.json"
    )

    assert set(result_metrics.keys()) == {
        "loss",
        "global_step",
        "loss_forked",
        "loss_forked_step",
        "val_acc",
        "val_loss",
        "train_acc",
        "loss_forked_epoch",
        "epoch",
    }

    loaded_latest_model = load_checkpoint(IrisClassification, run.info.run_id)
    assert isinstance(loaded_latest_model, IrisClassification)

    loaded_history_model = load_checkpoint(
        IrisClassification, run.info.run_id, epoch=NUM_EPOCHS // 2
    )
    assert isinstance(loaded_history_model, IrisClassification)


@pytest.mark.skipif(
    Version(pl.__version__) < Version("1.6.0"),
    reason="`Automatic model checkpointing doesn't exist in pytorch-lightning < 1.6.0",
)
def test_per_n_steps_automatic_model_checkpoint():
    mlflow.pytorch.autolog(
        checkpoint=True,
        checkpoint_monitor="loss_forked_step",
        checkpoint_mode="min",
        checkpoint_save_best_only=True,
        checkpoint_save_weights_only=True,
        checkpoint_save_freq="epoch",
    )

    model = IrisClassification()
    dm = IrisDataModule()
    dm.setup(stage="fit")
    trainer = pl.Trainer(max_epochs=4)

    with mlflow.start_run() as run:
        trainer.fit(model, dm)

    client = MlflowClient()
    artifacts = [artifact.path for artifact in client.list_artifacts(run.info.run_id)]

    assert "checkpoints" in artifacts

    result_metrics = mlflow.artifacts.load_dict(
        f"runs:/{run.info.run_id}/checkpoints/latest_checkpoint_metrics.json"
    )

    assert set(result_metrics.keys()) == {
        "loss",
        "global_step",
        "loss_forked",
        "loss_forked_step",
        "val_acc",
        "val_loss",
        "train_acc",
        "loss_forked_epoch",
        "epoch",
    }

    loaded_model = load_checkpoint(IrisClassification, run.info.run_id)
    assert isinstance(loaded_model, IrisClassification)

    # Test logging all history checkpoints
    mlflow.pytorch.autolog(
        checkpoint=True,
        checkpoint_monitor=None,
        checkpoint_mode=None,
        checkpoint_save_best_only=False,
        checkpoint_save_weights_only=True,
        checkpoint_save_freq=20,
    )

    model = IrisClassification()
    dm = IrisDataModule()
    dm.setup(stage="fit")
    trainer = pl.Trainer(max_epochs=4)

    with mlflow.start_run() as run:
        trainer.fit(model, dm)

    client = MlflowClient()
    artifacts = [artifact.path for artifact in client.list_artifacts(run.info.run_id)]

    assert "checkpoints" in artifacts

    result_metrics = mlflow.artifacts.load_dict(
        f"runs:/{run.info.run_id}/checkpoints/global_step_120/checkpoint_metrics.json"
    )

    assert set(result_metrics.keys()) == {
        "loss",
        "global_step",
        "loss_forked",
        "loss_forked_step",
        "val_acc",
        "val_loss",
        "train_acc",
        "loss_forked_epoch",
        "epoch",
    }

    loaded_latest_model = load_checkpoint(IrisClassification, run.info.run_id)
    assert isinstance(loaded_latest_model, IrisClassification)

    loaded_history_model = load_checkpoint(IrisClassification, run.info.run_id, global_step=40)
    assert isinstance(loaded_history_model, IrisClassification)
