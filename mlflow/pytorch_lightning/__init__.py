"""
The ``mlflow.pytorch_lightning`` module provides autologging for pytorch lightning.
It is compatible with mlflow.pytorch.

PyTorch (native) format
    This is the main flavor that can be loaded back into PyTorch.
:py:mod:`mlflow.pyfunc`
    Produced for use by generic pyfunc-based deployment tools and batch inference.
"""
from mlflow.utils.autologging_utils import autologging_integration, safe_patch

FLAVOR_NAME = "pytorch_lightning"


@autologging_integration(FLAVOR_NAME)
def autolog(
    log_every_n_epoch=1,
    log_every_n_step=None,
    log_models=True,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    registered_model_name=None,
):  # pylint: disable=unused-argument
    """
    Enables (or disables) and configures autologging from `PyTorch Lightning
    <https://pytorch-lightning.readthedocs.io/en/latest>`_ to MLflow.

    Autologging is performed when you call the `fit` method of
    `pytorch_lightning.Trainer() \
    <https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#>`_.

    Explore the complete `PyTorch MNIST \
    <https://github.com/mlflow/mlflow/tree/master/examples/pytorch/MNIST>`_ for
    an expansive example with implementation of additional lightning steps.

    **Note**: Autologging is only supported for PyTorch Lightning models,
    i.e., models that subclass
    `pytorch_lightning.LightningModule \
    <https://pytorch-lightning.readthedocs.io/en/latest/lightning_module.html>`_.
    In particular, autologging support for vanilla PyTorch models that only subclass
    `torch.nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`_
    is not yet available.

    :param log_every_n_epoch: If specified, logs metrics once every `n` epochs. By default, metrics
                       are logged after every epoch.
    :param log_every_n_step: If specified, logs batch metrics once every `n` global step.
                       By default, metrics are not logged for steps. Note that setting this to 1 can
                       cause performance issues and is not recommended.
    :param log_models: If ``True``, trained models are logged as MLflow model artifacts.
                       If ``False``, trained models are not logged.
    :param disable: If ``True``, disables the PyTorch Lightning autologging integration.
                    If ``False``, enables the PyTorch Lightning autologging integration.
    :param exclusive: If ``True``, autologged content is not logged to user-created fluent runs.
                      If ``False``, autologged content is logged to the active fluent run,
                      which may be user-created.
    :param disable_for_unsupported_versions: If ``True``, disable autologging for versions of
                      pytorch and pytorch-lightning that have not been tested against this version
                      of the MLflow client or are incompatible.
    :param silent: If ``True``, suppress all event logs and warnings from MLflow during PyTorch
                   Lightning autologging. If ``False``, show all events and warnings during
                   PyTorch Lightning autologging.
    :param registered_model_name: If given, each time a model is trained, it is registered as a
                                  new model version of the registered model with this name.
                                  The registered model is created if it does not already exist.

    .. code-block:: python
        :caption: Example

        import os

        import pytorch_lightning as pl
        import torch
        from torch.nn import functional as F
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from torchvision.datasets import MNIST

        try:
            from torchmetrics.functional import accuracy
        except ImportError:
            from pytorch_lightning.metrics.functional import accuracy

        import mlflow.pytorch
        from mlflow import MlflowClient

        # For brevity, here is the simplest most minimal example with just a training
        # loop step, (no validation, no testing). It illustrates how you can use MLflow
        # to auto log parameters, metrics, and models.

        class MNISTModel(pl.LightningModule):
            def __init__(self):
                super().__init__()
                self.l1 = torch.nn.Linear(28 * 28, 10)

            def forward(self, x):
                return torch.relu(self.l1(x.view(x.size(0), -1)))

            def training_step(self, batch, batch_nb):
                x, y = batch
                logits = self(x)
                loss = F.cross_entropy(logits, y)
                pred = logits.argmax(dim=1)
                acc = accuracy(pred, y)

                # Use the current of PyTorch logger
                self.log("train_loss", loss, on_epoch=True)
                self.log("acc", acc, on_epoch=True)
                return loss

            def configure_optimizers(self):
                return torch.optim.Adam(self.parameters(), lr=0.02)

        def print_auto_logged_info(r):

            tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
            artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
            print("run_id: {}".format(r.info.run_id))
            print("artifacts: {}".format(artifacts))
            print("params: {}".format(r.data.params))
            print("metrics: {}".format(r.data.metrics))
            print("tags: {}".format(tags))

        # Initialize our model
        mnist_model = MNISTModel()

        # Initialize DataLoader from MNIST Dataset
        train_ds = MNIST(os.getcwd(), train=True,
            download=True, transform=transforms.ToTensor())
        train_loader = DataLoader(train_ds, batch_size=32)

        # Initialize a trainer
        trainer = pl.Trainer(max_epochs=20, progress_bar_refresh_rate=20)

        # Auto log all MLflow entities
        mlflow.pytorch_lightning.autolog()

        # Train the model
        with mlflow.start_run() as run:
            trainer.fit(mnist_model, train_loader)

        # fetch the auto logged parameters and metrics
        print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

    .. code-block:: text
        :caption: Output

        run_id: 42caa17b60cb489c8083900fb52506a7
        artifacts: ['model/MLmodel', 'model/conda.yaml', 'model/data']
        params: {'betas': '(0.9, 0.999)',
                 'weight_decay': '0',
                 'epochs': '20',
                 'eps': '1e-08',
                 'lr': '0.02',
                 'optimizer_name': 'Adam', '
                 amsgrad': 'False'}
        metrics: {'acc_step': 0.0,
                  'train_loss_epoch': 1.0917967557907104,
                  'train_loss_step': 1.0794280767440796,
                  'train_loss': 1.0794280767440796,
                  'acc_epoch': 0.0033333334140479565,
                  'acc': 0.0}
        tags: {'Mode': 'training'}

    .. figure:: ../_static/images/pytorch_lightening_autolog.png

        PyTorch autologged MLflow entities
    """
    import pytorch_lightning as pl
    from mlflow.pytorch_lightning._pytorch_lightning_autolog import patched_fit

    safe_patch(FLAVOR_NAME, pl.Trainer, "fit", patched_fit, manage_run=True)
