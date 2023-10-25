# Location: mlflow/mlflow/pytorch/__init__.py:950
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/pytorch/__init__.py:950 '])
def test(_):
    import os

    import lightning as L
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Subset
    from torchmetrics import Accuracy
    from torchvision import transforms
    from torchvision.datasets import MNIST

    import mlflow.pytorch
    from mlflow import MlflowClient


    class MNISTModel(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(28 * 28, 10)
            self.accuracy = Accuracy("multiclass", num_classes=10)

        def forward(self, x):
            return torch.relu(self.l1(x.view(x.size(0), -1)))

        def training_step(self, batch, batch_nb):
            x, y = batch
            logits = self(x)
            loss = F.cross_entropy(logits, y)
            pred = logits.argmax(dim=1)
            acc = self.accuracy(pred, y)

            # PyTorch `self.log` will be automatically captured by MLflow.
            self.log("train_loss", loss, on_epoch=True)
            self.log("acc", acc, on_epoch=True)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.02)


    def print_auto_logged_info(r):
        tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
        artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
        print(f"run_id: {r.info.run_id}")
        print(f"artifacts: {artifacts}")
        print(f"params: {r.data.params}")
        print(f"metrics: {r.data.metrics}")
        print(f"tags: {tags}")


    # Initialize our model.
    mnist_model = MNISTModel()

    # Load MNIST dataset.
    train_ds = MNIST(
        os.getcwd(), train=True, download=True, transform=transforms.ToTensor()
    )
    # Only take a subset of the data for faster training.
    indices = torch.arange(32)
    train_ds = Subset(train_ds, indices)
    train_loader = DataLoader(train_ds, batch_size=8)

    # Initialize a trainer.
    trainer = L.Trainer(max_epochs=3)

    # Auto log all MLflow entities
    mlflow.pytorch.autolog()

    # Train the model.
    with mlflow.start_run() as run:
        trainer.fit(mnist_model, train_loader)

    # Fetch the auto logged parameters and metrics.
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))


if __name__ == "__main__":
    test()
