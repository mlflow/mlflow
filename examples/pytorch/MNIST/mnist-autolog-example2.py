#
# Trains an MNIST digit recognizer using PyTorch Lightning,
# and uses Mlflow to log metrics, params and artifacts
# NOTE: This example requires you to first install
# pytorch-lightning (using pip install pytorch-lightning)
#       and mlflow (using pip install mlflow).
#
import mlflow
import pytorch_lightning as pl
import torch
from mlflow.utils.autologging_utils import try_mlflow_log
from torchvision import datasets, transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from mlflow.pytorch.pytorch_lightning_without_patching \
    import __MLflowPLCallback
from sklearn.metrics import accuracy_score


class LightningMNISTClassifier(pl.LightningModule):
    def __init__(self):
        """
        Initializes the network
        """
        super(LightningMNISTClassifier, self).__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        """
        Forward Function
        """
        batch_size, channels, width, height = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # layer 1 (b, 1*28*28) -> (b, 128)
        x = self.layer_1(x)
        x = torch.relu(x)

        # layer 2 (b, 128) -> (b, 256)
        x = self.layer_2(x)
        x = torch.relu(x)

        # layer 3 (b, 256) -> (b, 10)
        x = self.layer_3(x)

        # probability distribution over labels
        x = torch.log_softmax(x, dim=1)

        return x

    def cross_entropy_loss(self, logits, labels):
        """
        Loss Fn to compute loss
        """
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        """
        training the data as batches and returns training loss on each batch
        """
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        """
        Performs validation of data in batches
        """
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        """
        Computes average validation accuracy
        """
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, test_batch, batch_idx):
        """
        Performs test and computes test accuracy
        """
        x, y = test_batch
        output = self.forward(x)
        a, y_hat = torch.max(output, dim=1)
        test_acc = accuracy_score(y_hat, y)
        return {"test_acc": torch.tensor(test_acc)}

    def test_epoch_end(self, outputs):
        """
        Computes average test accuracy score
        """
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        return {"avg_test_acc": avg_test_acc}

    def prepare_data(self):
        """
        Preprocess the input data.
        """
        # transforms for images
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        mnist_train = datasets.MNIST(
            "dataset", download=True, train=True, transform=transform
        )
        self.mnist_test = datasets.MNIST(
            "dataset", download=True, train=False, transform=transform
        )

        self.mnist_train, self.mnist_val = \
            random_split(mnist_train, [55000, 5000])

    def train_dataloader(self):
        """
        Loading training data as batches
        """
        return DataLoader(self.mnist_train, batch_size=64)

    def val_dataloader(self):
        """
        Loading validation data as batches
        """
        return DataLoader(self.mnist_val, batch_size=64)

    def test_dataloader(self):
        """
        Loading test data as batches
        """
        return DataLoader(self.mnist_test, batch_size=64)

    def configure_optimizers(self):
        """
        Creates and returns Optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    mlflow.tracking.set_tracking_uri("http://IP:5000/")
    if not mlflow.active_run():
        try_mlflow_log(mlflow.start_run)
    model = LightningMNISTClassifier()
    trainer = pl.Trainer(max_epochs=5, callbacks=[__MLflowPLCallback()])
    trainer.fit(model)
    trainer.test(model)

    try_mlflow_log(mlflow.end_run)
