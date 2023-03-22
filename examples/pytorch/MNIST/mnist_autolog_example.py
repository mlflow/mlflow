#
# Trains an MNIST digit recognizer using PyTorch Lightning,
# and uses Mlflow to log metrics, params and artifacts
# NOTE: This example requires you to first install
# pytorch-lightning (using pip install pytorch-lightning)
#       and mlflow (using pip install mlflow).
#
# pylint: disable=arguments-differ
# pylint: disable=unused-argument
# pylint: disable=abstract-method
import os

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from torchvision import datasets, transforms

import mlflow.pytorch


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=3):
        """
        Initialization of inherited lightning data module
        """
        super().__init__()
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.batch_size = batch_size
        self.num_workers = num_workers

        # transforms for images
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def setup(self, stage=None):
        """
        Downloads the data, parse it and split the data into train, test, validation data

        :param stage: Stage - training or testing
        """

        self.df_train = datasets.MNIST(
            "dataset", download=True, train=True, transform=self.transform
        )
        self.df_train, self.df_val = random_split(self.df_train, [55000, 5000])
        self.df_test = datasets.MNIST(
            "dataset", download=True, train=False, transform=self.transform
        )

    def create_data_loader(self, df):
        """
        Generic data loader function

        :param df: Input tensor

        :return: Returns the constructed dataloader
        """
        return DataLoader(df, batch_size=self.batch_size, num_workers=self.num_workers)

    def train_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        return self.create_data_loader(self.df_train)

    def val_dataloader(self):
        """
        :return: output - Validation data loader for the given input
        """
        return self.create_data_loader(self.df_val)

    def test_dataloader(self):
        """
        :return: output - Test data loader for the given input
        """
        return self.create_data_loader(self.df_test)


class LightningMNISTClassifier(L.LightningModule):
    def __init__(self, learning_rate=0.01):
        """
        Initializes the network
        """
        super().__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.optimizer = None
        self.scheduler = None
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)
        self.learning_rate = learning_rate
        self.val_outputs = []
        self.test_outputs = []

    def forward(self, x):
        """
        :param x: Input data

        :return: output - mnist digit label for the input image
        """
        batch_size = x.size()[0]

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
        Initializes the loss function

        :return: output - Initialized cross entropy loss function
        """
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        """
        Training the data as batches and returns training loss on each batch

        :param train_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - Training loss
        """
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        """
        Performs validation of data in batches

        :param val_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - valid step loss
        """
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.val_outputs.append(loss)
        return {"val_step_loss": loss}

    def on_validation_epoch_end(self):
        """
        Computes average validation loss
        """
        avg_loss = torch.stack(self.val_outputs).mean()
        self.log("val_loss", avg_loss, sync_dist=True)
        self.val_outputs.clear()

    def test_step(self, test_batch, batch_idx):
        """
        Performs test and computes the accuracy of the model

        :param test_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - Testing accuracy
        """
        x, y = test_batch
        output = self.forward(x)
        _, y_hat = torch.max(output, dim=1)
        test_acc = accuracy(y_hat.cpu(), y.cpu(), task="multiclass", num_classes=10)
        self.test_outputs.append(test_acc)
        return {"test_acc": test_acc}

    def on_test_epoch_end(self):
        """
        Computes average test accuracy score
        """
        avg_test_acc = torch.stack(self.test_outputs).mean()
        self.log("avg_test_acc", avg_test_acc, sync_dist=True)
        self.test_outputs.clear()

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler

        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.2,
                patience=2,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "val_loss",
        }
        return [self.optimizer], [self.scheduler]


def cli_main():
    early_stopping = EarlyStopping(
        monitor="val_loss",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.getcwd(), save_top_k=1, verbose=True, monitor="val_loss", mode="min"
    )
    lr_logger = LearningRateMonitor()
    cli = LightningCLI(
        LightningMNISTClassifier,
        MNISTDataModule,
        run=False,
        save_config_callback=None,
        trainer_defaults={"callbacks": [early_stopping, checkpoint_callback, lr_logger]},
    )
    if cli.trainer.global_rank == 0:
        mlflow.pytorch.autolog()
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)


if __name__ == "__main__":
    cli_main()
