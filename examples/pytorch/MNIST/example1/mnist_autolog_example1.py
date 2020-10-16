#
# Trains an MNIST digit recognizer using PyTorch Lightning,
# and uses Mlflow to log metrics, params and artifacts
# NOTE: This example requires you to first install
# pytorch-lightning (using pip install pytorch-lightning)
#       and mlflow (using pip install mlflow).
#
# pylint: disable=W0221
# pylint: disable=W0613
import pytorch_lightning as pl
import os
import mlflow
import torch
from argparse import ArgumentParser
from mlflow.pytorch.pytorch_autolog import autolog
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.metrics.functional import accuracy
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class LightningMNISTClassifier(pl.LightningModule):
    def __init__(self, **kwargs):
        """
        Initializes the network
        """
        super(LightningMNISTClassifier, self).__init__()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.optimizer = None
        self.scheduler = None
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)
        self.args = kwargs

        # transforms for images
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--batch-size",
            type=int,
            default=64,
            metavar="N",
            help="input batch size for training (default: 64)",
        )
        parser.add_argument(
            "--num-workers",
            type=int,
            default=1,
            metavar="N",
            help="number of workers (default: 0)",
        )
        parser.add_argument(
            "--lr", type=float, default=1e-3, metavar="LR", help="learning rate (default: 1e-3)",
        )
        return parser

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
        return {"val_step_loss": loss}

    def validation_epoch_end(self, outputs):
        """
        Computes average validation accuracy

        :param outputs: outputs after every epoch end

        :return: output - average valid loss
        """
        avg_loss = torch.stack([x["val_step_loss"] for x in outputs]).mean()
        return {"val_loss": avg_loss}

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
        test_acc = accuracy(y_hat.cpu(), y.cpu())
        return {"test_acc": test_acc}

    def test_epoch_end(self, outputs):
        """
        Computes average test accuracy score

        :param outputs: outputs after every epoch end

        :return: output - average test loss
        """
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        return {"avg_test_acc": avg_test_acc}

    def prepare_data(self):
        """
        Prepares the data for training and prediction
        """
        return {}

    def train_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        mnist_train = datasets.MNIST("dataset", download=True, train=True, transform=self.transform)
        return DataLoader(
            mnist_train, batch_size=self.args["batch_size"], num_workers=self.args["num_workers"],
        )

    def val_dataloader(self):
        """
        :return: output - Validation data loader for the given input
        """
        mnist_train = datasets.MNIST("dataset", download=True, train=True, transform=self.transform)
        mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

        return DataLoader(
            mnist_val, batch_size=self.args["batch_size"], num_workers=self.args["num_workers"],
        )

    def test_dataloader(self):
        """
        :return: output - Test data loader for the given input
        """
        mnist_test = datasets.MNIST("dataset", download=True, train=False, transform=self.transform)
        return DataLoader(
            mnist_test, batch_size=self.args["batch_size"], num_workers=self.args["num_workers"],
        )

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler

        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.args["lr"])
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.2, patience=2, min_lr=1e-6, verbose=True,
            )
        }
        return [self.optimizer], [self.scheduler]

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        second_order_closure=None,
        on_tpu=False,
        using_lbfgs=False,
        using_native_amp=False,
    ):
        """
        Training step function which runs for the given number of epochs

        :param epoch: Number of epochs to train
        :param batch_idx: batch indices
        :param optimizer: Optimizer to be used in training step
        """
        self.optimizer.step()
        self.optimizer.zero_grad()


if __name__ == "__main__":
    parser = ArgumentParser(description="PyTorch Autolog Mnist Example")

    # Add trainer specific arguments

    parser.add_argument(
        "--tracking-uri", type=str, default="http://localhost:5000/", help="mlflow tracking uri"
    )
    parser.add_argument(
        "--max-epochs", type=int, default=20, help="number of epochs to run (default: 20)"
    )
    parser.add_argument(
        "--gpus", type=int, default=0, help="Number of gpus - by default runs on CPU"
    )
    parser.add_argument(
        "--distributed-backend",
        type=str,
        default=None,
        help="Distributed Backend - (default: None)",
    )

    # Early stopping parameters

    parser.add_argument(
        "--es-monitor", type=str, default="val_loss", help="Early stopping monitor parameter"
    )

    parser.add_argument("--es-mode", type=str, default="min", help="Early stopping mode parameter")

    parser.add_argument(
        "--es-verbose", type=bool, default=True, help="Early stopping verbose parameter"
    )

    parser.add_argument(
        "--es-patience", type=int, default=3, help="Early stopping patience parameter"
    )

    parser = LightningMNISTClassifier.add_model_specific_args(parent_parser=parser)

    autolog()

    args = parser.parse_args()
    dict_args = vars(args)
    mlflow.set_tracking_uri(dict_args["tracking_uri"])

    model = LightningMNISTClassifier(**dict_args)
    early_stopping = EarlyStopping(
        monitor=dict_args["es_monitor"],
        mode=dict_args["es_mode"],
        verbose=dict_args["es_verbose"],
        patience=dict_args["es_patience"],
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(), save_top_k=1, verbose=True, monitor="val_loss", mode="min", prefix="",
    )
    lr_logger = LearningRateLogger()

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[lr_logger],
        early_stop_callback=early_stopping,
        checkpoint_callback=checkpoint_callback,
        train_percent_check=0.1,
    )
    trainer.fit(model)
    trainer.test()
