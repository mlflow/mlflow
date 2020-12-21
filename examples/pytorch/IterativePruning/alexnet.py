# pylint: disable=W0221
# pylint: disable=W0201
# pylint: disable=W0223
# pylint: disable=arguments-differ
# pylint: disable=abstract-method

import argparse
import torch
import time
import os
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import mlflow.pytorch
from pytorch_lightning.metrics.functional import accuracy


class DataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_transforms = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        self.test_transforms = transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

    def prepare_data(self):
        trainset = torchvision.datasets.CIFAR10(
            root="./CIFAR10", train=True, download=True, transform=self.train_transforms
        )
        testset = torchvision.datasets.CIFAR10(
            root="./CIFAR10", train=False, download=True, transform=self.test_transforms
        )
        return trainset, testset

    def setup(self, stage=None):
        if stage == "fit" or stage == "None":
            self.dataset, _ = self.prepare_data()
            self.train_set, self.val_set = random_split(self.dataset, [45000, 5000])

        if stage == "test" or stage == "None":
            _, self.test_set = self.prepare_data()

    def train_dataloader(self):

        return DataLoader(self.train_set, batch_size=64, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=64, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=64, shuffle=False)


class AlexNet(pl.LightningModule):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

    def cross_entropy_loss(self, logits, labels):
        """
        Loss Fn to compute loss
        """
        # labels = labels.squeeze(1)
        return F.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        """
        training the data as batches and returns training loss on each batch
        """
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.log("train_loss", loss)
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
        self.log("val_loss", avg_loss, sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        """
        Performs test and computes test accuracy
        """
        x, y = test_batch
        output = self.forward(x)
        loss = F.cross_entropy(output, y)
        a, y_hat = torch.max(output, dim=1)
        test_acc = accuracy(y_hat.cpu(), y.cpu())
        self.log("test_loss", loss)
        self.log("test_acc", test_acc)
        return {"test_loss": loss, "test_acc": test_acc}

    def test_epoch_end(self, outputs):
        """
        Computes average test accuracy score
        """
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        self.log("avg_test_loss", avg_loss)
        self.log("avg_test_acc", avg_test_acc)

    def configure_optimizers(self):
        """
        Creates and returns Optimizer
        """

        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.05, momentum=0.6)
        return self.optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parent_parser=parser)

    parser.add_argument(
        "--mlflow_run_name",
        default="BaseModel",
        help="Name of MLFLOW experiment run in which results would be dumped",
    )

    args = parser.parse_args()
    if "MLFLOW_TRACKING_URI" in os.environ:
        tracking_uri = os.environ["MLFLOW_TRACKING_URI"]

    else:
        tracking_uri = "http://localhost:5000/"

    mlflow.tracking.set_tracking_uri(tracking_uri)

    experiment_name = os.environ["MLFLOW_EXPERIMENT_NAME"]
    mlflow.set_experiment(experiment_name)
    run_name = args.mlflow_run_name
    mlflow.start_run(run_name=run_name)
    trainer = pl.Trainer(max_epochs=int(args.max_epochs))
    model = AlexNet()
    dm = DataModule()
    dm.setup("fit")
    mlflow.pytorch.autolog()
    start_time = time.time()
    trainer.fit(model, dm)
    training_time = round((time.time() - start_time) / 60, 2)
    testloader = dm.setup("test")
    trainer.test(datamodule=testloader)
    model = trainer.model
    mlflow.end_run()
