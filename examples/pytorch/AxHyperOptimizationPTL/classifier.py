import mlflow.pytorch
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from mlflow.utils.autologging_utils import try_mlflow_log
from sklearn.metrics import accuracy_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import transforms
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


class LeNet(pl.LightningModule):
    def __init__(self, **kwargs):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.lr = 0.0
        self.nesterov = False
        self.momentum = 0.0
        self.lr = kwargs.get("kwargs", {}).get("lr")
        self.momentum = kwargs.get("kwargs", {}).get("momentum")
        self.weight_decay = kwargs.get("kwargs", {}).get("weight_decay")
        self.nesterov = kwargs.get("kwargs", {}).get("nesterov")

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return F.log_softmax(out, dim=1)

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
        self.log("val_loss", avg_loss)

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

        self.optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            nesterov=self.nesterov,
            weight_decay=self.weight_decay,
        )
        return self.optimizer


def train_evaluate(parameterization=None, dm=None, model=None, max_epochs=1):
    trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[])
    dm.prepare_data()
    dm.setup("fit")
    mlflow.pytorch.autolog()
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)
    test_accuracy = trainer.callback_metrics.get("avg_test_acc")
    return test_accuracy
