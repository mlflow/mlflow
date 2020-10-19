# pylint: disable=W0221
# pylint: disable=W0613
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from sklearn.datasets import load_iris
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset


class IrisClassification(pl.LightningModule):
    def __init__(self):
        super(IrisClassification, self).__init__()
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 3)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(x, dim=1)
        return x

    def prepare_data(self):
        iris = load_iris()
        df = iris.data
        target = iris["target"]

        data = torch.Tensor(df).float()
        labels = torch.Tensor(target).long()

        data_set = TensorDataset(data, labels)
        self.train_set, self.val_set = random_split(data_set, [130, 20])
        self.train_set, self.test_set = random_split(self.train_set, [110, 20])

    def train_dataloader(self):
        train_loader = DataLoader(dataset=self.train_set, batch_size=8)
        return train_loader

    def val_dataloader(self):
        validation_loader = DataLoader(dataset=self.val_set, batch_size=8)
        return validation_loader

    def test_dataloader(self):
        test_loader = DataLoader(dataset=self.test_set, batch_size=8)
        return test_loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        logs = {"loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", avg_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        _, y_hat = torch.max(logits, dim=1)
        test_acc = accuracy(y_hat.cpu(), y.cpu())
        self.log("test_loss", loss)
        self.log("test_acc", test_acc)
        return {"test_loss": loss, "test_acc": test_acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        self.log("avg_test_loss", avg_loss)
        self.log("avg_test_acc", avg_test_acc)
