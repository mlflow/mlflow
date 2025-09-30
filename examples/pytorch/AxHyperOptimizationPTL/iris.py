import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class IrisClassification(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.args = kwargs

        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 3)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.lr = kwargs.get("lr", 0.01)
        self.momentum = kwargs.get("momentum", 0.9)
        self.weight_decay = kwargs.get("weight_decay", 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_acc", acc)
        self.log("val_loss", loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
