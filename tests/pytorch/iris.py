import tempfile

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.tensorboard
from packaging.version import Version
from torch import nn

tmpdir = tempfile.mkdtemp()
SUMMARY_WRITER = torch.utils.tensorboard.SummaryWriter(log_dir=tmpdir)


def create_multiclass_accuracy():
    # NB: Older versions of PyTorch Lightning define native APIs for metric computation,
    # (e.g., pytorch_lightning.metrics.Accuracy), while newer versions rely on the `torchmetrics`
    # package (e.g. `torchmetrics.Accuracy)
    try:
        import torchmetrics
        from torchmetrics import Accuracy

        if Version(torchmetrics.__version__) >= Version("0.11"):
            return Accuracy(task="multiclass", num_classes=3)
        else:
            return Accuracy()
    except ImportError:
        from pytorch_lightning.metrics import Accuracy

        return Accuracy()


class IrisClassificationBase(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()

        self.train_acc = create_multiclass_accuracy()
        self.val_acc = create_multiclass_accuracy()
        self.test_acc = create_multiclass_accuracy()
        self.args = kwargs

        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 3)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.relu(self.fc3(x))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), 0.01)


class IrisClassification(IrisClassificationBase):
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        # this should *not* get intercepted by "plain" pytorch autologging
        # since it is called from inside lightning's fit()
        SUMMARY_WRITER.add_scalar("plain_loss", loss.item())
        self.train_acc(torch.argmax(logits, dim=1), y)
        self.log("train_acc", self.train_acc.compute(), on_step=False, on_epoch=True)
        self.log("loss", loss)
        self.log("loss_forked", loss, on_epoch=True, on_step=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.val_acc(torch.argmax(logits, dim=1), y)
        self.log("val_acc", self.val_acc.compute())
        self.log("val_loss", loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.test_acc(torch.argmax(logits, dim=1), y)
        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc.compute())


class IrisClassificationWithoutValidation(IrisClassificationBase):
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.train_acc(torch.argmax(logits, dim=1), y)
        self.log("train_acc", self.train_acc.compute(), on_step=False, on_epoch=True)
        self.log("loss", loss)
        return {"loss": loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.test_acc(torch.argmax(logits, dim=1), y)
        self.log("test_loss", loss)
        self.log("test_acc", self.test_acc.compute())


class IrisClassificationMultiOptimizer(IrisClassificationBase):
    """
    Contrived lightning module that uses multiple optimizers. In real-world scenarios
    multiple optimizers might be used for Generative Adversarial Networks (GANs).
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt1, opt2 = self.optimizers()

        x, y = batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)

        opt1.zero_grad()
        opt2.zero_grad()

        self.manual_backward(loss)

        opt1.step()
        opt2.step()

        self.log("loss", loss, on_epoch=True, on_step=True)

    def configure_optimizers(self):
        opt1 = torch.optim.Adam(self.parameters(), 0.01)
        opt2 = torch.optim.Adam(self.parameters(), 0.01)
        return [opt1, opt2]


if __name__ == "__main__":
    pass
