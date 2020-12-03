# pylint: disable=W0221
# pylint: disable=W0201
# pylint: disable=W0223
# pylint: disable=arguments-differ
# pylint: disable=abstract-method

import pytorch_lightning as pl
import torch
from sklearn.datasets import load_iris
from torch.utils.data import DataLoader, random_split, TensorDataset


class IrisDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.columns = None

    def prepare_data(self):
        iris = load_iris()
        df = iris.data
        self.columns = iris.feature_names
        target = iris["target"]
        data = torch.Tensor(df).float()
        labels = torch.Tensor(target).long()
        data_set = TensorDataset(data, labels)
        return data_set

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            iris_full = self.prepare_data()
            self.train_set, self.val_set = random_split(iris_full, [130, 20])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.train_set, self.test_set = random_split(self.train_set, [110, 20])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=4)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=4)


if __name__ == "__main__":
    pass
