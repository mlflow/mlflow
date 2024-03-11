import torch
from sklearn.datasets import load_iris
from torch.utils.data import DataLoader, TensorDataset, random_split

# Handle import of either 'pytorch_lightning' or 'lightning'
try:
    import lightning.pytorch as pl
except ModuleNotFoundError:
    try:
        import pytorch_lightning as pl

        PYTORCH_LIGHTNING_LEGACY_NAMESPACE = True
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            message="""Unable to import 'lightning' or 'pytorch-lightning'.\n
            Please install 'lightning' into your environment ('pip install lightning')"""
        )


class IrisDataModuleBase(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.columns = None

    def _get_iris_as_tensor_dataset(self):
        iris = load_iris()
        df = iris.data
        self.columns = iris.feature_names
        target = iris["target"]
        data = torch.Tensor(df).float()
        labels = torch.Tensor(target).long()
        return TensorDataset(data, labels)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            iris_full = self._get_iris_as_tensor_dataset()
            self.train_set, self.val_set = random_split(iris_full, [130, 20])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.train_set, self.test_set = random_split(self.train_set, [110, 20])


class IrisDataModule(IrisDataModuleBase):
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=4)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=4)


class IrisDataModuleWithoutValidation(IrisDataModuleBase):
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=4)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=4)


if __name__ == "__main__":
    pass
