from __future__ import print_function

import unittest

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import mlflow.pytorch
from mlflow import pyfunc
from mlflow import tracking
from mlflow.utils.file_utils import TempDir


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class TestModelExport(unittest.TestCase):
    def setUp(self):

        torch.manual_seed(12345)

        num_train_samples = 100
        num_test_samples = 30

        train_dataset = [[torch.rand(3, 32, 32),
                          torch.randint(0, 10, size=(1,), dtype=torch.long).item()]
                         for _ in range(num_train_samples)]
        self.test_dataset = [[torch.rand(3, 32, 32),
                              torch.randint(0, 10, size=(1,), dtype=torch.long).item()]
                             for _ in range(num_test_samples)]

        batch_size = 16
        num_workers = 4
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                      num_workers=num_workers, shuffle=True, drop_last=True)

        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size,
                                          num_workers=num_workers, shuffle=False, drop_last=False)

        # Setup model
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 16, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 16, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            Flatten(),
            nn.Linear(64, 10)
        )

        # Train
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.model.train()
        for epoch in range(5):
            for batch in train_dataloader:
                optimizer.zero_grad()
                y_pred = self.model(batch[0])
                loss = criterion(y_pred, batch[1])
                loss.backward()
                optimizer.step()

        # Predict
        self.predictions = self._predict(self.model, self.test_dataloader)

    @staticmethod
    def _predict(model, test_dataloader):
        batch_size = test_dataloader.batch_size
        predictions = np.zeros((len(test_dataloader.sampler),))
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                y_probas = F.softmax(model(batch[0]), dim=1).numpy()
                y_preds = np.argmax(y_probas, axis=1)
                predictions[i * batch_size:(i + 1) * batch_size] = y_preds
        return predictions

    def test_log_model(self):
        old_uri = tracking.get_tracking_uri()
        # should_start_run tests whether or not calling log_model() automatically starts a run.
        for should_start_run in [False, True]:
            with TempDir(chdr=True, remove_on_exit=True) as tmp:
                try:
                    tracking.set_tracking_uri("test")
                    if should_start_run:
                        tracking.start_run()

                    mlflow.pytorch.log_model(self.model, artifact_path="pytorch")

                    # Load model
                    run_id = tracking.active_run().info.run_uuid
                    model_loaded = mlflow.pytorch.load_model("pytorch", run_id=run_id)

                    test_predictions = self._predict(model_loaded, self.test_dataloader)
                    assert all(test_predictions == self.predictions)
                finally:
                    tracking.end_run()
                    tracking.set_tracking_uri(old_uri)

    def test_save_and_load_model(self):
        with TempDir(chdr=True, remove_on_exit=True) as tmp:
            path = tmp.path("model")
            mlflow.pytorch.save_model(self.model, path)

            # Loading pytorch model
            model_loaded = mlflow.pytorch.load_model(path)
            assert all(self._predict(model_loaded, self.test_dataloader) == self.predictions)

            # Loading pyfunc model
            pyfunc_loaded = mlflow.pyfunc.load_pyfunc(path)

            with torch.no_grad():
                for i, dp in enumerate(self.test_dataset):
                    data = dp[0].numpy()
                    y_proba = pyfunc_loaded.predict(data)
                    y_pred = y_proba.argmax()
                    assert y_pred == self.predictions[i]
