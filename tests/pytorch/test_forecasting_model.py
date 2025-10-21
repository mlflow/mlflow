import mlflow
import numpy as np
import torch
from lightning.pytorch import Trainer
from pytorch_forecasting import DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data.examples import generate_ar_data


def test_forecasting_model_pyfunc_loader(model_path):
    data = generate_ar_data(seasonality=10.0, timesteps=100, n_series=10)
    max_encoder_length = 30
    max_prediction_length = 10

    time_series_dataset = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= 90],
        time_idx="time_idx",
        target="value",
        group_ids=["series"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=["value"],
    )
    deepar = DeepAR.from_dataset(
        time_series_dataset,
        learning_rate=1e-3,
        hidden_size=16,
        rnn_layers=2,
    )
    dataloader = time_series_dataset.to_dataloader(train=True, batch_size=32)
    trainer = Trainer(max_epochs=2, gradient_clip_val=0.1, accelerator="auto")
    trainer.fit(deepar, train_dataloaders=dataloader)
    torch.manual_seed(42)
    predicted = deepar.predict(data).numpy()

    mlflow.pytorch.save_model(deepar, model_path)

    pyfunc_loaded = mlflow.pyfunc.load_model(model_path)
    torch.manual_seed(42)
    np.testing.assert_array_almost_equal(pyfunc_loaded.predict(data), predicted, decimal=4)
