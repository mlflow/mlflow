import numpy as np
import paddle

import mlflow.paddle

train_dataset = paddle.text.datasets.UCIHousing(mode="train")
eval_dataset = paddle.text.datasets.UCIHousing(mode="test")


class UCIHousing(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.fc_ = paddle.nn.Linear(13, 1, None)

    def forward(self, inputs):
        pred = self.fc_(inputs)
        return pred


model = paddle.Model(UCIHousing())
optim = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())
model.prepare(optim, paddle.nn.MSELoss())

model.fit(train_dataset, epochs=6, batch_size=8, verbose=1)

with mlflow.start_run() as run:
    mlflow.paddle.log_model(model, name="model")
    print(f"Model saved in run {run.info.run_id}")

    # load model
    model_path = mlflow.get_artifact_uri("model")
    pd_model = mlflow.paddle.load_model(model_path)
    np_test_data = np.array([x[0] for x in eval_dataset])
    print(pd_model(np_test_data))
