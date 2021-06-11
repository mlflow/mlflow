import mlflow.paddle
import paddle
import numpy as np

train_dataset = paddle.text.datasets.UCIHousing(mode="train")
eval_dataset = paddle.text.datasets.UCIHousing(mode="test")


class UCIHousing(paddle.nn.Layer):
    def __init__(self):
        super(UCIHousing, self).__init__()
        self.fc_ = paddle.nn.Linear(13, 1, None)

    def forward(self, inputs):
        pred = self.fc_(inputs)
        return pred


model = paddle.Model(UCIHousing())
optim = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())
model.prepare(optim, paddle.nn.MSELoss())

model.fit(train_dataset, epochs=6, batch_size=8, verbose=1)

sk_path_dir = "./test-out"
mlflow.paddle.save_model(model, sk_path_dir)
print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

# load model
pd_model = mlflow.paddle.load_model("test-out")
np_test_data = np.array([x[0] for x in eval_dataset])
print(pd_model(np_test_data))
