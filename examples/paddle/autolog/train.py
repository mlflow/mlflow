import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from paddle.static import InputSpec

import mlflow.paddle
from mlflow.tracking import MlflowClient

from argparse import ArgumentParser


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


if __name__ == "__main__":
    parser = ArgumentParser(description="PaddlePaddle Autolog Mnist Example")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=2, help="Training epochs parameter")

    parser.add_argument("--batch_size", type=int, default=32, help="Training batch_size parameter")

    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Training learning_rate parameters"
    )

    parser.add_argument("--monitor", type=str, default="acc", help="Early checking mode parameters")

    parser.add_argument("--mode", type=str, default="auto", help="Early checking mode parameters")

    parser.add_argument(
        "--patience", type=int, default=2, help="Early checking patience parameters"
    )

    args = parser.parse_args()
    dict_args = vars(args)

    device = paddle.set_device("cpu")
    net = nn.Sequential(nn.Flatten(1), nn.Linear(784, 200), nn.Tanh(), nn.Linear(200, 10))
    # inputs and labels are not required for dynamic graph.
    input = InputSpec([None, 784], "float32", "x")
    label = InputSpec([None, 1], "int64", "label")
    model = paddle.Model(net, input, label)
    optim = paddle.optimizer.SGD(
        learning_rate=dict_args["learning_rate"], parameters=model.parameters()
    )
    model.prepare(optim, paddle.nn.CrossEntropyLoss(), paddle.metric.Accuracy())
    transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
    data = paddle.vision.datasets.MNIST(mode="train", transform=transform)

    mlflow.paddle.autolog()

    callbacks_earlystopping = paddle.callbacks.EarlyStopping(
        dict_args["monitor"],
        mode=dict_args["mode"],
        patience=dict_args["patience"],
        verbose=1,
        min_delta=0,
        baseline=0,
        save_best_model=True,
    )

    with mlflow.start_run() as run:
        model.fit(
            data,
            epochs=dict_args["epochs"],
            batch_size=dict_args["batch_size"],
            verbose=1,
            callbacks=[callbacks_earlystopping],
        )

    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
