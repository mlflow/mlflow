PyTorch within MLflow
=========================

In this guide we will walk you through how to use PyTorch within MLflow. We will demonstrate
how to track your PyTorch experiments and log your PyTorch models to MLflow.

Logging PyTorch Experiments to MLflow
-------------------------------------

Autologging PyTorch Experiments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unlike other deep learning flavors, MLflow does not have an autologging integration with PyTorch because
native PyTorch requires writing custom training loops. If you want to use autologging with PyTorch, please
use `Lightning <https://lightning.ai/>`_ to train your models. When Lightning is being used, you can turned
on autologging by calling :py:func:`mlflow.tensorflow.autolog()` or :py:func:`mlflow.autolog()`. For more
details, please refer to the MLflow Lightning Developer Guide.

Manually Logging PyTorch Experiments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To log your PyTorch experiments, you can insert MLflow logging into your PyTorch training loop, which relies
on the following APIs:

- :py:func:`mlflow.log_metric()` / :py:func:`mlflow.log_metrics()`: log metrics such as accuracy and loss
  during training.
- :py:func:`mlflow.log_param()` / :py:func:`mlflow.log_params()`: log parameters such as learning rate and
  batch size during training.
- :py:func:`mlflow.log_artifact()`: log artifacts such as model checkpoints and plots during training and
  saving the model at the end of training.

The following is an example of how to log your PyTorch experiments to MLflow:

.. code-block:: python

    import mlflow
    import torch

    from torch import nn
    from torch.utils.data import DataLoader
    from torchinfo import summary
    from torchmetrics import Accuracy
    from torchvision import datasets
    from torchvision.transforms import ToTensor

    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=64)

    # Get cpu, gpu or mps device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define model
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits

    def train(dataloader, model, loss_fn, metrics_fn, optimizer):
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)
            accuracy = metrics_fn(pred, y)

            # Backpropagation.
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), batch
                mlflow.log_metric("loss", f"{loss:3f}", step=(batch // 100))
                mlflow.log_metric("accuracy", f"{accuracy:3f}", step=(batch // 100))
                print(f"loss: {loss:3f} accuracy: {accuracy:3f} [{current} / {len(dataloader)}]")

    epochs = 3
    loss_fn = nn.CrossEntropyLoss()
    metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    model = NeuralNetwork().to(device)

    with mlflow.start_run():
        params = {
            "epochs": epochs,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "loss_function": loss_fn.__class__.__name__,
            "metric_function": metric_fn.__class__.__name__,
            "optimizer": "SGD",
            "model_summary": summary(model)
        }
        # Log model and training parameters.
        mlflow.log_params(params)
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, metric_fn, optimizer)

        # Save the trained model to MLflow.
        mlflow.pytorch.log_model(model, "model")

