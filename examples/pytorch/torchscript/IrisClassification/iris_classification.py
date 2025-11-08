import argparse

import torch
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn

import mlflow.pytorch
from mlflow.models import infer_signature


class IrisClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.2)
        x = self.fc3(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_data():
    iris = load_iris()
    data = iris.data
    labels = iris.target
    target_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels
    )

    X_train = torch.FloatTensor(X_train).to(device)
    X_test = torch.FloatTensor(X_test).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    y_test = torch.LongTensor(y_test).to(device)

    return X_train, X_test, y_train, y_test, target_names


def train_model(model, epochs, X_train, y_train):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        out = model(X_train)
        loss = criterion(out, y_train).to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print("number of epoch", epoch, "loss", float(loss))

    return model


def test_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predict_out = model(X_test)
        _, predict_y = torch.max(predict_out, 1)

        print("\nprediction accuracy", float(accuracy_score(y_test.cpu(), predict_y.cpu())))
        return infer_signature(X_test.numpy(), predict_out.numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iris Classification Torchscripted model")

    parser.add_argument(
        "--epochs", type=int, default=100, help="number of epochs to run (default: 100)"
    )

    args = parser.parse_args()

    model = IrisClassifier()
    model = model.to(device)
    X_train, X_test, y_train, y_test, target_names = prepare_data()
    scripted_model = torch.jit.script(model)  # scripting the model
    scripted_model = train_model(scripted_model, args.epochs, X_train, y_train)
    signature = test_model(scripted_model, X_test, y_test)

    with mlflow.start_run() as run:
        mlflow.pytorch.log_model(
            scripted_model, name="model", signature=signature
        )  # logging scripted model
        model_path = mlflow.get_artifact_uri("model")
        loaded_pytorch_model = mlflow.pytorch.load_model(model_path)  # loading scripted model
        model.eval()
        with torch.no_grad():
            test_datapoint = torch.Tensor([4.4000, 3.0000, 1.3000, 0.2000]).to(device)
            prediction = loaded_pytorch_model(test_datapoint)
            actual = "setosa"
            predicted = target_names[torch.argmax(prediction)]
            print(f"\nPREDICTION RESULT: ACTUAL: {actual}, PREDICTED: {predicted}")
