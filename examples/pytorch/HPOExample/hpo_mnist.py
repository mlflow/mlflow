"""
Hyperparameter Optimization Example with Pure PyTorch and MLflow

This example demonstrates:
- Using MLflow to track hyperparameter optimization trials
- Parent/child run structure for organizing HPO experiments
- Pure PyTorch training (no Lightning dependencies)
- Simple MNIST classification with configurable hyperparameters

Run with: python hpo_mnist.py --n-trials 5 --max-epochs 3
"""

import argparse

import optuna
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import mlflow


class SimpleNet(nn.Module):
    def __init__(self, hidden_size, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train_epoch(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy


def objective(trial, args, train_loader, test_loader, device):
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    hidden_size = trial.suggest_int("hidden_size", 64, 512, step=64)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Recreate data loaders with new batch size
    train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_loader.dataset, batch_size=batch_size, shuffle=False)

    # Start nested MLflow run for this trial
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        # Log hyperparameters
        mlflow.log_params(
            {
                "lr": lr,
                "hidden_size": hidden_size,
                "dropout_rate": dropout_rate,
                "batch_size": batch_size,
            }
        )

        # Create model and optimizer
        model = SimpleNet(hidden_size, dropout_rate).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training loop
        for epoch in range(args.max_epochs):
            train_epoch(model, device, train_loader, optimizer)
            test_loss, accuracy = evaluate(model, device, test_loader)

            # Log metrics for each epoch
            mlflow.log_metrics({"test_loss": test_loss, "accuracy": accuracy}, step=epoch)

        # Return final accuracy for optimization
        return accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=10, help="Number of HPO trials")
    parser.add_argument("--max-epochs", type=int, default=5, help="Epochs per trial")
    parser.add_argument("--batch-size", type=int, default=64, help="Initial batch size")
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MNIST data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Start parent MLflow run
    with mlflow.start_run(run_name="HPO_Parent"):
        mlflow.log_params({"n_trials": args.n_trials, "max_epochs": args.max_epochs})

        # Create Optuna study
        study = optuna.create_study(direction="maximize", study_name="mnist_hpo")

        # Run optimization
        study.optimize(
            lambda trial: objective(trial, args, train_loader, test_loader, device),
            n_trials=args.n_trials,
        )

        # Log best results to parent run
        mlflow.log_metrics(
            {
                "best_accuracy": study.best_value,
                "best_trial": study.best_trial.number,
            }
        )
        # Log best hyperparameters with 'best_' prefix to avoid conflicts
        best_params = {f"best_{k}": v for k, v in study.best_params.items()}
        mlflow.log_params(best_params)

        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best accuracy: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
