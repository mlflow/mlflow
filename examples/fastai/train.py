import argparse

from fastai.learner import Learner
from fastai.tabular.all import TabularDataLoaders
import numpy as np
from sklearn.datasets import load_iris
from torch import nn

import mlflow


def parse_args():
    parser = argparse.ArgumentParser(description="Fastai example")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="learning rate to update step size at each step (default: 0.01)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="number of epochs (default: 5). Note it takes about 1 min per epoch",
    )
    return parser.parse_args()


def get_data_loaders():
    X, y = load_iris(return_X_y=True, as_frame=True)
    y = y.astype(np.float32)
    return TabularDataLoaders.from_df(
        X.assign(target=y), cont_names=list(X.columns), y_names=y.name
    )


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 3)
        self.linear2 = nn.Linear(3, 1)

    def forward(self, _, x_cont):
        x = self.linear1(x_cont)
        return self.linear2(x)


def splitter(model):
    params = list(model.parameters())
    return [
        # weights and biases of the first linear layer
        params[:2],
        # weights and biases of the second linear layer
        params[2:],
    ]


def main():
    # Parse command-line arguments
    args = parse_args()

    # Enable auto logging
    mlflow.fastai.autolog()

    # Create Learner model
    learn = Learner(get_data_loaders(), Model(), loss_func=nn.MSELoss(), splitter=splitter)

    # Start MLflow session
    with mlflow.start_run():
        # Train and fit with default or supplied command line arguments
        learn.fit_one_cycle(args.epochs, args.lr)


if __name__ == "__main__":
    main()
