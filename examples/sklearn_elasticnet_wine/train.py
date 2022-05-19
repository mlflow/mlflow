# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys
import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def main(args):
    mlflow.autolog(log_input_examples=True)
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    # csv_url = (
    #     "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    # )
    try:
        data = pd.read_csv(args.wine_csv, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    with mlflow.start_run() as run:
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        mlflow.register_model(
            "runs:/{}/model".format(run.info.run_id),
            "ElasticnetWineModel"
        )

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--wine-csv", type=str, default="http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv")
    parser.add_argument("--alpha", type=int, default=0.5)
    parser.add_argument("--l1-ratio", type=str, default=0.5)

    # parse args
    args = parser.parse_args()

    # return args
    return args

if __name__ == "__main__":
    args = parse_args()

    # run main function
    main(args)