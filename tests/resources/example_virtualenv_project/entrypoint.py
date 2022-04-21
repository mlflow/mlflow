import argparse
import sys

import numpy as np
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import mlflow

parser = argparse.ArgumentParser()
parser.add_argument(
    "--test",
    action="store_true",
    help="Check python and sklearn versions are correct",
)
args = parser.parse_args()
if args.test:
    assert sys.version_info[:3] == (3, 8, 13)
    assert sklearn.__version__ == "1.0.2"

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])

clf = make_pipeline(StandardScaler(), SVC(gamma="auto"))
clf.fit(X, y)

with mlflow.start_run():
    mlflow.sklearn.log_model(clf, artifact_path="model")
