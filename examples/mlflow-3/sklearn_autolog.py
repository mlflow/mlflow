# ruff: noqa
"""
python examples/sklearn_autolog.py
"""

import logging
import os
import tempfile

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import mlflow

os.environ["MLFLOW_AUTOLOGGING_TESTING"] = "true"

mlflow.sklearn.autolog()

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = LogisticRegression()
with mlflow.start_run():
    model.fit(X_train, y_train)
