"""
python examples/mlflow-3/sklearn_autolog.py
"""

import os

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split

import mlflow

os.environ["MLFLOW_AUTOLOGGING_TESTING"] = "true"

mlflow.sklearn.autolog()

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# LogisticRegression
model = LogisticRegression()
with mlflow.start_run() as run:
    model.fit(X_train, y_train)

model = mlflow.last_logged_model()
print(model.params)
print(model.metrics)

# GridSearchCV
model = LogisticRegression()
params = {"C": [0.1, 1.0, 10.0], "max_iter": [100, 200]}
clf = GridSearchCV(model, params)
with mlflow.start_run() as run:
    clf.fit(X_train, y_train)

for model in mlflow.search_logged_models(
    experiment_ids=[run.info.experiment_id], output_format="list"
):
    print(f"----- model_id: {model.model_id} -----")
    print(model)
