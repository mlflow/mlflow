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

outputs = mlflow.get_run(run.info.run_id).outputs
print(outputs)
model_id = outputs.model_outputs[0].model_id
print(model_id)
model = mlflow.get_logged_model(model_id)
print(model)

# GridSearchCV
model = LogisticRegression()
params = {"C": [0.1, 1.0, 10.0], "max_iter": [100, 200]}
clf = GridSearchCV(model, params)
with mlflow.start_run() as run:
    clf.fit(X_train, y_train)

outputs = mlflow.get_run(run.info.run_id).outputs
print(outputs)
model_id = outputs.model_outputs[0].model_id
print(model_id)
model = mlflow.get_logged_model(model_id)
print(model)
