# Based on the official regression example:
# https://catboost.ai/docs/concepts/python-usages-examples.html#regression

import numpy as np
from catboost import CatBoostRegressor

import mlflow
from mlflow.models import infer_signature

# Initialize data
train_data = np.array([[1, 4, 5, 6], [4, 5, 6, 7], [30, 40, 50, 60]])
train_labels = np.array([10, 20, 30])
eval_data = np.array([[2, 4, 6, 8], [1, 4, 50, 60]])

# Initialize CatBoostRegressor
params = {
    "iterations": 2,
    "learning_rate": 1,
    "depth": 2,
    "allow_writing_files": False,
}
model = CatBoostRegressor(**params)

# Fit model
model.fit(train_data, train_labels)

# Log parameters and fitted model
with mlflow.start_run() as run:
    signature = infer_signature(eval_data, model.predict(eval_data))
    mlflow.log_params(params)
    model_info = mlflow.catboost.log_model(model, name="model", signature=signature)

# Load model
loaded_model = mlflow.catboost.load_model(model_info.model_uri)

# Get predictions
preds = loaded_model.predict(eval_data)
print("predictions:", preds)
