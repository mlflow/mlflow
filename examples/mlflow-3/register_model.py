import json

from sklearn.linear_model import LinearRegression

import mlflow

with mlflow.start_run():
    model = LinearRegression().fit([[1], [2]], [3, 4])
    model_info = mlflow.sklearn.log_model(model, "model")

mlflow.register_model(model_info.model_uri, name="model")
m = mlflow.get_logged_model(model_info.model_id)
assert len(json.loads(m.tags["mlflow.modelVersions"])) == 1
print(m.tags)

mlflow.register_model(model_info.model_uri, name="hello")
m = mlflow.get_logged_model(model_info.model_id)
assert len(json.loads(m.tags["mlflow.modelVersions"])) == 2
print(m.tags)

# Register model in log_model() directly
with mlflow.start_run():
    model_1 = LinearRegression().fit([[1], [2]], [3, 4])
    model_info_1 = mlflow.sklearn.log_model(model_1, "model_1", registered_model_name="model_1")

m = mlflow.get_logged_model(model_info_1.model_id)
assert len(json.loads(m.tags["mlflow.modelVersions"])) == 1
print(m.tags)

client = mlflow.MlflowClient()
m = client.get_model_version("model_1", 1)
