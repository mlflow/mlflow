import json

from sklearn.linear_model import LinearRegression

import mlflow

client = mlflow.MlflowClient()

with mlflow.start_run():
    model = LinearRegression().fit([[1], [2]], [3, 4])
    model_info = mlflow.sklearn.log_model(
        model,
        name="model",
        params={
            "alpha": 0.5,
            "l1_ratio": 0.5,
        },
    )
    model_info_2 = mlflow.sklearn.log_model(
        model,
        name="model",
        step=2,
        params={
            "alpha": 0.5,
            "l1_ratio": 0.5,
        },
    )

mlflow.register_model(model_info.model_uri, name="model")
m = mlflow.get_logged_model(model_info.model_id)
assert len(json.loads(m.tags["mlflow.modelVersions"])) == 1
print(m.tags)
assert m.model_id == model_info.model_id

mlflow.register_model(model_info.model_uri, name="hello")
m = mlflow.get_logged_model(model_info.model_id)
assert len(json.loads(m.tags["mlflow.modelVersions"])) == 2
print(m.tags)

client = mlflow.MlflowClient()

client.create_registered_model("model_client")
client.create_model_version("model_client", model_info.model_uri, model_id=model_info.model_id)
m = client.get_model_version("model_client", 1)
print(m)
assert m.model_id == model_info.model_id
assert m.params == {
    "alpha": "0.5",
    "l1_ratio": "0.5",
}

# Support backwards compatibility for runs:/... in addition to models:/...
model_uri = f"runs:/{model_info.run_id}/model"
mlflow.register_model(model_uri, name="model_from_runs_path")
mv = client.get_model_version("model_from_runs_path", 1)
assert mv.model_id == model_info_2.model_id  # model at largest step is registered

# Register model in log_model() directly
with mlflow.start_run():
    model_1 = LinearRegression().fit([[1], [2]], [3, 4])
    model_info_1 = mlflow.sklearn.log_model(
        model_1, name="model_1", registered_model_name="model_1"
    )

m = mlflow.get_logged_model(model_info_1.model_id)
assert len(json.loads(m.tags["mlflow.modelVersions"])) == 1
print(m.tags)

mv = client.get_model_version("model_1", 1)
assert mv.model_id == model_info_1.model_id
