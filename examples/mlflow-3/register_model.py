import json

from sklearn.linear_model import LinearRegression

import mlflow
from mlflow.entities.logged_model_parameter import LoggedModelParameter

with mlflow.start_run():
    model = LinearRegression().fit([[1], [2]], [3, 4])
    model_info = mlflow.sklearn.log_model(
        model, 
        "model", 
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
client.create_model_version("model_client", model_info.model_uri, model_id = model_info.model_id)
m = client.get_model_version("model_client", 1)
assert m.model_id == model_info.model_id
assert m.params == [
    LoggedModelParameter(key="alpha", value="0.5"),
    LoggedModelParameter(key="l1_ratio", value="0.5"),
]