"""
MLFLOW_SQLALCHEMYSTORE_ECHO=true python aliases.py
"""
import mlflow

mlflow.set_tracking_uri("sqlite:///:memory:")
with mlflow.start_run():
    for _ in range(3):
        mlflow.sklearn.log_model(
            "I am a model",
            "model",
            registered_model_name="sklearn",
        )

mlflow.MlflowClient().set_registered_model_alias(name="sklearn", alias="alias-1", version=1)
mlflow.MlflowClient().set_registered_model_alias(name="sklearn", alias="alias-2", version=1)
mlflow.MlflowClient().set_registered_model_alias(name="sklearn", alias="alias-3", version=2)
print("Fetching model versions")  # noqa: T201
for mv in mlflow.search_model_versions():
    print(mv.aliases)  # noqa: T201
