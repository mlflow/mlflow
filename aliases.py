"""
MLFLOW_SQLALCHEMYSTORE_ECHO=true python aliases.py
"""
import mlflow

mlflow.set_tracking_uri("sqlite:///:memory:")
with mlflow.start_run():
    mlflow.sklearn.log_model(
        "I am a model",
        "model",
        registered_model_name="sklearn",
    )

# set a registered model tag
mlflow.MlflowClient().set_model_version_tag(
    name="sklearn", version=1, key="tag-key", value="tag-value"
)
mlflow.MlflowClient().set_registered_model_alias(
    name="sklearn", alias="sklearn-model-alias", version=1
)
for mv in mlflow.search_model_versions():
    print(mv.aliases)  # noqa: T201
