# FAQs

## Can MLflow 3.x load runs/models/traces logged with MLflow 2.x?

Yes, MLflow 3.x can load runs/models/traces logged with MLflow 2.x.

## `load_model` throws a `ResourceNotFound` error when loading a model logged with MLflow 2.x. What's wrong?

For example, the following code fails to load the model in MLflow 3.x since the model artifacts are NOT stored as run artifacts:

```python
import mlflow

with mlflow.start_run() as run:
    mlflow.sklearn.log_model(my_model, "model")
    mlflow.sklearn.load_model(mlflow.get_artifact_uri("model"))
    # Throws a `ResourceNotFound` error.
```

To avoid this error, call `mlflow.<flavor>.load_model` with the model URI returned by `mlflow.<flavor>.log_model`:

```python
import mlflow

with mlflow.start_run() as run:
    info = mlflow.sklearn.log_model(my_model, "model")
    mlflow.sklearn.load_model(info.model_uri)
    # or if only `model_id` is available
    mlflow.sklearn.load_model(f"model:/{info.model_id}/model")
    # or neither `model_id` nor `model_uri` is available
    mlflow.sklearn.load_model(f"runs:/{run.info.run_id}/model")
```
