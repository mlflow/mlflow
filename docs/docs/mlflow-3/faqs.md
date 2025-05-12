# FAQs

## Can MLflow 3.x load runs/models/traces logged with MLflow 2.x?

Yes, MLflow 3.x can load runs/models/traces logged with MLflow 2.x. However, the reverse is not true.

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

Why does this happen? In MLflow 3.x, the model artifacts are stored in a different location than in MLflow 2.x. The following is a comparison of the two versions:

```bash
# MLflow 2.x
- experiments:
  - <experiment_id>
    - <run_id>
      - artifacts
        - ... # model artifacts are stored here

# MLflow 3.x
- experiments:
  - <experiment_id>
    - logged_models
      - <model_id>
        - artifacts
          - ... # model artifacts are stored here
```

## I'm still not ready to upgrade to MLflow 3.x. How can I pin my MLflow version to 2.x?

You can pin MLflow to the latest 2.x version by using the following command:

```bash
pip install 'mlflow<3'
```
