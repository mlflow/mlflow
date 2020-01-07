import mlflow

with mlflow.start_run(run_name="Plugin Example Run"):
    mlflow.log_metric("accuracy", 0.9)
    mlflow.log_param("learning_rate", 0.1)
