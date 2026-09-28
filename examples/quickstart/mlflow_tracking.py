import mlflow

if __name__ == "__main__":
    print("Running mlflow_tracking.py")

    mlflow.log_param("learning_rate", 0.01)

    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("accuracy", 0.88, step=1)
    mlflow.log_metric("accuracy", 0.92, step=2)

    with open("model_summary.txt", "w") as f:
        f.write("Model training completed successfully!")

    mlflow.log_artifact("model_summary.txt")
