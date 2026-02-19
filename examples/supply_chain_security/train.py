import sklearn

import mlflow

# Use explicit model logging to control the conda environment and pip requirements
mlflow.sklearn.autolog(log_models=False)

# Load data
X, y = sklearn.datasets.load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Train model
with mlflow.start_run() as run:
    print(f"MLflow run ID: {run.info.run_id}")

    model = sklearn.linear_model.Ridge(alpha=0.03)
    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(
        model,
        name="model",
        signature=mlflow.models.infer_signature(X_train[:10], y_train[:10]),
        input_example=X_train[:10],
    )
