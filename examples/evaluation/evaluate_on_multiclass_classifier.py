import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

mlflow.sklearn.autolog()

X, y = make_classification(n_samples=10000, n_classes=10, n_informative=5, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

with mlflow.start_run() as run:
    model = LogisticRegression(solver="liblinear").fit(X_train, y_train)
    model_uri = mlflow.get_artifact_uri("model")
    result = mlflow.evaluate(
        model_uri,
        X_test,
        targets=y_test,
        model_type="classifier",
        dataset_name="multiclass-classification-dataset",
        evaluators="default",
        evaluator_config={"log_model_explainability": True, "explainability_nsamples": 1000},
    )

print(f"run_id={run.info.run_id}")
print(f"metrics:\n{result.metrics}")
print(f"artifacts:\n{result.artifacts}")
