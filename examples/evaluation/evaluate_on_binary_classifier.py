import xgboost
import shap
import mlflow
from sklearn.model_selection import train_test_split

# train XGBoost model
X, y = shap.datasets.adult()

num_examples = len(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = xgboost.XGBClassifier().fit(X_train, y_train)

eval_data = X_test
eval_data["label"] = y_test

with mlflow.start_run() as run:
    mlflow.sklearn.log_model(model, "model")
    model_uri = mlflow.get_artifact_uri("model")
    result = mlflow.evaluate(
        model_uri,
        eval_data,
        targets="label",
        model_type="classifier",
        dataset_name="adult",
        evaluators=["default"],
    )

print(f"metrics:\n{result.metrics}")
print(f"artifacts:\n{result.artifacts}")
