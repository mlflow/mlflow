import xgboost
import shap
import mlflow
from sklearn.model_selection import train_test_split

# Load the UCI Adult Dataset
X, y = shap.datasets.adult()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Fit an XGBoost binary classifier on the training data split
model = xgboost.XGBClassifier().fit(X_train, y_train)

# Build the Evaluation Dataset from the test set
eval_data = X_test
eval_data["label"] = y_test

with mlflow.start_run() as run:

    # Log the XGBoost binary classifier model to MLflow
    mlflow.sklearn.log_model(model, "model")
    model_uri = mlflow.get_artifact_uri("model")

    # Evaluate the logged model
    result = mlflow.evaluate(
        model_uri,
        eval_data,
        targets="label",
        model_type="classifier",
        evaluators=["default"],
    )

print(f"metrics:\n{result.metrics}")
print(f"artifacts:\n{result.artifacts}")
