import shap
import xgboost
from sklearn.model_selection import train_test_split

import mlflow

# Load the UCI Adult Dataset
X, y = shap.datasets.adult()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Fit an XGBoost binary classifier on the training data split
model = xgboost.XGBClassifier().fit(X_train, y_train)

# Build the Evaluation Dataset from the test set
y_test_pred = model.predict(X=X_test)
eval_data = X_test
eval_data["label"] = y_test
eval_data["predictions"] = y_test_pred


with mlflow.start_run() as run:
    # Evaluate the static dataset without providing a model
    result = mlflow.evaluate(
        data=eval_data,
        targets="label",
        predictions="predictions",
        model_type="classifier",
    )

print(f"metrics:\n{result.metrics}")
print(f"artifacts:\n{result.artifacts}")
