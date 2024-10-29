from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.models import infer_signature

X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = LogisticRegression().fit(X_train, y_train)

predictions = model.predict(X_train)
signature = infer_signature(X_train, predictions)

with mlflow.start_run() as run:
    logged_model = mlflow.sklearn.log_model(model, "model", signature=signature)
    model_uri = f"models:/{logged_model.model_id}"
    result = mlflow.evaluate(
        model_uri,
        X_test.assign(label=y_test),
        targets="label",
        model_type="classifier",
        evaluators=["default"],
    )

    print(mlflow.get_logged_model(logged_model.model_id))
