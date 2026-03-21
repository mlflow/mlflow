from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import mlflow
from mlflow.models import infer_signature

X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_test_1, X_test_2, y_test_1, y_test_2 = train_test_split(
    X_test, y_test, test_size=0.5, random_state=42
)
model = LogisticRegression().fit(X_train, y_train)

predictions = model.predict(X_train)
signature = infer_signature(X_train, predictions)

with mlflow.start_run() as run:
    model_info = mlflow.sklearn.log_model(model, name="model", signature=signature)
    print(model_info.name)

    # Evaluate the model URI
    mlflow.evaluate(
        model_info.model_uri,
        X_test_1.assign(label=y_test_1),
        targets="label",
        model_type="classifier",
        evaluators=["default"],
    )
    print(mlflow.get_logged_model(model_info.model_id))

    # Evaluate the pyfunc model object
    model = mlflow.pyfunc.load_model(model_info.model_uri)
    assert model.model_id is not None
    mlflow.evaluate(
        model,
        X_test_2.assign(label=y_test_2),
        targets="label",
        model_type="classifier",
        evaluators=["default"],
    )
    print(mlflow.get_logged_model(model_info.model_id))
