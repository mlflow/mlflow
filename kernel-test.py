import mlflow
import mlflow.models
from mlflow.models.signature import infer_signature

mlflow.set_tracking_uri("http://localhost:5000")

with mlflow.start_run():
    import numpy as np
    from sklearn.linear_model import LinearRegression

    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])

    model = LinearRegression()
    model.fit(X, y)


    # Infer the input/output signature for the model using the training examples
    signature = infer_signature(X, model.predict(X))

    mlflow.sklearn.log_model(model, signature=signature, name="simple_linear_regression_model")