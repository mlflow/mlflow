from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

import mlflow

X, y = [[0, 1], [1, 2], [2, 3]], [0, 1, 2]

original = LogisticRegression.fit


def patch(self, X, y, **kwargs):
    fitted_model = original(self, X, y)
    params = self.get_params()
    model = mlflow.sklearn.log_model(
        sk_model=lr,
        name="test",
        params=params,
    )

    mse = mean_squared_error(y, fitted_model.predict(X))
    mlflow.log_metrics({"mse": mse}, model_id=model._model_id)
    return fitted_model


LogisticRegression.fit = patch  # mlflow.sklearn.autolog()

with mlflow.start_run():
    lr = LogisticRegression()
    lr.fit(X, y)
