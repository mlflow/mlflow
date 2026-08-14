import json

import numpy as np
from pmdarima import auto_arima, model_selection
from pmdarima.datasets import load_wineind

import mlflow
from mlflow.models import infer_signature

ARTIFACT_PATH = "model"


def calculate_cv_metrics(model, endog, metric, cv):
    cv_metric = model_selection.cross_val_score(model, endog, cv=cv, scoring=metric, verbose=0)
    return cv_metric[~np.isnan(cv_metric)].mean()


with mlflow.start_run():
    data = load_wineind()

    train, test = model_selection.train_test_split(data, train_size=150)

    print("Training AutoARIMA model...")
    arima = auto_arima(
        train,
        error_action="ignore",
        trace=False,
        suppress_warnings=True,
        maxiter=5,
        seasonal=True,
        m=12,
    )

    print("Model trained. \nExtracting parameters...")
    parameters = arima.get_params(deep=True)

    metrics = {x: getattr(arima, x)() for x in ["aicc", "aic", "bic", "hqic", "oob"]}

    # Cross validation backtesting
    cross_validator = model_selection.RollingForecastCV(h=10, step=20, initial=60)

    for x in ["smape", "mean_absolute_error", "mean_squared_error"]:
        metrics[x] = calculate_cv_metrics(arima, data, x, cross_validator)

    print(f"Metrics: \n{json.dumps(metrics, indent=2)}")
    print(f"Parameters: \n{json.dumps(parameters, indent=2)}")

    predictions = arima.predict(n_periods=30, return_conf_int=False)
    signature = infer_signature(train, predictions)

    model_info = mlflow.pmdarima.log_model(
        pmdarima_model=arima, name=ARTIFACT_PATH, signature=signature
    )
    mlflow.log_params(parameters)
    mlflow.log_metrics(metrics)

    print(f"Model artifact logged to: {model_info.model_uri}")

loaded_model = mlflow.pmdarima.load_model(model_info.model_uri)

forecast = loaded_model.predict(30)

print(f"Forecast: \n{forecast}")
