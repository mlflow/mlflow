import mlflow
import json

from sktime.datasets import load_longley
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import (
    mean_absolute_error,
    mean_absolute_percentage_error,
)

import flavor

ARTIFACT_PATH = "model"
COVERAGE = [0.8, 0.9]

with mlflow.start_run():
    y, X = load_longley()
    y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)

    forecaster = NaiveForecaster()
    forecaster.fit(
        y_train,
        X=X_train,
        fh=[1, 2, 3, 4],
    )

    # create prediction configuration used in pyfunc flavor
    pyfunc_predict_conf = {
        "predict_method": {
            "predict": {},
            "predict_interval": {"coverage": COVERAGE},
        }
    }

    # assign configuration dictionary to fitted model instance
    forecaster.pyfunc_predict_conf = pyfunc_predict_conf

    # extract parameters
    parameters = forecaster.get_params()

    # evaluate model
    y_pred = forecaster.predict(X=X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mape": mean_absolute_percentage_error(y_test, y_pred),
    }

    print(f"Parameters: \n{json.dumps(parameters, indent=2)}")
    print(f"Metrics: \n{json.dumps(metrics, indent=2)}")

    # log parameters and metrics
    mlflow.log_params(parameters)
    mlflow.log_metrics(metrics)

    # log model using custom model flavor
    flavor.log_model(sktime_model=forecaster, artifact_path=ARTIFACT_PATH)
    model_uri = mlflow.get_artifact_uri(ARTIFACT_PATH)

# load model in native sktime flavor and pyfunc flavor
loaded_model = flavor.load_model(model_uri)
loaded_pyfunc = flavor.pyfunc.load_model(model_uri)

print(f"\nNative sktime 'predict':\n${loaded_model.predict(X=X_test)}")
print(
    f"\nNative sktime 'predict_interval':\n${loaded_model.predict_interval(X=X_test, coverage=COVERAGE)}"
)
print(f"\nPyfunc 'predict':\n${loaded_pyfunc.predict(X_test)}")
