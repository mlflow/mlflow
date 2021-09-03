import mlflow
import pandas as pd
import numpy as np
from prophet import Prophet, serialize
from prophet.diagnostics import cross_validation, performance_metrics

SOURCE_DATA = (
    "https://raw.githubusercontent.com/facebook/prophet/master/examples/example_retail_sales.csv"
)
np.random.seed(12345)


def extract_params(model):
    return {attr: getattr(model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}


sales_data = pd.read_csv(SOURCE_DATA)

with mlflow.start_run():

    model = Prophet().fit(sales_data)

    params = extract_params(model)

    metric_values = ["mse", "rmse", "mae", "mape", "mdape", "smape", "coverage"]
    metrics_raw = cross_validation(
        model=model,
        horizon="365 days",
        period="180 days",
        initial="710 days",
        parallel="threads",
        disable_tqdm=True,
    )
    cv_metrics = performance_metrics(metrics_raw)
    metrics = {}
    [metrics.update({x: cv_metrics[x].mean()}) for x in metric_values]

    mlflow.prophet.save_model(model)
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

model_uri = mlflow.get_artifact_uri("model")

loaded_model = mlflow.prophet.load_model(model_uri)

forecast = loaded_model.predict(model.make_future_dataframe(60))
