import mlflow
import json
import pandas as pd
import numpy as np
from prophet import Prophet, serialize
from prophet.diagnostics import cross_validation, performance_metrics

SOURCE_DATA = (
    "https://raw.githubusercontent.com/facebook/prophet/master/examples/example_retail_sales.csv"
)
ARTIFACT_PATH = "model"
np.random.seed(12345)


def extract_params(pr_model):
    return {attr: getattr(pr_model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}


sales_data = pd.read_csv(SOURCE_DATA)

with mlflow.start_run():

    model = Prophet().fit(sales_data)

    params = extract_params(model)

    metric_keys = ["mse", "rmse", "mae", "mape", "mdape", "smape", "coverage"]
    metrics_raw = cross_validation(
        model=model,
        horizon="365 days",
        period="180 days",
        initial="710 days",
        parallel="threads",
        disable_tqdm=True,
    )
    cv_metrics = performance_metrics(metrics_raw)
    metrics = {k: cv_metrics[k].mean() for k in metric_keys}

    print(f"Logged Metrics: \n{json.dumps(metrics, indent=2)}")
    print(f"Logged Params: \n{json.dumps(params, indent=2)}")

    mlflow.prophet.log_model(model, artifact_path=ARTIFACT_PATH)
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    model_uri = mlflow.get_artifact_uri(ARTIFACT_PATH)
    print(f"Model artifact logged to: {model_uri}")


loaded_model = mlflow.prophet.load_model(model_uri)

forecast = loaded_model.predict(loaded_model.make_future_dataframe(60))

print(f"forecast:\n${forecast.head(30)}")
