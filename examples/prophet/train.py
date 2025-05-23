import numpy as np
import pandas as pd
from prophet import Prophet, serialize
from prophet.diagnostics import cross_validation, performance_metrics

import mlflow

SOURCE_DATA = (
    "https://raw.githubusercontent.com/facebook/prophet/master/examples/example_retail_sales.csv"
)
np.random.seed(12345)


def extract_params(pr_model):
    params = {attr: getattr(pr_model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}
    return {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}


sales_data = pd.read_csv(SOURCE_DATA)

with mlflow.start_run():
    model = Prophet().fit(sales_data)

    params = extract_params(model)

    metrics_raw = cross_validation(
        model=model,
        horizon="365 days",
        period="180 days",
        initial="710 days",
        parallel="threads",
        disable_tqdm=True,
    )

    cv_metrics = performance_metrics(metrics_raw)
    metrics = cv_metrics.drop(columns=["horizon"]).mean().to_dict()

    # The training data can be retrieved from the fit model for convenience
    train = model.history

    model_info = mlflow.prophet.log_model(
        model, name="prophet_model", input_example=train[["ds"]].head(10)
    )
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)


loaded_model = mlflow.prophet.load_model(model_info.model_uri)

forecast = loaded_model.predict(loaded_model.make_future_dataframe(60))

forecast = forecast[["ds", "yhat"]].tail(90)

print(f"forecast:\n${forecast.head(30)}")
