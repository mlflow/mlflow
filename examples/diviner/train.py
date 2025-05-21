import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
from diviner import GroupedProphet
from pmdarima import datasets

import mlflow.diviner


def generate_data(location_data, start_dt) -> pd.DataFrame:
    """
    Synthetic data generation utility to take a real data set and generate a 'stacked'
    representation that can be grouped by identifying keys within columns.
    Here we are taking a list of tuples as location data [(country, city)] and applying
    a random factor to each copy of the underlying series to generate data at different scales.

    Args:
        location_data: List[Tuple("country", "city")] for synthetic key grouping columns
            generation.
        start_dt: String datetime value in the form of `YYYY-mm-dd HH:MM:SS`

    Returns:
        A Pandas DataFrame that is a concatenation of each entry of the location_data tuple
        values with a datetime column added via hourly intervals for the duration of each
        series. The original data is in hourly electrical consumption values.
    """
    raw_data = datasets.load_taylor(as_series=True)
    start = datetime.strptime(start_dt, "%Y-%m-%d %H:%M:%S")
    dates = pd.date_range(start=start, periods=len(raw_data), freq="H").values

    generated_listing = []

    for country, city in location_data:
        generated_listing.append(
            pd.DataFrame(
                {
                    "watts": np.random.uniform(0.3, 0.8) * raw_data,
                    "datetime": dates,
                    "country": country,
                    "city": city,
                }
            )
        )

    return pd.concat(generated_listing).reset_index().drop("index", axis=1)


def grouped_prophet_example(locations, start_dt):
    print("Generating data...\n")
    data = generate_data(location_data=locations, start_dt=start_dt)
    grouping_keys = ["country", "city"]
    print("Data Generated.\nBuilding GroupedProphet Model...")

    model = GroupedProphet(n_changepoints=96, uncertainty_samples=0).fit(
        df=data, group_key_columns=grouping_keys, y_col="watts", datetime_col="datetime"
    )
    print("GroupedProphet model built.\n")

    params = model.extract_model_params()

    print(f"Params: \n{params.to_string()}")

    print("Running Cross Validation on all groups...\n")
    metrics = model.cross_validate_and_score(
        horizon="120 hours",
        period="480 hours",
        initial="960 hours",
        parallel="threads",
        rolling_window=0.05,
        monthly=False,
    )
    print(f"Cross Validation Metrics: \n{metrics.to_string()}")

    model_info = mlflow.diviner.log_model(diviner_model=model, name="diviner_model")

    # As an Alternative to saving metrics and params directly with a `log_dict()` function call,
    # Serializing the DataFrames to local as a .csv can be done as well, without requiring
    # column or object manipulation as shown below this block, utilizing a temporary directory
    # with a context wrapper to clean up the files from the local OS after the artifact logging
    # is complete:

    with tempfile.TemporaryDirectory() as tmpdir:
        params.to_csv(f"{tmpdir}/params.csv", index=False, header=True)
        metrics.to_csv(f"{tmpdir}/metrics.csv", index=False, header=True)
        mlflow.log_artifacts(tmpdir, artifact_path="run_data")

    # Saving the parameters and metrics as json without having to serialize to local
    # NOTE: this requires casting of fields that cannot be serialized to JSON
    # NOTE: Do not use both of these methods. These are shown as an either/or alternative based
    # on how you would choose to consume, view, or analyze the per-group metrics and parameters.

    # NB: There are object references present in the Prophet model parameters. Coerce to string if
    # using a JSON serialization approach with ``mlflow.log_dict()``.
    params = params.astype(dtype=str, errors="ignore")

    mlflow.log_dict(params.to_dict(), "params.json")

    mlflow.log_dict(metrics.to_dict(), "metrics.json")

    return model_info.model_uri


if __name__ == "__main__":
    locations = [
        ("US", "Raleigh"),
        ("US", "KansasCity"),
        ("CA", "Toronto"),
        ("CA", "Quebec"),
        ("MX", "Tijuana"),
        ("MX", "MexicoCity"),
    ]
    start_dt = "2022-02-01 04:11:35"

    with mlflow.start_run():
        uri = grouped_prophet_example(locations, start_dt)

    loaded_model = mlflow.diviner.load_model(model_uri=uri)

    forecast = loaded_model.forecast(horizon=12, frequency="H")

    print(f"Forecast: \n{forecast.to_string()}")
