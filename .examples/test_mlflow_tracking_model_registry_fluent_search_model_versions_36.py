# Location: mlflow/mlflow/tracking/_model_registry/fluent.py:270
import pytest


@pytest.mark.parametrize('_', [' mlflow/mlflow/tracking/_model_registry/fluent.py:270 '])
def test(_):
    import mlflow
    from sklearn.linear_model import LogisticRegression

    for _ in range(2):
        with mlflow.start_run():
            mlflow.sklearn.log_model(
                LogisticRegression(),
                "Cordoba",
                registered_model_name="CordobaWeatherForecastModel",
            )

    # Get all versions of the model filtered by name
    filter_string = "name = 'CordobaWeatherForecastModel'"
    results = mlflow.search_model_versions(filter_string=filter_string)
    print("-" * 80)
    for res in results:
        print(f"name={res.name}; run_id={res.run_id}; version={res.version}")

    # Get the version of the model filtered by run_id
    filter_string = "run_id = 'ae9a606a12834c04a8ef1006d0cff779'"
    results = mlflow.search_model_versions(filter_string=filter_string)
    print("-" * 80)
    for res in results:
        print(f"name={res.name}; run_id={res.run_id}; version={res.version}")


if __name__ == "__main__":
    test()
