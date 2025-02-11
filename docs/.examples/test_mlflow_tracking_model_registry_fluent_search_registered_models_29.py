# Location: mlflow/tracking/_model_registry/fluent.py:145
import pytest


@pytest.mark.parametrize('_', [' mlflow/tracking/_model_registry/fluent.py:145 '])
def test(_):
    import mlflow
    from sklearn.linear_model import LogisticRegression

    with mlflow.start_run():
        mlflow.sklearn.log_model(
            LogisticRegression(),
            "Cordoba",
            registered_model_name="CordobaWeatherForecastModel",
        )
        mlflow.sklearn.log_model(
            LogisticRegression(),
            "Boston",
            registered_model_name="BostonWeatherForecastModel",
        )

    # Get search results filtered by the registered model name
    filter_string = "name = 'CordobaWeatherForecastModel'"
    results = mlflow.search_registered_models(filter_string=filter_string)
    print("-" * 80)
    for res in results:
        for mv in res.latest_versions:
            print("name={}; run_id={}; version={}".format(mv.name, mv.run_id, mv.version))

    # Get search results filtered by the registered model name that matches
    # prefix pattern
    filter_string = "name LIKE 'Boston%'"
    results = mlflow.search_registered_models(filter_string=filter_string)
    print("-" * 80)
    for res in results:
        for mv in res.latest_versions:
            print("name={}; run_id={}; version={}".format(mv.name, mv.run_id, mv.version))

    # Get all registered models and order them by ascending order of the names
    results = mlflow.search_registered_models(order_by=["name ASC"])
    print("-" * 80)
    for res in results:
        for mv in res.latest_versions:
            print("name={}; run_id={}; version={}".format(mv.name, mv.run_id, mv.version))


if __name__ == "__main__":
    test()
