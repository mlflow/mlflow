# Location: mlflow/tracking/_model_registry/fluent.py:43
import pytest


@pytest.mark.parametrize('_', [' mlflow/tracking/_model_registry/fluent.py:43 '])
def test(_):
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor

    mlflow.set_tracking_uri("sqlite:////tmp/mlruns.db")
    params = {"n_estimators": 3, "random_state": 42}
    X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)

    # Log MLflow entities
    with mlflow.start_run() as run:
        rfr = RandomForestRegressor(**params).fit(X, y)
        signature = infer_signature(X, rfr.predict(X))
        mlflow.log_params(params)
        mlflow.sklearn.log_model(rfr, artifact_path="sklearn-model", signature=signature)

    model_uri = "runs:/{}/sklearn-model".format(run.info.run_id)
    mv = mlflow.register_model(model_uri, "RandomForestRegressionModel")
    print("Name: {}".format(mv.name))
    print("Version: {}".format(mv.version))


if __name__ == "__main__":
    test()
