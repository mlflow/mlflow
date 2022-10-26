import mlflow


def test_doctor():
    mlflow.doctor()
    with mlflow.start_run():
        mlflow.doctor()
