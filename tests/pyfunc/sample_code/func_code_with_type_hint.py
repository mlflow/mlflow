from mlflow.models import set_model


def predict(model_input: list[str]):
    return model_input


set_model(predict)
