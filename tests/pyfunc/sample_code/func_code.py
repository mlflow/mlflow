from mlflow.models import set_model


def predict(model_input: list[str]):
    return f"This was the input: {model_input[0]}"


set_model(predict)
