from mlflow.models import set_model


def predict(model_input):
    return f"This was the input: {model_input}"


set_model(predict)
