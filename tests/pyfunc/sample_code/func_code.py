from mlflow.models import set_model


def predict(model_input):
    return model_input


set_model(predict)
