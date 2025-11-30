from mlflow.models import ModelConfig, set_model


def predict(model_input):
    model_config = ModelConfig()
    timeout = model_config.get("timeout")
    return [timeout] * len(model_input)


set_model(predict)
