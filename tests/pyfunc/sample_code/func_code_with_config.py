from mlflow.models import ModelConfig, set_model


def predict(model_input):
    model_config = ModelConfig(development_config="tests/pyfunc/sample_code/config.yml")
    timeout = model_config.get("timeout")
    return f"This was the input: {model_input}, timeout {timeout}"


set_model(predict)
