def parse_model_name(model_name: str) -> str:
    """
    Parse model name from URI format to provider/model format.
    Accepts two formats:
    - URI format: 'openai:/gpt-4o' -> converted to 'openai/gpt-4o'
    - Standard format: 'openai/gpt-4o' -> returned unchanged
    Args:
        model_name: Model name in URI format or standard format
    Returns:
        Model name in standard 'provider/model' format
    Raises:
        MlflowException: If the model name format is invalid
    """
    from mlflow.exceptions import MlflowException
    from mlflow.metrics.genai.model_utils import _parse_model_uri

    if not model_name:
        raise MlflowException.invalid_parameter_value(
            "Model name cannot be empty. Please provide a model name in the format "
            "'<provider>:/<model>' or '<provider>/<model>'."
        )

    try:
        scheme, path = _parse_model_uri(model_name)
        return f"{scheme}/{path}"
    except MlflowException:
        if "/" in model_name and ":" not in model_name:
            parts = model_name.split("/")
            if len(parts) == 2 and parts[0] and parts[1]:
                return model_name

        raise MlflowException.invalid_parameter_value(
            f"Invalid model name format: '{model_name}'. "
            "Model name must be in one of the following formats:\n"
            "  - '<provider>/<model>' (e.g., 'openai/gpt-4')\n"
            "  - '<provider>:/<model>' (e.g., 'openai:/gpt-4')"
        )
