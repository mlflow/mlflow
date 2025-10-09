def parse_model_name(model_name: str) -> str:
    """
    Parse model name from MLflow format to provider/model format.

    Converts "provider:/model" to "provider/model" format.

    Args:
        model_name: Model name in either "provider:/model" or "provider/model" format

    Returns:
        Model name in "provider/model" format
    """
    if ":/" in model_name:
        return model_name.replace(":/", "/")
    return model_name
