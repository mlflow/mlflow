from unittest.mock import MagicMock, patch


def is_model_traced(model):
    """
    Check if a PyFuncModel is being traced without logging to the database.

    Args:
        model (PyFuncModel): The model to check.

    Returns:
        True if the model is being traced, False otherwise.
    """
    with patch("mlflow.tracing.provider._get_tracer", return_value=None) as mock_get_tracer:
        try:
            model.predict(MagicMock())
        except Exception:
            pass

        return 0 < mock_get_tracer.call_count
