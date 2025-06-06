from mlflow.utils import annotations


def databricks_api(func):
    """
    Decorator to prepend a Databricks-only note to the docstring of a function or class.
    """
    indent = annotations._get_min_indent_of_docstring(func.__doc__) if func.__doc__ else ""
    notice = (
        indent + ".. note:: This functionality is only available in Databricks. "
        "Please install `mlflow[databricks]` to use it.\n\n"
    )
    func.__doc__ = notice + (func.__doc__ or "")
    return func
