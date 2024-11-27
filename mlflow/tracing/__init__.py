from mlflow.tracing.display import get_display_handler
from mlflow.tracing.provider import disable, enable

__all__ = ["disable", "enable", "disable_notebook_display", "enable_notebook_display"]


def disable_notebook_display():
    get_display_handler().disable()


def enable_notebook_display():
    get_display_handler().enable()
