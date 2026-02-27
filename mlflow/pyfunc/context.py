import contextlib
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

# A thread local variable to store the context of the current prediction request.
# This is particularly used to associate logs/traces with a specific prediction request in the
# caller side. The context variable is intended to be set by the called before invoking the
# predict method, using the set_prediction_context context manager.
_PREDICTION_REQUEST_CTX = ContextVar("mlflow_prediction_request_context", default=None)


@dataclass
class Context:
    # A unique identifier for the current prediction request.
    request_id: str | None = None
    # Whether the current prediction request is as a part of MLflow model evaluation.
    is_evaluate: bool = False
    # The schema of the dependencies to be added into the tag of trace info.
    dependencies_schemas: dict[str, Any] | None = None
    # The logged model ID associated with the current prediction request
    model_id: str | None = None
    # The model serving endpoint name where the prediction request is made
    endpoint_name: str | None = None

    def __init__(
        self,
        request_id: str | None = None,
        is_evaluate: bool = False,
        dependencies_schemas: dict[str, Any] | None = None,
        model_id: str | None = None,
        endpoint_name: str | None = None,
        # Accept extra kwargs so we don't need to worry backward compatibility
        # when adding new attributes to the Context class
        **kwargs,
    ):
        self.request_id = request_id
        self.is_evaluate = is_evaluate
        self.dependencies_schemas = dependencies_schemas
        self.model_id = model_id
        self.endpoint_name = endpoint_name

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Context has no attribute named '{key}'")


@contextlib.contextmanager
def set_prediction_context(context: Context | None):
    """
    Set the context for the current prediction request. The context will be set as a thread-local
    variable and will be accessible globally within the same thread.

    Args:
        context: The context for the current prediction request.
    """
    if context and not isinstance(context, Context):
        raise TypeError(f"Expected context to be an instance of Context, but got: {context}")

    token = _PREDICTION_REQUEST_CTX.set(context)
    try:
        yield
    finally:
        _PREDICTION_REQUEST_CTX.reset(token)


def get_prediction_context() -> Context | None:
    """
    Get the context for the current prediction request. The context is thread-local and is set
    using the set_prediction_context context manager.

    Returns:
        The context for the current prediction request, or None if no context is set.
    """
    return _PREDICTION_REQUEST_CTX.get()
