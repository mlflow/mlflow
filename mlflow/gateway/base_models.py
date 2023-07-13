from pydantic import BaseModel, Extra


class RequestModel(
    BaseModel,
    # Allow extra fields for pydantic request models, e.g. to support
    # vendor-specific embeddings parameters
    extra=Extra.allow,
):
    """
    A pydantic model representing Gateway request data, such as a chat or completions request
    """


class ResponseModel(
    BaseModel,
    # Ignore extra fields for pydantic response models to ensure a consistent response
    # experience for clients across different backends
    extra=Extra.ignore,
):
    """
    A pydantic model representing Gateway response data, such as information about a Gateway
    Route returned in response to a GetRoute request
    """


class ConfigModel(
    BaseModel,
    # Ignore extra fields for pydantic config models, since they are unused
    extra=Extra.ignore,
):
    """
    A pydantic model representing Gateway configuration data, such as an OpenAI completions
    route definition including route name, model name, API keys, etc.
    """
