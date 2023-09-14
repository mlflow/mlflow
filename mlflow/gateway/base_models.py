from typing import Any, Dict, List

from pydantic import BaseModel


class RequestModel(
    BaseModel,
    # Allow extra fields for pydantic request models, e.g. to support
    # vendor-specific embeddings parameters
    extra="allow",
):
    """
    A pydantic model representing Gateway request data, such as a chat or completions request
    """


class ResponseModel(
    BaseModel,
    # Ignore extra fields for pydantic response models to ensure a consistent response
    # experience for clients across different backends
    extra="ignore",
):
    """
    A pydantic model representing Gateway response data, such as information about a Gateway
    Route returned in response to a GetRoute request
    """


class ConfigModel(
    BaseModel,
    # Ignore extra fields for pydantic config models, since they are unused
    extra="ignore",
):
    """
    A pydantic model representing Gateway configuration data, such as an OpenAI completions
    route definition including route name, model name, API keys, etc.
    """


class LimitModel(
    BaseModel,
    # Ignore extra fields for pydantic limit models, since they are unused
    extra="ignore",
):
    """
    A pydantic model representing Gateway Limit data, such as renewal period, limit
    key, limit value, etc.
    """


class SetLimitsModel(
    BaseModel,
    # Ignore extra fields for pydantic limit models, since they are unused
    extra="ignore",
):
    route: str
    limits: List[Dict[str, Any]]
    """
    A pydantic model representing Gateway SetLimits request body, containing route and limits.
    """
