"""Type definitions for MLflow webhook payloads.

This module contains TypedDict definitions for all webhook event payloads
that are sent when various model registry events occur.
"""

from typing import Optional, TypeAlias, TypedDict


class RegisteredModelCreatedPayload(TypedDict):
    """Payload sent when a new registered model is created."""

    name: str
    tags: dict[str, str]
    description: str


class ModelVersionCreatedPayload(TypedDict):
    """Payload sent when a new model version is created."""

    name: str
    version: str
    source: str
    run_id: Optional[str]
    tags: dict[str, str]
    description: Optional[str]


class ModelVersionTagSetPayload(TypedDict):
    """Payload sent when a tag is set on a model version."""

    name: str
    version: str
    key: str
    value: str


class ModelVersionTagDeletedPayload(TypedDict):
    """Payload sent when a tag is deleted from a model version."""

    name: str
    version: str
    key: str


class ModelVersionAliasCreatedPayload(TypedDict):
    """Payload sent when an alias is created for a model version."""

    name: str
    alias: str
    version: str


class ModelVersionAliasDeletedPayload(TypedDict):
    """Payload sent when an alias is deleted from a model version."""

    name: str
    alias: str


WebhookPayload: TypeAlias = (
    RegisteredModelCreatedPayload
    | ModelVersionCreatedPayload
    | ModelVersionTagSetPayload
    | ModelVersionTagDeletedPayload
    | ModelVersionAliasCreatedPayload
    | ModelVersionAliasDeletedPayload
)
