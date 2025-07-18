"""Type definitions for MLflow webhook payloads.

This module contains class definitions for all webhook event payloads
that are sent when various model registry events occur.
"""

from typing import Optional, TypeAlias, TypedDict

from mlflow.entities.webhook import WebhookEvent


class RegisteredModelCreatedPayload(TypedDict):
    """Payload sent when a new registered model is created."""

    name: str
    tags: dict[str, str]
    description: Optional[str]

    @classmethod
    def example(cls) -> "RegisteredModelCreatedPayload":
        return cls(
            name="example_model",
            tags={"example_key": "example_value"},
            description="An example registered model",
        )


class ModelVersionCreatedPayload(TypedDict):
    """Payload sent when a new model version is created."""

    name: str
    version: str
    source: str
    run_id: Optional[str]
    tags: dict[str, str]
    description: Optional[str]

    @classmethod
    def example(cls) -> "ModelVersionCreatedPayload":
        return cls(
            name="example_model",
            version="1",
            source="runs:/abcd1234abcd5678/model",
            run_id="abcd1234abcd5678",
            tags={"example_key": "example_value"},
            description="An example model version",
        )


class ModelVersionTagSetPayload(TypedDict):
    """Payload sent when a tag is set on a model version."""

    name: str
    version: str
    key: str
    value: str

    @classmethod
    def example(cls) -> "ModelVersionTagSetPayload":
        return cls(
            name="example_model",
            version="1",
            key="example_key",
            value="example_value",
        )


class ModelVersionTagDeletedPayload(TypedDict):
    """Payload sent when a tag is deleted from a model version."""

    name: str
    version: str
    key: str

    @classmethod
    def example(cls) -> "ModelVersionTagDeletedPayload":
        return cls(
            name="example_model",
            version="1",
            key="example_key",
        )


class ModelVersionAliasCreatedPayload(TypedDict):
    """Payload sent when an alias is created for a model version."""

    name: str
    alias: str
    version: str

    @classmethod
    def example(cls) -> "ModelVersionAliasCreatedPayload":
        return cls(
            name="example_model",
            alias="example_alias",
            version="1",
        )


class ModelVersionAliasDeletedPayload(TypedDict):
    """Payload sent when an alias is deleted from a model version."""

    name: str
    alias: str

    @classmethod
    def example(cls) -> "ModelVersionAliasDeletedPayload":
        return cls(
            name="example_model",
            alias="example_alias",
        )


WebhookPayload: TypeAlias = (
    RegisteredModelCreatedPayload
    | ModelVersionCreatedPayload
    | ModelVersionTagSetPayload
    | ModelVersionTagDeletedPayload
    | ModelVersionAliasCreatedPayload
    | ModelVersionAliasDeletedPayload
)

# Mapping of event types to their corresponding payload classes
EVENT_TO_PAYLOAD_CLASS = {
    WebhookEvent.REGISTERED_MODEL_CREATED: RegisteredModelCreatedPayload,
    WebhookEvent.MODEL_VERSION_CREATED: ModelVersionCreatedPayload,
    WebhookEvent.MODEL_VERSION_TAG_SET: ModelVersionTagSetPayload,
    WebhookEvent.MODEL_VERSION_TAG_DELETED: ModelVersionTagDeletedPayload,
    WebhookEvent.MODEL_VERSION_ALIAS_CREATED: ModelVersionAliasCreatedPayload,
    WebhookEvent.MODEL_VERSION_ALIAS_DELETED: ModelVersionAliasDeletedPayload,
}


def get_example_payload_for_event(event: WebhookEvent) -> WebhookPayload:
    """Get an example payload for the given webhook event type.

    Args:
        event: The webhook event type

    Returns:
        Example payload for the event type

    Raises:
        ValueError: If the event type is unknown
    """
    if payload_class := EVENT_TO_PAYLOAD_CLASS.get(event):
        return payload_class.example()

    raise ValueError(f"Unknown event type: {event}")
