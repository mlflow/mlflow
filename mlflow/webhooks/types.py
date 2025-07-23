"""Type definitions for MLflow webhook payloads.

This module contains class definitions for all webhook event payloads
that are sent when various model registry events occur.
"""

from typing import Optional, TypeAlias, TypedDict

from mlflow.entities.webhook import WebhookEvent


class RegisteredModelCreatedPayload(TypedDict):
    """Payload sent when a new registered model is created.

    Example payload:

    .. code-block:: python

        {
            "name": "example_model",
            "tags": {"example_key": "example_value"},
            "description": "An example registered model",
        }

    """

    name: str
    """The name of the registered model."""
    tags: dict[str, str]
    """Tags associated with the registered model."""
    description: Optional[str]
    """Description of the registered model."""

    @classmethod
    def example(cls) -> "RegisteredModelCreatedPayload":
        return cls(
            name="example_model",
            tags={"example_key": "example_value"},
            description="An example registered model",
        )


class ModelVersionCreatedPayload(TypedDict):
    """Payload sent when a new model version is created.

    Example payload:

    .. code-block:: python

        {
            "name": "example_model",
            "version": "1",
            "source": "models:/123",
            "run_id": "abcd1234abcd5678",
            "tags": {"example_key": "example_value"},
            "description": "An example model version",
        }

    """

    name: str
    """The name of the registered model."""
    version: str
    """The version of the model."""
    source: str
    """The source URI of the model version."""
    run_id: Optional[str]
    """The run ID associated with the model version, if applicable."""
    tags: dict[str, str]
    """Tags associated with the model version."""
    description: Optional[str]
    """Description of the model version."""

    @classmethod
    def example(cls) -> "ModelVersionCreatedPayload":
        return cls(
            name="example_model",
            version="1",
            source="models:/123",
            run_id="abcd1234abcd5678",
            tags={"example_key": "example_value"},
            description="An example model version",
        )


class ModelVersionTagSetPayload(TypedDict):
    """Payload sent when a tag is set on a model version.

    Example payload:

    .. code-block:: python

        {
            "name": "example_model",
            "version": "1",
            "key": "example_key",
            "value": "example_value",
        }

    """

    name: str
    """The name of the registered model."""
    version: str
    """The version of the model."""
    key: str
    """The tag key being set."""
    value: str
    """The tag value being set."""

    @classmethod
    def example(cls) -> "ModelVersionTagSetPayload":
        return cls(
            name="example_model",
            version="1",
            key="example_key",
            value="example_value",
        )


class ModelVersionTagDeletedPayload(TypedDict):
    """Payload sent when a tag is deleted from a model version.

    Example payload:

    .. code-block:: python

        {
            "name": "example_model",
            "version": "1",
            "key": "example_key",
        }

    """

    name: str
    """The name of the registered model."""
    version: str
    """The version of the model."""
    key: str
    """The tag key being deleted."""

    @classmethod
    def example(cls) -> "ModelVersionTagDeletedPayload":
        return cls(
            name="example_model",
            version="1",
            key="example_key",
        )


class ModelVersionAliasCreatedPayload(TypedDict):
    """
    Payload sent when an alias is created for a model version.

    Example payload:

    .. code-block:: python

        {
            "name": "example_model",
            "alias": "example_alias",
            "version": "1",
        }

    """

    name: str
    """The name of the registered model."""
    alias: str
    """The alias being created."""
    version: str
    """The version of the model the alias is being assigned to."""

    @classmethod
    def example(cls) -> "ModelVersionAliasCreatedPayload":
        return cls(
            name="example_model",
            alias="example_alias",
            version="1",
        )


class ModelVersionAliasDeletedPayload(TypedDict):
    """Payload sent when an alias is deleted from a model version.

    Example payload:

    .. code-block:: python

        {
            "name": "example_model",
            "alias": "example_alias",
        }

    """

    name: str
    """The name of the registered model."""
    alias: str
    """The alias being deleted."""

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
