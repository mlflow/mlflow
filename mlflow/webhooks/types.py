"""Type definitions for MLflow webhook payloads.

This module contains class definitions for all webhook event payloads
that are sent when various model registry events occur.
"""

from typing import TypeAlias, TypedDict

from mlflow.entities.webhook import WebhookAction, WebhookEntity, WebhookEvent


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
    description: str | None
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
    run_id: str | None
    """The run ID associated with the model version, if applicable."""
    tags: dict[str, str]
    """Tags associated with the model version."""
    description: str | None
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


class PromptCreatedPayload(TypedDict):
    """Payload sent when a new prompt is created.

    Example payload:

    .. code-block:: python

        {
            "name": "example_prompt",
            "tags": {"example_key": "example_value"},
            "description": "An example prompt",
        }

    """

    name: str
    """The name of the prompt."""
    tags: dict[str, str]
    """Tags associated with the prompt."""
    description: str | None
    """Description of the prompt."""

    @classmethod
    def example(cls) -> "PromptCreatedPayload":
        return cls(
            name="example_prompt",
            tags={"example_key": "example_value"},
            description="An example prompt",
        )


class PromptVersionCreatedPayload(TypedDict):
    """Payload sent when a new prompt version is created.

    Example payload:

    .. code-block:: python

        {
            "name": "example_prompt",
            "version": "1",
            "template": "Hello {{name}}!",
            "tags": {"example_key": "example_value"},
            "description": "An example prompt version",
        }

    """

    name: str
    """The name of the prompt."""
    version: str
    """The version of the prompt."""
    template: str
    """The template content of the prompt version."""
    tags: dict[str, str]
    """Tags associated with the prompt version."""
    description: str | None
    """Description of the prompt version."""

    @classmethod
    def example(cls) -> "PromptVersionCreatedPayload":
        return cls(
            name="example_prompt",
            version="1",
            template="Hello {{name}}!",
            tags={"example_key": "example_value"},
            description="An example prompt version",
        )


class PromptTagSetPayload(TypedDict):
    """Payload sent when a tag is set on a prompt.

    Example payload:

    .. code-block:: python

        {
            "name": "example_prompt",
            "key": "example_key",
            "value": "example_value",
        }

    """

    name: str
    """The name of the prompt."""
    key: str
    """The tag key being set."""
    value: str
    """The tag value being set."""

    @classmethod
    def example(cls) -> "PromptTagSetPayload":
        return cls(
            name="example_prompt",
            key="example_key",
            value="example_value",
        )


class PromptTagDeletedPayload(TypedDict):
    """Payload sent when a tag is deleted from a prompt.

    Example payload:

    .. code-block:: python

        {
            "name": "example_prompt",
            "key": "example_key",
        }

    """

    name: str
    """The name of the prompt."""
    key: str
    """The tag key being deleted."""

    @classmethod
    def example(cls) -> "PromptTagDeletedPayload":
        return cls(
            name="example_prompt",
            key="example_key",
        )


class PromptVersionTagSetPayload(TypedDict):
    """Payload sent when a tag is set on a prompt version.

    Example payload:

    .. code-block:: python

        {
            "name": "example_prompt",
            "version": "1",
            "key": "example_key",
            "value": "example_value",
        }

    """

    name: str
    """The name of the prompt."""
    version: str
    """The version of the prompt."""
    key: str
    """The tag key being set."""
    value: str
    """The tag value being set."""

    @classmethod
    def example(cls) -> "PromptVersionTagSetPayload":
        return cls(
            name="example_prompt",
            version="1",
            key="example_key",
            value="example_value",
        )


class PromptVersionTagDeletedPayload(TypedDict):
    """Payload sent when a tag is deleted from a prompt version.

    Example payload:

    .. code-block:: python

        {
            "name": "example_prompt",
            "version": "1",
            "key": "example_key",
        }

    """

    name: str
    """The name of the prompt."""
    version: str
    """The version of the prompt."""
    key: str
    """The tag key being deleted."""

    @classmethod
    def example(cls) -> "PromptVersionTagDeletedPayload":
        return cls(
            name="example_prompt",
            version="1",
            key="example_key",
        )


class PromptAliasCreatedPayload(TypedDict):
    """Payload sent when an alias is created for a prompt version.

    Example payload:

    .. code-block:: python

        {
            "name": "example_prompt",
            "alias": "example_alias",
            "version": "1",
        }

    """

    name: str
    """The name of the prompt."""
    alias: str
    """The alias being created."""
    version: str
    """The version of the prompt the alias is being assigned to."""

    @classmethod
    def example(cls) -> "PromptAliasCreatedPayload":
        return cls(
            name="example_prompt",
            alias="example_alias",
            version="1",
        )


class PromptAliasDeletedPayload(TypedDict):
    """Payload sent when an alias is deleted from a prompt.

    Example payload:

    .. code-block:: python

        {
            "name": "example_prompt",
            "alias": "example_alias",
        }

    """

    name: str
    """The name of the prompt."""
    alias: str
    """The alias being deleted."""

    @classmethod
    def example(cls) -> "PromptAliasDeletedPayload":
        return cls(
            name="example_prompt",
            alias="example_alias",
        )


WebhookPayload: TypeAlias = (
    RegisteredModelCreatedPayload
    | ModelVersionCreatedPayload
    | ModelVersionTagSetPayload
    | ModelVersionTagDeletedPayload
    | ModelVersionAliasCreatedPayload
    | ModelVersionAliasDeletedPayload
    | PromptCreatedPayload
    | PromptVersionCreatedPayload
    | PromptTagSetPayload
    | PromptTagDeletedPayload
    | PromptVersionTagSetPayload
    | PromptVersionTagDeletedPayload
    | PromptAliasCreatedPayload
    | PromptAliasDeletedPayload
)

# Mapping of (entity, action) tuples to their corresponding payload classes
EVENT_TO_PAYLOAD_CLASS: dict[tuple[WebhookEntity, WebhookAction], type[WebhookPayload]] = {
    (WebhookEntity.REGISTERED_MODEL, WebhookAction.CREATED): RegisteredModelCreatedPayload,
    (WebhookEntity.MODEL_VERSION, WebhookAction.CREATED): ModelVersionCreatedPayload,
    (WebhookEntity.MODEL_VERSION_TAG, WebhookAction.SET): ModelVersionTagSetPayload,
    (WebhookEntity.MODEL_VERSION_TAG, WebhookAction.DELETED): ModelVersionTagDeletedPayload,
    (WebhookEntity.MODEL_VERSION_ALIAS, WebhookAction.CREATED): ModelVersionAliasCreatedPayload,
    (WebhookEntity.MODEL_VERSION_ALIAS, WebhookAction.DELETED): ModelVersionAliasDeletedPayload,
    (WebhookEntity.PROMPT, WebhookAction.CREATED): PromptCreatedPayload,
    (WebhookEntity.PROMPT_VERSION, WebhookAction.CREATED): PromptVersionCreatedPayload,
    (WebhookEntity.PROMPT_TAG, WebhookAction.SET): PromptTagSetPayload,
    (WebhookEntity.PROMPT_TAG, WebhookAction.DELETED): PromptTagDeletedPayload,
    (WebhookEntity.PROMPT_VERSION_TAG, WebhookAction.SET): PromptVersionTagSetPayload,
    (WebhookEntity.PROMPT_VERSION_TAG, WebhookAction.DELETED): PromptVersionTagDeletedPayload,
    (WebhookEntity.PROMPT_ALIAS, WebhookAction.CREATED): PromptAliasCreatedPayload,
    (WebhookEntity.PROMPT_ALIAS, WebhookAction.DELETED): PromptAliasDeletedPayload,
}


def get_example_payload_for_event(event: WebhookEvent) -> WebhookPayload:
    """Get an example payload for the given webhook event type.

    Args:
        event: The webhook event instance

    Returns:
        Example payload for the event type

    Raises:
        ValueError: If the event type is unknown
    """
    event_key = (event.entity, event.action)
    if payload_class := EVENT_TO_PAYLOAD_CLASS.get(event_key):
        return payload_class.example()

    raise ValueError(f"Unknown event type: {event.entity}.{event.action}")


def get_payload_class_for_event(event: WebhookEvent) -> type[WebhookPayload] | None:
    """Get the payload class for the given webhook event type.

    Args:
        event: The webhook event instance

    Returns:
        Payload class for the event type, or None if unknown
    """
    return EVENT_TO_PAYLOAD_CLASS.get((event.entity, event.action))
