"""
Base class for OTEL semantic convention translators.

This module provides a base class that implements common translation logic.
Subclasses only need to define the attribute keys and mappings as class attributes.
"""

import json
import logging
from typing import Any

_logger = logging.getLogger(__name__)


class OtelSchemaTranslator:
    """
    Base class for OTEL schema translators.

    Each OTEL semantic convention (OpenInference, Traceloop, GenAI, etc.)
    should extend this class and override class attributes if needed.
    """

    SPAN_KIND_ATTRIBUTE_KEY: str | None = None
    SPAN_KIND_TO_MLFLOW_TYPE: dict[str, str] | None = None
    INPUT_TOKEN_KEY: str | None = None
    OUTPUT_TOKEN_KEY: str | None = None
    TOTAL_TOKEN_KEY: str | None = None
    INPUT_VALUE_KEYS: list[str] | None = None
    OUTPUT_VALUE_KEYS: list[str] | None = None

    def get_message_format(self, attributes: dict[str, Any]) -> str | None:
        """
        Get message format identifier for chat UI rendering.

        Subclasses should override this method to return their format identifier
        when they can handle the given attributes.

        Args:
            attributes: Dictionary of span attributes

        Returns:
            Message format string or None if not applicable
        """

    def translate_span_type(self, attributes: dict[str, Any]) -> str | None:
        """
        Translate OTEL span kind attribute to MLflow span type.

        Args:
            attributes: Dictionary of span attributes

        Returns:
            MLflow span type string or None if not found
        """
        if self.SPAN_KIND_ATTRIBUTE_KEY and (
            span_kind := attributes.get(self.SPAN_KIND_ATTRIBUTE_KEY)
        ):
            # Handle JSON-serialized values
            if isinstance(span_kind, str):
                try:
                    span_kind = json.loads(span_kind)
                except (json.JSONDecodeError, TypeError):
                    pass  # Use the string value as-is

            mlflow_type = self.SPAN_KIND_TO_MLFLOW_TYPE.get(span_kind)
            if mlflow_type is None:
                _logger.debug(
                    f"{self.__class__.__name__}: span kind '{span_kind}' "
                    f"is not supported by MLflow Span Type"
                )
            return mlflow_type

    def get_input_tokens(self, attributes: dict[str, Any]) -> int | None:
        """
        Get input token count from OTEL attributes.

        Args:
            attributes: Dictionary of span attributes

        Returns:
            Input token count or None if not found
        """
        if self.INPUT_TOKEN_KEY:
            return attributes.get(self.INPUT_TOKEN_KEY)

    def get_output_tokens(self, attributes: dict[str, Any]) -> int | None:
        """
        Get output token count from OTEL attributes.

        Args:
            attributes: Dictionary of span attributes

        Returns:
            Output token count or None if not found
        """
        if self.OUTPUT_TOKEN_KEY:
            return attributes.get(self.OUTPUT_TOKEN_KEY)

    def get_total_tokens(self, attributes: dict[str, Any]) -> int | None:
        """
        Get total token count from OTEL attributes.

        Args:
            attributes: Dictionary of span attributes

        Returns:
            Total token count or None if not found
        """
        if self.TOTAL_TOKEN_KEY:
            return attributes.get(self.TOTAL_TOKEN_KEY)

    def get_input_value(self, attributes: dict[str, Any]) -> Any:
        """
        Get input value from OTEL attributes.

        Args:
            attributes: Dictionary of span attributes

        Returns:
            Input value or None if not found
        """
        return self.get_attribute_value(attributes, self.INPUT_VALUE_KEYS)

    def get_output_value(self, attributes: dict[str, Any]) -> Any:
        """
        Get output value from OTEL attributes.

        Args:
            attributes: Dictionary of span attributes

        Returns:
            Output value or None if not found
        """
        return self.get_attribute_value(attributes, self.OUTPUT_VALUE_KEYS)

    def get_attribute_value(
        self, attributes: dict[str, Any], keys_to_check: list[str] | None = None
    ) -> Any:
        """
        Get attribute value from OTEL attributes by checking whether
        the keys in keys_to_check are present in the attributes.
        Always use this function to get the existing attribute value in the OTel Span.

        Args:
            attributes: Dictionary of span attributes
            keys_to_check: List of attribute keys to check

        Returns:
            Attribute value or None if not found
        """
        if keys_to_check:
            for key in keys_to_check:
                if value := self._get_and_check_attribute_value(attributes, key):
                    return value

    def _get_and_check_attribute_value(self, attributes: dict[str, Any], key: str) -> Any:
        """
        Get attribute value from OTEL attributes by checking whether the value is valid or not.
        This avoids fetching the value if it's empty dictionary or null.

        Args:
            attributes: Dictionary of span attributes
            key: Attribute key

        Returns:
            Attribute value or None if not found
        """
        value = attributes.get(key)
        if isinstance(value, str):
            try:
                return value if json.loads(value) else None
            except json.JSONDecodeError:
                pass  # Use the string value as-is
        return value
