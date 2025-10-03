"""
Base models and mixins for MLflow Insights.

This module contains the foundational components used by the main entity models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class DatetimeFieldsMixin:
    """Mixin providing datetime parsing utilities.

    Classes using this mixin can call the parse_datetime class method
    in their field validators to convert string timestamps to datetime objects.
    """

    @classmethod
    def parse_datetime(cls, v: str | datetime) -> datetime:
        """Convert string to datetime if needed.

        Args:
            v: Either a string in ISO format or a datetime object

        Returns:
            datetime object
        """
        if isinstance(v, str):
            # Handle 'Z' suffix (UTC timezone) by replacing with '+00:00'
            if v.endswith("Z"):
                v = v[:-1] + "+00:00"
            return datetime.fromisoformat(v)
        return v


class SerializableModel(BaseModel):
    """Mixin for models that can be serialized to/from YAML."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return self.model_dump(mode="json")

    def to_yaml(self) -> str:
        """Convert to YAML string."""
        import yaml

        return yaml.safe_dump(self.to_dict(), default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "SerializableModel":
        """Create instance from YAML string.

        Args:
            yaml_str: YAML formatted string

        Returns:
            Instance of the model class
        """
        import yaml

        data = yaml.safe_load(yaml_str)
        return cls(**data)
