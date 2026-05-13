from enum import Enum, EnumMeta


class MetaEnum(EnumMeta):
    """Metaclass for Enum classes that allows to check if a value is a valid member of the Enum."""

    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class StrEnum(str, Enum, metaclass=MetaEnum):
    def __str__(self):
        """Return the string representation of the enum using its value."""
        return self.value

    @classmethod
    def values(cls) -> list[str]:
        """Return a list of all string values of the Enum."""
        return [str(member) for member in cls]
