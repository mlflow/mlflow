from dataclasses import dataclass
from datetime import datetime, timedelta


def _format_datetime_for_repr(value: datetime) -> str:
    formatted = value.isoformat(sep=" ", timespec="seconds")
    if value.utcoffset() == timedelta(0):
        return formatted.removesuffix("+00:00") + " UTC"
    return formatted


@dataclass(frozen=True)
class EvaluationDatasetVersion:
    version: int
    created_at: datetime | None = None
    created_by: str | None = None
    operation: str | None = None

    def __repr__(self) -> str:
        created_at = (
            repr(_format_datetime_for_repr(self.created_at))
            if self.created_at is not None
            else None
        )
        return (
            f"{type(self).__name__}("
            f"version={self.version!r}, "
            f"created_at={created_at}, "
            f"created_by={self.created_by!r}, "
            f"operation={self.operation!r}"
            ")"
        )


@dataclass(frozen=True)
class EvaluationDatasetAlias:
    alias: str
    version: EvaluationDatasetVersion
