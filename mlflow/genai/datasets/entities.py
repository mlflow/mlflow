from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class EvaluationDatasetVersion:
    version: int
    created_at: datetime | None = None
    created_by: str | None = None
    operation: str | None = None


@dataclass(frozen=True)
class EvaluationDatasetAlias:
    alias: str
    version: EvaluationDatasetVersion
