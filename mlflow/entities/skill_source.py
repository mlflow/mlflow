from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental


class SkillSourceType(str, Enum):
    GIT = "git"
    OCI = "oci"
    ZIP = "zip"
    MLFLOW = "mlflow"
    ASSEMBLED = "assembled"

    def __str__(self):
        return self.value


@experimental(version="3.16.0")
@dataclass(frozen=True)
class GitSource:
    url: str
    ref: str | None = None
    subpath: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            k: v
            for k, v in {"url": self.url, "ref": self.ref, "subpath": self.subpath}.items()
            if v is not None
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GitSource:
        try:
            url = data["url"]
        except KeyError:
            raise MlflowException.invalid_parameter_value(
                "Missing required key 'url' in GitSource dictionary"
            ) from None
        return cls(url=url, ref=data.get("ref"), subpath=data.get("subpath"))


@experimental(version="3.16.0")
@dataclass(frozen=True)
class OCISource:
    image: str
    subpath: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            k: v for k, v in {"image": self.image, "subpath": self.subpath}.items() if v is not None
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OCISource:
        try:
            image = data["image"]
        except KeyError:
            raise MlflowException.invalid_parameter_value(
                "Missing required key 'image' in OCISource dictionary"
            ) from None
        return cls(image=image, subpath=data.get("subpath"))


@experimental(version="3.16.0")
@dataclass(frozen=True)
class ZipSource:
    url: str
    subpath: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            k: v for k, v in {"url": self.url, "subpath": self.subpath}.items() if v is not None
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ZipSource:
        try:
            url = data["url"]
        except KeyError:
            raise MlflowException.invalid_parameter_value(
                "Missing required key 'url' in ZipSource dictionary"
            ) from None
        return cls(url=url, subpath=data.get("subpath"))


SourceValue = GitSource | OCISource | ZipSource | str

_SOURCE_CLASSES = {
    SkillSourceType.GIT: GitSource,
    SkillSourceType.OCI: OCISource,
    SkillSourceType.ZIP: ZipSource,
}


def source_to_dict(source: SourceValue | None) -> Any:
    if source is None or isinstance(source, str):
        return source
    return source.to_dict()


def source_from_dict(source: Any, source_type: SkillSourceType | None) -> SourceValue | None:
    if source is None or isinstance(source, str):
        return source
    if source_type is not None and not isinstance(source_type, SkillSourceType):
        source_type = SkillSourceType(source_type)
    cls = _SOURCE_CLASSES.get(source_type)
    if cls is None:
        raise MlflowException.invalid_parameter_value(
            f"Cannot deserialize a structured source for source_type {source_type!r}; only "
            "git, oci, and zip sources use structured payloads (mlflow and assembled sources "
            "are represented as strings)."
        )
    return cls.from_dict(source)
