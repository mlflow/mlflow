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


def build_source(
    source_type: SkillSourceType | None,
    source: str | None,
    ref: str | None = None,
    subpath: str | None = None,
) -> GitSource | OCISource | ZipSource | str | None:
    """Reconstruct a typed source from the flat wire/DB fields.

    git -> GitSource(url, ref, subpath); oci -> OCISource(image, subpath);
    zip -> ZipSource(url, subpath); mlflow/assembled -> the plain string pointer.
    """
    if source is None:
        return None
    if source_type is not None and not isinstance(source_type, SkillSourceType):
        source_type = SkillSourceType(source_type)
    if source_type == SkillSourceType.GIT:
        return GitSource(url=source, ref=ref, subpath=subpath)
    if source_type == SkillSourceType.OCI:
        return OCISource(image=source, subpath=subpath)
    if source_type == SkillSourceType.ZIP:
        return ZipSource(url=source, subpath=subpath)
    return source
