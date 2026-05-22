from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version as OriginalVersion
from pydantic import BaseModel, ConfigDict, field_validator

DEV_VERSION = "dev"
# Treat "dev" as "newer than any existing versions"
DEV_NUMERIC = "9999.9999.9999"


class Version(OriginalVersion):
    def __init__(self, version: str, release_date: datetime | None = None):
        self._is_dev = version == DEV_VERSION
        self._release_date = release_date
        super().__init__(DEV_NUMERIC if self._is_dev else version)

    def __str__(self):
        return DEV_VERSION if self._is_dev else super().__str__()

    @classmethod
    def create_dev(cls):
        return cls(DEV_VERSION, datetime.now(timezone.utc))

    @property
    def days_since_release(self) -> int | None:
        """
        Compute the number of days since this version was released.
        Returns None if release date is not available.
        """
        if self._release_date is None:
            return None
        delta = datetime.now(timezone.utc) - self._release_date
        return delta.days


class PackageInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pip_release: str
    install_dev: str | None = None
    module_name: str | None = None
    genai: bool = False
    repo: str | None = None


class TestConfig(BaseModel):
    minimum: Version
    maximum: Version
    unsupported: list[SpecifierSet] | None = None
    requirements: dict[str, list[str]] | None = None
    python: dict[str, str] | None = None
    runs_on: dict[str, str] | None = None
    java: dict[str, str] | None = None
    run: str
    allow_unreleased_max_version: bool | None = None
    pre_test: str | None = None
    test_every_n_versions: int = 1
    test_tracing_sdk: bool = False
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @field_validator("minimum", mode="before")
    @classmethod
    def validate_minimum(cls, v):
        return Version(v)

    @field_validator("maximum", mode="before")
    @classmethod
    def validate_maximum(cls, v):
        return Version(v)

    @field_validator("unsupported", mode="before")
    @classmethod
    def validate_unsupported(cls, v):
        return [SpecifierSet(x) for x in v] if v else None

    @field_validator("python", mode="before")
    @classmethod
    def validate_python_requirements(cls, v):
        if v is None:
            return v

        # Read the minimum Python version from .python-version file
        python_version_file = Path(".python-version")
        min_python_version = python_version_file.read_text().strip()

        # Check if any value in the python dict matches the minimum version
        for version in v.values():
            if version == min_python_version:
                raise ValueError(f"Unnecessary Python version requirement: {version}")

        return v


class FlavorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    package_info: PackageInfo
    models: TestConfig | None = None
    autologging: TestConfig | None = None

    @property
    def categories(self) -> list[tuple[str, TestConfig]]:
        cs = []
        if self.models:
            cs.append(("models", self.models))
        if self.autologging:
            cs.append(("autologging", self.autologging))
        return cs
