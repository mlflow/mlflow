"""Parse and format ``skills:/`` and ``agent-plugins:/`` URIs.

Versions in URIs must already be canonical: skill versions are ASCII integers
without leading zeros, and agent-plugin versions are exact SemVer strings.

The invariant is that a parsed object round-trips exactly
(``format(parse(format(x))) == format(x)``), and every canonical URI satisfies
``format(parse(s)) == s``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mlflow.exceptions import MlflowException
from mlflow.utils.annotations import experimental
from mlflow.utils.semver_utils import parse_semver
from mlflow.utils.validation import (
    MAX_SKILL_VERSION,
    _validate_agent_plugin_name,
    _validate_organization_name,
    _validate_skill_alias,
    _validate_skill_name,
    _validate_skill_version,
)

_SKILL_SCHEME = "skills:/"
_AGENT_PLUGIN_SCHEME = "agent-plugins:/"
_SKILL_VERSION_RE = re.compile(r"^[1-9][0-9]*$")


@experimental(version="3.16.0")
@dataclass(frozen=True)
class ParsedSkillUri:
    name: str
    organization: str = ""
    version: int | None = None
    alias: str | None = None

    def __post_init__(self):
        if not self.name:
            raise MlflowException.invalid_parameter_value("Skill URI must include a name.")
        if self.version is not None and self.alias is not None:
            raise MlflowException.invalid_parameter_value(
                "Skill URI cannot specify both a version and an alias."
            )
        _validate_organization_name(self.organization)
        _validate_skill_name(self.name)
        if self.version is not None:
            _validate_skill_version(self.version)
        if self.alias is not None:
            _validate_skill_alias(self.alias)


@experimental(version="3.16.0")
@dataclass(frozen=True)
class ParsedAgentPluginUri:
    name: str
    organization: str = ""
    version: str | None = None
    alias: str | None = None

    def __post_init__(self):
        if not self.name:
            raise MlflowException.invalid_parameter_value("Agent plugin URI must include a name.")
        if self.version is not None and self.alias is not None:
            raise MlflowException.invalid_parameter_value(
                "Agent plugin URI cannot specify both a version and an alias."
            )
        _validate_organization_name(self.organization)
        _validate_agent_plugin_name(self.name)
        if self.version is not None:
            parse_semver(self.version, param_name="agent plugin version")
        if self.alias is not None:
            _validate_skill_alias(self.alias)


def _split_uri(uri: str, scheme: str) -> tuple[str, str, str | None, str | None]:
    """Return (organization, name, version_str, alias) for a URI under ``scheme``."""
    if not isinstance(uri, str) or not uri.startswith(scheme):
        raise MlflowException.invalid_parameter_value(
            f"Invalid URI {uri!r}: expected it to start with {scheme!r}."
        )
    remainder = uri[len(scheme) :]
    if not remainder:
        raise MlflowException.invalid_parameter_value(f"Invalid URI {uri!r}: missing name.")

    organization = ""
    if remainder.startswith("@"):
        match remainder[1:].split("/", 1):
            case [org, rest] if rest and org:
                organization = org
                remainder = rest
            case ["", rest] if rest:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid URI {uri!r}: organization marker '@' must be followed by a "
                    "non-empty organization name."
                )
            case _:
                raise MlflowException.invalid_parameter_value(
                    f"Invalid URI {uri!r}: organization must be followed by a name."
                )

    version_str: str | None = None
    alias: str | None = None
    if "@" in remainder:
        name, _, alias = remainder.partition("@")
    elif "/" in remainder:
        name, _, version_str = remainder.partition("/")
    else:
        name = remainder
    return organization, name, version_str, alias


def _format_uri(
    scheme: str, organization: str, name: str, version: object | None, alias: str | None
) -> str:
    prefix = scheme
    if organization:
        prefix += f"@{organization}/"
    result = f"{prefix}{name}"
    if version is not None:
        result += f"/{version}"
    elif alias is not None:
        result += f"@{alias}"
    return result


@experimental(version="3.16.0")
def parse_skill_uri(uri: str) -> ParsedSkillUri:
    organization, name, version_str, alias = _split_uri(uri, _SKILL_SCHEME)
    _validate_organization_name(organization)
    _validate_skill_name(name)
    version: int | None = None
    if version_str is not None:
        if _SKILL_VERSION_RE.fullmatch(version_str) is None:
            raise MlflowException.invalid_parameter_value(
                f"Invalid skill version {version_str!r}: must be a positive integer with no "
                "leading zeros, sign, underscores, or non-ASCII digits."
            )
        max_version = str(MAX_SKILL_VERSION)
        if len(version_str) > len(max_version) or (
            len(version_str) == len(max_version) and version_str > max_version
        ):
            raise MlflowException.invalid_parameter_value(
                f"Invalid skill version: must be <= {MAX_SKILL_VERSION}."
            )
        version = int(version_str)
        _validate_skill_version(version)
    if alias is not None:
        _validate_skill_alias(alias)
    return ParsedSkillUri(name=name, organization=organization, version=version, alias=alias)


@experimental(version="3.16.0")
def format_skill_uri(parsed: ParsedSkillUri) -> str:
    return _format_uri(
        _SKILL_SCHEME, parsed.organization, parsed.name, parsed.version, parsed.alias
    )


@experimental(version="3.16.0")
def parse_agent_plugin_uri(uri: str) -> ParsedAgentPluginUri:
    organization, name, version_str, alias = _split_uri(uri, _AGENT_PLUGIN_SCHEME)
    _validate_organization_name(organization)
    _validate_agent_plugin_name(name)
    if version_str is not None:
        parse_semver(version_str, param_name="agent plugin version")
    version = version_str
    if alias is not None:
        _validate_skill_alias(alias)
    return ParsedAgentPluginUri(name=name, organization=organization, version=version, alias=alias)


@experimental(version="3.16.0")
def format_agent_plugin_uri(parsed: ParsedAgentPluginUri) -> str:
    return _format_uri(
        _AGENT_PLUGIN_SCHEME, parsed.organization, parsed.name, parsed.version, parsed.alias
    )
