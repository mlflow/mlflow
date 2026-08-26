"""Parse and format ``skills:/`` and ``agent-plugins:/`` URIs.

Parsing is normalizing: a skill version must be a canonical ASCII integer (no
leading zeros, sign, underscores, or non-ASCII digits), and an agent-plugin
version is coerced to canonical SemVer via
:func:`mlflow.utils.semver_utils.normalize_semver` (so, e.g.,
``agent-plugins:/pr-workflow/1`` intentionally round-trips to ``.../1.0.0``).

The invariant is that a parsed object round-trips exactly
(``format(parse(format(x))) == format(x)``), and every canonical URI satisfies
``format(parse(s)) == s``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from mlflow.exceptions import MlflowException
from mlflow.utils.semver_utils import normalize_semver
from mlflow.utils.validation import (
    _validate_agent_plugin_name,
    _validate_organization_name,
    _validate_skill_alias,
    _validate_skill_name,
    _validate_skill_version,
)

_SKILL_SCHEME = "skills:/"
_AGENT_PLUGIN_SCHEME = "agent-plugins:/"
_SKILL_VERSION_RE = re.compile(r"^[1-9][0-9]*$")


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
            case [org, rest] if rest:
                organization = org
                remainder = rest
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
        version = int(version_str)
        _validate_skill_version(version)
    if alias is not None:
        _validate_skill_alias(alias)
    return ParsedSkillUri(name=name, organization=organization, version=version, alias=alias)


def format_skill_uri(parsed: ParsedSkillUri) -> str:
    return _format_uri(
        _SKILL_SCHEME, parsed.organization, parsed.name, parsed.version, parsed.alias
    )


def parse_agent_plugin_uri(uri: str) -> ParsedAgentPluginUri:
    organization, name, version_str, alias = _split_uri(uri, _AGENT_PLUGIN_SCHEME)
    _validate_organization_name(organization)
    _validate_agent_plugin_name(name)
    version = normalize_semver(version_str) if version_str is not None else None
    if alias is not None:
        _validate_skill_alias(alias)
    return ParsedAgentPluginUri(name=name, organization=organization, version=version, alias=alias)


def format_agent_plugin_uri(parsed: ParsedAgentPluginUri) -> str:
    return _format_uri(
        _AGENT_PLUGIN_SCHEME, parsed.organization, parsed.name, parsed.version, parsed.alias
    )
