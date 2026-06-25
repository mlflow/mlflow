from __future__ import annotations

import re
from dataclasses import dataclass

from mlflow.exceptions import MlflowException

_MAX_SEMVER_LENGTH = 128
_MAX_SEMVER_CORE_VALUE = 2_147_483_647
_RELEASE_PRERELEASE_SORT_KEY = "2"
_ASCII_CODE_WIDTH = 3
_TEXT_IDENTIFIER_TERMINATOR = "000"
# Recommended SemVer 2.0.0 regex from semver.org, adapted for Python.
_SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)"
    r"\.(?P<minor>0|[1-9]\d*)"
    r"\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*))*))?"
    r"(?:\+(?P<buildmetadata>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)


@dataclass(frozen=True)
class SemVer:
    major: int
    minor: int
    patch: int
    prerelease: tuple[str, ...] = ()
    build: str | None = None


def parse_semver(version: str, *, param_name: str = "version") -> SemVer:
    if len(version) > _MAX_SEMVER_LENGTH:
        raise MlflowException.invalid_parameter_value(
            f"Invalid semantic version for {param_name}: '{version}' "
            f"(maximum length is {_MAX_SEMVER_LENGTH} characters)"
        )
    match = _SEMVER_RE.fullmatch(version)
    if not match:
        raise MlflowException.invalid_parameter_value(
            f"Invalid semantic version for {param_name}: '{version}'"
        )
    major = int(match.group("major"))
    minor = int(match.group("minor"))
    patch = int(match.group("patch"))
    for label, value in (("major", major), ("minor", minor), ("patch", patch)):
        if value > _MAX_SEMVER_CORE_VALUE:
            raise MlflowException.invalid_parameter_value(
                f"Invalid semantic version for {param_name}: '{version}' "
                f"({label} must be <= {_MAX_SEMVER_CORE_VALUE})"
            )
    return SemVer(
        major=major,
        minor=minor,
        patch=patch,
        prerelease=(
            tuple(match.group("prerelease").split(".")) if match.group("prerelease") else ()
        ),
        build=match.group("buildmetadata"),
    )


def compare_semver(left: SemVer, right: SemVer) -> int:
    for left_part, right_part in (
        (left.major, right.major),
        (left.minor, right.minor),
        (left.patch, right.patch),
    ):
        if left_part != right_part:
            return -1 if left_part < right_part else 1

    if not left.prerelease and not right.prerelease:
        return 0
    if not left.prerelease:
        return 1
    if not right.prerelease:
        return -1

    for left_identifier, right_identifier in zip(left.prerelease, right.prerelease):
        left_numeric = left_identifier.isdigit()
        right_numeric = right_identifier.isdigit()
        if left_numeric and right_numeric:
            left_value = int(left_identifier)
            right_value = int(right_identifier)
            if left_value != right_value:
                return -1 if left_value < right_value else 1
            continue
        if left_numeric != right_numeric:
            return -1 if left_numeric else 1
        if left_identifier != right_identifier:
            return -1 if left_identifier < right_identifier else 1

    if len(left.prerelease) != len(right.prerelease):
        return -1 if len(left.prerelease) < len(right.prerelease) else 1

    return 0


def _encode_numeric_prerelease_identifier(identifier: str) -> str:
    # Numeric identifiers compare by numeric value under SemVer. Prefixing the
    # base-10 payload with its digit count makes plain lexicographic comparison
    # match numeric comparison for all non-negative integer strings.
    return f"0{len(identifier):03d}{identifier}"


def _encode_ascii_sort_key(value: str) -> str:
    return "".join(f"{ord(char):0{_ASCII_CODE_WIDTH}d}" for char in value)


def _encode_text_prerelease_identifier(identifier: str) -> str:
    # Encode ASCII bytes as decimal digits so SQL collations cannot make text
    # prerelease ordering case-insensitive or locale-dependent.
    return f"1{_encode_ascii_sort_key(identifier)}{_TEXT_IDENTIFIER_TERMINATOR}"


def encode_prerelease_sort_key(parsed: SemVer) -> str:
    """Encode prerelease identifiers into a lexicographically sortable key.

    The SQL latest-resolution path orders this field descending, so the encoded
    string must preserve SemVer precedence under plain lexicographic comparison
    without depending on database collation rules.

    Reference: SemVer 2.0.0 https://semver.org/spec/v2.0.0.html

    This is an order-preserving tuple encoding:

    - release versions encode to the sentinel ``"2"``
    - numeric prerelease identifiers encode as ``0[length][digits]``
    - text prerelease identifiers encode as ``1[ASCII-code-points]000``

    Why this works:

    - ``2`` sorts above ``0`` and ``1``, so release versions sort above all
      prerelease encodings.
    - ``0`` sorts below ``1``, so numeric prerelease identifiers sort below
      non-numeric ones.
    - Numeric identifiers use a length-prefix, so lexicographic comparison of
      the encoded payload matches numeric comparison.
    - Text identifiers encode each ASCII code point as a fixed-width decimal
      field, so SQL collation cannot change ordering of ``A``/``a`` or ``-``.
    - The ``000`` terminator sorts below every encoded valid SemVer identifier
      character, so prefix cases like ``alpha < alpha1``,
      ``alpha < alpha-1``, and ``alpha < alpha.1`` are preserved.
    - Concatenating the encoded identifiers preserves left-to-right comparison
      and ensures that a shorter equal-prefix identifier list sorts lower than a
      longer one.

    Examples:
    - ``1.0.0`` -> ``2``
    - ``1.0.0-alpha.2`` -> ``109710811210409700000012``
    - ``1.0.0-alpha.10`` -> ``1097108112104097000000210``
    - ``1.0.0-beta.1`` -> ``109810111609700000011``

    With descending SQL ordering this gives:
    ``release`` > ``beta.1`` > ``alpha.10`` > ``alpha.2``, which matches
    SemVer precedence.
    """
    if not parsed.prerelease:
        return _RELEASE_PRERELEASE_SORT_KEY

    parts = []
    for identifier in parsed.prerelease:
        if identifier.isdigit():
            parts.append(_encode_numeric_prerelease_identifier(identifier))
        else:
            parts.append(_encode_text_prerelease_identifier(identifier))
    return "".join(parts)
