from functools import cmp_to_key
from itertools import product

import pytest

from mlflow.exceptions import MlflowException
from mlflow.utils.semver_utils import (
    SemVer,
    compare_semver,
    encode_prerelease_sort_key,
    parse_semver,
)


def _compare_identifiers(left: str, right: str) -> int:
    left_is_numeric = left.isdigit()
    right_is_numeric = right.isdigit()

    if left_is_numeric and right_is_numeric:
        left_num = int(left)
        right_num = int(right)
        return (left_num > right_num) - (left_num < right_num)

    if left_is_numeric != right_is_numeric:
        return -1 if left_is_numeric else 1

    return (left > right) - (left < right)


def _compare_prerelease(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    if not left:
        return 0 if not right else 1
    if not right:
        return -1

    for left_id, right_id in zip(left, right):
        if comparison := _compare_identifiers(left_id, right_id):
            return comparison

    if len(left) != len(right):
        return -1 if len(left) < len(right) else 1

    return 0


def _compare_semver_precedence(left: SemVer, right: SemVer) -> int:
    left_core = (left.major, left.minor, left.patch)
    right_core = (right.major, right.minor, right.patch)
    if left_core != right_core:
        return (left_core > right_core) - (left_core < right_core)
    return _compare_prerelease(left.prerelease, right.prerelease)


def _compare_encoded_keys(left: str, right: str) -> int:
    return (left > right) - (left < right)


def _compare_sql_order(left: SemVer, right: SemVer) -> int:
    left_tuple = (
        left.major,
        left.minor,
        left.patch,
        encode_prerelease_sort_key(left),
    )
    right_tuple = (
        right.major,
        right.minor,
        right.patch,
        encode_prerelease_sort_key(right),
    )
    return (left_tuple > right_tuple) - (left_tuple < right_tuple)


def test_parse_semver_returns_structured_components():
    assert parse_semver("1.2.3") == SemVer(major=1, minor=2, patch=3)


def test_parse_semver_preserves_prerelease_and_build_metadata():
    assert parse_semver("1.0.0-beta.2+exp.sha.5114f85") == SemVer(
        major=1,
        minor=0,
        patch=0,
        prerelease=("beta", "2"),
        build="exp.sha.5114f85",
    )


@pytest.mark.parametrize(
    "valid",
    [
        "0.0.4",
        "1.2.3",
        "10.20.30",
        "1.1.2-prerelease+meta",
        "1.1.2+meta",
        "1.1.2+meta-valid",
        "1.0.0-alpha",
        "1.0.0-beta",
        "1.0.0-alpha.beta",
        "1.0.0-alpha.beta.1",
        "1.0.0-alpha.1",
        "1.0.0-alpha0.valid",
        "1.0.0-alpha.0valid",
        "1.0.0-alpha-a.b-c-somethinglong+build.1-aef.1-its-okay",
        "1.0.0-rc.1+build.1",
        "2.0.0-rc.1+build.123",
        "1.2.3-beta",
        "10.2.3-DEV-SNAPSHOT",
        "1.2.3-SNAPSHOT-123",
        "1.0.0",
        "2.0.0",
        "1.1.7",
        "2.0.0+build.1848",
        "2.0.1-alpha.1227",
        "1.0.0-alpha+beta",
        "1.0.0-0.3.7",
        "1.0.0-x.7.z.92",
        "1.2.3--",
        "1.2.3--.-.-",
        "1.2.3-----",
        "1.2.3-4.3.2.1",
        "1.2.3--1.-.abc+000001.-2.3e7",
        "1.2.3----RC-SNAPSHOT.12.9.1--.12+788",
        "1.2.3----R-S.12.9.1--.12+meta",
        "1.2.3----RC-SNAPSHOT.12.9.1--.12",
        "1.0.0+0.build.1-rc.10000aaa-kk-0.1",
        "1.0.0--1",
        "1.0.0-1.-1",
        "1.0.0-0A.is.legal",
    ],
)
def test_parse_semver_accepts_valid_edge_cases(valid):
    parse_semver(valid)


@pytest.mark.parametrize(
    "invalid",
    [
        "1.0",
        "latest",
        "v1.0.0",
        "01.0.0",
        "1.0.0.0",
        "",
        "1.2.3-00",
        "1.2.3-01",
        "1.2.3-.1",
        "1.2.3-a.",
        "1.2.3-...",
        "1.1.2+.123",
        "alpha",
        "alpha.beta",
        "alpha.beta.1",
        "1.0.0-alpha_beta",
        "1.0.0-alpha..1",
        "1.0.0-0123",
        "1.0.0-0123.0123",
        "+invalid",
        "-invalid",
        "-invalid+invalid",
        "-invalid.01",
        "alpha.1",
        "alpha+beta",
        "alpha_beta",
        "alpha.",
        "alpha..",
        "beta",
        "-alpha.",
        "1.0.0-alpha...1",
        "1.0.0-alpha....1",
        "1.0.0-alpha.....1",
        "1.0.0-alpha......1",
        "1.0.0-alpha.......1",
        "1.01.1",
        "1.1.01",
        "2147483648.0.0",
        "0.2147483648.0",
        "0.0.2147483648",
        "1.2.3.DEV",
        "1.2-SNAPSHOT",
        "1.2.31.2.3----RC-SNAPSHOT.12.09.1--..12+788",
        "1.2-RC-SNAPSHOT",
        "-1.0.3-gamma+b7718",
        "+justmeta",
        "9.8.7+meta+meta",
        "9.8.7-whatever+meta+meta",
        "99999999999999999999999.999999999999999999.99999999999999999----RC-SNAPSHOT.12.09.1--------------------------------..12",
    ],
)
def test_parse_semver_rejects_invalid_versions(invalid):
    with pytest.raises(MlflowException, match="Invalid semantic version"):
        parse_semver(invalid)


def test_parse_semver_rejects_versions_longer_than_storage_limit():
    too_long = "1.0.0-" + ("a" * 123)
    with pytest.raises(MlflowException, match="maximum length is 128 characters"):
        parse_semver(too_long)


def test_parse_semver_rejects_core_components_exceeding_db_integer_limit():
    with pytest.raises(MlflowException, match="major must be <= 2147483647"):
        parse_semver("2147483648.0.0")


def test_encode_prerelease_sort_key_matches_docstring_examples():
    alpha_2 = encode_prerelease_sort_key(parse_semver("1.0.0-alpha.2"))
    alpha_10 = encode_prerelease_sort_key(parse_semver("1.0.0-alpha.10"))
    beta_1 = encode_prerelease_sort_key(parse_semver("1.0.0-beta.1"))

    assert alpha_2 == "109710811210409700000012"
    assert alpha_10 == "1097108112104097000000210"
    assert beta_1 == "109810111609700000011"

    assert beta_1 > alpha_10 > alpha_2


def test_encoded_prerelease_keys_preserve_ascii_order_without_raw_text():
    upper_z = encode_prerelease_sort_key(parse_semver("1.0.0-Z"))
    lower_a = encode_prerelease_sort_key(parse_semver("1.0.0-a"))

    assert upper_z < lower_a
    assert "Z" not in upper_z
    assert "a" not in lower_a


@pytest.mark.parametrize(
    ("lower", "higher"),
    [
        ("1.0.0-alpha", "1.0.0-alpha1"),
        ("1.0.0-alpha", "1.0.0-alpha-1"),
        ("1.0.0-alpha", "1.0.0-alpha.1"),
        ("1.0.0-alpha.1", "1.0.0-alpha.beta"),
        ("1.0.0-alpha.beta", "1.0.0-beta"),
        ("1.0.0-beta", "1.0.0-beta.2"),
        ("1.0.0-beta.2", "1.0.0-beta.11"),
        ("1.0.0-beta.11", "1.0.0-rc.1"),
        ("1.0.0-rc.1", "1.0.0"),
        ("1.0.0-0.0", "1.0.0-0.0.0"),
        ("1.0.0-99", "1.0.0-100"),
        ("1.0.0-0", "1.0.0--1"),
        ("1.0.0-0", "1.0.0-1"),
        ("1.0.0-1.0", "1.0.0-1.-1"),
        ("0.9.0", "0.10.0"),
        ("0.9.99", "1.0.0"),
    ],
)
def test_encoded_prerelease_keys_match_semver_precedence_examples(lower, higher):
    lower_parsed = parse_semver(lower)
    higher_parsed = parse_semver(higher)

    assert _compare_semver_precedence(lower_parsed, higher_parsed) == -1
    assert _compare_sql_order(lower_parsed, higher_parsed) == -1


def test_encoded_prerelease_keys_ignore_build_metadata_for_precedence():
    left = parse_semver("1.0.0-alpha+001")
    right = parse_semver("1.0.0-alpha+exp.sha.5114f85")

    assert _compare_semver_precedence(left, right) == 0
    assert encode_prerelease_sort_key(left) == encode_prerelease_sort_key(right)


def test_encoded_prerelease_keys_match_reference_comparator_for_generated_corpus():
    identifier_pool = [
        "0",
        "1",
        "2",
        "10",
        "Z",
        "a",
        "alpha",
        "alpha1",
        "alpha-1",
        "beta",
        "rc",
        "--1",
    ]
    versions = [parse_semver("1.0.0")]
    versions.extend(parse_semver(f"1.0.0-{identifier}") for identifier in identifier_pool)
    versions.extend(
        parse_semver(f"1.0.0-{first}.{second}")
        for first, second in product(identifier_pool, repeat=2)
    )

    encoded_keys = {version: encode_prerelease_sort_key(version) for version in versions}

    for left in versions:
        for right in versions:
            assert _compare_encoded_keys(encoded_keys[left], encoded_keys[right]) == (
                _compare_semver_precedence(left, right)
            )


def test_sql_order_matches_reference_sort_for_broader_generated_corpus():
    identifier_pool = [
        "0",
        "1",
        "2",
        "10",
        "Z",
        "a",
        "alpha",
        "alpha1",
        "alpha-1",
        "beta",
        "--1",
    ]
    versions = [parse_semver("1.0.0")]
    for length in (1, 2, 3):
        versions.extend(
            parse_semver("1.0.0-" + ".".join(parts))
            for parts in product(identifier_pool, repeat=length)
        )

    reference_sorted = sorted(versions, key=cmp_to_key(_compare_semver_precedence))
    sql_sorted = sorted(versions, key=cmp_to_key(_compare_sql_order))

    assert sql_sorted == reference_sorted


def test_compare_semver_matches_reference_comparator_for_generated_corpus():
    identifier_pool = [
        "0",
        "1",
        "2",
        "10",
        "Z",
        "a",
        "alpha",
        "alpha1",
        "alpha-1",
        "beta",
        "--1",
    ]
    versions = [parse_semver("1.0.0")]
    for length in (1, 2, 3):
        versions.extend(
            parse_semver("1.0.0-" + ".".join(parts))
            for parts in product(identifier_pool, repeat=length)
        )

    for left in versions:
        for right in versions:
            assert compare_semver(left, right) == _compare_semver_precedence(left, right)
