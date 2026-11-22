import pytest

from mlflow.exceptions import MlflowException
from mlflow.utils.skill_uris import (
    ParsedSkillUri,
    format_skill_uri,
    parse_skill_uri,
)

SKILL_URIS = [
    "skills:/code-review",
    "skills:/code-review/1",
    "skills:/code-review@production",
    "skills:/code-review@latest",
    "skills:/@acme/code-review",
    "skills:/@acme/code-review/1",
    "skills:/@acme/code-review@production",
    "skills:/@acme/code-review@latest",
    "skills:/@acme.com/123",
]


@pytest.mark.parametrize("uri", SKILL_URIS)
def test_skill_uri_roundtrip_from_string(uri):
    assert format_skill_uri(parse_skill_uri(uri)) == uri


@pytest.mark.parametrize(
    "parsed",
    [
        ParsedSkillUri(name="code-review"),
        ParsedSkillUri(name="code-review", organization="acme", version=3),
        ParsedSkillUri(name="code-review", alias="production"),
        ParsedSkillUri(name="code-review", alias="latest"),
    ],
)
def test_skill_object_roundtrip(parsed):
    assert parse_skill_uri(format_skill_uri(parsed)) == parsed


def test_parse_skill_uri_fields():
    parsed = parse_skill_uri("skills:/@acme/code-review/2")
    assert parsed == ParsedSkillUri(name="code-review", organization="acme", version=2)


def test_version_and_alias_mutually_exclusive():
    with pytest.raises(MlflowException, match="both a version and an alias"):
        ParsedSkillUri(name="x", version=1, alias="production")


@pytest.mark.parametrize(
    ("uri", "match"),
    [
        ("models:/code-review", "expected it to start with"),  # wrong scheme
        ("skills:/", "missing name"),  # missing name
        ("skills:/@acme", "organization must be followed by a name"),  # org without name
        ("skills:/code-review/abc", "positive integer"),  # non-integer version
        ("skills:/code-review/1_000", "positive integer"),  # underscore digit group
        ("skills:/code-review/+5", "positive integer"),  # signed integer
        ("skills:/code-review/01", "positive integer"),  # leading zero
        ("skills:/code-review/٥", "positive integer"),  # Arabic-Indic digit
        ("skills:/@bad@org/code-review", "Invalid organization name"),  # invalid org
    ],
)
def test_parse_skill_uri_invalid(uri, match):
    with pytest.raises(MlflowException, match=match):
        parse_skill_uri(uri)


def test_parse_uri_rejects_empty_marked_organization():
    with pytest.raises(MlflowException, match="organization marker"):
        parse_skill_uri("skills:/@/skill")


def test_skill_uri_version_must_fit_storage_bound():
    assert parse_skill_uri("skills:/code-review/2147483647").version == 2_147_483_647
    with pytest.raises(MlflowException, match="2147483647"):
        parse_skill_uri("skills:/code-review/2147483648")


def test_skill_uri_huge_version_raises_mlflow_exception():
    with pytest.raises(MlflowException, match="skill version"):
        parse_skill_uri(f"skills:/code-review/{'9' * 4301}")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": "bad/name"}, "skill name"),
        ({"name": "code-review", "version": -1}, "positive integer"),
        ({"name": "code-review", "alias": "not a valid alias!"}, "alias"),
        ({"name": "code-review", "organization": "Bad_Org"}, "[Oo]rganization"),
    ],
)
def test_parsed_skill_uri_rejects_invalid_fields_on_construction(kwargs, match):
    with pytest.raises(MlflowException, match=match):
        ParsedSkillUri(**kwargs)
