import pytest

from mlflow.exceptions import MlflowException
from mlflow.utils.skill_uris import (
    ParsedAgentPluginUri,
    ParsedSkillUri,
    format_agent_plugin_uri,
    format_skill_uri,
    parse_agent_plugin_uri,
    parse_skill_uri,
)

SKILL_URIS = [
    "skills:/code-review",
    "skills:/code-review/1",
    "skills:/code-review@production",
    "skills:/@acme/code-review",
    "skills:/@acme/code-review/1",
    "skills:/@acme/code-review@production",
    "skills:/@acme.com/123",
]

PLUGIN_URIS = [
    "agent-plugins:/pr-workflow",
    "agent-plugins:/pr-workflow/1.0.0-beta.11",
    "agent-plugins:/pr-workflow@production",
    "agent-plugins:/@acme/pr-workflow",
    "agent-plugins:/@acme/pr-workflow/1.0.0",
    "agent-plugins:/@acme/pr-workflow@production",
]


@pytest.mark.parametrize("uri", SKILL_URIS)
def test_skill_uri_roundtrip_from_string(uri):
    assert format_skill_uri(parse_skill_uri(uri)) == uri


@pytest.mark.parametrize("uri", PLUGIN_URIS)
def test_plugin_uri_roundtrip_from_string(uri):
    assert format_agent_plugin_uri(parse_agent_plugin_uri(uri)) == uri


@pytest.mark.parametrize(
    "parsed",
    [
        ParsedSkillUri(name="code-review"),
        ParsedSkillUri(name="code-review", organization="acme", version=3),
        ParsedSkillUri(name="code-review", alias="production"),
    ],
)
def test_skill_object_roundtrip(parsed):
    assert parse_skill_uri(format_skill_uri(parsed)) == parsed


@pytest.mark.parametrize(
    "parsed",
    [
        ParsedAgentPluginUri(name="pr-workflow"),
        ParsedAgentPluginUri(name="pr-workflow", organization="acme", version="1.0.0"),
        ParsedAgentPluginUri(name="pr-workflow", alias="production"),
    ],
)
def test_plugin_object_roundtrip(parsed):
    assert parse_agent_plugin_uri(format_agent_plugin_uri(parsed)) == parsed


def test_parse_skill_uri_fields():
    parsed = parse_skill_uri("skills:/@acme/code-review/2")
    assert parsed == ParsedSkillUri(name="code-review", organization="acme", version=2)


def test_plugin_version_is_normalized_on_parse():
    parsed = parse_agent_plugin_uri("agent-plugins:/pr-workflow/1.0")
    assert parsed.version == "1.0.0"


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


def test_parse_plugin_uri_rejects_non_semver_version():
    with pytest.raises(MlflowException, match="[Ss]em[Vv]er|version"):
        parse_agent_plugin_uri("agent-plugins:/pr-workflow/not-a-version")
