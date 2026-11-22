import json
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

from mlflow.assistant.skill_installer import BundledSkill
from mlflow.cli.skills import commands
from mlflow.entities import (
    GitSource,
    OCISource,
    Skill,
    SkillSourceType,
    SkillStatus,
    SkillVersion,
    ZipSource,
)
from mlflow.exceptions import MlflowException
from mlflow.store.entities.paged_list import PagedList


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_bundled_skills():
    with mock.patch("mlflow.cli.skills.list_bundled_skills", return_value=[]) as m:
        yield m


@pytest.fixture
def mock_search_skills():
    with mock.patch("mlflow.cli.skills.search_skills") as m:
        yield m


@pytest.fixture
def mock_register_skill():
    with mock.patch("mlflow.cli.skills.register_skill") as m:
        m.return_value = SkillVersion(
            name="code-review", version=1, source_type=SkillSourceType.GIT
        )
        yield m


@pytest.fixture
def mock_import_skills():
    with mock.patch("mlflow.cli.skills.import_skills") as m:
        m.return_value = [
            SkillVersion(name="code-review", version=1, source_type=SkillSourceType.GIT),
            SkillVersion(name="style-check", version=2, source_type=SkillSourceType.GIT),
        ]
        yield m


@pytest.fixture
def mock_mlflow_client():
    with mock.patch("mlflow.cli.skills.MlflowClient") as client_class:
        yield client_class.return_value


def test_list_command_renders_skills(runner, mock_bundled_skills):
    skills = [
        BundledSkill(name="alpha-skill", description="Does alpha things.", path=Path("/s/alpha")),
        BundledSkill(name="beta-skill", description="", path=Path("/s/beta")),
    ]
    mock_bundled_skills.return_value = skills

    result = runner.invoke(commands, ["list"])

    assert result.exit_code == 0
    assert "alpha-skill" in result.output
    assert "Does alpha things." in result.output
    assert "beta-skill" in result.output
    assert str(Path("/s/alpha")) in result.output
    assert str(Path("/s/beta")) in result.output


def test_list_command_handles_no_skills(runner, mock_bundled_skills):
    result = runner.invoke(commands, ["list"])

    assert result.exit_code == 0
    assert "No MLflow skills found" in result.output
    assert "git submodule update --init" in result.output


def test_view_command_renders_skill(runner, mock_bundled_skills):
    skills = [
        BundledSkill(name="alpha-skill", description="Does alpha things.", path=Path("/s/alpha")),
        BundledSkill(name="beta-skill", description="Does beta things.", path=Path("/s/beta")),
    ]
    mock_bundled_skills.return_value = skills

    result = runner.invoke(commands, ["view", "alpha-skill"])

    assert result.exit_code == 0
    assert "alpha-skill" in result.output
    assert "Does alpha things." in result.output
    assert str(Path("/s/alpha")) in result.output
    assert "beta-skill" not in result.output


def test_view_command_skill_not_found(runner, mock_bundled_skills):
    result = runner.invoke(commands, ["view", "missing-skill"])

    assert result.exit_code != 0
    assert "Skill missing-skill not found." in result.output


def test_view_command_skill_without_description(runner, mock_bundled_skills):
    skills = [BundledSkill(name="alpha-skill", description="", path=Path("/s/alpha"))]
    mock_bundled_skills.return_value = skills

    result = runner.invoke(commands, ["view", "alpha-skill"])

    assert result.exit_code == 0
    assert "alpha-skill" in result.output
    assert str(Path("/s/alpha")) in result.output
    assert "\n  \n" not in result.output


def test_search_command_forwards_options_and_renders_table(runner, mock_search_skills):
    mock_search_skills.return_value = PagedList(
        [Skill(name="code-review", organization="acme", latest_version=2)], "next-page"
    )

    result = runner.invoke(
        commands,
        [
            "search",
            "--filter-string",
            "organization = 'acme'",
            "--max-results",
            "10",
            "--order-by",
            "name ASC, latest_version DESC",
            "--page-token",
            "previous-page",
        ],
    )

    assert result.exit_code == 0
    mock_search_skills.assert_called_once_with(
        filter_string="organization = 'acme'",
        max_results=10,
        order_by=["name ASC", "latest_version DESC"],
        page_token="previous-page",
    )
    assert "code-review" in result.output
    assert "acme" in result.output
    assert "Next page token: next-page" in result.output


def test_search_command_renders_json(runner, mock_search_skills):
    mock_search_skills.return_value = PagedList(
        [Skill(name="code-review", organization="acme", latest_version=2)], None
    )

    result = runner.invoke(commands, ["search", "--output", "json"])

    assert result.exit_code == 0
    mock_search_skills.assert_called_once_with(
        filter_string=None,
        max_results=100,
        order_by=None,
        page_token=None,
    )
    assert json.loads(result.output)["skills"][0]["name"] == "code-review"
    assert json.loads(result.output)["next_page_token"] is None


def test_search_command_converts_mlflow_errors_to_click_errors(runner, mock_search_skills):
    mock_search_skills.side_effect = MlflowException("search failed")

    result = runner.invoke(commands, ["search"])

    assert result.exit_code != 0
    assert "Error: search failed" in result.output


def test_register_command_registers_local_directory(runner, tmp_path, mock_register_skill):
    skill_path = tmp_path / "code-review"
    skill_path.mkdir()
    mock_register_skill.return_value = SkillVersion(
        name="code-review", version=1, status=SkillStatus.DRAFT
    )

    result = runner.invoke(
        commands,
        [
            "register",
            "--source",
            str(skill_path),
            "--name",
            "code-review",
            "--organization",
            "acme",
            "--status",
            "draft",
        ],
    )

    assert result.exit_code == 0
    mock_register_skill.assert_called_once_with(
        source=str(skill_path), name="code-review", organization="acme", status="draft"
    )
    assert "code-review" in result.output
    assert "draft" in result.output


@pytest.mark.parametrize(
    ("source_command", "source_options", "expected_source"),
    [
        (
            "git",
            [
                "--url",
                "https://github.com/acme/skills.git",
                "--ref",
                "main",
                "--subpath",
                "reviewer",
            ],
            GitSource(url="https://github.com/acme/skills.git", ref="main", subpath="reviewer"),
        ),
        (
            "oci",
            ["--image", "ghcr.io/acme/skills:latest", "--subpath", "reviewer"],
            OCISource(image="ghcr.io/acme/skills:latest", subpath="reviewer"),
        ),
        (
            "zip",
            ["--url", "https://example.com/skills.zip", "--subpath", "reviewer"],
            ZipSource(url="https://example.com/skills.zip", subpath="reviewer"),
        ),
    ],
)
def test_register_command_constructs_typed_source(
    runner, mock_register_skill, source_command, source_options, expected_source
):
    result = runner.invoke(
        commands,
        ["register", source_command, *source_options, "--name", "reviewer"],
    )

    assert result.exit_code == 0
    mock_register_skill.assert_called_once_with(
        source=expected_source, name="reviewer", organization="", status="active"
    )


def test_register_git_command_reports_fetch_errors(runner, mock_register_skill):
    mock_register_skill.side_effect = MlflowException(
        "Git is required to fetch Git skill sources. Install Git and ensure it is available "
        "on PATH."
    )

    result = runner.invoke(
        commands,
        ["register", "git", "--url", "https://github.com/acme/skills.git"],
    )

    assert result.exit_code != 0
    assert "Git is required to fetch Git skill sources" in result.output
    assert "Traceback" not in result.output
    mock_register_skill.assert_called_once_with(
        source=GitSource(url="https://github.com/acme/skills.git"),
        name=None,
        organization="",
        status="active",
    )


@pytest.mark.parametrize(
    ("option", "value"),
    [("--name", "requested"), ("--status", "active"), ("--output", "table")],
)
def test_register_command_rejects_options_before_typed_source_subcommand(
    runner, mock_register_skill, option, value
):
    result = runner.invoke(
        commands,
        [
            "register",
            option,
            value,
            "git",
            "--url",
            "https://github.com/acme/skills.git",
        ],
    )

    assert result.exit_code != 0
    assert "must follow the source subcommand" in result.output
    mock_register_skill.assert_not_called()


def test_import_command_forwards_selection_and_source_options(runner, mock_import_skills):
    result = runner.invoke(
        commands,
        [
            "import",
            "--source",
            "https://github.com/acme/skills.git",
            "--ref",
            "main",
            "--subpath",
            "skills",
            "--skill-name",
            "code-review",
            "--skill-name",
            "style-check",
            "--organization",
            "acme",
            "--status",
            "draft",
        ],
    )

    assert result.exit_code == 0
    mock_import_skills.assert_called_once_with(
        source=GitSource(url="https://github.com/acme/skills.git", ref="main", subpath="skills"),
        organization="acme",
        skill_names=["code-review", "style-check"],
        status="draft",
    )
    assert "code-review" in result.output
    assert "style-check" in result.output


def test_import_command_documents_status_scope(runner):
    result = runner.invoke(commands, ["import", "--help"])

    assert result.exit_code == 0
    assert "Status for new versions; exact existing matches keep their current status." in " ".join(
        result.output.split()
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("skills:/code-review", ("", "code-review", None, None)),
        ("skills:/@acme/code-review", ("acme", "code-review", None, None)),
        ("skills:/code-review/2", ("", "code-review", 2, None)),
        ("skills:/@acme/code-review/2", ("acme", "code-review", 2, None)),
        ("skills:/code-review@stable", ("", "code-review", None, "stable")),
        ("skills:/@acme/code-review@stable", ("acme", "code-review", None, "stable")),
    ],
)
def test_skill_uri_parser(runner, value, expected):
    from mlflow.cli.skills import _parse_skill_target

    parsed = _parse_skill_target(value)
    assert (parsed.organization, parsed.name, parsed.version, parsed.alias) == expected


def test_get_command_accepts_positional_skill_uri(runner, mock_mlflow_client):
    mock_mlflow_client.get_skill.return_value = Skill(name="code-review", organization="acme")

    result = runner.invoke(commands, ["get", "skills:/@acme/code-review", "--output", "json"])

    assert result.exit_code == 0
    mock_mlflow_client.get_skill.assert_called_once_with(name="code-review", organization="acme")
    assert json.loads(result.output)["skill"]["name"] == "code-review"


def test_get_command_accepts_skill_uri_option(runner, mock_mlflow_client):
    mock_mlflow_client.get_skill.return_value = Skill(name="code-review")

    result = runner.invoke(commands, ["get", "--skill-uri", "skills:/code-review"])

    assert result.exit_code == 0
    mock_mlflow_client.get_skill.assert_called_once_with(name="code-review", organization="")


def test_get_command_rejects_version_target(runner, mock_mlflow_client):
    result = runner.invoke(commands, ["get", "skills:/code-review/1"])

    assert result.exit_code != 0
    assert "requires a parent Skill URI" in result.output
    mock_mlflow_client.get_skill.assert_not_called()


def test_update_command_parses_icons(runner, mock_mlflow_client):
    mock_mlflow_client.update_skill.return_value = Skill(
        name="code-review", description="Review code"
    )
    icons = '[{"src":"https://example.com/icon.svg","sizes":["32x32"]}]'

    result = runner.invoke(
        commands,
        [
            "update",
            "--skill-uri",
            "skills:/code-review",
            "--description",
            "Review code",
            "--icons",
            icons,
        ],
    )

    assert result.exit_code == 0
    mock_mlflow_client.update_skill.assert_called_once_with(
        name="code-review",
        organization="",
        description="Review code",
        icons=[{"src": "https://example.com/icon.svg", "sizes": ["32x32"]}],
    )


def test_update_command_requires_update_fields(runner, mock_mlflow_client):
    result = runner.invoke(commands, ["update", "skills:/code-review"])

    assert result.exit_code != 0
    assert "Provide --description and/or --icons" in result.output
    mock_mlflow_client.update_skill.assert_not_called()


def test_get_version_command_uses_exact_version(runner, mock_mlflow_client):
    mock_mlflow_client.get_skill_version.return_value = SkillVersion(name="code-review", version=2)

    result = runner.invoke(commands, ["get-version", "skills:/code-review/2", "--output", "json"])

    assert result.exit_code == 0
    mock_mlflow_client.get_skill_version.assert_called_once_with(
        name="code-review", version=2, organization=""
    )
    assert json.loads(result.output)["skill_versions"][0]["version"] == 2


def test_get_version_command_uses_alias(runner, mock_mlflow_client):
    mock_mlflow_client.get_skill_version_by_alias.return_value = SkillVersion(
        name="code-review",
        version=2,
        source=GitSource(url="https://github.com/acme/skills.git", ref="main", subpath="reviewer"),
    )

    result = runner.invoke(commands, ["get-version", "skills:/code-review@stable"])

    assert result.exit_code == 0
    mock_mlflow_client.get_skill_version_by_alias.assert_called_once_with(
        name="code-review", alias="stable", organization=""
    )
    assert "https://github.com/acme/skills.git" in result.output
    assert "main" in result.output
    assert "reviewer" in result.output


@pytest.mark.parametrize(
    "arguments",
    [
        ["skills:/@acme/code-review@latest"],
        ["--skill-uri", "skills:/@acme/code-review@latest"],
    ],
)
def test_get_version_command_uses_latest_alias(runner, mock_mlflow_client, arguments):
    mock_mlflow_client.get_skill_version_by_alias.return_value = SkillVersion(
        name="code-review", organization="acme", version=2
    )

    result = runner.invoke(commands, ["get-version", *arguments])

    assert result.exit_code == 0, result.output
    mock_mlflow_client.get_skill_version_by_alias.assert_called_once_with(
        name="code-review", alias="latest", organization="acme"
    )


def test_get_version_command_uses_latest_for_parent_uri(runner, mock_mlflow_client):
    mock_mlflow_client.get_latest_skill_version.return_value = SkillVersion(
        name="code-review", version=2
    )

    result = runner.invoke(commands, ["get-version", "skills:/code-review"])

    assert result.exit_code == 0
    mock_mlflow_client.get_latest_skill_version.assert_called_once_with(
        name="code-review", organization=""
    )


def test_set_alias_command_requires_parent_and_version_and_calls_client(runner, mock_mlflow_client):
    result = runner.invoke(
        commands,
        ["set-alias", "skills:/@acme/code-review", "--alias", "stable", "--version", "2"],
    )

    assert result.exit_code == 0
    mock_mlflow_client.set_skill_alias.assert_called_once_with(
        name="code-review", alias="stable", version=2, organization="acme"
    )
    assert "Set alias 'stable'" in result.output


def test_set_alias_command_requires_version(runner, mock_mlflow_client):
    result = runner.invoke(
        commands,
        ["set-alias", "skills:/code-review", "--alias", "stable"],
    )

    assert result.exit_code != 0
    assert "Missing option '--version'" in result.output
    mock_mlflow_client.set_skill_alias.assert_not_called()


def test_delete_alias_command_calls_client(runner, mock_mlflow_client):
    result = runner.invoke(
        commands,
        ["delete-alias", "--skill-uri", "skills:/@acme/code-review", "--alias", "stable"],
    )

    assert result.exit_code == 0
    mock_mlflow_client.delete_skill_alias.assert_called_once_with(
        name="code-review", alias="stable", organization="acme"
    )


def test_set_tag_command_supports_json_output(runner, mock_mlflow_client):
    result = runner.invoke(
        commands,
        [
            "set-tag",
            "skills:/code-review",
            "--key",
            "team",
            "--value",
            "platform",
            "--output",
            "json",
        ],
    )

    assert result.exit_code == 0
    mock_mlflow_client.set_skill_tag.assert_called_once_with(
        name="code-review", key="team", value="platform", organization=""
    )
    assert json.loads(result.output) == {"success": True}


def test_delete_tag_command_calls_client(runner, mock_mlflow_client):
    result = runner.invoke(
        commands,
        ["delete-tag", "skills:/code-review", "--key", "team"],
    )

    assert result.exit_code == 0
    mock_mlflow_client.delete_skill_tag.assert_called_once_with(
        name="code-review", key="team", organization=""
    )


def test_set_version_tag_command_calls_client(runner, mock_mlflow_client):
    result = runner.invoke(
        commands,
        [
            "set-version-tag",
            "skills:/code-review/3",
            "--key",
            "channel",
            "--value",
            "stable",
        ],
    )

    assert result.exit_code == 0
    mock_mlflow_client.set_skill_version_tag.assert_called_once_with(
        name="code-review", version=3, key="channel", value="stable", organization=""
    )


def test_delete_version_tag_command_rejects_alias_uri(runner, mock_mlflow_client):
    result = runner.invoke(
        commands,
        ["delete-version-tag", "skills:/code-review@stable", "--key", "channel"],
    )

    assert result.exit_code != 0
    assert "exact integer version" in result.output
    mock_mlflow_client.delete_skill_version_tag.assert_not_called()


def test_search_versions_command_forwards_options(runner, mock_mlflow_client):
    mock_mlflow_client.search_skill_versions.return_value = PagedList(
        [SkillVersion(name="code-review", version=2)], "next-page"
    )

    result = runner.invoke(
        commands,
        [
            "search-versions",
            "--skill-uri",
            "skills:/@acme/code-review",
            "--filter-string",
            "status = 'active'",
            "--max-results",
            "5",
            "--order-by",
            "version DESC",
            "--page-token",
            "previous-page",
        ],
    )

    assert result.exit_code == 0
    mock_mlflow_client.search_skill_versions.assert_called_once_with(
        name="code-review",
        organization="acme",
        filter_string="status = 'active'",
        max_results=5,
        order_by=["version DESC"],
        page_token="previous-page",
    )
    assert "Next page token: next-page" in result.output


def test_update_version_requires_exact_version(runner, mock_mlflow_client):
    result = runner.invoke(
        commands,
        ["update-version", "skills:/code-review@stable", "--status", "deprecated"],
    )

    assert result.exit_code != 0
    assert "exact integer version" in result.output
    mock_mlflow_client.update_skill_version.assert_not_called()


def test_update_version_forwards_status(runner, mock_mlflow_client):
    mock_mlflow_client.update_skill_version.return_value = SkillVersion(
        name="code-review", version=2
    )

    result = runner.invoke(
        commands,
        ["update-version", "skills:/code-review/2", "--status", "deprecated"],
    )

    assert result.exit_code == 0
    mock_mlflow_client.update_skill_version.assert_called_once_with(
        name="code-review", version=2, organization="", status="deprecated"
    )


def test_delete_version_soft_deletes_exact_version(runner, mock_mlflow_client):
    result = runner.invoke(commands, ["delete-version", "skills:/code-review/2"])

    assert result.exit_code == 0
    mock_mlflow_client.delete_skill_version.assert_called_once_with(
        name="code-review", version=2, organization=""
    )
    assert "Soft-deleted code-review version 2" in result.output
