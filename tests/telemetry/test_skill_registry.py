import json
from unittest import mock

import pytest

from mlflow.entities.skill import SkillStatus
from mlflow.entities.skill_source import GitSource, OCISource, SkillSourceType, ZipSource
from mlflow.environment_variables import MLFLOW_DISABLE_TELEMETRY
from mlflow.exceptions import MlflowException
from mlflow.genai import skills
from mlflow.genai.skill_content.fetchers import FetchedContent
from mlflow.genai.skill_content.sources import resolve_source_type
from mlflow.store.tracking.sqlalchemy_store import SqlAlchemyStore
from mlflow.store.tracking.sqlalchemy_workspace_store import WorkspaceAwareSqlAlchemyStore
from mlflow.telemetry.events import (
    SkillRegistryBulkRegisterSkillsEvent,
    SkillRegistryCreateSkillVersionEvent,
    SkillRegistryImportSkillsEvent,
    SkillRegistryPullEvent,
    SkillRegistryRegisterSkillEvent,
)
from mlflow.tracking.client import MlflowClient
from mlflow.utils.workspace_context import WorkspaceContext

from tests.telemetry.helper_functions import validate_telemetry_record


@pytest.mark.parametrize(
    ("source_type", "expected"),
    [(None, None), *[(t, t.value) for t in SkillSourceType], ("private-type", "other")],
)
@pytest.mark.parametrize("status", ["active", SkillStatus.DRAFT, "private-status"])
def test_create_version_parameters(source_type, expected, status):
    params = SkillRegistryCreateSkillVersionEvent.parse({
        "source_type": source_type,
        "status": status,
        "name": "private-skill",
        "organization": "private-org",
        "source": "https://private.example/skills.git",
        "ref": "private-ref",
        "digest": "a" * 64,
    })
    assert params == {
        "source_type": expected,
        "status": status if status in ("active", "draft") else "other",
    }
    assert SkillRegistryCreateSkillVersionEvent.parse({}) == {
        "source_type": None,
        "status": "active",
    }


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("./private-skill", "local"),
        ("https://private.example/skills.git", "git"),
        (GitSource("https://private.example/skills", ref="private-ref"), "git"),
        (OCISource("private.example/team/skill:v1"), "oci"),
        ("oci://private.example/team/skill:v1", "oci"),
        (ZipSource("https://private.example/skill.zip"), "zip"),
        ("https://private.example/skill.zip", "zip"),
        ("mlflow-artifacts:/private-path", "other"),
        ("https://private.example/unknown", "other"),
        (None, "other"),
    ],
)
def test_register_parameters(source, expected):
    assert SkillRegistryRegisterSkillEvent.parse({"source": source}) == {"source_type": expected}


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("skills:/private-skill/3", "version"),
        ("skills:/@private-org/private-skill@private-alias", "alias"),
        ("skills:/private-skill", "latest"),
        ("skills:/private-skill@latest", "latest"),
        ("private-invalid-uri", "other"),
        (None, "other"),
    ],
)
def test_pull_parameters(uri, expected):
    assert SkillRegistryPullEvent.parse({"uri": uri, "destination": "/private/path"}) == {
        "reference_type": expected,
    }


@pytest.mark.parametrize("definitions", [None, [], [{"name": "private-skill"}]])
def test_bulk_parameters(definitions):
    assert SkillRegistryBulkRegisterSkillsEvent.parse({"skill_definitions": definitions}) == {
        "num_skills": len(definitions) if definitions is not None else None,
    }


@pytest.mark.parametrize("names", [None, [], ["private-skill"]])
def test_import_parameters(names):
    assert SkillRegistryImportSkillsEvent.parse({"skill_names": names}) == {
        "filters_by_name": names is not None,
    }
    assert SkillRegistryImportSkillsEvent.parse_result([object(), object()]) == {"num_returned": 2}
    assert SkillRegistryImportSkillsEvent.parse_result([]) == {"num_returned": 0}
    assert SkillRegistryImportSkillsEvent.parse_result(None) == {}


@pytest.fixture
def telemetry(mock_telemetry_client):
    with mock.patch(
        "mlflow.telemetry.track.get_telemetry_client", return_value=mock_telemetry_client
    ):
        yield mock_telemetry_client


@pytest.fixture(params=[False, True], ids=["default-store", "workspace-store"])
def skill_store(request, monkeypatch, tmp_path, db_uri):
    monkeypatch.setenv("MLFLOW_ENABLE_WORKSPACES", str(request.param).lower())
    store_cls = WorkspaceAwareSqlAlchemyStore if request.param else SqlAlchemyStore
    with WorkspaceContext("default"):
        store = store_cls(db_uri, tmp_path.as_uri())
        with mock.patch(
            "mlflow.genai.skills.MlflowClient", return_value=MlflowClient(tracking_uri=db_uri)
        ):
            yield store


@pytest.fixture
def skill_directory(tmp_path):
    directory = tmp_path / "skill"
    directory.mkdir()
    (directory / "SKILL.md").write_text(
        "---\nname: review\ndescription: Reviews code\n---\nPrivate skill content\n"
    )
    return directory


def _skill_events(telemetry, mock_requests):
    telemetry.flush()
    return [
        (data["event_name"], data["status"], json.loads(data["params"]))
        for record in mock_requests
        if (data := record["data"])["event_name"].startswith("skill_registry_")
    ]


def test_store_create_version(telemetry, mock_requests, skill_store):
    version = skill_store.create_skill_version(
        name="review",
        organization="private-org",
        source_type="git",
        source="https://private.example/skills.git",
        ref="private-ref",
        status="draft",
    )
    assert version.name == "review"
    validate_telemetry_record(
        telemetry,
        mock_requests,
        SkillRegistryCreateSkillVersionEvent.name,
        {"source_type": "git", "status": "draft"},
    )
    with pytest.raises(MlflowException, match="status"):
        skill_store.create_skill_version(name="invalid", status="private-status")
    validate_telemetry_record(
        telemetry,
        mock_requests,
        SkillRegistryCreateSkillVersionEvent.name,
        {"source_type": None, "status": "other"},
        status="failure",
    )


def test_bulk_registration_counts_requests_including_reuse(telemetry, mock_requests, skill_store):
    definitions = [
        {
            "name": "review",
            "source_type": "git",
            "source": "https://private.example/skills.git",
            "ref": "main",
            "digest": "a" * 64,
        }
    ]
    first = skill_store.bulk_register_skills(definitions, organization="private-org")
    second = skill_store.bulk_register_skills(definitions, organization="private-org")
    assert first == second
    assert _skill_events(telemetry, mock_requests) == [
        (SkillRegistryBulkRegisterSkillsEvent.name, "success", {"num_skills": 1}),
        (SkillRegistryBulkRegisterSkillsEvent.name, "success", {"num_skills": 1}),
    ]


def test_bulk_failure(telemetry, mock_requests, skill_store):
    with pytest.raises(MlflowException, match="nonempty"):
        skill_store.bulk_register_skills([])
    validate_telemetry_record(
        telemetry,
        mock_requests,
        SkillRegistryBulkRegisterSkillsEvent.name,
        {"num_skills": 0},
        status="failure",
    )


def test_register_workflow(telemetry, mock_requests, skill_store, skill_directory):
    source = GitSource("https://private.example/skills.git", ref="main")
    with mock.patch("mlflow.genai.skills.fetch_source") as fetch:
        fetch.return_value.__enter__.return_value = FetchedContent(
            skill_directory, resolve_source_type(source)
        )
        version = skills.register_skill(source=source, organization="private-org")
        fetch.assert_called_once_with(source)
    assert (
        skill_store.get_skill_version("review", version.version, organization="private-org")
        == version
    )
    assert _skill_events(telemetry, mock_requests) == [
        (
            SkillRegistryCreateSkillVersionEvent.name,
            "success",
            {"source_type": "git", "status": "active"},
        ),
        (SkillRegistryRegisterSkillEvent.name, "success", {"source_type": "git"}),
    ]


def test_register_fetch_failure(telemetry, mock_requests):
    with mock.patch("mlflow.genai.skills.fetch_source", side_effect=RuntimeError("private error")):
        with pytest.raises(RuntimeError, match="private error"):
            skills.register_skill(source="./private-skill")
    assert _skill_events(telemetry, mock_requests) == [
        (SkillRegistryRegisterSkillEvent.name, "failure", {"source_type": "local"}),
    ]


def test_import_workflow(telemetry, mock_requests, skill_store, skill_directory):
    source = GitSource("https://private.example/skills.git", ref="main")
    with mock.patch("mlflow.genai.skills.fetch_source") as fetch:
        fetch.return_value.__enter__.return_value = FetchedContent(
            skill_directory.parent, resolve_source_type(source)
        )
        first = skills.import_skills(source=source, skill_names=["review"])
        second = skills.import_skills(source=source, skill_names=["review"])
        assert (
            fetch.call_args_list
            == [
                mock.call(
                    source,
                    max_bytes=skills.get_max_decompressed_size() * skills._MAX_BULK_REGISTER_SKILLS,
                )
            ]
            * 2
        )
    assert first == second
    assert (
        _skill_events(telemetry, mock_requests)
        == [
            (SkillRegistryBulkRegisterSkillsEvent.name, "success", {"num_skills": 1}),
            (
                SkillRegistryImportSkillsEvent.name,
                "success",
                {"filters_by_name": True, "num_returned": 1},
            ),
        ]
        * 2
    )


def test_import_failure_has_no_returned_count(telemetry, mock_requests):
    with pytest.raises(MlflowException, match="nonempty"):
        skills.import_skills(source="https://private.example/skills.git", skill_names=[])
    validate_telemetry_record(
        telemetry,
        mock_requests,
        SkillRegistryImportSkillsEvent.name,
        {"filters_by_name": True},
        status="failure",
    )


@pytest.mark.parametrize("failure", [False, True])
def test_pull_workflow(telemetry, mock_requests, tmp_path, failure):
    target = tmp_path / "private-destination"
    with (
        mock.patch("mlflow.genai.skills.resolve_skill_version") as resolve,
        mock.patch("mlflow.genai.skills.pull_skill_version", return_value=target) as download,
    ):
        if failure:
            download.side_effect = RuntimeError("digest mismatch")
            with pytest.raises(RuntimeError, match="digest mismatch"):
                skills.pull("skills:/review@private-alias", destination=target)
        else:
            assert skills.pull("skills:/review@private-alias", destination=target) == str(target)
        resolve.assert_called_once_with("skills:/review@private-alias")
        download.assert_called_once_with(resolve.return_value, target)
    validate_telemetry_record(
        telemetry,
        mock_requests,
        SkillRegistryPullEvent.name,
        {"reference_type": "alias"},
        status="failure" if failure else "success",
    )


def test_telemetry_opt_out_preserves_operation(
    telemetry, mock_requests, skill_store, monkeypatch, bypass_env_check
):
    monkeypatch.setenv(MLFLOW_DISABLE_TELEMETRY.name, "true")
    assert skill_store.create_skill_version(name="review").version == 1
    assert _skill_events(telemetry, mock_requests) == []


def test_event_disabled_preserves_operation(telemetry, mock_requests, skill_store):
    telemetry.config.disable_events.add(SkillRegistryCreateSkillVersionEvent.name)
    assert skill_store.create_skill_version(name="review").version == 1
    assert _skill_events(telemetry, mock_requests) == []
