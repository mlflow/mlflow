# Skill pull through the SDK and the CLI, against a running registry and real source servers.
#
# test_pull.py covers the pull engine with hand-built versions, and
# tests/genai/test_skills_pull.py pulls uploaded content and one Git source through a server.
# These tests register Git, OCI, and ZIP sources in a live registry and pull them back through
# both public entry points, so the source, subpath, and digest that are fetched are the ones
# the server stored and returned.

import base64
import json
import os
import re
import uuid

import pytest
from click.testing import CliRunner

import mlflow
from mlflow.cli import cli
from mlflow.entities.skill_source import GitSource, OCISource, ZipSource
from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content.digest import compute_tree_digest
from mlflow.tracking import MlflowClient

from tests.tracking.integration_test_utils import _init_server

pytestmark = pytest.mark.usefixtures("isolated_credentials")

SUBPATH = "skills/demo"
REMOTE_KINDS = ["git", "oci", "zip"]
SURFACES = ["sdk", "cli"]
FAILURES = {
    "digest-mismatch": ("INVALID_STATE", "does not match its recorded digest"),
    "missing-source": ("RESOURCE_DOES_NOT_EXIST", "Failed to fetch skill content"),
    "rejected-credentials": ("UNAUTHENTICATED", "Failed to fetch skill content"),
}


@pytest.fixture(scope="module")
def tracking_server(tmp_path_factory):
    root = tmp_path_factory.mktemp("server")
    with _init_server(
        backend_uri=f"sqlite:///{root / 'mlflow.db'}",
        root_artifact_uri=(root / "artifacts").as_uri(),
    ) as url:
        yield url


@pytest.fixture
def client(tracking_server):
    mlflow.set_tracking_uri(tracking_server)
    try:
        yield MlflowClient()
    finally:
        mlflow.set_tracking_uri(None)


@pytest.fixture
def org():
    # The module-scoped server keeps every registration, so each test uses its own organization.
    return f"pull-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def broken_source(request, make_source, tmp_path, monkeypatch):
    def make(failure, kind):
        match (failure, kind):
            case ("digest-mismatch", _):
                return make_source(kind, SUBPATH)
            case ("missing-source", "git"):
                return GitSource(url=make_source("git").url, ref="no-such-ref")
            case ("missing-source", "oci"):
                return OCISource(image=make_source("oci").image.replace(":v1", ":v9"))
            case ("missing-source", "zip"):
                return ZipSource(url=make_source("zip").url.replace("skills.zip", "gone.zip"))
            case ("rejected-credentials", "git"):
                # The HTTP server answers 401 under /private/, and no credential helper is set.
                base_url, _ = request.getfixturevalue("zip_server")
                return GitSource(url=f"{base_url}/private/skills.git")
            case ("rejected-credentials", "oci"):
                source = make_source("oci")
                host = source.image.removeprefix("oci://").split("/")[0]
                docker = tmp_path / "wrong-login"
                docker.mkdir()
                auth = base64.b64encode(b"user:wrong-secret").decode()
                (docker / "config.json").write_text(json.dumps({"auths": {host: {"auth": auth}}}))
                monkeypatch.setenv("DOCKER_CONFIG", str(docker))
                return source
            case ("rejected-credentials", "zip"):
                base_url, _ = request.getfixturevalue("zip_server")
                return ZipSource(
                    url=f"{base_url}/private/skills.zip?X-Amz-Signature=presigned-secret"
                )
            case _:
                raise ValueError(f"Unknown failure: {failure!r} for {kind!r}")

    return make


def _listing(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def _pull(surface, uri, destination):
    if surface == "sdk":
        assert mlflow.genai.pull(uri, destination=destination) == str(destination)
        return
    result = CliRunner().invoke(cli, ["skills", "pull", uri, "-d", str(destination)])
    assert result.exit_code == 0, result.output
    assert f"Pulled {uri} to {destination}" in result.output


def _failed_pull(surface, uri, destination, *, error_code, match):
    # Returns everything the caller is shown: the exception message, or the command's output.
    if surface == "sdk":
        with pytest.raises(MlflowException, match=match) as exc:
            mlflow.genai.pull(uri, destination=destination)
        assert exc.value.error_code == error_code
        return exc.value.message
    result = CliRunner().invoke(cli, ["skills", "pull", uri, "-d", str(destination)])
    assert result.exit_code == 1, result.output
    # A reported error, not a crash: the command exits instead of leaking an exception.
    assert isinstance(result.exception, SystemExit)
    assert re.search(match, result.output)
    return result.output


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("kind", REMOTE_KINDS)
def test_pull_registered_source_by_version_alias_and_latest(
    kind, surface, client, org, make_source, skill_tree, tmp_path
):
    expected = skill_tree / SUBPATH
    source = make_source(kind, SUBPATH)
    version = mlflow.genai.register_skill(source=source, organization=org)
    assert (version.name, version.version, version.digest) == (
        "demo",
        1,
        compute_tree_digest(expected),
    )
    assert type(version.source) is type(source)
    assert version.source.subpath == SUBPATH
    # A higher draft whose digest cannot match: an alias or latest lookup that resolved to it
    # instead of version 1 would fail the pull.
    client.create_skill_version(
        name="demo", organization=org, source=source, digest="0" * 64, status="draft"
    )
    client.set_skill_alias(name="demo", alias="production", version=1, organization=org)

    for index, suffix in enumerate(["/1", "@production", ""]):
        destination = tmp_path / f"out-{index}"
        _pull(surface, f"skills:/@{org}/demo{suffix}", destination)
        assert _listing(destination) == _listing(expected)
        assert compute_tree_digest(destination) == version.digest

    draft = tmp_path / "draft"
    _failed_pull(
        surface,
        f"skills:/@{org}/demo/2",
        draft,
        error_code="INVALID_STATE",
        match="does not match its recorded digest",
    )
    assert not draft.exists()


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("kind", REMOTE_KINDS)
@pytest.mark.parametrize("failure", list(FAILURES))
@pytest.mark.parametrize("initially_empty_dir", [False, True])
def test_failed_pull_leaves_destination_as_found(
    initially_empty_dir, failure, kind, surface, client, org, broken_source, skill_tree, tmp_path
):
    digest = "0" * 64 if failure == "digest-mismatch" else compute_tree_digest(skill_tree / SUBPATH)
    client.create_skill_version(
        name="demo", organization=org, source=broken_source(failure, kind), digest=digest
    )
    destination = tmp_path / "out"
    if initially_empty_dir:
        destination.mkdir()
    error_code, match = FAILURES[failure]

    shown = _failed_pull(
        surface, f"skills:/@{org}/demo/1", destination, error_code=error_code, match=match
    )

    assert "secret" not in shown
    if initially_empty_dir:
        assert list(destination.iterdir()) == []
    else:
        assert not os.path.lexists(destination)
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".mlflow-skill-pull-")]
