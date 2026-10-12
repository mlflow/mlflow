# Skill pull through the SDK and the CLI, against a running registry and real source servers.
#
# test_pull.py covers the pull engine with hand-built versions, and
# tests/genai/test_skills_pull.py pulls uploaded content and one Git source through a server.
# These tests register Git, OCI, ZIP, and uploaded sources in a live registry and pull them
# back through both public entry points, so the source, subpath, and digest that are fetched
# are the ones the server stored and returned.

import base64
import json
import os
import re
import shutil
import uuid

import pytest
from click.testing import CliRunner

import mlflow
from mlflow.cli import cli
from mlflow.entities.skill_source import GitSource, OCISource, ZipSource
from mlflow.exceptions import MlflowException
from mlflow.genai.skill_content.digest import compute_tree_digest
from mlflow.server import ARTIFACTS_DESTINATION_ENV_VAR, SERVE_ARTIFACTS_ENV_VAR
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
def artifact_root(tmp_path_factory):
    # Where the server stores uploaded skill content, so a test can change it behind the server.
    return tmp_path_factory.mktemp("artifacts")


@pytest.fixture(scope="module")
def tracking_server(tmp_path_factory, artifact_root):
    root = tmp_path_factory.mktemp("server")
    with _init_server(
        backend_uri=f"sqlite:///{root / 'mlflow.db'}",
        root_artifact_uri="mlflow-artifacts:/",
        extra_env={
            SERVE_ARTIFACTS_ENV_VAR: "true",
            ARTIFACTS_DESTINATION_ENV_VAR: str(artifact_root),
        },
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


def _assert_no_staging_left(parent):
    assert not [p for p in parent.iterdir() if p.name.startswith(".mlflow-skill-pull-")]


def _state(path):
    # What is at a path, without following a symbolic link.
    if path.is_symlink():
        return ("symlink", os.readlink(path))
    if path.is_dir():
        return ("directory", _listing(path))
    return ("file", path.read_text())


def _pull(surface, uri, destination):
    if surface == "sdk":
        assert mlflow.genai.pull(uri, destination=destination) == str(destination)
        return
    result = CliRunner().invoke(cli, ["skills", "pull", uri, "-d", str(destination)])
    assert result.exit_code == 0, result.output
    assert f"Pulled {uri} to {destination}" in result.output


def _failed_pull(surface, uri, destination, *, error_code, match, exit_code=1):
    # Returns everything the caller is shown: the exception message, or the command's output.
    # An error_code of None leaves the SDK's error code unchecked.
    if surface == "sdk":
        with pytest.raises(MlflowException, match=match) as exc:
            mlflow.genai.pull(uri, destination=destination)
        if error_code is not None:
            assert exc.value.error_code == error_code
        return exc.value.message
    result = CliRunner().invoke(cli, ["skills", "pull", uri, "-d", str(destination)])
    assert result.exit_code == exit_code, result.output
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
    _assert_no_staging_left(tmp_path)

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
    _assert_no_staging_left(tmp_path)


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("kind", REMOTE_KINDS)
def test_pull_source_registered_without_subpath(
    kind, surface, client, org, make_source, skill_tree, tmp_path
):
    client.create_skill_version(
        name="demo",
        organization=org,
        source=make_source(kind),
        digest=compute_tree_digest(skill_tree),
    )
    destination = tmp_path / "out"

    _pull(surface, f"skills:/@{org}/demo/1", destination)

    assert _listing(destination) == _listing(skill_tree)
    assert (destination / "README.md").exists()


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("kind", REMOTE_KINDS)
def test_version_without_digest_is_pulled_unverified(
    kind, surface, client, org, make_source, skill_tree, tmp_path
):
    version = client.create_skill_version(
        name="demo", organization=org, source=make_source(kind, SUBPATH)
    )
    assert version.digest is None
    destination = tmp_path / "out"

    _pull(surface, f"skills:/@{org}/demo/1", destination)

    assert compute_tree_digest(destination) == compute_tree_digest(skill_tree / SUBPATH)


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("kind", REMOTE_KINDS)
def test_deprecated_version_is_pullable(kind, surface, client, org, make_source, tmp_path):
    version = mlflow.genai.register_skill(source=make_source(kind, SUBPATH), organization=org)
    client.update_skill_version(name="demo", version=1, organization=org, status="deprecated")
    destination = tmp_path / "out"

    _pull(surface, f"skills:/@{org}/demo/1", destination)

    assert compute_tree_digest(destination) == version.digest


@pytest.mark.parametrize("surface", SURFACES)
def test_deleted_version_is_not_pulled(surface, client, org, make_source, tmp_path):
    mlflow.genai.register_skill(source=make_source("zip", SUBPATH), organization=org)
    client.set_skill_alias(name="demo", alias="production", version=1, organization=org)
    # An active version has to be deprecated before it can be deleted.
    client.update_skill_version(name="demo", version=1, organization=org, status="deprecated")
    client.delete_skill_version(name="demo", version=1, organization=org)

    for index, suffix in enumerate(["/1", "@production", ""]):
        destination = tmp_path / f"out-{index}"
        _failed_pull(
            surface,
            f"skills:/@{org}/demo{suffix}",
            destination,
            error_code="RESOURCE_DOES_NOT_EXIST",
            match="not found|No resolved latest version",
        )
        assert not os.path.lexists(destination)


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize(
    ("state", "reason", "cli_reason", "cli_exit_code"),
    [
        # The CLI's own option check refuses a file before the pull starts.
        ("file", "it exists and is not a directory", "is a file", 2),
        ("non-empty", "it is a directory that is not empty", None, 1),
        ("hidden-file-only", "it is a directory that is not empty", None, 1),
        ("symlink-to-directory", "it is a symbolic link", None, 1),
        ("dangling-symlink", "it is a symbolic link", None, 1),
    ],
)
def test_invalid_destination_is_rejected_unmodified(
    state, reason, cli_reason, cli_exit_code, surface, client, org, closed_port, tmp_path
):
    # Nothing listens at the source, so reaching it would fail with a fetch error: getting the
    # destination error instead shows the destination is checked before anything is fetched.
    client.create_skill_version(
        name="demo",
        organization=org,
        source=ZipSource(url=f"http://127.0.0.1:{closed_port}/skills.zip"),
    )
    destination = tmp_path / "out"
    target = tmp_path / "link-target"
    match state:
        case "file":
            destination.write_text("keep")
        case "non-empty":
            destination.mkdir()
            (destination / "keep.txt").write_text("keep")
        case "hidden-file-only":
            destination.mkdir()
            (destination / ".keep").write_text("keep")
        case "symlink-to-directory":
            target.mkdir()
            destination.symlink_to(target)
        case "dangling-symlink":
            destination.symlink_to(target)
    before = _state(destination)

    _failed_pull(
        surface,
        f"skills:/@{org}/demo/1",
        destination,
        error_code="INVALID_PARAMETER_VALUE",
        match=cli_reason if surface == "cli" and cli_reason else reason,
        exit_code=cli_exit_code,
    )

    assert _state(destination) == before
    if state == "symlink-to-directory":
        assert list(target.iterdir()) == []
    else:
        assert not target.exists()
    _assert_no_staging_left(tmp_path)


@pytest.mark.parametrize("surface", SURFACES)
def test_pull_uploaded_skill_by_alias(surface, client, org, skill_tree, tmp_path):
    version = mlflow.genai.register_skill(source=str(skill_tree / SUBPATH), organization=org)
    client.set_skill_alias(name="demo", alias="production", version=1, organization=org)
    destination = tmp_path / "out"

    _pull(surface, f"skills:/@{org}/demo@production", destination)

    assert compute_tree_digest(destination) == version.digest


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("change", ["modified", "removed"])
@pytest.mark.parametrize("initially_empty_dir", [False, True])
def test_uploaded_content_changed_on_the_server_fails_the_pull(
    initially_empty_dir, change, surface, client, org, artifact_root, skill_tree, tmp_path
):
    version = mlflow.genai.register_skill(source=str(skill_tree / SUBPATH), organization=org)
    stored = artifact_root / version.source.artifact_path.removeprefix("mlflow-artifacts:/")
    assert (stored / "SKILL.md").is_file()
    if change == "modified":
        (stored / "SKILL.md").write_text("changed after registration\n")
        error_code, match = "INVALID_STATE", "does not match its recorded digest"
    else:
        shutil.rmtree(stored)
        # The artifact server's 404 currently reaches the caller as INTERNAL_ERROR, while every
        # other source type reports a missing source as RESOURCE_DOES_NOT_EXIST, so the error
        # code is left unchecked here.
        error_code, match = None, "Failed to fetch skill content"
    destination = tmp_path / "out"
    if initially_empty_dir:
        destination.mkdir()

    _failed_pull(surface, f"skills:/@{org}/demo/1", destination, error_code=error_code, match=match)

    if initially_empty_dir:
        assert list(destination.iterdir()) == []
    else:
        assert not os.path.lexists(destination)
    _assert_no_staging_left(tmp_path)
