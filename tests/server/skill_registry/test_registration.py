import io
import re
import tarfile
import threading
from unittest import mock

import pytest

from mlflow.entities.skill import SkillStatus
from mlflow.entities.skill_source import GitSource, MlflowSource, OCISource, SkillSourceType
from mlflow.environment_variables import (
    MLFLOW_ENABLE_WORKSPACES,
    MLFLOW_SKILL_CONTENT_MAX_DECOMPRESSED_SIZE,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INTERNAL_ERROR, RESOURCE_ALREADY_EXISTS
from mlflow.server.skill_registry import SkillVersionRegistration, register_skill_version
from mlflow.server.skill_registry import registration as registration_module
from mlflow.utils.workspace_context import WorkspaceContext

from tests.server.skill_registry.conftest import (
    SKILL_FILES,
    skill_archive,
    stored_files,
    version_rows,
)

pytestmark = pytest.mark.notrackingurimock

_DIGEST = "ab" * 32
_UPLOAD = SkillVersionRegistration(name="reviewer", organization="acme", digest=_DIGEST)


def _special_member(name, member_type, linkname=""):
    member = tarfile.TarInfo(name)
    member.type = member_type
    member.linkname = linkname
    return member


# --- upload flow ------------------------------------------------------------------------------


def test_upload_stores_content_then_commits_a_complete_version(store, artifact_root):
    version = register_skill_version(_UPLOAD, content=skill_archive(), multipart=True)

    assert version.version == 1
    assert version.status == SkillStatus.ACTIVE
    assert version.source_type == SkillSourceType.MLFLOW
    assert version.digest == _DIGEST
    assert isinstance(version.source, MlflowSource)
    assert version.source.subpath is None
    match = re.fullmatch(
        r"mlflow-artifacts:/(skills/@acme/reviewer/[0-9a-f]{32})", version.source.artifact_path
    )
    assert match is not None
    stored = artifact_root / match.group(1)
    assert stored_files(stored) == sorted(SKILL_FILES)
    assert (stored / "SKILL.md").read_bytes() == SKILL_FILES["SKILL.md"]
    assert store.get_skill_version("reviewer", 1, organization="acme") == version


def test_upload_without_organization_omits_the_segment(store, artifact_root):
    version = register_skill_version(
        SkillVersionRegistration(name="reviewer"), content=skill_archive(), multipart=True
    )
    assert re.fullmatch(
        r"mlflow-artifacts:/skills/reviewer/[0-9a-f]{32}", version.source.artifact_path
    )


def test_upload_can_register_a_draft(store, artifact_root):
    draft = SkillVersionRegistration(name="reviewer", status="draft")
    version = register_skill_version(draft, content=skill_archive(), multipart=True)
    assert version.status == SkillStatus.DRAFT
    # A deliberately registered draft is a complete, retrievable version.
    assert store.get_skill_version("reviewer", 1).status == SkillStatus.DRAFT


def test_every_upload_gets_its_own_path_and_the_next_version(store, artifact_root):
    versions = [
        register_skill_version(_UPLOAD, content=skill_archive(), multipart=True) for _ in range(3)
    ]
    assert [v.version for v in versions] == [1, 2, 3]
    assert len({v.source.artifact_path for v in versions}) == 3


def test_upload_is_scoped_to_the_request_workspace(store, artifact_root, monkeypatch):
    monkeypatch.setenv(MLFLOW_ENABLE_WORKSPACES.name, "true")
    with WorkspaceContext("team-a"):
        with mock.patch.object(
            store, "create_skill_version", wraps=store.create_skill_version
        ) as create:
            register_skill_version(_UPLOAD, content=skill_archive(), multipart=True)
    create.assert_called_once()
    # The recorded source stays workspace-relative, like any mlflow-artifacts URI a client
    # sees; only the bytes live under the workspace's prefix.
    source = create.call_args.kwargs["source"]
    assert source.startswith("mlflow-artifacts:/skills/@acme/reviewer/")
    relative = source.removeprefix("mlflow-artifacts:/")
    assert stored_files(artifact_root / "workspaces" / "team-a" / relative) == sorted(SKILL_FILES)
    assert not (artifact_root / relative).exists()


# --- body and source must agree ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("registration", "kwargs", "message"),
    [
        # JSON with a null source would create a content-less version.
        (_UPLOAD, {"content": None, "multipart": False}, "requires a multipart/form-data body"),
        # Multipart with the content part missing.
        (_UPLOAD, {"content": None, "multipart": True}, "requires a multipart/form-data body"),
        # Multipart that also names a remote source.
        (
            SkillVersionRegistration(name="reviewer", source="https://example.com/r.git"),
            {"content": skill_archive(), "multipart": True},
            "cannot also carry uploaded content",
        ),
        (
            SkillVersionRegistration(name="reviewer", source="https://example.com/r.git"),
            {"content": None, "multipart": True},
            "must use an application/json body",
        ),
        # Fields that only make sense for a remote pointer.
        (
            SkillVersionRegistration(name="reviewer", subpath="skills/a"),
            {"content": skill_archive(), "multipart": True},
            "'subpath' does not apply to uploaded content",
        ),
        (
            SkillVersionRegistration(name="reviewer", ref="main"),
            {"content": skill_archive(), "multipart": True},
            "'ref' does not apply to uploaded content",
        ),
        (
            SkillVersionRegistration(name="reviewer", source_type="git"),
            {"content": skill_archive(), "multipart": True},
            "'source_type' does not apply to uploaded content",
        ),
    ],
)
def test_mismatched_body_and_source_are_rejected_without_a_version(
    store, artifact_root, registration, kwargs, message
):
    with pytest.raises(MlflowException, match=message) as exc:
        register_skill_version(registration, **kwargs)
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    assert version_rows(store) == 0
    assert stored_files(artifact_root) == []


@pytest.mark.parametrize(
    ("registration", "message"),
    [
        (SkillVersionRegistration(name=None), "'name' must be provided explicitly"),
        (SkillVersionRegistration(name=""), "'name' must be provided explicitly"),
        (SkillVersionRegistration(name="Bad_Name"), "Invalid skill name"),
        (SkillVersionRegistration(name="reviewer", organization="../x"), "Invalid organization"),
        (SkillVersionRegistration(name="reviewer", status="deleted"), "'active' or 'draft'"),
        (SkillVersionRegistration(name="reviewer", status="deprecated"), "'active' or 'draft'"),
        (SkillVersionRegistration(name="reviewer", digest="xyz"), "64 lowercase hex"),
        (SkillVersionRegistration(name="reviewer", digest="AB" * 32), "64 lowercase hex"),
        (SkillVersionRegistration(name="reviewer", source_type="mlflow"), "set by the server"),
        (SkillVersionRegistration(name="reviewer", source_type="assembled"), "set by the server"),
    ],
)
def test_invalid_metadata_is_rejected_without_a_version(
    store, artifact_root, registration, message
):
    with pytest.raises(MlflowException, match=message):
        register_skill_version(registration, content=skill_archive(), multipart=True)
    assert version_rows(store) == 0
    assert stored_files(artifact_root) == []


# --- malformed content ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (io.BytesIO(b""), "'content' part is empty"),
        (io.BytesIO(b"this is not a gzip tar"), "not a readable tar archive"),
        (skill_archive({}), "contains no files"),
        (skill_archive({"../escape.txt": b"x"}), "unsafe path"),
        (skill_archive({"/etc/passwd": b"x"}), "unsafe path"),
        (
            skill_archive(extra_members=[_special_member("link", tarfile.SYMTYPE, "SKILL.md")]),
            "symbolic link|not a regular file",
        ),
        (
            skill_archive(extra_members=[_special_member("hard", tarfile.LNKTYPE, "SKILL.md")]),
            "hard link|not a regular file",
        ),
        (
            skill_archive(extra_members=[_special_member("pipe", tarfile.FIFOTYPE)]),
            "not a regular file|special",
        ),
    ],
)
def test_malformed_content_is_rejected_without_a_version(store, artifact_root, content, message):
    with pytest.raises(MlflowException, match=message) as exc:
        register_skill_version(_UPLOAD, content=content, multipart=True)
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    assert version_rows(store) == 0
    assert stored_files(artifact_root) == []


def test_oversize_content_is_rejected_without_a_version(store, artifact_root, monkeypatch):
    monkeypatch.setenv(MLFLOW_SKILL_CONTENT_MAX_DECOMPRESSED_SIZE.name, "1000")
    big = skill_archive({"SKILL.md": b"x", "big.bin": b"\0" * 5000})
    with pytest.raises(MlflowException, match="exceeds the skill content size limit"):
        register_skill_version(_UPLOAD, content=big, multipart=True)
    assert version_rows(store) == 0
    assert stored_files(artifact_root) == []


def test_upload_is_cut_off_on_the_wire(store, artifact_root, monkeypatch):
    monkeypatch.setenv(MLFLOW_SKILL_CONTENT_MAX_DECOMPRESSED_SIZE.name, "1000")
    monkeypatch.setattr(registration_module, "_UPLOAD_SIZE_SLACK", 0)
    read_sizes = []

    class Endless(io.RawIOBase):
        def read(self, size=-1):
            read_sizes.append(size)
            return b"\x1f" * 600

    with pytest.raises(MlflowException, match="exceeds the size limit of 1000 bytes"):
        register_skill_version(_UPLOAD, content=Endless(), multipart=True)
    # Two chunks are enough to pass the cap; the rest of the stream is never read.
    assert len(read_sizes) == 2
    assert version_rows(store) == 0


# --- failure injection ------------------------------------------------------------------------


def test_storage_failure_leaves_no_version_and_cleans_the_partial_write(store, artifact_root):
    attempted = []

    def fail_midway(local_dir, artifact_path):
        attempted.append(artifact_path)
        partial = artifact_root / artifact_path
        partial.mkdir(parents=True)
        (partial / "SKILL.md").write_bytes(b"partial")
        raise OSError("disk full")

    with mock.patch.object(
        registration_module, "store_skill_tree", side_effect=fail_midway
    ) as store_tree:
        with pytest.raises(OSError, match="disk full"):
            register_skill_version(_UPLOAD, content=skill_archive(), multipart=True)
    store_tree.assert_called_once()
    assert version_rows(store) == 0
    assert stored_files(artifact_root) == []
    with pytest.raises(MlflowException, match="not found"):
        store.get_skill_version("reviewer", 1, organization="acme")
    assert len(attempted) == 1


def test_definite_commit_rejection_reclaims_the_stored_content(store, artifact_root):
    rejection = MlflowException("giving up", error_code=RESOURCE_ALREADY_EXISTS)
    with mock.patch.object(store, "create_skill_version", side_effect=rejection) as create:
        with pytest.raises(MlflowException, match="giving up"):
            register_skill_version(_UPLOAD, content=skill_archive(), multipart=True)
    create.assert_called_once()
    # The store reported that nothing was committed, so nothing can reference the bytes.
    assert stored_files(artifact_root) == []
    assert version_rows(store) == 0


def test_ambiguous_commit_failure_keeps_residue_that_no_version_can_address(store, artifact_root):
    # After an unknown failure the row may exist, so the content must not be deleted. The
    # bytes that are left are an accepted leak: their path is never handed out again.
    failure = MlflowException("connection dropped", error_code=INTERNAL_ERROR)
    with mock.patch.object(store, "create_skill_version", side_effect=failure) as create:
        with pytest.raises(MlflowException, match="connection dropped"):
            register_skill_version(_UPLOAD, content=skill_archive(), multipart=True)
    residue_source = create.call_args.kwargs["source"]
    residue = residue_source.removeprefix("mlflow-artifacts:/")
    assert stored_files(artifact_root / residue) == sorted(SKILL_FILES)
    assert version_rows(store) == 0

    version = register_skill_version(_UPLOAD, content=skill_archive(), multipart=True)

    assert version.version == 1
    assert version.source.artifact_path != residue_source
    assert stored_files(artifact_root / residue) == sorted(SKILL_FILES)


def test_version_collision_retries_without_rewriting_content(store, artifact_root):
    # Occupy version 1 after the store computed MAX(version) but before it inserts.
    original = store._persist_skill_version
    calls = []

    def collide_once(*args, **kwargs):
        calls.append(kwargs["version"])
        if len(calls) == 1:
            raise MlflowException("taken", error_code=RESOURCE_ALREADY_EXISTS)
        return original(*args, **kwargs)

    with (
        mock.patch.object(store, "_persist_skill_version", side_effect=collide_once) as persist,
        mock.patch.object(
            registration_module, "store_skill_tree", wraps=registration_module.store_skill_tree
        ) as store_tree,
    ):
        version = register_skill_version(_UPLOAD, content=skill_archive(), multipart=True)
    assert persist.call_count == 2
    store_tree.assert_called_once()
    assert version.version == 1
    assert len(list((artifact_root / "skills" / "@acme" / "reviewer").iterdir())) == 1


def test_concurrent_registrations_never_share_an_artifact_path(store, artifact_root):
    workers = 8
    all_storing = threading.Barrier(workers, timeout=30)
    commit_lock = threading.Lock()
    original_store_tree = registration_module.store_skill_tree
    original_create = store.create_skill_version
    paths = []
    results = []
    errors = []

    def store_together(local_dir, artifact_path):
        # Every worker has its path and is writing content at the same moment.
        all_storing.wait()
        paths.append(artifact_path)
        original_store_tree(local_dir, artifact_path)

    def commit_in_turn(**kwargs):
        # SQLite serializes writers; version allocation under contention is the store's
        # concern and has its own tests.
        with commit_lock:
            return original_create(**kwargs)

    def work():
        try:
            results.append(register_skill_version(_UPLOAD, content=skill_archive(), multipart=True))
        except Exception as e:
            errors.append(e)

    with (
        mock.patch.object(
            registration_module, "store_skill_tree", side_effect=store_together
        ) as store_tree,
        mock.patch.object(store, "create_skill_version", side_effect=commit_in_turn) as create,
    ):
        threads = [
            threading.Thread(target=work, name=f"skill-registration-{i}") for i in range(workers)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
    assert errors == []
    assert store_tree.call_count == workers
    assert create.call_count == workers
    assert len(set(paths)) == workers
    assert sorted(v.version for v in results) == list(range(1, workers + 1))
    assert len({v.source.artifact_path for v in results}) == workers
    for version in results:
        relative = version.source.artifact_path.removeprefix("mlflow-artifacts:/")
        assert stored_files(artifact_root / relative) == sorted(SKILL_FILES)


# --- deployment mode --------------------------------------------------------------------------


def test_upload_is_refused_when_the_server_does_not_serve_artifacts(store, no_artifact_serving):
    with pytest.raises(MlflowException, match="does not serve artifacts") as exc:
        register_skill_version(_UPLOAD, content=skill_archive(), multipart=True)
    assert exc.value.error_code == "NOT_IMPLEMENTED"
    assert exc.value.get_http_status_code() == 501
    assert version_rows(store) == 0


def test_remote_registration_does_not_need_artifact_serving(store, no_artifact_serving):
    remote = SkillVersionRegistration(name="reviewer", source="https://example.com/r.git")
    version = register_skill_version(remote)
    assert version.source == GitSource(url="https://example.com/r.git")


# --- remote pointers --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fields", "expected_type", "expected_source"),
    [
        (
            {"source": "https://example.com/r.git", "ref": "v1", "subpath": "skills/a/"},
            SkillSourceType.GIT,
            GitSource(url="https://example.com/r.git", ref="v1", subpath="skills/a"),
        ),
        (
            {"source": "oci://ghcr.io/acme/skills:v1"},
            SkillSourceType.OCI,
            OCISource(image="ghcr.io/acme/skills:v1"),
        ),
        (
            # No distinguishing shape, so the explicit type is what classifies it.
            {"source": "https://example.com/acme/skills", "source_type": "git"},
            SkillSourceType.GIT,
            GitSource(url="https://example.com/acme/skills"),
        ),
        (
            {"source": "oci://ghcr.io/acme/skills:v1", "source_type": "oci"},
            SkillSourceType.OCI,
            OCISource(image="ghcr.io/acme/skills:v1"),
        ),
    ],
)
def test_remote_registration(store, artifact_root, fields, expected_type, expected_source):
    version = register_skill_version(SkillVersionRegistration(name="reviewer", **fields))
    assert version.source_type == expected_type
    assert version.source == expected_source
    assert stored_files(artifact_root) == []


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"source": "https://example.com/acme/skills"}, "Cannot infer the source type"),
        ({"source": "oci://ghcr.io/a/b:v1", "source_type": "git"}, "contradicts the 'oci' scheme"),
        ({"source": "git://example.com/r", "source_type": "zip"}, "contradicts the 'git' scheme"),
        ({"source": "https://example.com/a.zip", "source_type": "svn"}, "Unknown 'source_type'"),
        ({"source": "https://example.com/a.zip", "ref": "main"}, "'ref' applies to Git"),
        (
            {"source": "https://example.com/a.zip", "source_type": "zip", "ref": "main"},
            "'ref' applies to git sources only",
        ),
        # The server never treats a source as a path on its own filesystem.
        ({"source": "/etc/skills/reviewer"}, "must be a remote git, oci, or zip location"),
        ({"source": "./reviewer"}, "must be a remote git, oci, or zip location"),
        ({"source": "C:\\skills\\reviewer"}, "must be a remote git, oci, or zip location"),
        ({"source": "https://user:pw@example.com/a.zip"}, "publicly accessible"),
        ({"source": "mlflow-artifacts:/skills/reviewer/x", "source_type": "mlflow"}, "set by the"),
    ],
)
def test_invalid_remote_registration_is_rejected_without_a_version(store, fields, message):
    with pytest.raises(MlflowException, match=message):
        register_skill_version(SkillVersionRegistration(name="reviewer", **fields))
    assert version_rows(store) == 0
