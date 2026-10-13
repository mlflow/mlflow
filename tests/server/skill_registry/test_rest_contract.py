import json
import tarfile

import pytest

from mlflow.environment_variables import MLFLOW_SKILL_CONTENT_MAX_DECOMPRESSED_SIZE

from tests.server.skill_registry.conftest import (
    SKILL_FILES,
    skill_archive,
    skill_rows,
    stored_files,
    version_rows,
)

pytestmark = pytest.mark.notrackingurimock

_DIGEST = "ab" * 32


def _remote_registration_fields() -> dict[str, str]:
    return {
        "source_type": "git",
        "source": "https://github.com/acme/skills.git",
        "ref": "v1.0.0",
        "subpath": "skills/reviewer",
        "digest": _DIGEST,
    }


def _version_url(prefix: str, name: str = "reviewer", organization: str = "") -> str:
    identity_path = f"@{organization}/{name}" if organization else name
    return f"{prefix}/{identity_path}/versions"


def _create_remote_version(client, prefix: str, name: str = "reviewer", organization: str = "acme"):
    response = client.post(
        _version_url(prefix, name, organization),
        json=_remote_registration_fields(),
    )
    assert response.status_code == 200, response.text
    return response


def _archive_member(name: str, member_type: int, linkname: str = "") -> tarfile.TarInfo:
    member = tarfile.TarInfo(name)
    member.type = member_type
    member.linkname = linkname
    return member


@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
@pytest.mark.parametrize("organization", ["", "acme"])
def test_parent_skill_lifecycle_over_http(
    prefix: str,
    organization: str,
    skill_registry_client,
    store,
    skill_registry_api_prefixes,
    mock_icon_hostname_resolution,
):
    assert prefix in skill_registry_api_prefixes
    identity_path = f"@{organization}/code-review" if organization else "code-review"

    response = skill_registry_client.post(
        prefix,
        json={
            "name": "code-review",
            "organization": organization,
            "description": "Reviews pull requests",
            "icons": [{"src": "https://example.com/icon.svg", "theme": "light"}],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["organization"] == organization

    response = skill_registry_client.get(f"{prefix}/{identity_path}")
    assert response.status_code == 200, response.text
    assert response.json()["description"] == "Reviews pull requests"

    response = skill_registry_client.patch(
        f"{prefix}/{identity_path}",
        json={"description": "Updated description"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["description"] == "Updated description"
    assert response.json()["icons"] == [{"src": "https://example.com/icon.svg", "theme": "light"}]

    response = skill_registry_client.patch(
        f"{prefix}/{identity_path}",
        json={"icons": None},
    )
    assert response.status_code == 200, response.text
    assert response.json()["icons"] is None

    response = skill_registry_client.delete(f"{prefix}/{identity_path}")
    assert response.status_code == 200, response.text
    assert response.json() == {}

    response = skill_registry_client.get(f"{prefix}/{identity_path}")
    assert response.status_code == 404, response.text


@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
@pytest.mark.parametrize("organization", ["", "acme"])
def test_parent_delete_cascades_versions_over_http(
    prefix: str, organization: str, skill_registry_client, store
):
    identity_path = f"@{organization}/code-review" if organization else "code-review"

    response = skill_registry_client.post(
        prefix,
        json={"name": "code-review", "organization": organization},
    )
    assert response.status_code == 200, response.text

    response = skill_registry_client.post(
        f"{prefix}/{identity_path}/versions",
        json=_remote_registration_fields(),
    )
    assert response.status_code == 200, response.text
    assert skill_rows(store) == 1
    assert version_rows(store) == 1

    response = skill_registry_client.delete(f"{prefix}/{identity_path}")
    assert response.status_code == 200, response.text
    assert response.json() == {}
    assert skill_rows(store) == 0
    assert version_rows(store) == 0

    response = skill_registry_client.get(f"{prefix}/{identity_path}")
    assert response.status_code == 404, response.text
    response = skill_registry_client.get(f"{prefix}/{identity_path}/versions/1")
    assert response.status_code == 404, response.text


@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
@pytest.mark.parametrize("organization", ["", "acme"])
def test_skill_version_lifecycle_over_http(prefix, organization, skill_registry_client, store):
    identity_path = f"@{organization}/code-review" if organization else "code-review"
    create_response = skill_registry_client.post(
        prefix,
        json={"name": "code-review", "organization": organization},
    )
    assert create_response.status_code == 200, create_response.text

    response = skill_registry_client.post(
        f"{prefix}/{identity_path}/versions",
        json=_remote_registration_fields(),
    )
    assert response.status_code == 200, response.text
    assert response.json()["version"] == 1
    assert response.json()["source_type"] == "git"
    assert response.json()["status"] == "active"

    response = skill_registry_client.get(f"{prefix}/{identity_path}/versions/1")
    assert response.status_code == 200, response.text
    assert response.json()["digest"] == _DIGEST

    response = skill_registry_client.patch(
        f"{prefix}/{identity_path}/versions/1",
        json={"status": "draft"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "draft"

    response = skill_registry_client.delete(f"{prefix}/{identity_path}/versions/1")
    assert response.status_code == 200, response.text
    assert response.json() == {}

    response = skill_registry_client.get(f"{prefix}/{identity_path}/versions/1")
    assert response.status_code == 404, response.text


@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
def test_parent_search_filters_and_paginates_sql_results(prefix, skill_registry_client, store):
    store.create_skill("code-review", organization="acme")
    store.create_skill_version(
        "code-review",
        organization="acme",
        source_type="git",
        source="https://github.com/acme/skills.git",
    )
    store.set_skill_tag("code-review", "team", "platform", organization="acme")

    store.create_skill("lint-check", organization="acme")
    store.create_skill_version(
        "lint-check",
        organization="acme",
        source_type="git",
        source="https://github.com/acme/skills.git",
    )
    store.set_skill_tag("lint-check", "team", "platform", organization="acme")

    store.create_skill("deploy", organization="acme")
    store.create_skill_version(
        "deploy",
        organization="acme",
        source_type="zip",
        source="https://example.com/deploy.zip",
    )

    response = skill_registry_client.get(
        prefix,
        params=[
            ("filter_string", "tags.team = 'platform'"),
            ("max_results", "1"),
            ("order_by", "name ASC"),
        ],
    )
    assert response.status_code == 200, response.text
    assert [skill["name"] for skill in response.json()["skills"]] == ["code-review"]
    page_token = response.json()["next_page_token"]
    assert page_token

    response = skill_registry_client.get(
        prefix,
        params=[
            ("filter_string", "tags.team = 'platform'"),
            ("max_results", "1"),
            ("order_by", "name ASC"),
            ("page_token", page_token),
        ],
    )
    assert response.status_code == 200, response.text
    assert [skill["name"] for skill in response.json()["skills"]] == ["lint-check"]
    assert response.json()["next_page_token"] is None


@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
def test_parent_search_free_text_paginates_without_duplicates(prefix, skill_registry_client, store):
    store.create_skill("code-review", description="Code review tool")
    store.create_skill("lint-check", description="Linting tool")
    store.create_skill("deploy", description="Deployment automation")

    names = []
    page_token = None
    while True:
        params = [
            ("filter_string", "search_text ILIKE '%TOOL%'"),
            ("max_results", "1"),
            ("order_by", "name ASC"),
        ]
        if page_token:
            params.append(("page_token", page_token))

        response = skill_registry_client.get(prefix, params=params)
        assert response.status_code == 200, response.text
        names.extend(skill["name"] for skill in response.json()["skills"])
        page_token = response.json()["next_page_token"]
        if page_token is None:
            break

    assert names == ["code-review", "lint-check"]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
def test_parent_search_uses_latest_version_source_type_filter(prefix, skill_registry_client, store):
    store.create_skill("git-skill")
    store.create_skill_version(
        "git-skill",
        source_type="git",
        source="https://github.com/acme/skills.git",
    )
    store.create_skill("zip-skill")
    store.create_skill_version(
        "zip-skill",
        source_type="zip",
        source="https://example.com/skills.zip",
    )

    response = skill_registry_client.get(
        prefix,
        params={"filter_string": "source_type = 'git'"},
    )
    assert response.status_code == 200, response.text
    assert [skill["name"] for skill in response.json()["skills"]] == ["git-skill"]


@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
def test_parent_source_type_filter_tracks_latest_non_deleted_version(
    prefix, skill_registry_client, store
):
    store.create_skill("evolving-skill")
    store.create_skill_version(
        "evolving-skill",
        source_type="git",
        source="https://github.com/acme/skills.git",
        ref="v1.0.0",
    )
    store.create_skill_version(
        "evolving-skill",
        source_type="zip",
        source="https://example.com/skills.zip",
    )

    response = skill_registry_client.get(
        prefix,
        params={"filter_string": "source_type = 'zip'"},
    )
    assert response.status_code == 200, response.text
    assert [skill["name"] for skill in response.json()["skills"]] == ["evolving-skill"]

    response = skill_registry_client.patch(
        f"{prefix}/evolving-skill/versions/2",
        json={"status": "deprecated"},
    )
    assert response.status_code == 200, response.text

    response = skill_registry_client.delete(f"{prefix}/evolving-skill/versions/2")
    assert response.status_code == 200, response.text

    response = skill_registry_client.get(
        prefix,
        params={"filter_string": "source_type = 'git'"},
    )
    assert response.status_code == 200, response.text
    assert [skill["name"] for skill in response.json()["skills"]] == ["evolving-skill"]


@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
@pytest.mark.parametrize("organization", ["", "acme"])
def test_skill_version_search_filters_and_orders_sql_results(
    prefix, organization, skill_registry_client, store
):
    identity_path = f"@{organization}/code-review" if organization else "code-review"
    store.create_skill("code-review", organization=organization)
    for version, status in enumerate(("active", "draft", "active"), start=1):
        store.create_skill_version(
            "code-review",
            organization=organization,
            source_type="git",
            source="https://github.com/acme/skills.git",
            ref=f"v{version}",
            status=status,
        )

    response = skill_registry_client.get(
        f"{prefix}/{identity_path}/versions",
        params=[
            ("filter_string", "status = 'active'"),
            ("max_results", "1"),
            ("order_by", "version DESC"),
        ],
    )
    assert response.status_code == 200, response.text
    assert [version["version"] for version in response.json()["skill_versions"]] == [3]
    page_token = response.json()["next_page_token"]
    assert page_token

    response = skill_registry_client.get(
        f"{prefix}/{identity_path}/versions",
        params=[
            ("filter_string", "status = 'active'"),
            ("max_results", "1"),
            ("order_by", "version DESC"),
            ("page_token", page_token),
        ],
    )
    assert response.status_code == 200, response.text
    assert [version["version"] for version in response.json()["skill_versions"]] == [1]
    assert response.json()["next_page_token"] is None


@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
def test_remote_registration_creates_parent_and_first_version(prefix, skill_registry_client, store):
    response = skill_registry_client.post(
        _version_url(prefix, organization="acme"),
        json=_remote_registration_fields(),
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["name"] == "reviewer"
    assert body["organization"] == "acme"
    assert body["version"] == 1
    assert body["status"] == "active"

    parent = skill_registry_client.get(f"{prefix}/@acme/reviewer")
    assert parent.status_code == 200, parent.text
    assert parent.json()["latest_version"] == 1


@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
def test_multipart_registration_persists_uploaded_skill_tree(
    prefix, skill_registry_client, store, artifact_root
):
    response = skill_registry_client.post(
        _version_url(prefix, organization="acme"),
        files={
            "metadata": (
                "metadata.json",
                json.dumps({}),
                "application/json",
            ),
            "content": (
                "skill.tar.gz",
                skill_archive().getvalue(),
                "application/gzip",
            ),
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["name"] == "reviewer"
    assert body["source_type"] == "mlflow"
    assert body["status"] == "active"
    artifact_path = body["source"].removeprefix("mlflow-artifacts:/")
    assert stored_files(artifact_root / artifact_path) == sorted(SKILL_FILES)


@pytest.mark.parametrize(
    ("fields", "expected_source_type"),
    [
        (
            {"source": "https://example.com/reviewer.git", "ref": "main"},
            "git",
        ),
        (
            {"source": "oci://ghcr.io/acme/skills:v1"},
            "oci",
        ),
        (
            {"source": "https://example.com/reviewer.zip"},
            "zip",
        ),
        (
            {"source": "https://example.com/reviewer", "source_type": "git"},
            "git",
        ),
    ],
)
@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
def test_remote_registration_resolves_source_type_over_http(
    prefix, fields, expected_source_type, skill_registry_client, store
):
    response = skill_registry_client.post(
        _version_url(prefix),
        json=fields,
    )

    assert response.status_code == 200, response.text
    assert response.json()["source_type"] == expected_source_type
    assert response.json()["status"] == "active"
    assert version_rows(store) == 1


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (
            {"source": "https://example.com/reviewer"},
            "Cannot infer the source type",
        ),
        (
            {"source": "oci://ghcr.io/acme/skills:v1", "source_type": "git"},
            "contradicts the 'oci' scheme",
        ),
        (
            {"source": "https://example.com/reviewer.zip", "source_type": "svn"},
            "Unknown 'source_type'",
        ),
        (
            {"source": "https://example.com/reviewer.zip", "ref": "main"},
            "'ref' applies to Git",
        ),
        (
            {"source": "/etc/skills/reviewer"},
            "must be a remote git, oci, or zip location",
        ),
        (
            {
                "source": "mlflow-artifacts:/skills/reviewer/x",
                "source_type": "mlflow",
            },
            "set by the server",
        ),
    ],
)
@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
def test_remote_registration_rejects_invalid_source_metadata_over_http(
    prefix, fields, message, skill_registry_client, store
):
    response = skill_registry_client.post(
        _version_url(prefix),
        json=fields,
    )

    assert response.status_code == 400, response.text
    assert message in response.json()["message"]
    assert skill_rows(store) == 0
    assert version_rows(store) == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "different-skill"},
        {"organization": "different-org"},
    ],
)
@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
def test_version_registration_rejects_conflicting_body_identity(
    payload, prefix, skill_registry_client, store
):
    response = skill_registry_client.post(
        _version_url(prefix, organization="acme"),
        json={**payload, **_remote_registration_fields()},
    )

    assert response.status_code == 400, response.text
    assert "must match the path identity" in response.json()["message"]
    assert skill_rows(store) == 0
    assert version_rows(store) == 0


@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
def test_registration_rejects_malformed_json(prefix, skill_registry_client, store):
    response = skill_registry_client.post(
        _version_url(prefix),
        content=b"not-json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400, response.text
    assert "valid JSON" in response.json()["message"]


@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
def test_registration_rejects_unsupported_content_type(prefix, skill_registry_client, store):
    response = skill_registry_client.post(
        _version_url(prefix),
        content=b"name=reviewer",
        headers={"content-type": "text/plain"},
    )

    assert response.status_code == 400, response.text
    assert "application/json" in response.json()["message"]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"name": "reviewer", "organization": "InvalidOrg"}, "Invalid organization name"),
        ({"name": "invalid_name"}, "Invalid skill name"),
    ],
)
@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
def test_parent_creation_rejects_invalid_identity(
    payload, message, prefix, skill_registry_client, store
):
    response = skill_registry_client.post(prefix, json=payload)

    assert response.status_code == 400, response.text
    assert message in response.json()["message"]
    assert skill_rows(store) == 0
    assert version_rows(store) == 0


@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
def test_json_registration_without_source_requires_multipart_content(
    prefix, skill_registry_client, store
):
    response = skill_registry_client.post(
        _version_url(prefix, organization="acme"),
        json={},
    )

    assert response.status_code == 400, response.text
    assert "multipart/form-data" in response.json()["message"]
    assert version_rows(store) == 0


@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
def test_multipart_registration_rejects_remote_source_metadata(
    prefix, skill_registry_client, store, artifact_root
):
    response = skill_registry_client.post(
        _version_url(prefix, organization="acme"),
        files={
            "metadata": (
                "metadata.json",
                json.dumps(_remote_registration_fields()),
                "application/json",
            ),
            "content": ("skill.tar.gz", skill_archive().getvalue(), "application/gzip"),
        },
    )

    assert response.status_code == 400, response.text
    assert "remote 'source'" in response.json()["message"]
    assert skill_rows(store) == 0
    assert version_rows(store) == 0
    assert stored_files(artifact_root) == []


@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
def test_parent_creation_rejects_duplicate_identity(prefix, skill_registry_client, store):
    payload = {"name": "reviewer", "organization": "acme"}
    first = skill_registry_client.post(prefix, json=payload)
    second = skill_registry_client.post(prefix, json=payload)

    assert first.status_code == 200, first.text
    assert second.status_code == 400, second.text
    assert skill_rows(store) == 1
    assert version_rows(store) == 0


@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
def test_aliases_are_organization_scoped(prefix, skill_registry_client, store):
    _create_remote_version(skill_registry_client, prefix, organization="acme")
    response = skill_registry_client.post(
        f"{prefix}/@acme/reviewer/aliases",
        json={"alias": "stable", "version": 1},
    )
    assert response.status_code == 200, response.text

    response = skill_registry_client.get(f"{prefix}/reviewer/aliases/stable")
    assert response.status_code == 404, response.text

    response = skill_registry_client.get(f"{prefix}/@acme/reviewer/aliases/stable")
    assert response.status_code == 200, response.text
    assert response.json()["organization"] == "acme"


@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
def test_multipart_registration_rejects_missing_content(
    prefix, skill_registry_client, store, artifact_root
):
    response = skill_registry_client.post(
        _version_url(prefix, organization="acme"),
        files={
            "metadata": (
                "metadata.json",
                json.dumps({}),
                "application/json",
            )
        },
    )

    assert response.status_code == 400, response.text
    assert "content" in response.json()["message"]
    assert skill_rows(store) == 0
    assert version_rows(store) == 0
    assert stored_files(artifact_root) == []


@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
def test_multipart_registration_rejects_missing_metadata(
    prefix, skill_registry_client, store, artifact_root
):
    response = skill_registry_client.post(
        _version_url(prefix, organization="acme"),
        files={
            "content": (
                "skill.tar.gz",
                skill_archive().getvalue(),
                "application/gzip",
            )
        },
    )

    assert response.status_code == 400, response.text
    assert "exactly one 'metadata' part" in response.json()["message"]
    assert skill_rows(store) == 0
    assert version_rows(store) == 0
    assert stored_files(artifact_root) == []


@pytest.mark.parametrize(
    ("member_name", "member_type", "linkname"),
    [
        ("../escape.txt", tarfile.REGTYPE, ""),
        ("/absolute.txt", tarfile.REGTYPE, ""),
        ("linked.txt", tarfile.SYMTYPE, "SKILL.md"),
        ("fifo", tarfile.FIFOTYPE, ""),
    ],
)
@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
def test_multipart_registration_rejects_unsafe_archive_members(
    member_name,
    member_type,
    linkname,
    prefix,
    skill_registry_client,
    store,
    artifact_root,
):
    archive = skill_archive(extra_members=(_archive_member(member_name, member_type, linkname),))
    response = skill_registry_client.post(
        _version_url(prefix, organization="acme"),
        files={
            "metadata": (
                "metadata.json",
                json.dumps({}),
                "application/json",
            ),
            "content": ("skill.tar.gz", archive.getvalue(), "application/gzip"),
        },
    )

    assert response.status_code == 400, response.text
    assert skill_rows(store) == 0
    assert version_rows(store) == 0
    assert stored_files(artifact_root) == []


@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
def test_multipart_registration_rejects_oversized_archive(
    prefix, skill_registry_client, store, artifact_root, monkeypatch
):
    monkeypatch.setenv(MLFLOW_SKILL_CONTENT_MAX_DECOMPRESSED_SIZE.name, "1")
    response = skill_registry_client.post(
        _version_url(prefix, organization="acme"),
        files={
            "metadata": (
                "metadata.json",
                json.dumps({}),
                "application/json",
            ),
            "content": ("skill.tar.gz", skill_archive().getvalue(), "application/gzip"),
        },
    )

    assert response.status_code == 400, response.text
    assert "size limit" in response.json()["message"]
    assert skill_rows(store) == 0
    assert version_rows(store) == 0
    assert stored_files(artifact_root) == []


@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
@pytest.mark.parametrize("organization", ["", "acme"])
def test_skill_alias_lifecycle_over_http(prefix, organization, skill_registry_client, store):
    _create_remote_version(skill_registry_client, prefix, organization=organization)
    identity_path = f"@{organization}/reviewer" if organization else "reviewer"

    response = skill_registry_client.post(
        f"{prefix}/{identity_path}/aliases",
        json={"alias": "stable", "version": 1},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {}

    response = skill_registry_client.get(f"{prefix}/{identity_path}/aliases/stable")
    assert response.status_code == 200, response.text
    assert response.json()["version"] == 1

    response = skill_registry_client.get(f"{prefix}/{identity_path}/aliases/latest")
    assert response.status_code == 200, response.text
    assert response.json()["version"] == 1

    response = skill_registry_client.delete(f"{prefix}/{identity_path}/aliases/stable")
    assert response.status_code == 200, response.text
    assert response.json() == {}

    response = skill_registry_client.get(f"{prefix}/{identity_path}/aliases/stable")
    assert response.status_code == 404, response.text


@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
@pytest.mark.parametrize("method", ["post", "delete"])
def test_latest_alias_cannot_be_mutated(prefix, skill_registry_client, store, method):
    _create_remote_version(skill_registry_client, prefix)
    if method == "post":
        url = f"{prefix}/@acme/reviewer/aliases"
        response = skill_registry_client.post(url, json={"alias": "latest", "version": 1})
    else:
        url = f"{prefix}/@acme/reviewer/aliases/latest"
        response = skill_registry_client.delete(url)

    assert response.status_code == 400, response.text
    assert "reserved" in response.json()["message"] or "automatically" in response.json()["message"]


@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
@pytest.mark.parametrize("organization", ["", "acme"])
def test_skill_tags_persist_with_case_and_long_values_over_http(
    prefix, organization, skill_registry_client, store
):
    identity_path = f"@{organization}/reviewer" if organization else "reviewer"
    store.create_skill("reviewer", organization=organization)
    store.create_skill_version(
        "reviewer",
        organization=organization,
        source_type="git",
        source="https://github.com/acme/skills.git",
    )
    long_value = "x" * 9000

    response = skill_registry_client.post(
        f"{prefix}/{identity_path}/tags",
        json={"key": "Team", "value": long_value},
    )
    assert response.status_code == 200, response.text
    response = skill_registry_client.post(
        f"{prefix}/{identity_path}/tags",
        json={"key": "team", "value": "platform"},
    )
    assert response.status_code == 200, response.text
    response = skill_registry_client.post(
        f"{prefix}/{identity_path}/versions/1/tags",
        json={"key": "Team", "value": "version-platform"},
    )
    assert response.status_code == 200, response.text

    response = skill_registry_client.get(f"{prefix}/{identity_path}")
    assert response.status_code == 200, response.text
    assert response.json()["tags"] == {"Team": long_value, "team": "platform"}

    response = skill_registry_client.get(f"{prefix}/{identity_path}/versions/1")
    assert response.status_code == 200, response.text
    assert response.json()["tags"] == {"Team": "version-platform"}

    response = skill_registry_client.delete(f"{prefix}/{identity_path}/tags/Team")
    assert response.status_code == 200, response.text
    response = skill_registry_client.get(f"{prefix}/{identity_path}")
    assert response.status_code == 200, response.text
    assert response.json()["tags"] == {"team": "platform"}

    response = skill_registry_client.delete(f"{prefix}/{identity_path}/versions/1/tags/Team")
    assert response.status_code == 200, response.text
    response = skill_registry_client.get(f"{prefix}/{identity_path}/versions/1")
    assert response.status_code == 200, response.text
    assert response.json()["tags"] == {}


def _bulk_registration(name: str, *, ref: str = "v1.0.0") -> dict[str, str]:
    return {
        "name": name,
        "source_type": "git",
        "source": "https://github.com/acme/skills.git",
        "ref": ref,
        "subpath": f"skills/{name}",
        "digest": _DIGEST,
    }


@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
def test_bulk_registration_creates_multiple_skills(prefix, skill_registry_client, store):
    response = skill_registry_client.post(
        f"{prefix}/bulk-register",
        json={
            "organization": "acme",
            "skills": [_bulk_registration("reviewer"), _bulk_registration("triage")],
        },
    )

    assert response.status_code == 200, response.text
    versions = response.json()["skill_versions"]
    assert [(version["name"], version["version"]) for version in versions] == [
        ("reviewer", 1),
        ("triage", 1),
    ]
    assert skill_registry_client.get(f"{prefix}/@acme/reviewer").status_code == 200
    assert skill_registry_client.get(f"{prefix}/@acme/triage").status_code == 200


@pytest.mark.parametrize("prefix", ["/ajax-api/3.0/mlflow/skills", "/api/3.0/mlflow/skills"])
def test_bulk_registration_is_idempotent_over_http(prefix, skill_registry_client, store):
    payload = {
        "organization": "acme",
        "skills": [_bulk_registration("reviewer")],
    }

    first = skill_registry_client.post(f"{prefix}/bulk-register", json=payload)
    second = skill_registry_client.post(f"{prefix}/bulk-register", json=payload)

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert second.json()["skill_versions"][0]["version"] == 1
    assert version_rows(store) == 1


@pytest.mark.parametrize(
    ("prefix", "skills"),
    [
        (
            "/ajax-api/3.0/mlflow/skills",
            [_bulk_registration("reviewer"), _bulk_registration("reviewer")],
        ),
        (
            "/ajax-api/3.0/mlflow/skills",
            [_bulk_registration("reviewer"), _bulk_registration("triage", ref="v2.0.0")],
        ),
        (
            "/api/3.0/mlflow/skills",
            [_bulk_registration("reviewer"), _bulk_registration("reviewer")],
        ),
        (
            "/api/3.0/mlflow/skills",
            [_bulk_registration("reviewer"), _bulk_registration("triage", ref="v2.0.0")],
        ),
    ],
)
def test_bulk_registration_rejects_invalid_batch_without_persisting(
    prefix, skills, skill_registry_client, store
):
    response = skill_registry_client.post(
        f"{prefix}/bulk-register",
        json={"organization": "acme", "skills": skills},
    )

    assert response.status_code == 400, response.text
    assert version_rows(store) == 0
