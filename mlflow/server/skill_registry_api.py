from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer
from starlette.datastructures import UploadFile
from starlette.types import Message, Receive

from mlflow.entities.skill import RegistryIcon, Skill, SkillStatus
from mlflow.entities.skill_source import (
    GitSource,
    MlflowSource,
    OCISource,
    ZipSource,
)
from mlflow.entities.skill_version import SkillVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    PERMISSION_DENIED,
    TEMPORARILY_UNAVAILABLE,
)
from mlflow.server.constants import ARTIFACTS_ONLY_ENV_VAR
from mlflow.server.skill_registry.registration import (
    SkillVersionRegistration,
    bulk_register_skill_versions,
    get_max_upload_size,
    register_skill_version,
)
from mlflow.store.tracking import NOT_SET
from mlflow.utils.validation import (
    _MAX_BULK_REGISTER_SKILLS,
    _MAX_REGISTRY_ICONS_PER_LIST,
    _validate_icon_mime_type,
    _validate_icon_url,
    _validate_organization_name,
    _validate_skill_name,
    _validate_skill_version,
)

_SKILL_REGISTRY_AJAX_API_PREFIX = "/ajax-api/3.0/mlflow/skills"
_SKILL_REGISTRY_API_PREFIX = "/api/3.0/mlflow/skills"

# Reserve a leading ``@`` for the organization marker in paths such as
# ``/@org/versions``; skill-name segments therefore cannot begin with ``@``.
_SKILL_NAME_PATH_PATTERN = r"^[^@/][^/]*$"
SkillNamePath = Annotated[str, Path(pattern=_SKILL_NAME_PATH_PATTERN)]

# Multipart bodies include the archive plus the metadata part, part headers, and boundaries.
# Keep this allowance bounded so the transport limit remains close to the archive limit while
# allowing normal multipart requests to reach the registration-level checks.
_MULTIPART_REQUEST_OVERHEAD = 2 * 1024 * 1024
_MAX_REGISTRATION_METADATA_SIZE = 1 * 1024 * 1024
_MAX_MULTIPART_FILES = 2
_MAX_MULTIPART_FIELDS = 1


def get_skill_registry_api_route_prefixes() -> tuple[str, ...]:
    from mlflow.server.handlers import _add_static_prefix

    return (
        _add_static_prefix(_SKILL_REGISTRY_AJAX_API_PREFIX),
        _add_static_prefix(_SKILL_REGISTRY_API_PREFIX),
    )


def is_skill_registry_api_path(path: str) -> bool:
    return any(
        path == prefix or path.startswith(f"{prefix}/")
        for prefix in get_skill_registry_api_route_prefixes()
    )


class _BaseSkillIconPayload(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    src: str
    sizes: list[str] | None = None
    mimeType: str | None = None
    theme: str | None = None

    @model_serializer(mode="plain")
    def serialize(self) -> dict[str, Any]:
        icon = dict(self.model_extra or {})
        icon["src"] = self.src
        if self.sizes is not None:
            icon["sizes"] = self.sizes
        if self.mimeType is not None:
            icon["mimeType"] = self.mimeType
        if self.theme is not None:
            icon["theme"] = self.theme
        return icon


class SkillIconRequestPayload(_BaseSkillIconPayload):
    @field_validator("src")
    @classmethod
    def _validate_src(cls, value: str) -> str:
        _validate_icon_url(value)
        return value

    @field_validator("mimeType")
    @classmethod
    def _validate_mime_type(cls, value: str | None) -> str | None:
        _validate_icon_mime_type(value)
        return None if value is None else value.strip().lower()


class SkillIconResponsePayload(_BaseSkillIconPayload):
    """Icon payload used when returning a stored Skill."""


class CreateSkillRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    organization: str = ""
    description: str | None = None
    icons: list[SkillIconRequestPayload] | None = Field(
        default=None, max_length=_MAX_REGISTRY_ICONS_PER_LIST
    )


class UpdateSkillRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str | None = None
    icons: list[SkillIconRequestPayload] | None = Field(
        default=None, max_length=_MAX_REGISTRY_ICONS_PER_LIST
    )


class SkillAliasResponse(BaseModel):
    alias: str
    version: int


class SkillResponse(BaseModel):
    name: str
    organization: str = ""
    workspace: str | None = None
    description: str | None = None
    icons: list[SkillIconResponsePayload] | None = None
    status: str | None = None
    latest_version: int | None = None
    source_type: str | None = None
    aliases: list[SkillAliasResponse] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    @classmethod
    def from_entity(cls, entity: Skill) -> SkillResponse:
        return cls(
            name=entity.name,
            organization=entity.organization,
            workspace=entity.workspace,
            description=entity.description,
            icons=(
                None
                if entity.icons is None
                else [SkillIconResponsePayload.model_validate(icon) for icon in entity.icons]
            ),
            status=str(entity.status) if entity.status else None,
            latest_version=entity.latest_version,
            source_type=str(entity.source_type) if entity.source_type else None,
            aliases=[
                SkillAliasResponse(alias=alias, version=version)
                for alias, version in entity.aliases.items()
            ],
            tags=entity.tags,
            created_by=entity.created_by,
            last_updated_by=entity.last_updated_by,
            creation_timestamp=entity.creation_timestamp,
            last_updated_timestamp=entity.last_updated_timestamp,
        )


class SearchSkillsResponse(BaseModel):
    skills: list[SkillResponse]
    next_page_token: str | None = None


class SetSkillAliasRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alias: str
    version: int


class SetTagRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    value: str


class CreateSkillVersionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    organization: str = ""
    source_type: str | None = None
    source: str | None = None
    ref: str | None = None
    subpath: str | None = None
    digest: str | None = None
    status: str = SkillStatus.ACTIVE.value


class RegisterSkillRequest(CreateSkillVersionRequest):
    pass


class BulkRegisterSkillRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    source_type: str | None = None
    source: str
    ref: str | None = None
    subpath: str | None = None
    digest: str
    status: str = SkillStatus.ACTIVE.value


class BulkRegisterSkillsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organization: str = ""
    skills: list[BulkRegisterSkillRequest] = Field(
        min_length=1, max_length=_MAX_BULK_REGISTER_SKILLS
    )


class UpdateSkillVersionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str | None = None


def _skill_source_response_fields(
    source: GitSource | OCISource | ZipSource | MlflowSource | str | None,
) -> tuple[str | None, str | None, str | None]:
    if isinstance(source, GitSource):
        return source.url, source.ref, source.subpath
    if isinstance(source, OCISource):
        return source.image, None, source.subpath
    if isinstance(source, ZipSource):
        return source.url, None, source.subpath
    if isinstance(source, MlflowSource):
        return source.artifact_path, None, source.subpath
    return source, None, None


class SkillVersionResponse(BaseModel):
    name: str
    version: int
    organization: str = ""
    workspace: str | None = None
    source_type: str | None = None
    source: str | None = None
    ref: str | None = None
    subpath: str | None = None
    digest: str | None = None
    status: str = SkillStatus.ACTIVE.value
    aliases: list[str] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)
    created_by: str | None = None
    last_updated_by: str | None = None
    creation_timestamp: int | None = None
    last_updated_timestamp: int | None = None

    @classmethod
    def from_entity(cls, entity: SkillVersion) -> SkillVersionResponse:
        source_value, ref, subpath = _skill_source_response_fields(entity.source)

        return cls(
            name=entity.name,
            version=entity.version,
            organization=entity.organization,
            workspace=entity.workspace,
            source_type=str(entity.source_type) if entity.source_type else None,
            source=source_value,
            ref=ref,
            subpath=subpath,
            digest=entity.digest,
            status=str(entity.status),
            aliases=entity.aliases,
            tags=entity.tags,
            created_by=entity.created_by,
            last_updated_by=entity.last_updated_by,
            creation_timestamp=entity.creation_timestamp,
            last_updated_timestamp=entity.last_updated_timestamp,
        )


class SearchSkillVersionsResponse(BaseModel):
    skill_versions: list[SkillVersionResponse]
    next_page_token: str | None = None


class BulkRegisterSkillsResponse(BaseModel):
    skill_versions: list[SkillVersionResponse]


def _skill_version_create_openapi_extra(*, require_name: bool = False) -> dict[str, Any]:
    json_schema = CreateSkillVersionRequest.model_json_schema()
    if require_name:
        json_schema["required"] = ["name"]
        json_schema["properties"]["name"] = {
            "type": "string",
            "title": "Name",
        }

    return {
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": json_schema,
                },
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["metadata", "content"],
                        "properties": {
                            "metadata": {"type": "string", "format": "binary"},
                            "content": {"type": "string", "format": "binary"},
                        },
                    },
                },
            },
        }
    }


_SKILL_VERSION_CREATE_OPENAPI_EXTRA = _skill_version_create_openapi_extra()
_REGISTER_SKILL_OPENAPI_EXTRA = _skill_version_create_openapi_extra(require_name=True)


def _icons_to_entities(icons: list[SkillIconRequestPayload] | None) -> list[RegistryIcon] | None:
    if icons is None:
        return None
    return [icon.model_dump(exclude_none=True) for icon in icons]


def _validate_skill_path_identity(organization: str, name: str) -> None:
    _validate_organization_name(organization)
    _validate_skill_name(name)


def _ensure_tracking_server_enabled() -> None:
    if os.environ.get(ARTIFACTS_ONLY_ENV_VAR):
        raise MlflowException(
            "Skill Registry endpoints are disabled when the MLflow server is running in "
            "`--artifacts-only` mode. To enable tracking server functionality, run "
            "`mlflow server` without `--artifacts-only`.",
            error_code=TEMPORARILY_UNAVAILABLE,
        )


def _require_skill_capability(
    request: Request, name: str, organization: str, capability: str
) -> None:
    username = getattr(request.state, "username", None)
    if username is None:
        return
    from mlflow.server import auth

    if auth.store.get_user(username).is_admin:
        return
    permission = auth._get_skill_permission(organization, name, username)
    if not getattr(permission, f"can_{capability}"):
        raise MlflowException("Permission denied", PERMISSION_DENIED)


def _require_skill_read(request: Request, name: SkillNamePath, organization: str = "") -> None:
    _require_skill_capability(request, name, organization, "read")


def _require_skill_update(request: Request, name: SkillNamePath, organization: str = "") -> None:
    _require_skill_capability(request, name, organization, "update")


def _require_skill_manage(request: Request, name: SkillNamePath, organization: str = "") -> None:
    _require_skill_capability(request, name, organization, "manage")


def _require_skill_create(request: Request) -> None:
    username = getattr(request.state, "username", None)
    if username is None:
        return
    from mlflow.server import auth

    if not auth.store.get_user(username).is_admin and not auth.validate_can_create_skill(username):
        raise MlflowException("Permission denied", PERMISSION_DENIED)


def _authorize_registration(request: Request, organization: str, name: str) -> bool:
    """Return whether registration will create the parent, after checking its grant."""
    username = getattr(request.state, "username", None)
    if username is None:
        # Authentication is supplied by the basic-auth FastAPI middleware when enabled.
        return False
    from mlflow.server import auth

    parent_exists = auth._skill_exists_for_auth(organization, name)
    if not auth.validate_can_register_skill(
        username, organization, name, parent_exists=parent_exists
    ):
        raise MlflowException("Permission denied", PERMISSION_DENIED)
    return not parent_exists


def _grant_creator_if_new(request: Request, organization: str, name: str, new: bool) -> None:
    if new and (username := getattr(request.state, "username", None)):
        from mlflow.server import auth
        from mlflow.server.handlers import _get_tracking_store

        # Registration can race with another creator after its preflight lookup.
        # The caller must never inherit MANAGE on a parent that won that race.
        parent = _get_tracking_store().get_skill(name=name, organization=organization)
        if parent.created_by == username:
            auth.grant_manage_for_created_skills(username, organization, [name])


async def _create_skill_version(
    name: str,
    request: Request,
    organization: str = "",
) -> SkillVersionResponse:
    async with _parse_registration_request(
        request,
        name=name,
        organization=organization,
    ) as (registration, content, multipart):
        parent_missing = _authorize_registration(request, organization, name)
        version = await asyncio.to_thread(
            register_skill_version,
            registration,
            content=content,
            multipart=multipart,
            expected_parent_exists=(
                not parent_missing if getattr(request.state, "username", None) else None
            ),
        )
    _grant_creator_if_new(request, organization, name, parent_missing)
    return SkillVersionResponse.from_entity(version)


def _get_skill_version(name: str, version: int, organization: str = "") -> SkillVersionResponse:
    from mlflow.server.handlers import _get_tracking_store

    _validate_skill_path_identity(organization, name)
    _validate_skill_version(version)
    return SkillVersionResponse.from_entity(
        _get_tracking_store().get_skill_version(
            name=name,
            version=version,
            organization=organization,
        )
    )


def _get_skill_version_by_alias(
    name: str,
    alias: str,
    organization: str = "",
) -> SkillVersionResponse:
    from mlflow.server.handlers import _get_tracking_store

    _validate_skill_path_identity(organization, name)
    return SkillVersionResponse.from_entity(
        _get_tracking_store().get_skill_version_by_alias(
            name=name,
            alias=alias,
            organization=organization,
        )
    )


def _set_skill_alias(
    name: str,
    alias: str,
    version: int,
    organization: str = "",
) -> dict[str, Any]:
    from mlflow.server.handlers import _get_tracking_store

    _validate_skill_path_identity(organization, name)
    _get_tracking_store().set_skill_alias(
        name=name,
        alias=alias,
        version=version,
        organization=organization,
    )
    return {}


def _delete_skill_alias(name: str, alias: str, organization: str = "") -> dict[str, Any]:
    from mlflow.server.handlers import _get_tracking_store

    _validate_skill_path_identity(organization, name)
    _get_tracking_store().delete_skill_alias(
        name=name,
        alias=alias,
        organization=organization,
    )
    return {}


def _search_skill_versions(
    name: str,
    organization: str = "",
    filter_string: str | None = None,
    max_results: int = 100,
    order_by: list[str] | None = None,
    page_token: str | None = None,
) -> SearchSkillVersionsResponse:
    from mlflow.server.handlers import _get_tracking_store

    _validate_skill_path_identity(organization, name)
    results = _get_tracking_store().search_skill_versions(
        name=name,
        organization=organization,
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by,
        page_token=page_token,
    )
    return SearchSkillVersionsResponse(
        skill_versions=[SkillVersionResponse.from_entity(version) for version in results],
        next_page_token=results.token,
    )


def _set_skill_tag(
    name: str,
    key: str,
    value: str,
    organization: str = "",
) -> dict[str, Any]:
    from mlflow.server.handlers import _get_tracking_store

    _validate_skill_path_identity(organization, name)
    _get_tracking_store().set_skill_tag(
        name=name,
        key=key,
        value=value,
        organization=organization,
    )
    return {}


def _delete_skill_tag(name: str, key: str, organization: str = "") -> dict[str, Any]:
    from mlflow.server.handlers import _get_tracking_store

    _validate_skill_path_identity(organization, name)
    _get_tracking_store().delete_skill_tag(
        name=name,
        key=key,
        organization=organization,
    )
    return {}


def _set_skill_version_tag(
    name: str,
    version: int,
    key: str,
    value: str,
    organization: str = "",
) -> dict[str, Any]:
    from mlflow.server.handlers import _get_tracking_store

    _validate_skill_path_identity(organization, name)
    _validate_skill_version(version)
    _get_tracking_store().set_skill_version_tag(
        name=name,
        version=version,
        key=key,
        value=value,
        organization=organization,
    )
    return {}


def _delete_skill_version_tag(
    name: str,
    version: int,
    key: str,
    organization: str = "",
) -> dict[str, Any]:
    from mlflow.server.handlers import _get_tracking_store

    _validate_skill_path_identity(organization, name)
    _validate_skill_version(version)
    _get_tracking_store().delete_skill_version_tag(
        name=name,
        version=version,
        key=key,
        organization=organization,
    )
    return {}


def _delete_skill_version(
    name: str,
    version: int,
    request: Request,
    organization: str = "",
) -> dict[str, Any]:
    from mlflow.server.handlers import _get_tracking_store

    _validate_skill_path_identity(organization, name)
    _validate_skill_version(version)
    _get_tracking_store().delete_skill_version(
        name=name,
        version=version,
        organization=organization,
        last_updated_by=getattr(request.state, "username", None),
    )
    return {}


def _delete_skill(name: str, organization: str = "") -> dict[str, Any]:
    from mlflow.server.skill_registry.deletion import delete_skill

    _validate_skill_path_identity(organization, name)
    try:
        from mlflow.server import auth
    except ImportError as e:
        missing_module = e if isinstance(e, ModuleNotFoundError) else e.__cause__
        if (
            not isinstance(missing_module, ModuleNotFoundError)
            or missing_module.name != "flask_wtf"
        ):
            raise
        # Basic auth is an optional extra. A plain MLflow server has no grants to clean.
        auth = None
    delete_skill(name=name, organization=organization)

    if auth is not None and auth.is_auth_enabled():
        auth.delete_skill_permissions(organization, name)
    return {}


def _update_skill_version(
    name: str,
    version: int,
    body: UpdateSkillVersionRequest,
    request: Request,
    organization: str = "",
) -> SkillVersionResponse:
    from mlflow.server.handlers import _get_tracking_store

    _validate_skill_path_identity(organization, name)
    _validate_skill_version(version)
    username = getattr(request.state, "username", None)
    status = body.status if "status" in body.model_fields_set else NOT_SET
    if username is not None and status == SkillStatus.DELETED.value:
        from mlflow.server import auth

        if (
            not auth.store.get_user(username).is_admin
            and not auth._get_skill_permission(organization, name, username).can_manage
        ):
            raise MlflowException("Permission denied", PERMISSION_DENIED)
    return SkillVersionResponse.from_entity(
        _get_tracking_store().update_skill_version(
            name=name,
            version=version,
            organization=organization,
            status=status,
            last_updated_by=username,
        )
    )


def _registration_from_metadata(
    metadata: bytes | str | dict[str, Any],
    username: str | None,
    name: str | None = None,
    organization: str | None = None,
) -> SkillVersionRegistration:
    if isinstance(metadata, (bytes, str)):
        try:
            metadata = json.loads(metadata)
        except (TypeError, ValueError) as e:
            raise MlflowException.invalid_parameter_value(
                "The 'metadata' part must contain a valid JSON object."
            ) from e
    try:
        registration = RegisterSkillRequest.model_validate(metadata)
    except ValueError as e:
        raise MlflowException.invalid_parameter_value(f"Invalid registration metadata: {e}") from e
    resolved_name = registration.name if name is None else name
    resolved_organization = registration.organization if organization is None else organization
    if resolved_name is None:
        raise MlflowException.invalid_parameter_value(
            "'name' must be provided explicitly for registration."
        )
    _validate_skill_path_identity(resolved_organization, resolved_name)
    return SkillVersionRegistration(
        name=resolved_name,
        organization=resolved_organization,
        source_type=registration.source_type,
        source=registration.source,
        ref=registration.ref,
        subpath=registration.subpath,
        digest=registration.digest,
        status=registration.status,
        created_by=username,
    )


def _get_multipart_request_size_limit() -> int:
    return get_max_upload_size() + _MULTIPART_REQUEST_OVERHEAD


def _validate_registration_metadata_size(metadata: bytes | str) -> None:
    metadata_size = len(metadata) if isinstance(metadata, bytes) else len(metadata.encode("utf-8"))
    if metadata_size > _MAX_REGISTRATION_METADATA_SIZE:
        raise HTTPException(
            status_code=413,
            detail=(
                "The registration metadata exceeds the maximum allowed size of "
                f"{_MAX_REGISTRATION_METADATA_SIZE} bytes."
            ),
        )


def _request_with_multipart_size_limit(request: Request) -> Request:
    max_bytes = _get_multipart_request_size_limit()
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "The multipart request body exceeds the maximum allowed size of "
                        f"{max_bytes} bytes."
                    ),
                )
        except ValueError:
            # The receive wrapper below remains the authoritative check for malformed or absent
            # Content-Length headers.
            pass

    received = 0
    receive: Receive = request.receive

    async def limited_receive() -> Message:
        nonlocal received
        message = await receive()
        if message["type"] == "http.request":
            received += len(message.get("body", b""))
            if received > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "The multipart request body exceeds the maximum allowed size of "
                        f"{max_bytes} bytes."
                    ),
                )
        return message

    return Request(request.scope, receive=limited_receive)


@asynccontextmanager
async def _parse_registration_request(
    request: Request,
    name: str | None = None,
    organization: str | None = None,
) -> AsyncIterator[tuple[SkillVersionRegistration, Any | None, bool]]:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    username = getattr(request.state, "username", None)
    if content_type == "application/json":
        try:
            metadata = await request.json()
        except ValueError as e:
            raise MlflowException.invalid_parameter_value(
                "The request body must contain a valid JSON object."
            ) from e
        yield (
            _registration_from_metadata(
                metadata,
                username,
                name=name,
                organization=organization,
            ),
            None,
            False,
        )
        return
    if content_type != "multipart/form-data":
        raise MlflowException.invalid_parameter_value(
            "Skill registration requires an application/json or multipart/form-data request."
        )

    request = _request_with_multipart_size_limit(request)
    async with request.form(
        max_files=_MAX_MULTIPART_FILES,
        max_fields=_MAX_MULTIPART_FIELDS,
    ) as form:
        parts = list(form.multi_items())
        part_names = [name for name, _ in parts]
        if len(parts) != 2 or set(part_names) != {"metadata", "content"}:
            raise MlflowException.invalid_parameter_value(
                "Multipart registration requires exactly one 'metadata' part "
                "and one 'content' part."
            )

        metadata = form.get("metadata")
        content = form.get("content")
        if isinstance(metadata, UploadFile):
            metadata = await metadata.read(_MAX_REGISTRATION_METADATA_SIZE + 1)
        if isinstance(metadata, (bytes, str)):
            _validate_registration_metadata_size(metadata)
        if not isinstance(content, UploadFile):
            raise MlflowException.invalid_parameter_value(
                "Multipart registration requires a 'content' file part."
            )
        yield (
            _registration_from_metadata(
                metadata or "",
                username,
                name=name,
                organization=organization,
            ),
            content.file,
            True,
        )


skill_registry_router = APIRouter(
    tags=["Skill Registry"],
    dependencies=[Depends(_ensure_tracking_server_enabled)],
)


@skill_registry_router.post(
    "", response_model=SkillResponse, dependencies=[Depends(_require_skill_create)]
)
def create_skill(body: CreateSkillRequest, request: Request) -> SkillResponse:
    from mlflow.server.handlers import _get_tracking_store

    _validate_skill_path_identity(body.organization, body.name)
    username = getattr(request.state, "username", None)
    skill = _get_tracking_store().create_skill(
        name=body.name,
        organization=body.organization,
        description=body.description,
        icons=_icons_to_entities(body.icons),
        created_by=username,
    )
    _grant_creator_if_new(request, body.organization, body.name, True)
    return SkillResponse.from_entity(skill)


def _update_skill(
    name: str,
    body: UpdateSkillRequest,
    request: Request,
    organization: str = "",
) -> SkillResponse:
    from mlflow.server.handlers import _get_tracking_store

    _validate_skill_path_identity(organization, name)
    username = getattr(request.state, "username", None)
    description = body.description if "description" in body.model_fields_set else NOT_SET
    icons = _icons_to_entities(body.icons) if "icons" in body.model_fields_set else NOT_SET
    skill = _get_tracking_store().update_skill(
        name=name,
        organization=organization,
        description=description,
        icons=icons,
        last_updated_by=username,
    )
    return SkillResponse.from_entity(skill)


@skill_registry_router.get("", response_model=SearchSkillsResponse)
def search_skills(
    request: Request,
    filter_string: str | None = Query(None),
    max_results: int = Query(100),
    order_by: list[str] | None = Query(None),
    page_token: str | None = Query(None),
) -> SearchSkillsResponse:
    from mlflow.server.handlers import _get_tracking_store

    include_skill_identities = None
    exclude_skill_identities = None
    if username := getattr(request.state, "username", None):
        from mlflow.server import auth

        include_skill_identities, exclude_skill_identities = auth.skill_search_permission_scope(
            username
        )
    results = _get_tracking_store().search_skills(
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by,
        page_token=page_token,
        include_skill_identities=include_skill_identities,
        exclude_skill_identities=exclude_skill_identities,
    )
    return SearchSkillsResponse(
        skills=[SkillResponse.from_entity(skill) for skill in results],
        next_page_token=results.token,
    )


@skill_registry_router.get(
    "/@{organization}/{name}",
    response_model=SkillResponse,
    dependencies=[Depends(_require_skill_read)],
)
def get_organization_skill(organization: str, name: SkillNamePath) -> SkillResponse:
    return _get_skill(name=name, organization=organization)


@skill_registry_router.get(
    "/@{organization}/{name}/versions",
    response_model=SearchSkillVersionsResponse,
    dependencies=[Depends(_require_skill_read)],
)
def search_organization_skill_versions(
    organization: str,
    name: SkillNamePath,
    filter_string: str | None = Query(None),
    max_results: int = Query(100),
    order_by: list[str] | None = Query(None),
    page_token: str | None = Query(None),
) -> SearchSkillVersionsResponse:
    return _search_skill_versions(
        name=name,
        organization=organization,
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by,
        page_token=page_token,
    )


@skill_registry_router.get(
    "/{name}/versions",
    response_model=SearchSkillVersionsResponse,
    dependencies=[Depends(_require_skill_read)],
)
def search_skill_versions(
    name: SkillNamePath,
    filter_string: str | None = Query(None),
    max_results: int = Query(100),
    order_by: list[str] | None = Query(None),
    page_token: str | None = Query(None),
) -> SearchSkillVersionsResponse:
    return _search_skill_versions(
        name=name,
        filter_string=filter_string,
        max_results=max_results,
        order_by=order_by,
        page_token=page_token,
    )


@skill_registry_router.post(
    "/@{organization}/{name}/tags", dependencies=[Depends(_require_skill_update)]
)
def set_organization_skill_tag(
    organization: str,
    name: SkillNamePath,
    body: SetTagRequest,
) -> dict[str, Any]:
    return _set_skill_tag(
        name=name,
        organization=organization,
        key=body.key,
        value=body.value,
    )


@skill_registry_router.delete(
    "/@{organization}/{name}/tags/{key:path}", dependencies=[Depends(_require_skill_update)]
)
def delete_organization_skill_tag(
    organization: str,
    name: SkillNamePath,
    key: str,
) -> dict[str, Any]:
    return _delete_skill_tag(name=name, organization=organization, key=key)


@skill_registry_router.post(
    "/@{organization}/{name}/versions/{version}/tags",
    dependencies=[Depends(_require_skill_update)],
)
def set_organization_skill_version_tag(
    organization: str,
    name: SkillNamePath,
    version: int,
    body: SetTagRequest,
) -> dict[str, Any]:
    return _set_skill_version_tag(
        name=name,
        organization=organization,
        version=version,
        key=body.key,
        value=body.value,
    )


@skill_registry_router.delete(
    "/@{organization}/{name}/versions/{version}/tags/{key:path}",
    dependencies=[Depends(_require_skill_manage)],
)
def delete_organization_skill_version_tag(
    organization: str,
    name: SkillNamePath,
    version: int,
    key: str,
) -> dict[str, Any]:
    return _delete_skill_version_tag(
        name=name,
        organization=organization,
        version=version,
        key=key,
    )


@skill_registry_router.post("/{name}/tags", dependencies=[Depends(_require_skill_update)])
def set_skill_tag(name: SkillNamePath, body: SetTagRequest) -> dict[str, Any]:
    return _set_skill_tag(name=name, key=body.key, value=body.value)


@skill_registry_router.delete(
    "/{name}/tags/{key:path}", dependencies=[Depends(_require_skill_update)]
)
def delete_skill_tag(name: SkillNamePath, key: str) -> dict[str, Any]:
    return _delete_skill_tag(name=name, key=key)


@skill_registry_router.post(
    "/{name}/versions/{version}/tags", dependencies=[Depends(_require_skill_update)]
)
def set_skill_version_tag(
    name: SkillNamePath,
    version: int,
    body: SetTagRequest,
) -> dict[str, Any]:
    return _set_skill_version_tag(
        name=name,
        version=version,
        key=body.key,
        value=body.value,
    )


@skill_registry_router.delete(
    "/{name}/versions/{version}/tags/{key:path}",
    dependencies=[Depends(_require_skill_manage)],
)
def delete_skill_version_tag(
    name: SkillNamePath,
    version: int,
    key: str,
) -> dict[str, Any]:
    return _delete_skill_version_tag(name=name, version=version, key=key)


@skill_registry_router.patch(
    "/{name}", response_model=SkillResponse, dependencies=[Depends(_require_skill_update)]
)
def update_skill(
    name: SkillNamePath,
    body: UpdateSkillRequest,
    request: Request,
) -> SkillResponse:
    return _update_skill(name=name, body=body, request=request)


@skill_registry_router.patch(
    "/@{organization}/{name}",
    response_model=SkillResponse,
    dependencies=[Depends(_require_skill_update)],
)
def update_organization_skill(
    organization: str,
    name: SkillNamePath,
    body: UpdateSkillRequest,
    request: Request,
) -> SkillResponse:
    return _update_skill(
        name=name,
        organization=organization,
        body=body,
        request=request,
    )


def _get_skill(name: str, organization: str = "") -> SkillResponse:
    from mlflow.server.handlers import _get_tracking_store

    _validate_skill_path_identity(organization, name)
    return SkillResponse.from_entity(
        _get_tracking_store().get_skill(name=name, organization=organization)
    )


@skill_registry_router.get(
    "/{name}", response_model=SkillResponse, dependencies=[Depends(_require_skill_read)]
)
def get_skill(name: SkillNamePath) -> SkillResponse:
    return _get_skill(name=name)


@skill_registry_router.post(
    "/{name}/versions",
    response_model=SkillVersionResponse,
    openapi_extra=_SKILL_VERSION_CREATE_OPENAPI_EXTRA,
)
async def create_skill_version(
    name: SkillNamePath,
    request: Request,
) -> SkillVersionResponse:
    return await _create_skill_version(name=name, request=request)


@skill_registry_router.post(
    "/@{organization}/{name}/versions",
    response_model=SkillVersionResponse,
    openapi_extra=_SKILL_VERSION_CREATE_OPENAPI_EXTRA,
)
async def create_organization_skill_version(
    organization: str,
    name: SkillNamePath,
    request: Request,
) -> SkillVersionResponse:
    return await _create_skill_version(
        name=name,
        organization=organization,
        request=request,
    )


@skill_registry_router.post(
    "/register",
    response_model=SkillVersionResponse,
    openapi_extra=_REGISTER_SKILL_OPENAPI_EXTRA,
)
async def register_skill(request: Request) -> SkillVersionResponse:
    async with _parse_registration_request(request) as (registration, content, multipart):
        parent_missing = _authorize_registration(
            request, registration.organization, registration.name
        )
        version = await asyncio.to_thread(
            register_skill_version,
            registration,
            content=content,
            multipart=multipart,
            expected_parent_exists=(
                not parent_missing if getattr(request.state, "username", None) else None
            ),
        )
    _grant_creator_if_new(request, registration.organization, registration.name, parent_missing)
    return SkillVersionResponse.from_entity(version)


@skill_registry_router.post(
    "/bulk-register",
    response_model=BulkRegisterSkillsResponse,
)
async def bulk_register_skills(
    body: BulkRegisterSkillsRequest,
    request: Request,
) -> BulkRegisterSkillsResponse:
    username = getattr(request.state, "username", None)
    registrations = []
    new_parents = []
    for skill in body.skills:
        _validate_skill_path_identity(body.organization, skill.name)
        if _authorize_registration(request, body.organization, skill.name):
            new_parents.append(skill.name)
        registrations.append(
            SkillVersionRegistration(
                name=skill.name,
                organization=body.organization,
                source_type=skill.source_type,
                source=skill.source,
                ref=skill.ref,
                subpath=skill.subpath,
                digest=skill.digest,
                status=skill.status,
                created_by=username,
            )
        )

    versions = await asyncio.to_thread(
        bulk_register_skill_versions,
        registrations,
        expected_parent_exists=(
            {
                registration.name: registration.name not in new_parents
                for registration in registrations
            }
            if username
            else None
        ),
    )
    if new_parents and username:
        from mlflow.server import auth
        from mlflow.server.handlers import _get_tracking_store

        # A parent may have been created by another request after preflight. Resolve
        # ownership in one query, then commit the creator grants in one auth transaction.
        parents = _get_tracking_store().search_skills(
            max_results=len(new_parents),
            include_skill_identities=[(body.organization, name) for name in new_parents],
        )
        owned_names = [parent.name for parent in parents if parent.created_by == username]
        auth.grant_manage_for_created_skills(username, body.organization, owned_names)
    return BulkRegisterSkillsResponse(
        skill_versions=[SkillVersionResponse.from_entity(version) for version in versions]
    )


@skill_registry_router.get(
    "/{name}/versions/{version}",
    response_model=SkillVersionResponse,
    dependencies=[Depends(_require_skill_read)],
)
def get_skill_version(name: SkillNamePath, version: int) -> SkillVersionResponse:
    return _get_skill_version(name=name, version=version)


@skill_registry_router.get(
    "/@{organization}/{name}/versions/{version}",
    response_model=SkillVersionResponse,
    dependencies=[Depends(_require_skill_read)],
)
def get_organization_skill_version(
    organization: str,
    name: SkillNamePath,
    version: int,
) -> SkillVersionResponse:
    return _get_skill_version(name=name, organization=organization, version=version)


@skill_registry_router.get(
    "/{name}/aliases/{alias}",
    response_model=SkillVersionResponse,
    dependencies=[Depends(_require_skill_read)],
)
def get_skill_version_by_alias(name: SkillNamePath, alias: str) -> SkillVersionResponse:
    return _get_skill_version_by_alias(name=name, alias=alias)


@skill_registry_router.get(
    "/@{organization}/{name}/aliases/{alias}",
    response_model=SkillVersionResponse,
    dependencies=[Depends(_require_skill_read)],
)
def get_organization_skill_version_by_alias(
    organization: str,
    name: SkillNamePath,
    alias: str,
) -> SkillVersionResponse:
    return _get_skill_version_by_alias(
        name=name,
        organization=organization,
        alias=alias,
    )


@skill_registry_router.post("/{name}/aliases", dependencies=[Depends(_require_skill_update)])
def set_skill_alias(name: SkillNamePath, body: SetSkillAliasRequest) -> dict[str, Any]:
    return _set_skill_alias(
        name=name,
        alias=body.alias,
        version=body.version,
    )


@skill_registry_router.post(
    "/@{organization}/{name}/aliases", dependencies=[Depends(_require_skill_update)]
)
def set_organization_skill_alias(
    organization: str,
    name: SkillNamePath,
    body: SetSkillAliasRequest,
) -> dict[str, Any]:
    return _set_skill_alias(
        name=name,
        organization=organization,
        alias=body.alias,
        version=body.version,
    )


@skill_registry_router.delete(
    "/{name}/aliases/{alias}", dependencies=[Depends(_require_skill_manage)]
)
def delete_skill_alias(name: SkillNamePath, alias: str) -> dict[str, Any]:
    return _delete_skill_alias(name=name, alias=alias)


@skill_registry_router.delete(
    "/@{organization}/{name}/aliases/{alias}", dependencies=[Depends(_require_skill_manage)]
)
def delete_organization_skill_alias(
    organization: str,
    name: SkillNamePath,
    alias: str,
) -> dict[str, Any]:
    return _delete_skill_alias(name=name, organization=organization, alias=alias)


@skill_registry_router.delete(
    "/{name}/versions/{version}", dependencies=[Depends(_require_skill_manage)]
)
def delete_skill_version(name: SkillNamePath, version: int, request: Request) -> dict[str, Any]:
    return _delete_skill_version(name=name, version=version, request=request)


@skill_registry_router.delete(
    "/@{organization}/{name}/versions/{version}", dependencies=[Depends(_require_skill_manage)]
)
def delete_organization_skill_version(
    organization: str,
    name: SkillNamePath,
    version: int,
    request: Request,
) -> dict[str, Any]:
    return _delete_skill_version(
        name=name,
        organization=organization,
        version=version,
        request=request,
    )


@skill_registry_router.delete("/{name}", dependencies=[Depends(_require_skill_manage)])
def delete_skill(name: SkillNamePath) -> dict[str, Any]:
    return _delete_skill(name=name)


@skill_registry_router.delete(
    "/@{organization}/{name}", dependencies=[Depends(_require_skill_manage)]
)
def delete_organization_skill(organization: str, name: SkillNamePath) -> dict[str, Any]:
    return _delete_skill(name=name, organization=organization)


@skill_registry_router.patch(
    "/{name}/versions/{version}",
    response_model=SkillVersionResponse,
    dependencies=[Depends(_require_skill_update)],
)
def update_skill_version(
    name: SkillNamePath,
    version: int,
    body: UpdateSkillVersionRequest,
    request: Request,
) -> SkillVersionResponse:
    return _update_skill_version(
        name=name,
        version=version,
        body=body,
        request=request,
    )


@skill_registry_router.patch(
    "/@{organization}/{name}/versions/{version}",
    response_model=SkillVersionResponse,
    dependencies=[Depends(_require_skill_update)],
)
def update_organization_skill_version(
    organization: str,
    name: SkillNamePath,
    version: int,
    body: UpdateSkillVersionRequest,
    request: Request,
) -> SkillVersionResponse:
    return _update_skill_version(
        name=name,
        organization=organization,
        version=version,
        body=body,
        request=request,
    )
