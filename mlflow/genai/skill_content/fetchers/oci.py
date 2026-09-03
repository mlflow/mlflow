from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from mlflow.genai.skill_content.archive import extract_skill_archive
from mlflow.genai.skill_content.errors import (
    error_code_for_http_status,
    invalid_content,
    source_unavailable,
)
from mlflow.genai.skill_content.paths import (
    canonical_relative_path,
    ensure_within,
    is_under_subpath,
    normalize_subpath,
    tree_size,
)
from mlflow.protos.databricks_pb2 import TEMPORARILY_UNAVAILABLE, UNAUTHENTICATED

_MANIFEST_MEDIA_TYPES = (
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
)
_INDEX_MEDIA_TYPES = (
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
)
_DEFAULT_PLATFORM = ("linux", "amd64")
_TITLE_ANNOTATION = "org.opencontainers.image.title"
_DOCKER_HUB_HOSTS = ("docker.io", "index.docker.io")
_DOCKER_HUB_REGISTRY = "registry-1.docker.io"
_DOCKER_HUB_AUTH_KEY = "https://index.docker.io/v1/"
_IDENTITY_TOKEN_USERNAME = "<token>"
_REQUEST_TIMEOUT_SECONDS = 60
_CREDENTIAL_HELPER_TIMEOUT_SECONDS = 30
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_CHALLENGE_PARAM_PATTERN = re.compile(r'(\w+)="([^"]*)"')
_SHA256_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class ImageReference:
    registry: str
    repository: str
    reference: str

    @property
    def display(self) -> str:
        separator = "@" if self.reference.startswith("sha256:") else ":"
        return f"{self.registry}/{self.repository}{separator}{self.reference}"


def _validate_digest(value: Any, what: str) -> str:
    if not isinstance(value, str) or _SHA256_DIGEST_PATTERN.match(value) is None:
        raise invalid_content(
            f"{what} must be a sha256 digest of 64 hex characters, got {value!r}."
        )
    return value


def parse_image_reference(image: str) -> ImageReference:
    """
    Split ``registry/repository:tag`` or ``registry/repository@digest`` into its parts.

    Docker Hub conventions apply when no registry is given: ``docker.io`` with a ``library/``
    prefix for single-segment names. The tag defaults to ``latest``.
    """
    value = image.strip()
    if not value:
        raise invalid_content("OCI image reference must not be empty.")
    if "@" in value:
        name, reference = value.split("@", 1)
        _validate_digest(reference, f"OCI image digest in '{image}'")
    else:
        head, _, tail = value.rpartition("/")
        if ":" in tail:
            tail, reference = tail.split(":", 1)
        else:
            reference = "latest"
        name = f"{head}/{tail}" if head else tail
    if not name:
        raise invalid_content(f"OCI image reference '{image}' has no repository.")
    first, _, rest = name.partition("/")
    if rest and ("." in first or ":" in first or first == "localhost"):
        registry = first
        repository = rest
    else:
        registry = "docker.io"
        repository = name
    if registry in _DOCKER_HUB_HOSTS:
        registry = _DOCKER_HUB_REGISTRY
        if "/" not in repository:
            repository = f"library/{repository}"
    if not repository or not reference or "" in repository.split("/"):
        raise invalid_content(f"OCI image reference '{image}' is malformed.")
    return ImageReference(registry=registry, repository=repository, reference=reference)


def _docker_config_path() -> Path:
    config_dir = os.environ.get("DOCKER_CONFIG") or os.path.join(os.path.expanduser("~"), ".docker")
    return Path(config_dir) / "config.json"


def _credentials_from_auths(auths: dict[str, Any], keys: list[str]) -> tuple[str, str] | None:
    for key in keys:
        entry = auths.get(key)
        if not isinstance(entry, dict):
            continue
        username = entry.get("username")
        password = entry.get("password")
        if isinstance(username, str) and username and isinstance(password, str):
            return username, password
        encoded = entry.get("auth")
        if not isinstance(encoded, str) or not encoded:
            continue
        try:
            decoded = base64.b64decode(encoded).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue
        username, _, password = decoded.partition(":")
        return username, password
    return None


def _run_credential_helper(helper: str, server: str) -> tuple[str, str] | None:
    """
    Ask ``docker-credential-<helper> get`` for ``server``; any failure means no credentials.

    A ``Username`` of ``<token>`` marks an identity token, which is exchanged for a registry
    token through the OAuth2 refresh-token grant instead of Basic authentication.
    """
    executable = shutil.which(f"docker-credential-{helper}")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [executable, "get"],
            input=server,
            capture_output=True,
            text=True,
            timeout=_CREDENTIAL_HELPER_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    try:
        payload = json.loads(completed.stdout)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    username = payload.get("Username")
    secret = payload.get("Secret")
    if not isinstance(secret, str) or not secret:
        return None
    return (username if isinstance(username, str) else ""), secret


def _load_docker_credentials(registry: str) -> tuple[str, str] | None:
    """
    Resolve credentials for ``registry`` the way the Docker CLI does.

    Order: a matching ``auths`` entry, then the registry-specific ``credHelpers`` entry, then
    the global ``credsStore`` helper. Missing or unreadable config means no credentials.
    """
    config_path = _docker_config_path()
    if not config_path.is_file():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    keys = [registry, f"https://{registry}", f"http://{registry}"]
    server = registry
    if registry == _DOCKER_HUB_REGISTRY:
        keys.append(_DOCKER_HUB_AUTH_KEY)
        server = _DOCKER_HUB_AUTH_KEY
    auths = config.get("auths")
    if isinstance(auths, dict) and (found := _credentials_from_auths(auths, keys)):
        return found
    helpers = config.get("credHelpers")
    helper = None
    if isinstance(helpers, dict):
        helper = next((helpers[k] for k in keys if isinstance(helpers.get(k), str)), None)
    if helper is None and isinstance(config.get("credsStore"), str):
        helper = config["credsStore"]
    if helper:
        return _run_credential_helper(helper, server)
    return None


def _parse_challenge(header: str) -> tuple[str, dict[str, str]]:
    scheme, _, params = header.strip().partition(" ")
    return scheme.lower(), dict(_CHALLENGE_PARAM_PATTERN.findall(params))


def _read_bounded(response: requests.Response, cap: int, what: str) -> bytes:
    declared = response.headers.get("Content-Length")
    if declared and declared.isdigit() and int(declared) > cap:
        raise invalid_content(f"{what} is {declared} bytes, larger than the {cap} byte cap.")
    chunks = []
    total = 0
    for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
        total += len(chunk)
        if total > cap:
            raise invalid_content(f"{what} is larger than the {cap} byte cap.")
        chunks.append(chunk)
    return b"".join(chunks)


def _parse_json(body: bytes, what: str) -> dict[str, Any]:
    try:
        parsed = json.loads(body)
    except ValueError as e:
        raise invalid_content(f"{what} is not valid JSON: {e}")
    if not isinstance(parsed, dict):
        raise invalid_content(f"{what} must be a JSON object.")
    return parsed


class RegistryClient:
    """Minimal OCI Distribution v2 client with Bearer and Basic authentication."""

    def __init__(self, registry: str, *, session: requests.Session | None = None):
        self.registry = registry
        host = registry.rsplit(":", 1)[0] if re.search(r":\d+$", registry) else registry
        insecure = host in ("localhost", "127.0.0.1", "::1")
        self.base_url = f"{'http' if insecure else 'https'}://{registry}"
        self._session = session or requests.Session()
        self._credentials = _load_docker_credentials(registry)
        self._token: str | None = None

    def _headers(self, accept: str | None) -> dict[str, str]:
        headers = {}
        if accept:
            headers["Accept"] = accept
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _request_token(self, realm: str, params: dict[str, str]) -> str | None:
        query = {k: v for k, v in params.items() if k in ("service", "scope")}
        if self._credentials and self._credentials[0] == _IDENTITY_TOKEN_USERNAME:
            form = {
                "grant_type": "refresh_token",
                "refresh_token": self._credentials[1],
                "client_id": "mlflow",
                **query,
            }
            response = self._session.post(realm, data=form, timeout=_REQUEST_TIMEOUT_SECONDS)
        else:
            response = self._session.get(
                realm, params=query, auth=self._credentials, timeout=_REQUEST_TIMEOUT_SECONDS
            )
        with response:
            if response.status_code >= 400:
                return None
            body = _parse_json(
                _read_bounded(response, _MAX_MANIFEST_BYTES, "Token response"), "Token response"
            )
        token = body.get("token") or body.get("access_token")
        return token if isinstance(token, str) and token else None

    def _acquire_token(self, challenge: str) -> bool:
        scheme, params = _parse_challenge(challenge)
        if scheme == "basic":
            if self._credentials is None:
                return False
            self._token = None
            self._session.auth = self._credentials
            return True
        if scheme != "bearer" or "realm" not in params:
            return False
        if (token := self._request_token(params["realm"], params)) is None:
            return False
        self._token = token
        return True

    def get(
        self, path: str, *, accept: str | None = None, stream: bool = False
    ) -> requests.Response:
        url = f"{self.base_url}{path}"
        try:
            response = self._session.get(
                url,
                headers=self._headers(accept),
                stream=stream,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code == 401 and (
                challenge := response.headers.get("WWW-Authenticate")
            ):
                response.close()
                if not self._acquire_token(challenge):
                    raise source_unavailable(
                        url,
                        "authentication required and no usable credentials were found",
                        error_code=UNAUTHENTICATED,
                    )
                response = self._session.get(
                    url,
                    headers=self._headers(accept),
                    stream=stream,
                    timeout=_REQUEST_TIMEOUT_SECONDS,
                )
        except requests.RequestException as e:
            raise source_unavailable(url, str(e), error_code=TEMPORARILY_UNAVAILABLE)
        if response.status_code >= 400:
            response.close()
            raise source_unavailable(
                url,
                f"HTTP {response.status_code} {response.reason}",
                error_code=error_code_for_http_status(response.status_code),
            )
        return response


def _fetch_manifest(client: RegistryClient, ref: ImageReference) -> tuple[dict[str, Any], str]:
    path = f"/v2/{ref.repository}/manifests/{ref.reference}"
    with client.get(path, accept=", ".join(_MANIFEST_MEDIA_TYPES), stream=True) as response:
        body = _read_bounded(response, _MAX_MANIFEST_BYTES, f"OCI manifest for '{ref.display}'")
        content_type = response.headers.get("Content-Type", "")
    if ref.reference.startswith("sha256:"):
        actual = f"sha256:{hashlib.sha256(body).hexdigest()}"
        if actual != ref.reference:
            raise invalid_content(
                f"OCI manifest for '{ref.display}' did not match its digest (got {actual})."
            )
    manifest = _parse_json(body, f"OCI manifest for '{ref.display}'")
    media_type = str(manifest.get("mediaType") or content_type).split(";")[0].strip()
    return manifest, media_type


def _select_manifest(client: RegistryClient, ref: ImageReference) -> dict[str, Any]:
    """
    Resolve ``ref`` to a single image manifest.

    A multi-platform index is resolved one level deep to its ``linux/amd64`` entry (or the
    first entry without a platform); nested indexes are rejected. Manifests requested by
    digest are verified against that digest.
    """
    manifest, media_type = _fetch_manifest(client, ref)
    if media_type in _INDEX_MEDIA_TYPES or "manifests" in manifest:
        chosen = None
        for candidate in manifest.get("manifests", []):
            if not isinstance(candidate, dict):
                continue
            platform = candidate.get("platform") or {}
            if (
                not platform
                or (platform.get("os"), platform.get("architecture")) == _DEFAULT_PLATFORM
            ):
                chosen = candidate
                break
        if chosen is None:
            raise invalid_content(
                f"OCI index for '{ref.display}' has no manifest for "
                f"{'/'.join(_DEFAULT_PLATFORM)}; pin a single-platform manifest by digest."
            )
        digest = _validate_digest(chosen.get("digest"), f"OCI index entry in '{ref.display}'")
        child_ref = ImageReference(ref.registry, ref.repository, digest)
        child, child_type = _fetch_manifest(client, child_ref)
        if child_type in _INDEX_MEDIA_TYPES or "manifests" in child:
            raise invalid_content(
                f"OCI index for '{ref.display}' nests another index; not supported."
            )
        manifest = child
    if not isinstance(manifest.get("layers"), list):
        raise invalid_content(f"OCI manifest for '{ref.display}' contains no layers.")
    return manifest


def _download_blob(
    client: RegistryClient,
    ref: ImageReference,
    layer: dict[str, Any],
    target: Path,
    *,
    max_bytes: int,
) -> None:
    digest = _validate_digest(layer.get("digest"), f"OCI layer digest in '{ref.display}'")
    declared = layer.get("size")
    if isinstance(declared, int) and declared > max_bytes:
        raise invalid_content(
            f"OCI layer {digest} is {declared} bytes, which exceeds the skill content size "
            f"limit of {max_bytes} bytes."
        )
    hasher = hashlib.sha256()
    received = 0
    url = f"/v2/{ref.repository}/blobs/{digest}"
    try:
        with client.get(url, stream=True) as response, open(target, "wb") as out:
            for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                received += len(chunk)
                if received > max_bytes:
                    raise invalid_content(
                        f"OCI layer {digest} exceeds the skill content size limit of "
                        f"{max_bytes} bytes."
                    )
                hasher.update(chunk)
                out.write(chunk)
    except requests.RequestException as e:
        raise source_unavailable(
            f"{client.base_url}{url}", str(e), error_code=TEMPORARILY_UNAVAILABLE
        )
    if f"sha256:{hasher.hexdigest()}" != digest:
        raise invalid_content(f"OCI layer {digest} did not match its digest after download.")


def _merge_tree(source: Path, dest: Path) -> None:
    """Move an extracted layer into ``dest``; later layers replace files but never change kinds."""
    for item in source.iterdir():
        target = dest / item.name
        if item.is_dir():
            if target.exists() and not target.is_dir():
                raise invalid_content(
                    f"OCI layers disagree about '{item.name}': one has a file, another a directory."
                )
            target.mkdir(exist_ok=True)
            _merge_tree(item, target)
        else:
            if target.is_dir():
                raise invalid_content(
                    f"OCI layers disagree about '{item.name}': one has a directory, another a file."
                )
            os.replace(item, target)


def _place_file_layer(blob: Path, layer: dict[str, Any], dest: Path, prefix: str | None) -> int:
    """Write a non-tar layer as the file named by its title annotation; returns bytes placed."""
    title = (layer.get("annotations") or {}).get(_TITLE_ANNOTATION)
    media_type = layer.get("mediaType")
    if not isinstance(title, str) or not title:
        raise invalid_content(
            f"OCI layer {layer.get('digest')} with media type '{media_type}' has no "
            f"'{_TITLE_ANNOTATION}' annotation, so its file name is unknown."
        )
    relative = canonical_relative_path(title)
    if relative is None:
        raise invalid_content(f"OCI layer title '{title}' is not a valid file path.")
    if not is_under_subpath(relative, prefix):
        blob.unlink()
        return 0
    target = dest.joinpath(*relative.split("/"))
    ensure_within(dest, target)
    if target.is_dir():
        raise invalid_content(f"OCI layer title '{title}' names an existing directory.")
    target.parent.mkdir(parents=True, exist_ok=True)
    size = blob.stat().st_size
    os.replace(blob, target)
    return size


def fetch_oci(image: str, dest: Path, *, max_bytes: int, subpath: str | None = None) -> Path:
    """
    Pull the layers of ``image`` into ``dest``.

    Layers whose media type is a tar (optionally gzip-compressed) are extracted with the skill
    archive rules; any other layer is written as a single file named by its
    ``org.opencontainers.image.title`` annotation, which is how ORAS publishes plain files.
    Multi-platform indexes resolve to ``linux/amd64``. Credentials come from the Docker config
    file: ``auths`` entries, then ``credHelpers`` or ``credsStore`` helpers. The decompressed
    limit applies to the content at ``subpath``; each layer download is also bounded by the
    limit on the wire.
    """
    ref = parse_image_reference(image)
    client = RegistryClient(ref.registry)
    manifest = _select_manifest(client, ref)
    prefix = normalize_subpath(subpath)
    dest.mkdir(parents=True, exist_ok=True)
    remaining = max_bytes
    with tempfile.TemporaryDirectory(prefix="mlflow-oci-layer-") as tmp:
        tmp_path = Path(tmp)
        for index, layer in enumerate(manifest["layers"]):
            if not isinstance(layer, dict):
                raise invalid_content(f"OCI manifest for '{ref.display}' has a malformed layer.")
            media_type = str(layer.get("mediaType", ""))
            blob = tmp_path / f"layer-{index}"
            _download_blob(client, ref, layer, blob, max_bytes=max_bytes)
            if "tar" in media_type:
                if "zstd" in media_type:
                    raise invalid_content(
                        f"OCI layer media type '{media_type}' (zstd) is not supported."
                    )
                if remaining <= 0:
                    raise invalid_content(
                        f"OCI image '{ref.display}' exceeds the skill content size limit of "
                        f"{max_bytes} bytes."
                    )
                extracted = tmp_path / f"extracted-{index}"
                extract_skill_archive(
                    blob,
                    extracted,
                    max_bytes=remaining,
                    compressed="gzip" in media_type,
                    subpath=prefix,
                )
                remaining -= tree_size(extracted)
                _merge_tree(extracted, dest)
            else:
                remaining -= _place_file_layer(blob, layer, dest, prefix)
            if remaining < 0:
                raise invalid_content(
                    f"OCI image '{ref.display}' exceeds the skill content size limit of "
                    f"{max_bytes} bytes."
                )
    return dest
