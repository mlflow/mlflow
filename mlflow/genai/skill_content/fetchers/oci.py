from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

from mlflow.genai.skill_content.archive import (
    DecompressionBudget,
    _iter_tar_members,
    _member_relative_path,
    _open_tar,
    default_decompression_budget,
    extract_skill_archive,
)
from mlflow.genai.skill_content.errors import (
    error_code_for_http_status,
    invalid_content,
    source_unavailable,
)
from mlflow.genai.skill_content.fetchers.zip import _discard_redirect_body, _no_auth
from mlflow.genai.skill_content.paths import (
    canonical_relative_path,
    collect_tree,
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
_HELPER_NOT_FOUND_MARKER = "credentials not found"
_REQUEST_TIMEOUT_SECONDS = 60
_CREDENTIAL_HELPER_TIMEOUT_SECONDS = 30
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
# Docker's own layer limit is 127; anything near this is not a skill image.
_MAX_LAYERS = 256
_LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")
# OCI layers mark deletions of lower-layer content with these entries (the "whiteout" rules
# of the image spec): ``.wh.<name>`` removes ``<name>``; ``.wh..wh..opq`` inside a directory
# removes everything the lower layers put there.
_WHITEOUT_PREFIX = ".wh."
_OPAQUE_WHITEOUT = ".wh..wh..opq"
_CHALLENGE_PARAM_PATTERN = re.compile(r'(\w+)="([^"]*)"')
_SHA256_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
# The Docker/OCI reference grammar. Every part is interpolated into a request URL, so anything
# outside it (``#``, ``?``, ``..``, spaces) must be refused rather than sent.
_DOMAIN_COMPONENT = r"(?:[A-Za-z0-9]|[A-Za-z0-9][A-Za-z0-9-]*[A-Za-z0-9])"
_REGISTRY_PATTERN = re.compile(
    rf"^(?:{_DOMAIN_COMPONENT}(?:\.{_DOMAIN_COMPONENT})*|\[[0-9A-Fa-f:.]+\])(?::[0-9]{{1,5}})?$"
)
_REPOSITORY_COMPONENT_PATTERN = re.compile(r"^[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*$")
_TAG_PATTERN = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class ImageReference:
    registry: str
    repository: str
    reference: str

    @property
    def display(self) -> str:
        separator = "@" if self.reference.startswith("sha256:") else ":"
        return f"{self.registry}/{self.repository}{separator}{self.reference}"


def _optional_object(container: dict[str, Any], key: str, what: str) -> dict[str, Any]:
    """A nested JSON object from registry-controlled data, or empty when absent."""
    value = container.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise invalid_content(f"{what} has a malformed '{key}' field; expected an object.")
    return value


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
        # ``repo:tag@digest`` is valid; the digest wins and the tag is informational.
        head, _, tail = name.rpartition("/")
        tail = tail.split(":", 1)[0]
        name = f"{head}/{tail}" if head else tail
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
    if _REGISTRY_PATTERN.match(registry) is None:
        raise invalid_content(f"OCI image reference '{image}' has an invalid registry host.")
    for component in repository.split("/"):
        if _REPOSITORY_COMPONENT_PATTERN.match(component) is None:
            raise invalid_content(
                f"OCI image reference '{image}' has an invalid repository component "
                f"'{component}'; components are lowercase alphanumerics separated by "
                "'.', '_', or '-'."
            )
    if not reference.startswith("sha256:") and _TAG_PATTERN.match(reference) is None:
        raise invalid_content(f"OCI image reference '{image}' has an invalid tag '{reference}'.")
    return ImageReference(registry=registry, repository=repository, reference=reference)


def _docker_config_path() -> Path:
    if config_dir := os.environ.get("DOCKER_CONFIG"):
        return Path(config_dir) / "config.json"
    return Path.home() / ".docker" / "config.json"


def _credential_files() -> list[Path]:
    """
    Credential files in lookup order: an explicit ``REGISTRY_AUTH_FILE``, the Docker config,
    then the default ``containers/auth.json`` locations Podman and other container runtimes
    write to. All share the Docker config format.
    """
    files = []
    if explicit := os.environ.get("REGISTRY_AUTH_FILE"):
        files.append(Path(explicit))
    files.append(_docker_config_path())
    if runtime_dir := os.environ.get("XDG_RUNTIME_DIR"):
        files.append(Path(runtime_dir) / "containers" / "auth.json")
    config_home = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    files.append(Path(config_home) / "containers" / "auth.json")
    return files


def _read_config_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return config if isinstance(config, dict) else None


def _is_loopback(netloc: str) -> bool:
    """Whether ``host`` or ``host:port`` (IPv6 in brackets) names the local machine."""
    return (urlsplit(f"//{netloc}").hostname or "") in _LOOPBACK_HOSTS


def _credentials_from_auths(auths: dict[str, Any], keys: list[str]) -> tuple[str, str] | None:
    for key in keys:
        entry = auths.get(key)
        if not isinstance(entry, dict):
            continue
        # `docker login` with an identity token stores the refresh token here, next to an
        # `auth` entry that carries only the username; the token is what authenticates.
        identity_token = entry.get("identitytoken")
        if isinstance(identity_token, str) and identity_token:
            return _IDENTITY_TOKEN_USERNAME, identity_token
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


class _CredentialHelperError(Exception):
    """The helper could not be run or gave an unusable answer (as opposed to "not found")."""


def _run_credential_helper(helper: str, server: str) -> tuple[str, str] | None:
    """
    Ask ``docker-credential-<helper> get`` for ``server``.

    ``None`` means the helper answered that it holds no credentials for ``server``, which is
    final. A helper that is not installed, fails to run, or prints something unusable raises
    ``_CredentialHelperError`` so the caller can fall back the way the Docker CLI does. A
    ``Username`` of ``<token>`` marks an identity token, which is exchanged for a registry
    token through the OAuth2 refresh-token grant instead of Basic authentication.
    """
    executable = shutil.which(f"docker-credential-{helper}")
    if executable is None:
        raise _CredentialHelperError(f"docker-credential-{helper} is not on PATH")
    try:
        completed = subprocess.run(
            [executable, "get"],
            input=server,
            capture_output=True,
            text=True,
            timeout=_CREDENTIAL_HELPER_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise _CredentialHelperError(str(e))
    if completed.returncode != 0:
        # The helpers print this exact phrase when the store has no entry; anything else is
        # a failure of the helper itself.
        if _HELPER_NOT_FOUND_MARKER in (completed.stdout + completed.stderr).lower():
            return None
        raise _CredentialHelperError(f"exit status {completed.returncode}")
    try:
        payload = json.loads(completed.stdout)
    except ValueError:
        raise _CredentialHelperError("output is not JSON")
    if not isinstance(payload, dict):
        raise _CredentialHelperError("output is not a JSON object")
    username = payload.get("Username")
    secret = payload.get("Secret")
    if not isinstance(secret, str) or not secret:
        return None
    return (username if isinstance(username, str) else ""), secret


def _load_docker_credentials(registry: str) -> tuple[str, str] | None:
    """
    Resolve credentials for ``registry`` from the container tooling's config files.

    Files are consulted in ``_credential_files`` order and the first one that yields
    credentials wins. Within a file the Docker CLI's rules apply: a configured helper (the
    registry-specific ``credHelpers`` entry, else the global ``credsStore``) decides, and an
    inline ``auths`` entry applies only when no helper is configured or the helper cannot be
    run. Missing or unreadable files are skipped.
    """
    for path in _credential_files():
        if (config := _read_config_file(path)) is None:
            continue
        if found := _config_credentials(config, registry):
            return found
    return None


def _config_credentials(config: dict[str, Any], registry: str) -> tuple[str, str] | None:
    keys = [registry, f"https://{registry}", f"http://{registry}"]
    server = registry
    if registry == _DOCKER_HUB_REGISTRY:
        keys.append(_DOCKER_HUB_AUTH_KEY)
        server = _DOCKER_HUB_AUTH_KEY
    helpers = config.get("credHelpers")
    helper = None
    if isinstance(helpers, dict):
        helper = next((helpers[k] for k in keys if isinstance(helpers.get(k), str)), None)
    if helper is None and isinstance(config.get("credsStore"), str):
        helper = config["credsStore"]
    if helper:
        # A working helper is authoritative, even when it has nothing: Docker pulls
        # anonymously rather than using an inline entry the user replaced with a helper. Only
        # a helper that cannot be run at all leaves the inline entry in play.
        try:
            return _run_credential_helper(helper, server)
        except _CredentialHelperError:
            pass
    auths = config.get("auths")
    if isinstance(auths, dict):
        return _credentials_from_auths(auths, keys)
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


class _RegistrySession(requests.Session):
    """
    A session whose credentials come only from the registry client.

    ``requests`` attaches a matching ``~/.netrc`` entry to any request made without explicit
    auth and again to every redirect target. Registry credentials are resolved from the
    container tooling's config instead, so ambient netrc entries must never be sent; proxy
    and CA handling from the environment is kept.
    """

    def rebuild_auth(self, prepared_request, response):
        # Same as the base implementation minus the netrc lookup: the Authorization header is
        # dropped when a redirect leaves the registry host and never replaced.
        if self.should_strip_auth(response.request.url, prepared_request.url):
            prepared_request.headers.pop("Authorization", None)


class RegistryClient:
    """Minimal OCI Distribution v2 client with Bearer and Basic authentication."""

    def __init__(self, registry: str, *, session: requests.Session | None = None):
        self.registry = registry
        # Plain http is only ever used for a registry on the local machine.
        self.base_url = f"{'http' if _is_loopback(registry) else 'https'}://{registry}"
        self._session = session or _RegistrySession()
        # Blob downloads are commonly redirected to a CDN; `requests` would otherwise buffer
        # every redirect body in full before following it.
        if _discard_redirect_body not in self._session.hooks["response"]:
            self._session.hooks["response"].append(_discard_redirect_body)
        self._credentials = _load_docker_credentials(registry)
        self._token: str | None = None

    def _from_registry(self, response: requests.Response) -> bool:
        """Whether ``response`` came from the registry itself rather than a redirect target."""
        parts = urlsplit(response.url or "")
        return f"{parts.scheme}://{parts.netloc}".lower() == self.base_url.lower()

    def _auth(self):
        # Basic credentials are set on the session once the registry asks for them; until
        # then, and for Bearer requests, an explicit no-op handler keeps `requests` from
        # falling back to netrc.
        return self._session.auth or _no_auth

    def _headers(self, accept: str | None) -> dict[str, str]:
        headers = {}
        if accept:
            headers["Accept"] = accept
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _request_token(self, realm: str, params: dict[str, str]) -> str | None:
        # The realm is chosen by the registry; credentials only ever travel to it over TLS,
        # except for a token endpoint on the local machine.
        parts = urlsplit(realm)
        if parts.scheme != "https" and not (parts.scheme == "http" and _is_loopback(parts.netloc)):
            raise source_unavailable(
                realm,
                "the registry's token endpoint must use https",
                error_code=UNAUTHENTICATED,
            )
        query = {k: v for k, v in params.items() if k in ("service", "scope")}
        if self._credentials and self._credentials[0] == _IDENTITY_TOKEN_USERNAME:
            form = {
                "grant_type": "refresh_token",
                "refresh_token": self._credentials[1],
                "client_id": "mlflow",
                **query,
            }
            response = self._session.post(
                realm,
                data=form,
                auth=_no_auth,
                stream=True,
                allow_redirects=False,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        else:
            response = self._session.get(
                realm,
                params=query,
                auth=self._credentials or _no_auth,
                stream=True,
                allow_redirects=False,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        with response:
            if response.is_redirect:
                # Following it would resend the credentials to a URL the https check above
                # never saw.
                raise source_unavailable(
                    realm,
                    "the registry's token endpoint redirected; credentials are only sent to "
                    "the realm the registry named",
                    error_code=UNAUTHENTICATED,
                )
            if response.status_code >= 400:
                # Keep the distinction between bad credentials, a scope the account lacks,
                # and an outage of the token service; each maps to its own error code.
                detail = f"token endpoint returned HTTP {response.status_code} {response.reason}"
                if self._credentials is None:
                    detail += "; no credentials were found for this registry"
                raise source_unavailable(
                    realm, detail, error_code=error_code_for_http_status(response.status_code)
                )
            body = _parse_json(
                _read_bounded(response, _MAX_MANIFEST_BYTES, "Token response"), "Token response"
            )
        token = body.get("token") or body.get("access_token")
        return token if isinstance(token, str) and token else None

    def _acquire_token(self, challenge: str) -> bool:
        scheme, params = _parse_challenge(challenge)
        if scheme == "basic":
            # An identity token is only good for the refresh-token grant; sending it as a
            # Basic password would disclose it to a registry that cannot use it.
            if self._credentials is None or self._credentials[0] == _IDENTITY_TOKEN_USERNAME:
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
                auth=self._auth(),
                stream=stream,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code == 401 and (
                challenge := response.headers.get("WWW-Authenticate")
            ):
                response.close()
                # Blob pulls are redirected to content servers that must never get to choose
                # where the registry credentials are sent; only the registry's own challenge
                # is honored.
                if not self._from_registry(response):
                    raise source_unavailable(
                        url,
                        f"a redirect target ('{response.url}') requested authentication",
                        error_code=UNAUTHENTICATED,
                    )
                if not self._acquire_token(challenge):
                    raise source_unavailable(
                        url,
                        "authentication required and no usable credentials were found",
                        error_code=UNAUTHENTICATED,
                    )
                response = self._session.get(
                    url,
                    headers=self._headers(accept),
                    auth=self._auth(),
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
    try:
        with client.get(path, accept=", ".join(_MANIFEST_MEDIA_TYPES), stream=True) as response:
            body = _read_bounded(response, _MAX_MANIFEST_BYTES, f"OCI manifest for '{ref.display}'")
            content_type = response.headers.get("Content-Type", "")
    except requests.RequestException as e:
        # The body streams after the request itself succeeded, so a drop here is not caught
        # by the client.
        raise source_unavailable(
            f"{client.base_url}{path}", str(e), error_code=TEMPORARILY_UNAVAILABLE
        )
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
        entries = manifest.get("manifests")
        if not isinstance(entries, list):
            raise invalid_content(f"OCI index for '{ref.display}' has a malformed manifest list.")
        # The exact platform wins wherever it appears; a platform-independent entry is only
        # a fallback for indexes that have no linux/amd64 manifest at all.
        chosen = None
        fallback = None
        for candidate in entries:
            if not isinstance(candidate, dict):
                continue
            platform = _optional_object(
                candidate, "platform", f"OCI index entry in '{ref.display}'"
            )
            if (platform.get("os"), platform.get("architecture")) == _DEFAULT_PLATFORM:
                chosen = candidate
                break
            if not platform and fallback is None:
                fallback = candidate
        if chosen is None:
            chosen = fallback
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
    layers = manifest.get("layers")
    if not isinstance(layers, list):
        raise invalid_content(f"OCI manifest for '{ref.display}' contains no layers.")
    if len(layers) > _MAX_LAYERS:
        raise invalid_content(
            f"OCI manifest for '{ref.display}' has {len(layers)} layers; the maximum is "
            f"{_MAX_LAYERS}."
        )
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


def _clear_directory(path: Path) -> None:
    if not path.is_dir():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


@dataclass
class _LayerEntries:
    """Canonical paths of a tar layer's entries: content paths and whiteout markers."""

    paths: list[str]
    whiteouts: list[str]


def _scan_layer(blob: Path, *, compressed: bool, work: DecompressionBudget) -> _LayerEntries:
    entries = _LayerEntries(paths=[], whiteouts=[])
    with _open_tar(blob, compressed=compressed, work=work) as (tar, bounded):
        for member in _iter_tar_members(tar, bounded):
            relative = _member_relative_path(member.name)
            if relative is None:
                continue
            if relative.rsplit("/", 1)[-1].startswith(_WHITEOUT_PREFIX):
                entries.whiteouts.append(relative)
            else:
                entries.paths.append(relative)
    return entries


class _MergedPaths:
    """
    Canonical paths merged into the destination so far, checked for collisions across layers.

    Each layer is validated on its own by the archive rules, but two layers can still carry
    paths that differ only by letter case. On a case-insensitive filesystem the merge would
    silently overwrite one with the other, so the check happens here, before the filesystem
    is touched, and the same image is accepted or rejected on every platform.
    """

    def __init__(self):
        self._by_key: dict[str, str] = {}

    @staticmethod
    def _key(path: str) -> str:
        return unicodedata.normalize("NFC", path).casefold()

    def add(self, path: str) -> None:
        parts = path.split("/")
        for depth in range(1, len(parts) + 1):
            candidate = unicodedata.normalize("NFC", "/".join(parts[:depth]))
            key = candidate.casefold()
            previous = self._by_key.get(key)
            if previous is not None and previous != candidate:
                raise invalid_content(
                    f"OCI layers carry paths '{previous}' and '{candidate}' that differ only "
                    "by letter case; the skill tree is ambiguous on case-insensitive "
                    "filesystems."
                )
            self._by_key[key] = candidate

    def remove(self, path: str) -> None:
        """Forget ``path`` and everything beneath it."""
        key = self._key(path)
        for existing in list(self._by_key):
            if existing == key or existing.startswith(key + "/"):
                del self._by_key[existing]

    def clear_children(self, directory: str) -> None:
        """Forget everything beneath ``directory`` (the whole tree when empty)."""
        marker = self._key(directory) + "/" if directory else ""
        for existing in list(self._by_key):
            if existing.startswith(marker) and existing != self._key(directory):
                del self._by_key[existing]


def _apply_whiteouts(
    whiteouts: list[str], dest: Path, prefix: str | None, merged: _MergedPaths
) -> None:
    """
    Delete lower-layer content named by ``whiteouts`` from ``dest`` and from ``merged``.

    The subpath filter only extracts entries beneath ``prefix``, so a deletion of the
    subpath itself or of one of its ancestors is applied by clearing the whole selected tree;
    deletions inside the subpath remove just that path. Whiteouts elsewhere are irrelevant.
    """
    content_root = dest if prefix is None else dest.joinpath(*prefix.split("/"))
    for entry in whiteouts:
        directory, _, name = entry.rpartition("/")
        if name == _OPAQUE_WHITEOUT:
            # Applies to ``directory`` (the layer root when empty).
            covers_root = prefix is not None and (
                not directory or is_under_subpath(prefix, directory)
            )
            if covers_root or (prefix is None and not directory):
                _clear_directory(content_root)
                merged.clear_children(prefix or "")
            elif is_under_subpath(directory, prefix):
                _clear_directory(dest.joinpath(*directory.split("/")))
                merged.clear_children(directory)
            continue
        target_name = name[len(_WHITEOUT_PREFIX) :]
        if not target_name:
            # Segment validation already refuses names ending in a period, so a bare marker
            # cannot reach this point; the guard keeps the deletion root safe regardless.
            raise invalid_content(f"OCI layer whiteout '{entry}' names nothing to delete.")
        deleted = f"{directory}/{target_name}" if directory else target_name
        if prefix is not None and is_under_subpath(prefix, deleted):
            _clear_directory(content_root)
            merged.clear_children(prefix)
        elif is_under_subpath(deleted, prefix):
            target = dest.joinpath(*deleted.split("/"))
            ensure_within(dest, target)
            _remove_path(target)
            merged.remove(deleted)


def _merge_tree(source: Path, dest: Path) -> None:
    """
    Move an extracted layer into ``dest``; later layers replace files but never change kinds.

    Whiteout markers were applied before the merge and are never copied.
    """
    for item in source.iterdir():
        if item.name.startswith(_WHITEOUT_PREFIX):
            continue
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


def _place_file_layer(
    blob: Path, layer: dict[str, Any], dest: Path, prefix: str | None, merged: _MergedPaths
) -> int:
    """Write a non-tar layer as the file named by its title annotation; returns bytes placed."""
    media_type = layer.get("mediaType")
    annotations = _optional_object(layer, "annotations", f"OCI layer {layer.get('digest')}")
    title = annotations.get(_TITLE_ANNOTATION)
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
    merged.add(relative)
    target = dest.joinpath(*relative.split("/"))
    ensure_within(dest, target)
    if target.is_dir():
        raise invalid_content(f"OCI layer title '{title}' names an existing directory.")
    parent = target.parent
    while not parent.exists():
        parent = parent.parent
    if not parent.is_dir():
        # An earlier layer put a file where this title needs a directory.
        raise invalid_content(
            f"OCI layers disagree about '{parent.relative_to(dest).as_posix()}': one has a "
            f"file, another needs a directory for '{title}'."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    size = blob.stat().st_size
    os.replace(blob, target)
    return size


def fetch_oci(
    image: str, dest: Path, *, scratch: Path, max_bytes: int, subpath: str | None = None
) -> Path:
    """
    Pull the layers of ``image`` into ``dest``.

    Layers whose media type is a tar (optionally gzip-compressed) are extracted with the skill
    archive rules and merged in order, applying the image spec's whiteout deletions to the
    lower layers first; any other layer is written as a single file named by its
    ``org.opencontainers.image.title`` annotation, which is how ORAS publishes plain files.
    Multi-platform indexes resolve to ``linux/amd64``. Credentials come from the Docker config
    or container runtime auth files: a configured ``credHelpers`` or ``credsStore`` helper
    first, inline ``auths`` entries only without a working helper. The decompressed
    limit applies to the content at ``subpath``; each layer download is also bounded by the
    limit on the wire, the decompression work across all layers by a multiple of it, and only
    one layer at a time occupies ``scratch``.
    """
    ref = parse_image_reference(image)
    client = RegistryClient(ref.registry)
    manifest = _select_manifest(client, ref)
    prefix = normalize_subpath(subpath)
    dest.mkdir(parents=True, exist_ok=True)
    remaining = max_bytes
    # One decompression budget for the whole image: every pass over every layer draws on it.
    work = default_decompression_budget(max_bytes)
    merged = _MergedPaths()
    tmp_path = scratch / "oci-layers"
    tmp_path.mkdir(parents=True, exist_ok=True)
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
            compressed = "gzip" in media_type
            extracted = tmp_path / f"extracted-{index}"
            extract_skill_archive(
                blob,
                extracted,
                max_bytes=remaining,
                compressed=compressed,
                subpath=prefix,
                decompression_budget=work,
            )
            remaining -= tree_size(extracted)
            entries = _scan_layer(blob, compressed=compressed, work=work)
            _apply_whiteouts(entries.whiteouts, dest, prefix, merged)
            for path in entries.paths:
                if is_under_subpath(path, prefix):
                    merged.add(path)
            _merge_tree(extracted, dest)
            # Scratch space is bounded by one layer at a time, not by the whole image.
            blob.unlink()
            shutil.rmtree(extracted)
        else:
            remaining -= _place_file_layer(blob, layer, dest, prefix, merged)
        if remaining < 0:
            raise invalid_content(
                f"OCI image '{ref.display}' exceeds the skill content size limit of "
                f"{max_bytes} bytes."
            )
    # Cross-layer case collisions were rejected before the merge; this re-walk is a final
    # consistency check of the assembled tree under the shared path rules.
    collect_tree(dest)
    return dest
