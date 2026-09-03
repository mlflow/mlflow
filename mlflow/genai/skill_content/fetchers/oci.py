from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from mlflow.genai.skill_content.archive import extract_skill_archive
from mlflow.genai.skill_content.errors import invalid_content, source_unavailable
from mlflow.genai.skill_content.paths import normalize_subpath

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
_REQUEST_TIMEOUT_SECONDS = 60
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_CHALLENGE_PARAM_PATTERN = re.compile(r'(\w+)="([^"]*)"')


@dataclass(frozen=True)
class ImageReference:
    registry: str
    repository: str
    reference: str

    @property
    def display(self) -> str:
        separator = "@" if self.reference.startswith("sha256:") else ":"
        return f"{self.registry}/{self.repository}{separator}{self.reference}"


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
        if not reference.startswith("sha256:"):
            raise invalid_content(f"OCI image digest must be a sha256 digest: '{image}'.")
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


def _load_docker_credentials(registry: str) -> tuple[str, str] | None:
    """Read ``auths`` from the Docker config file for ``registry``; helpers are not consulted."""
    config_dir = os.environ.get("DOCKER_CONFIG") or os.path.join(os.path.expanduser("~"), ".docker")
    config_path = Path(config_dir) / "config.json"
    if not config_path.is_file():
        return None
    try:
        auths = json.loads(config_path.read_text(encoding="utf-8")).get("auths") or {}
    except (OSError, ValueError):
        return None
    candidates = [registry, f"https://{registry}", f"http://{registry}"]
    if registry == _DOCKER_HUB_REGISTRY:
        candidates.append(_DOCKER_HUB_AUTH_KEY)
    for key in candidates:
        entry = auths.get(key)
        if not isinstance(entry, dict):
            continue
        if entry.get("username") and entry.get("password") is not None:
            return entry["username"], entry["password"]
        if encoded := entry.get("auth"):
            try:
                decoded = base64.b64decode(encoded).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                continue
            username, _, password = decoded.partition(":")
            return username, password
    return None


def _parse_challenge(header: str) -> tuple[str, dict[str, str]]:
    scheme, _, params = header.strip().partition(" ")
    return scheme.lower(), dict(_CHALLENGE_PARAM_PATTERN.findall(params))


class RegistryClient:
    """Minimal OCI Distribution v2 client with Bearer and Basic authentication."""

    def __init__(self, registry: str, *, session: requests.Session | None = None):
        self.registry = registry
        insecure = registry.startswith(("localhost", "127.0.0.1"))
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

    def _acquire_token(self, challenge: str) -> bool:
        scheme, params = _parse_challenge(challenge)
        if scheme == "basic":
            if self._credentials is None:
                return False
            username, password = self._credentials
            self._token = None
            self._session.auth = (username, password)
            return True
        if scheme != "bearer" or "realm" not in params:
            return False
        query = {k: v for k, v in params.items() if k in ("service", "scope")}
        auth = self._credentials or None
        response = self._session.get(
            params["realm"], params=query, auth=auth, timeout=_REQUEST_TIMEOUT_SECONDS
        )
        if response.status_code >= 400:
            return False
        body = response.json()
        token = body.get("token") or body.get("access_token")
        if not token:
            return False
        self._token = token
        return True

    def get(
        self, path: str, *, accept: str | None = None, stream: bool = False
    ) -> requests.Response:
        url = f"{self.base_url}{path}"
        response = self._session.get(
            url, headers=self._headers(accept), stream=stream, timeout=_REQUEST_TIMEOUT_SECONDS
        )
        if response.status_code == 401 and (challenge := response.headers.get("WWW-Authenticate")):
            response.close()
            if not self._acquire_token(challenge):
                raise source_unavailable(
                    url, "authentication required and no usable credentials were found"
                )
            response = self._session.get(
                url,
                headers=self._headers(accept),
                stream=stream,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        if response.status_code >= 400:
            response.close()
            raise source_unavailable(url, f"HTTP {response.status_code} {response.reason}")
        return response


def _select_manifest(client: RegistryClient, ref: ImageReference) -> dict[str, Any]:
    path = f"/v2/{ref.repository}/manifests/{ref.reference}"
    response = client.get(path, accept=", ".join(_MANIFEST_MEDIA_TYPES))
    if len(response.content) > _MAX_MANIFEST_BYTES:
        raise invalid_content(f"OCI manifest for '{ref.display}' is unexpectedly large.")
    manifest = response.json()
    media_type = manifest.get("mediaType") or response.headers.get("Content-Type", "")
    if media_type.split(";")[0] in _INDEX_MEDIA_TYPES or "manifests" in manifest:
        chosen = None
        for candidate in manifest.get("manifests", []):
            platform = candidate.get("platform") or {}
            if (
                not platform
                or (platform.get("os"), platform.get("architecture")) == _DEFAULT_PLATFORM
            ):
                chosen = candidate
                break
        if chosen is None or not chosen.get("digest"):
            raise invalid_content(
                f"OCI index for '{ref.display}' has no manifest for "
                f"{'/'.join(_DEFAULT_PLATFORM)}; pin a single-platform manifest by digest."
            )
        return _select_manifest(
            client, ImageReference(ref.registry, ref.repository, chosen["digest"])
        )
    if "layers" not in manifest:
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
    digest = layer.get("digest") or ""
    if not digest.startswith("sha256:"):
        raise invalid_content(f"OCI layer in '{ref.display}' has an unsupported digest '{digest}'.")
    declared = layer.get("size")
    if isinstance(declared, int) and declared > max_bytes:
        raise invalid_content(
            f"OCI layer {digest} is {declared} bytes, which exceeds the skill content size "
            f"limit of {max_bytes} bytes."
        )
    hasher = hashlib.sha256()
    received = 0
    with client.get(f"/v2/{ref.repository}/blobs/{digest}", stream=True) as response:
        with open(target, "wb") as out:
            for chunk in response.iter_content(chunk_size=_DOWNLOAD_CHUNK_SIZE):
                received += len(chunk)
                if received > max_bytes:
                    raise invalid_content(
                        f"OCI layer {digest} exceeds the skill content size limit of "
                        f"{max_bytes} bytes."
                    )
                hasher.update(chunk)
                out.write(chunk)
    if f"sha256:{hasher.hexdigest()}" != digest:
        raise invalid_content(f"OCI layer {digest} did not match its digest after download.")


def _merge_tree(source: Path, dest: Path) -> None:
    for item in source.iterdir():
        target = dest / item.name
        if item.is_dir():
            target.mkdir(exist_ok=True)
            _merge_tree(item, target)
        else:
            shutil.move(str(item), str(target))


def fetch_oci(image: str, dest: Path, *, max_bytes: int) -> Path:
    """
    Pull the layers of ``image`` into ``dest``.

    Layers whose media type is a tar (optionally gzip-compressed) are extracted with the skill
    archive rules; any other layer is written as a single file named by its
    ``org.opencontainers.image.title`` annotation, which is how ORAS publishes plain files.
    Multi-platform indexes resolve to ``linux/amd64``. Credentials come from the Docker config
    file's ``auths`` entries; credential helpers are not consulted.
    """
    ref = parse_image_reference(image)
    client = RegistryClient(ref.registry)
    manifest = _select_manifest(client, ref)
    dest.mkdir(parents=True, exist_ok=True)
    remaining = max_bytes
    with tempfile.TemporaryDirectory(prefix="mlflow-oci-layer-") as tmp:
        tmp_path = Path(tmp)
        for index, layer in enumerate(manifest["layers"]):
            media_type = str(layer.get("mediaType", ""))
            blob = tmp_path / f"layer-{index}"
            _download_blob(client, ref, layer, blob, max_bytes=remaining)
            if "tar" in media_type:
                if "zstd" in media_type:
                    raise invalid_content(
                        f"OCI layer media type '{media_type}' (zstd) is not supported."
                    )
                extracted = tmp_path / f"extracted-{index}"
                extract_skill_archive(
                    blob, extracted, max_bytes=remaining, compressed="gzip" in media_type
                )
                remaining -= sum(f.stat().st_size for f in extracted.rglob("*") if f.is_file())
                _merge_tree(extracted, dest)
            else:
                title = (layer.get("annotations") or {}).get(_TITLE_ANNOTATION)
                if not title:
                    raise invalid_content(
                        f"OCI layer {layer.get('digest')} with media type '{media_type}' has no "
                        f"'{_TITLE_ANNOTATION}' annotation, so its file name is unknown."
                    )
                relative = normalize_subpath(title)
                if relative is None:
                    raise invalid_content(f"OCI layer title '{title}' is not a valid file path.")
                remaining -= blob.stat().st_size
                target = dest.joinpath(*relative.split("/"))
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(blob), str(target))
            if remaining < 0:
                raise invalid_content(
                    f"OCI image '{ref.display}' exceeds the skill content size limit of "
                    f"{max_bytes} bytes."
                )
    return dest
