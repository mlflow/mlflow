"""
Policy for artifact URIs that make MLflow connect to the host named in the URI, or store files
under a local directory the client chose rather than the server's own storage.

The artifact repositories for ``ftp``, ``sftp``, ``hdfs``, ``viewfs``, ``r2``, ``b2`` and
``abfss`` connect to the host and port in the URI, and so do ``http``, ``https`` and
``mlflow-artifacts://host`` whenever the URI is not proxied by the tracking server. A tracking
server that builds one of these repositories from a location a client stored would therefore
connect to any host the client named (GHSA-mr9f-g8qf-4w4j). Inside a server process, and in the
job subprocesses that inherit its environment, only the hosts of the server's own
``--default-artifact-root`` and ``--artifacts-destination`` are connected to, unless the operator
allows a scheme via ``MLFLOW_ALLOWED_HOST_ADDRESSED_ARTIFACT_SCHEMES``.

A ``file:`` or scheme-less location names a directory on the server's own filesystem. An
experiment ``artifact_location`` set to an arbitrary local directory would otherwise make the
server create run directories and write uploaded artifacts there, outside the storage it was
configured with. Inside a server process, a local location that a client submits for a new
experiment must resolve inside the server's own ``--default-artifact-root`` or
``--artifacts-destination`` directory, unless the operator opts out via
``MLFLOW_ALLOW_LOCAL_ARTIFACT_LOCATION_OUTSIDE_ROOT``. Stored local locations are not re-checked
when used: experiments created directly against the backend store, such as runs logged to a local
database before a server is started on it, legitimately keep local locations of their own.

Trusted hosts and local roots, and the server-process marker, come from the environment variables
that ``mlflow server`` exports to its workers. A deployment that runs the WSGI app directly must
export the same variables to get the same trusted hosts and roots and registry-level enforcement.
"""

import os
import pathlib
import urllib.parse

from mlflow.environment_variables import (
    MLFLOW_ALLOW_LOCAL_ARTIFACT_LOCATION_OUTSIDE_ROOT,
    MLFLOW_ALLOWED_HOST_ADDRESSED_ARTIFACT_SCHEMES,
)
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.file_utils import local_file_uri_to_path
from mlflow.utils.os import is_windows
from mlflow.utils.uri import get_uri_scheme

# Schemes that name a directory on the server's own filesystem. ``get_uri_scheme`` reports a
# scheme-less path (``/data`` or ``data``) and a Windows drive path (``C:\\data``) as ``""``.
_LOCAL_ARTIFACT_SCHEMES = frozenset({"", "file"})

# Mirrors `mlflow.server.constants`, which the store layer cannot import because the `mlflow.server`
# package pulls in Flask. `tests/store/artifact/test_host_policy.py` pins the two to stay equal.
_SERVER_ARTIFACT_ROOT_ENV_VAR = "_MLFLOW_SERVER_ARTIFACT_ROOT"
_SERVER_ARTIFACTS_DESTINATION_ENV_VAR = "_MLFLOW_SERVER_ARTIFACT_DESTINATION"
_SERVER_SERVE_ARTIFACTS_ENV_VAR = "_MLFLOW_SERVER_SERVE_ARTIFACTS"

# `r2`, `b2` and `abfss` take their endpoint from the authority after the last "@"
# (`<bucket>@<endpoint>` and `<filesystem>@<account>.<domain>`), which `urlparse` reports as the
# hostname. `wasbs` is left out because its repository only accepts a letters-and-digits account
# under `*.blob.core.*` (`AzureBlobArtifactRepository.parse_wasbs_uri`).
HOST_ADDRESSED_ARTIFACT_SCHEMES = frozenset({
    "ftp",
    "sftp",
    "hdfs",
    "viewfs",
    "http",
    "https",
    "mlflow-artifacts",
    "r2",
    "b2",
    "abfss",
})
# `viewfs` reaches the same Hadoop service as `hdfs`, and `mlflow-artifacts://host:port` resolves
# to an `http(s)://host:port` request, so these schemes are compared as one family.
_SCHEME_FAMILIES = {"viewfs": "hdfs", "https": "http", "mlflow-artifacts": "http"}
# An omitted port connects to the scheme's well-known port, so `https://host` and `http://host`
# are different targets while `http://host` and `http://host:80` are the same one. A portless
# `mlflow-artifacts://host` resolves to whichever of `http` or `https` the tracking transport
# uses, so it stands for both well-known ports. `hdfs`/`viewfs` have no entry: a portless HDFS URI
# takes its port from the Hadoop client configuration, which cannot be read here, so it only
# matches a configured root that is also portless (fail closed).
_DEFAULT_PORTS: dict[str, tuple[int, ...]] = {
    "ftp": (21,),
    "sftp": (22,),
    "http": (80,),
    "https": (443,),
    "mlflow-artifacts": (80, 443),
    "r2": (443,),
    "b2": (443,),
    "abfss": (443,),
}
_MALFORMED = "malformed"

Target = tuple[str, str, int | None]


def host_addressed_uri_targets(uri: str) -> frozenset[Target] | None:
    """
    Return the ``(scheme family, hostname, port)`` targets that ``uri`` may make MLflow connect
    to, or None when the URI does not select a host. A URI yields more than one target only when
    its effective port cannot be known from the URI alone.
    """
    scheme = get_uri_scheme(uri)
    if scheme not in HOST_ADDRESSED_ARTIFACT_SCHEMES:
        return None
    parsed = urllib.parse.urlparse(uri)
    if scheme == "mlflow-artifacts" and not parsed.netloc:
        # `mlflow-artifacts:/path` resolves against the tracking server itself.
        return None
    if "\\" in parsed.netloc or any(c.isspace() for c in parsed.netloc):
        # HTTP clients end the authority at a backslash where `urlparse` does not, so the
        # hostname reported here would not be the one contacted. Never treat it as trusted.
        return frozenset({(_MALFORMED, parsed.netloc, None)})
    try:
        port = parsed.port
    except ValueError:
        return frozenset({(_MALFORMED, parsed.netloc, None)})
    family = _SCHEME_FAMILIES.get(scheme, scheme)
    # A missing host is kept as "" rather than None: `ftp:///path` still connects, to localhost.
    hostname = (parsed.hostname or "").lower()
    ports = (port,) if port is not None else _DEFAULT_PORTS.get(scheme, (None,))
    return frozenset((family, hostname, p) for p in ports)


def _trusted_targets() -> set[Target]:
    targets = set()
    for env_var in (_SERVER_ARTIFACT_ROOT_ENV_VAR, _SERVER_ARTIFACTS_DESTINATION_ENV_VAR):
        if uri := os.environ.get(env_var):
            targets |= host_addressed_uri_targets(uri) or set()
    return targets


def rejected_host_addressed_scheme(uri: str) -> str | None:
    """
    Return the scheme of ``uri`` when a tracking server must not connect to the host it names,
    or None when the URI is acceptable: it does not select a host, its scheme is allowed by
    ``MLFLOW_ALLOWED_HOST_ADDRESSED_ARTIFACT_SCHEMES``, or it targets the host of the server's
    default artifact root or artifacts destination.
    """
    targets = host_addressed_uri_targets(uri)
    if targets is None:
        return None
    scheme = get_uri_scheme(uri)
    if scheme in {s.lower() for s in MLFLOW_ALLOWED_HOST_ADDRESSED_ARTIFACT_SCHEMES.get()}:
        return None
    if targets & _trusted_targets():
        return None
    return scheme


def host_addressed_rejection_message(
    scheme: str, *, uri: str | None = None, field_name: str | None = None
) -> str:
    """
    Build the error for a rejected URI, phrased for a request field when ``field_name`` is given
    and for a stored location otherwise.
    """
    reason = (
        f"the '{scheme}' scheme addresses a host other than the tracking server's configured "
        "artifact storage, so the server would connect to the host named in the URI"
    )
    hint = (
        f"Set the {MLFLOW_ALLOWED_HOST_ADDRESSED_ARTIFACT_SCHEMES.name} environment variable on "
        "the server to allow it."
    )
    if field_name is not None:
        return f"'{field_name}' cannot use the '{scheme}' scheme: {reason}. {hint}"
    return f"The tracking server does not connect to artifact location '{uri}': {reason}. {hint}"


def validate_artifact_uri_host(artifact_uri: str) -> None:
    """Raise when a tracking server must not connect to the host named in ``artifact_uri``."""
    scheme = rejected_host_addressed_scheme(artifact_uri)
    if scheme is None:
        return
    raise MlflowException(
        host_addressed_rejection_message(scheme, uri=artifact_uri),
        error_code=INVALID_PARAMETER_VALUE,
    )


def _local_path_of(uri: str) -> str | None:
    """
    Return the local filesystem path that ``uri`` resolves to for ``LocalArtifactRepository``, or
    None when ``uri`` is not a local location. The conversion mirrors the repository's own
    ``local_file_uri_to_path`` so the path checked here is the path later written to, including a
    ``file://host/path`` form whose authority the repository drops on POSIX.
    """
    scheme = get_uri_scheme(uri)
    if scheme not in _LOCAL_ARTIFACT_SCHEMES:
        return None
    if is_windows() and scheme == "file" and urllib.parse.urlparse(uri).netloc:
        # ``file://server/share`` is a UNC network location on Windows, not a server-local path.
        return None
    return local_file_uri_to_path(uri)


def _trusted_local_roots() -> set[pathlib.Path]:
    roots = set()
    for env_var in (_SERVER_ARTIFACT_ROOT_ENV_VAR, _SERVER_ARTIFACTS_DESTINATION_ENV_VAR):
        if uri := os.environ.get(env_var):
            if (path := _local_path_of(uri)) is not None:
                roots.add(pathlib.Path(path).resolve())
    return roots


def rejected_local_artifact_path(uri: str) -> str | None:
    """
    Return the local filesystem path of ``uri`` when a tracking server must not accept it as a
    client-submitted artifact location, or None when the URI is acceptable: it is not a local
    location, the server is not the one that configured the storage, the operator opted out via
    ``MLFLOW_ALLOW_LOCAL_ARTIFACT_LOCATION_OUTSIDE_ROOT``, or it resolves inside one of the
    server's own local artifact roots.

    A server that stores only on remote backends has no local root, so every local location is
    rejected. A process that is not acting for a configured server cannot know the roots and so
    enforces nothing, matching the host-addressed policy.
    """
    if not is_tracking_server_process():
        return None
    if MLFLOW_ALLOW_LOCAL_ARTIFACT_LOCATION_OUTSIDE_ROOT.get():
        return None
    path = _local_path_of(uri)
    if path is None:
        return None
    resolved = pathlib.Path(path).resolve()
    for root in _trusted_local_roots():
        if resolved == root or resolved.is_relative_to(root):
            return None
    return path


def local_artifact_rejection_message(path: str, *, field_name: str) -> str:
    """Build the error for a rejected local location submitted in the request field."""
    return (
        f"'{field_name}' cannot use a local path outside the artifact root: the local path "
        f"'{path}' is outside the tracking server's configured artifact storage. Set the "
        f"{MLFLOW_ALLOW_LOCAL_ARTIFACT_LOCATION_OUTSIDE_ROOT.name} environment variable on the "
        "server to allow it."
    )


def is_tracking_server_process() -> bool:
    """
    The tracking server exports its default artifact root to its workers and to the job
    subprocesses they spawn, so its presence marks code running on the server's behalf.
    """
    return _SERVER_ARTIFACT_ROOT_ENV_VAR in os.environ


def enforce_server_artifact_uri_host_policy(artifact_uri: str) -> None:
    """
    Raise when called on behalf of a tracking server for a URI whose host the server must not
    connect to. Client processes are unaffected.
    """
    if is_tracking_server_process():
        validate_artifact_uri_host(artifact_uri)
