"""
Policy for artifact URIs whose scheme makes MLflow connect to the host named in the URI.

The artifact repositories for ``ftp``, ``sftp``, ``hdfs`` and ``viewfs`` connect to the host and
port in the URI, and so do ``http``, ``https`` and ``mlflow-artifacts://host`` whenever the URI is
not proxied by the tracking server. A tracking server that builds one of these repositories from
a location a client stored would therefore connect to any host the client named
(GHSA-mr9f-g8qf-4w4j). Inside a server process, and in the job subprocesses that inherit its
environment, only the hosts of the server's own ``--default-artifact-root`` and
``--artifacts-destination`` are connected to, unless the operator allows a scheme via
``MLFLOW_ALLOWED_HOST_ADDRESSED_ARTIFACT_SCHEMES``.
"""

import os
import urllib.parse

from mlflow.environment_variables import MLFLOW_ALLOWED_HOST_ADDRESSED_ARTIFACT_SCHEMES
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.uri import get_uri_scheme

# Mirrors the names in `mlflow.server.constants`, which cannot be imported from the store layer.
_SERVER_ARTIFACT_ROOT_ENV_VAR = "_MLFLOW_SERVER_ARTIFACT_ROOT"
_SERVER_ARTIFACTS_DESTINATION_ENV_VAR = "_MLFLOW_SERVER_ARTIFACT_DESTINATION"

HOST_ADDRESSED_ARTIFACT_SCHEMES = frozenset({
    "ftp",
    "sftp",
    "hdfs",
    "viewfs",
    "http",
    "https",
    "mlflow-artifacts",
})
# `viewfs` reaches the same Hadoop service as `hdfs`, and `mlflow-artifacts://host:port` resolves
# to an `http(s)://host:port` request, so these schemes are compared as one family.
_SCHEME_FAMILIES = {"viewfs": "hdfs", "https": "http", "mlflow-artifacts": "http"}
# An omitted port connects to the scheme's well-known port, so `https://host` and `http://host`
# are different targets while `http://host` and `http://host:80` are the same one.
_DEFAULT_PORTS = {"ftp": 21, "sftp": 22, "http": 80, "https": 443}
_MALFORMED = "malformed"


def host_addressed_uri_target(uri: str) -> tuple[str, str, int | None] | None:
    """
    Return ``(scheme family, hostname, port)`` for the host that ``uri`` would make MLflow
    connect to, or None when the URI does not select a host.
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
        return (_MALFORMED, parsed.netloc, None)
    try:
        port = parsed.port
    except ValueError:
        return (_MALFORMED, parsed.netloc, None)
    if port is None:
        port = _DEFAULT_PORTS.get(scheme)
    # A missing host is kept as "" rather than None: `ftp:///path` still connects, to localhost.
    return (_SCHEME_FAMILIES.get(scheme, scheme), (parsed.hostname or "").lower(), port)


def _trusted_targets() -> set[tuple[str, str, int | None]]:
    targets = set()
    for env_var in (_SERVER_ARTIFACT_ROOT_ENV_VAR, _SERVER_ARTIFACTS_DESTINATION_ENV_VAR):
        if uri := os.environ.get(env_var):
            if (target := host_addressed_uri_target(uri)) is not None:
                targets.add(target)
    return targets


def rejected_host_addressed_scheme(uri: str) -> str | None:
    """
    Return the scheme of ``uri`` when a tracking server must not connect to the host it names,
    or None when the URI is acceptable: it does not select a host, its scheme is allowed by
    ``MLFLOW_ALLOWED_HOST_ADDRESSED_ARTIFACT_SCHEMES``, or it targets the host of the server's
    default artifact root or artifacts destination.
    """
    target = host_addressed_uri_target(uri)
    if target is None:
        return None
    scheme = get_uri_scheme(uri)
    if scheme in {s.lower() for s in MLFLOW_ALLOWED_HOST_ADDRESSED_ARTIFACT_SCHEMES.get()}:
        return None
    if target in _trusted_targets():
        return None
    return scheme


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
    if not is_tracking_server_process():
        return
    scheme = rejected_host_addressed_scheme(artifact_uri)
    if scheme is None:
        return
    raise MlflowException(
        f"The tracking server does not connect to artifact location '{artifact_uri}': the "
        f"'{scheme}' scheme addresses a host other than the server's configured artifact "
        f"storage. Set the {MLFLOW_ALLOWED_HOST_ADDRESSED_ARTIFACT_SCHEMES.name} environment "
        "variable on the server to allow it.",
        error_code=INVALID_PARAMETER_VALUE,
    )
