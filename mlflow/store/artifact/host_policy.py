"""
Policy for artifact URIs whose scheme makes MLflow connect to the host named in the URI.

The artifact repositories for ``ftp``, ``sftp``, ``hdfs``, ``viewfs``, ``r2``, ``b2`` and
``abfss`` connect to the host and port in the URI, and so do ``http``, ``https`` and
``mlflow-artifacts://host`` whenever the URI is not proxied by the tracking server. A tracking
server that builds one of these repositories from a location a client stored would therefore
connect to any host the client named (GHSA-mr9f-g8qf-4w4j). Inside a server process, and in the
job subprocesses that inherit its environment, only the hosts of the server's own
``--default-artifact-root`` and ``--artifacts-destination`` are connected to, unless the operator
allows a scheme via ``MLFLOW_ALLOWED_HOST_ADDRESSED_ARTIFACT_SCHEMES``.

Both trusted hosts and the server-process marker come from the environment variables that
``mlflow server`` exports to its workers. A deployment that runs the WSGI app directly must export
the same variables to get the same trusted hosts and registry-level enforcement.
"""

import os
import urllib.parse

from mlflow.environment_variables import MLFLOW_ALLOWED_HOST_ADDRESSED_ARTIFACT_SCHEMES
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.uri import get_uri_scheme

# Mirrors `mlflow.server.constants`, which the store layer cannot import because the `mlflow.server`
# package pulls in Flask. `tests/store/artifact/test_host_policy.py` pins the two to stay equal.
_SERVER_ARTIFACT_ROOT_ENV_VAR = "_MLFLOW_SERVER_ARTIFACT_ROOT"
_SERVER_ARTIFACTS_DESTINATION_ENV_VAR = "_MLFLOW_SERVER_ARTIFACT_DESTINATION"

# `r2`, `b2` and `abfss` take their endpoint from the authority after the last "@"
# (`<bucket>@<endpoint>` and `<filesystem>@<account>.<domain>`), which `urlparse` reports as the
# hostname. `wasbs` is left out because its repository pins the domain to `*.blob.core.*`.
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
