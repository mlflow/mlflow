"""Hardened Docker container primitive for server-side sandboxing.

This module knows how to run a single command inside a locked-down container and
return its output. It is deliberately contract-agnostic: it knows nothing about
assistants, scorers, or jobs. Callers supply the command, the environment, and an
optional host directory to mount; the hardening recipe (no capabilities, no new
privileges, read-only root filesystem, non-root user, memory / CPU / PID caps) and
the container lifecycle (build image, run, wait with timeout, always clean up) live
here so there is one place to audit and tighten the sandbox security posture.

Network posture: sandbox containers run on a dedicated bridge network (isolating them from
other containers on the host) with an added ``host.docker.internal`` route so a command can
reach an MLflow tracking server on the host. An optional egress proxy
(``MLFLOW_SANDBOX_EGRESS_PROXY``) steers HTTP(S) egress from cooperating clients through an
operator-controlled chokepoint. Neither is a hard egress boundary: the container has network
access, so code that opens raw sockets or ignores the proxy env (e.g. Node's fetch, which does
not read ``HTTP_PROXY`` by default) can still reach other hosts, including the cloud metadata
endpoint. A true egress boundary requires host-level firewalling. The sandbox is off by default
and gated behind an experimental flag.
"""

import logging
import os
import re
import shutil
import tempfile
import urllib.parse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import requests
from packaging.version import Version

from mlflow.environment_variables import (
    _MLFLOW_SERVER_BOOT_ID,
    MLFLOW_SANDBOX_DOCKER_IMAGE,
    MLFLOW_SANDBOX_EGRESS_PROXY,
)
from mlflow.utils import PYTHON_VERSION, get_major_minor_py_version
from mlflow.version import VERSION

_logger = logging.getLogger(__name__)

# Host names that refer to the host loopback interface. A tracking server bound to any of
# these on the host is only reachable from inside the container via host.docker.internal.
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0", "::1"}

# Resource caps applied to every sandbox container. Conservative defaults that let an
# `mlflow` CLI call or a small Python snippet run while bounding a runaway command.
_MEMORY_LIMIT = "1g"
_NANO_CPUS = 1_000_000_000  # 1 CPU
_PIDS_LIMIT = 256

# Where the caller's working directory is mounted inside the container.
_WORKSPACE_MOUNT = "/workspace"

# Marker exit code used when the container is killed for exceeding its timeout.
_TIMEOUT_EXIT_CODE = -1

# How many lines of ``docker build`` output to include when the fallback image build fails.
_BUILD_LOG_TAIL_LINES = 30

# The ``mlflow`` package directory this module belongs to. For a source checkout, its parent is
# the checkout root.
_MLFLOW_PACKAGE_DIR = Path(__file__).resolve().parents[2]
_MLFLOW_PROJECT_NAME = re.compile(r'^name = "mlflow"$', re.MULTILINE)
# What ``pip install`` needs from an MLflow source tree, besides the ``mlflow`` package itself.
_SOURCE_INSTALL_FILES = ("pyproject.toml", "README.md", "LICENSE.txt")
# Never copied from a source tree into the image: the non-Python parts of the package (the
# frontend and the Java and R clients), caches, and local data such as a server's SQLite tracking
# or auth database.
_SOURCE_COPY_IGNORE = shutil.ignore_patterns(
    "js",
    "java",
    "R",
    "node_modules",
    ".*",
    "__pycache__",
    "*.pyc",
    "*.db",
    "*.db-*",
    "*.sqlite",
    "*.sqlite3",
    "*.sqlite-*",
    "mlruns",
    "mlartifacts",
)
_SOURCE_DIR_IN_CONTEXT = "mlflow-source"

# Labels stamped on every sandbox container. The first marks it as a sandbox container; the
# second carries the server's boot id so startup cleanup removes only containers from a
# previous server generation, never one a sibling worker in the current generation launched.
SANDBOX_CONTAINER_LABEL = "mlflow.sandbox"
SANDBOX_BOOT_LABEL = "mlflow.sandbox.boot"


def sandbox_container_labels() -> dict[str, str]:
    labels = {SANDBOX_CONTAINER_LABEL: "1"}
    if boot_id := _MLFLOW_SERVER_BOOT_ID.get():
        labels[SANDBOX_BOOT_LABEL] = boot_id
    return labels


# Dedicated bridge network for sandbox containers, so they are isolated from other containers
# on the host's default bridge (they can still reach the host via host.docker.internal).
SANDBOX_NETWORK_NAME = "mlflow-sandbox"
# Hosts a sandboxed command reaches directly instead of through the egress proxy: the MLflow
# tracking host and loopback. The metadata endpoint is deliberately NOT here, so a
# proxy-respecting client cannot reach it directly.
_EGRESS_PROXY_BYPASS_HOSTS = ("host.docker.internal", "localhost", "127.0.0.1")


def ensure_sandbox_network(client) -> str:
    """Return the name of the dedicated sandbox bridge network, creating it if missing.

    Handles the concurrent-first-launch race: a duplicate-create conflict is treated as the
    network already existing rather than an error. Falls back to the default ``bridge`` network
    (with a warning, since that reduces isolation) if the dedicated one truly cannot be used, so
    a restricted Docker setup still runs the sandbox rather than failing outright.
    """
    import docker.errors

    try:
        try:
            client.networks.get(SANDBOX_NETWORK_NAME)
            return SANDBOX_NETWORK_NAME
        except docker.errors.NotFound:
            pass
        try:
            client.networks.create(SANDBOX_NETWORK_NAME, driver="bridge", check_duplicate=True)
        except docker.errors.APIError:
            # Lost a create race with a concurrent launch; the network exists now.
            client.networks.get(SANDBOX_NETWORK_NAME)
        return SANDBOX_NETWORK_NAME
    except Exception as e:
        _logger.warning(
            "Falling back to the default bridge network for the sandbox; isolation from other "
            "containers is reduced: %s",
            e,
        )
        return "bridge"


def sandbox_egress_env() -> dict[str, str]:
    """Environment that routes a sandbox container's HTTP(S) egress through the configured proxy.

    Empty when no proxy is configured (the container then has unrestricted egress). This only
    shapes egress for clients that honor the standard proxy env vars; it is not an enforcement
    boundary — code that opens raw sockets, or a runtime that ignores the proxy env (e.g. Node's
    fetch), can still reach any host. Only the fixed self-host bypass list (``host.docker.internal``
    + loopback) is exempted. The caller's tracking host is deliberately NOT auto-exempted: it is
    request-derived (from ``request.base_url``), so a remote caller could set it to an internal
    host to carve that destination out of the proxy. A local tracking server is reached via
    ``host.docker.internal`` (already exempt); a remote one must be allowlisted in the proxy itself.
    """
    proxy = MLFLOW_SANDBOX_EGRESS_PROXY.get()
    if not proxy:
        return {}
    no_proxy = ",".join(_EGRESS_PROXY_BYPASS_HOSTS)
    return {
        "HTTP_PROXY": proxy,
        "HTTPS_PROXY": proxy,
        "http_proxy": proxy,
        "https_proxy": proxy,
        "NO_PROXY": no_proxy,
        "no_proxy": no_proxy,
    }


class SandboxUnavailableError(Exception):
    """Raised when the sandbox cannot run because Docker is not usable."""


@dataclass
class SandboxResult:
    """Outcome of a single sandboxed command.

    ``exit_code`` is the container's exit status, or ``_TIMEOUT_EXIT_CODE`` when the
    command was killed for exceeding ``timeout``. ``output`` is the combined
    stdout/stderr captured from the container.
    """

    exit_code: int
    output: str
    timed_out: bool = False


def _get_client():
    """Return a Docker client, or raise SandboxUnavailableError if the daemon is unreachable."""
    try:
        import docker
    except ImportError as e:
        raise SandboxUnavailableError(f"The docker package is required for sandboxing: {e}") from e
    try:
        client = docker.from_env()
        client.ping()
    except Exception as e:
        raise SandboxUnavailableError(f"Docker daemon is not reachable: {e}") from e
    return client


def _mlflow_source_root() -> Path | None:
    """Return the root of the MLflow source checkout the server runs from, or ``None``.

    Only a development build running from a checkout (for example an editable install) has one.
    A released version, or a development build installed into site-packages, returns ``None``.
    """
    if not Version(VERSION).is_devrelease:
        return None
    root = _MLFLOW_PACKAGE_DIR.parent
    pyproject = root / "pyproject.toml"
    if pyproject.is_file() and _MLFLOW_PROJECT_NAME.search(pyproject.read_text()):
        return root
    return None


def _copy_mlflow_source(source_root: Path, context_dir: str) -> str:
    """Copy what ``pip install`` needs from an MLflow source tree into the build context.

    Only the package and its build metadata are copied, never the whole tree: a checkout a server
    runs from can also hold that server's local data (for example ``mlflow.db``), which must not
    end up in an image that runs untrusted code. Returns the copy's path in the context.
    """
    destination = Path(context_dir, _SOURCE_DIR_IN_CONTEXT)
    destination.mkdir()
    for name in _SOURCE_INSTALL_FILES:
        shutil.copy2(source_root / name, destination / name)
    # Keep symlinks as links so a link pointing outside the tree never copies its target.
    shutil.copytree(
        source_root / "mlflow",
        destination / "mlflow",
        symlinks=True,
        ignore=_SOURCE_COPY_IGNORE,
    )
    return _SOURCE_DIR_IN_CONTEXT


def _minimal_sandbox_dockerfile(context_dir: str) -> str:
    """Build the Dockerfile for the fallback sandbox image, matched to the server's environment.

    The image is built to match the server that launches it, so a sandboxed ``mlflow`` command
    speaks the same API as the server rather than whatever the base image ships or the latest PyPI
    release happens to be: the base image uses the server's Python minor version. MLflow is
    installed from ``MLFLOW_HOME`` if set. Otherwise a released version installs the same MLflow
    version from PyPI, and a development build installs from the checkout the server runs from,
    falling back to MLflow's development branch only when there is none. A source install copies
    the package into ``context_dir``.
    """
    from mlflow.models.docker_utils import PYTHON_SLIM_BASE_IMAGE, _pip_mlflow_install_step

    base_image = PYTHON_SLIM_BASE_IMAGE.format(version=get_major_minor_py_version(PYTHON_VERSION))
    mlflow_home = os.environ.get("MLFLOW_HOME")
    source_root = Path(mlflow_home) if mlflow_home else _mlflow_source_root()
    if source_root is not None:
        _logger.info("Installing MLflow into the sandbox image from %s.", source_root)
        source_dir = _copy_mlflow_source(source_root, context_dir)
        install_step = f"COPY {source_dir} /opt/mlflow\nRUN pip install /opt/mlflow"
    else:
        if Version(VERSION).is_devrelease:
            _logger.warning(
                "MLflow %s is a development build that is not running from a source checkout, so "
                "the sandbox image installs MLflow's development branch, which may not match this "
                "server. Set MLFLOW_HOME to this server's MLflow source tree to match it.",
                VERSION,
            )
        install_step = _pip_mlflow_install_step(context_dir, None)
    return f"FROM {base_image}\n{install_step}\n"


def _build_log_tail(build_log: Iterable[dict[str, object]] | None) -> str:
    """Return the last lines of ``docker build`` output from ``BuildError.build_log``."""
    lines = []
    for chunk in build_log or []:
        if isinstance(text := chunk.get("stream"), str):
            lines.extend(line for line in text.splitlines() if line.strip())
    return "\n".join(lines[-_BUILD_LOG_TAIL_LINES:])


def _ensure_image(client, image: str) -> None:
    """Build a minimal sandbox image if ``image`` is not already present locally."""
    import docker.errors

    try:
        client.images.get(image)
        return
    except docker.errors.ImageNotFound:
        pass

    _logger.info(
        "Sandbox image %s not found locally; building a minimal one. It is not rebuilt when MLflow "
        "changes: delete the image to rebuild it.",
        image,
    )
    # This fallback build does not forward the operator's PIP_INDEX_URL/PIP_EXTRA_INDEX_URL: a
    # private-index URL can embed credentials (https://user:pass@mirror/...) and Docker records
    # build args in the image history, so forwarding them would persist those credentials in the
    # built image. Operators behind a private mirror or an air-gapped index should build and
    # provide their own image instead.
    with tempfile.TemporaryDirectory(prefix="mlflow-sandbox-image-") as ctx:
        Path(ctx, "Dockerfile").write_text(_minimal_sandbox_dockerfile(ctx))
        try:
            client.images.build(path=ctx, tag=image, rm=True)
        except docker.errors.BuildError as e:
            # The error itself only names the failed step; the reason (for example pip's error)
            # is in the build output.
            raise SandboxUnavailableError(
                f"{e.msg}\nBuild output:\n{_build_log_tail(e.build_log)}"
            ) from e


def to_container_host_uri(uri: str | None) -> str | None:
    """Rewrite a loopback URI so it is reachable from inside a sandbox container.

    A command that shells out to the ``mlflow`` CLI talks to the tracking server via
    ``MLFLOW_TRACKING_URI``. A loopback host (``localhost``/``127.0.0.1``/``0.0.0.0``/``::1``)
    inside the container points at the container itself, not the host, so it is rewritten to
    ``host.docker.internal``, which is routed to the host via the ``extra_hosts`` entry set on
    the container. Only the URI's host component is rewritten, and only when it is exactly a
    loopback name, so hosts that merely contain "localhost" (e.g. ``localhost.example.com``)
    and any path/query text are left untouched. A non-loopback or unparsable URI is returned
    unchanged.
    """
    if not uri:
        return uri
    try:
        parts = urllib.parse.urlsplit(uri)
        host = parts.hostname
    except ValueError:
        return uri
    if host is None or host.lower() not in _LOOPBACK_HOSTS:
        return uri

    # Swap only the host token in the netloc. Keep any userinfo bytes exactly as they were
    # (split on the last '@') and reuse the parsed port, which is bracket-aware so IPv6 loopbacks
    # like ``[::1]`` are handled. The original loopback host is discarded — it is what we replace.
    userinfo, sep, _ = parts.netloc.rpartition("@")
    # parts.port raises ValueError on an invalid port (e.g. "localhost:bad"); honor the
    # "unparsable URI returned unchanged" contract rather than letting it propagate.
    try:
        port = f":{parts.port}" if parts.port else ""
    except ValueError:
        return uri
    new_netloc = f"{userinfo}{sep}host.docker.internal{port}"
    return urllib.parse.urlunsplit(parts._replace(netloc=new_netloc))


def run_in_sandbox(
    command: list[str],
    *,
    workdir: Path | None = None,
    environment: dict[str, str] | None = None,
    timeout: float = 120.0,
    use_shell: bool = False,
) -> SandboxResult:
    """Run ``command`` inside a hardened, disposable Docker container.

    The first call with no prebuilt image builds one, which is not bounded by ``timeout``
    (only the command's own run is) and can take minutes; operators should prebuild the
    image to avoid that latency, and concurrent first calls may each build the same tag.

    Args:
        command: The command to run. When ``use_shell`` is False (the default) this is
            an argv list executed directly with no shell. When ``use_shell`` is True the
            single element is a shell command string run via ``sh -c``.
        workdir: Host directory to bind-mount read-write at ``/workspace`` and use as the
            container's working directory. When None the container has no workspace mount.
        environment: Extra environment variables to set inside the container.
        timeout: Seconds to wait before the container is killed and the result is marked
            as timed out.
        use_shell: Whether ``command`` is a shell string (True) or an argv list (False).

    Returns:
        A SandboxResult with the container's exit code and combined output.

    Raises:
        SandboxUnavailableError: If Docker is not available or the sandbox cannot be
            prepared or started (so the caller can surface a clear message rather than
            silently falling back to host execution).
    """
    if os.name == "nt":
        # user=uid:gid below relies on os.getuid/os.getgid, which do not exist on Windows,
        # and the hardening flags target Linux containers.
        raise SandboxUnavailableError("The sandbox requires a POSIX host running Docker.")

    client = _get_client()
    image = MLFLOW_SANDBOX_DOCKER_IMAGE.get()
    try:
        _ensure_image(client, image)
    except Exception as e:
        raise SandboxUnavailableError(f"Failed to prepare sandbox image {image!r}: {e}") from e

    env = dict(environment or {})
    # Read-only rootfs plus a writable /tmp: give the command a HOME it can write to
    # (the mlflow CLI and pip both write under HOME) without loosening the rootfs.
    env.setdefault("HOME", "/tmp")
    env.update(sandbox_egress_env())

    container_command = command
    if use_shell:
        # sh (not bash -l): matches the host path's /bin/sh semantics and avoids assuming
        # bash exists in a custom operator image.
        container_command = ["sh", "-c", command[0] if command else ""]

    volumes = {}
    working_dir = None
    if workdir is not None:
        volumes[str(workdir)] = {"bind": _WORKSPACE_MOUNT, "mode": "rw"}
        working_dir = _WORKSPACE_MOUNT

    try:
        container = client.containers.run(
            image,
            command=container_command,
            detach=True,
            labels=sandbox_container_labels(),
            network=ensure_sandbox_network(client),
            extra_hosts={"host.docker.internal": "host-gateway"},
            mem_limit=_MEMORY_LIMIT,
            memswap_limit=_MEMORY_LIMIT,
            nano_cpus=_NANO_CPUS,
            pids_limit=_PIDS_LIMIT,
            read_only=True,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            tmpfs={"/tmp": ""},
            # Run as the server's user so anything written to the bind-mounted workspace is owned
            # by the server, not root. NOTE: if the MLflow server itself runs as root (uid 0) this
            # is 0:gid and the container is root too — the non-root posture holds only when the
            # server runs as a non-root user.
            user=f"{os.getuid()}:{os.getgid()}",
            working_dir=working_dir,
            environment=env,
            volumes=volumes,
        )
    except Exception as e:
        raise SandboxUnavailableError(f"Failed to start sandbox container: {e}") from e

    # Always remove the container and reap its output, even if wait() raises: a container
    # left behind would leak both a process slot and disk.
    try:
        try:
            outcome = container.wait(timeout=timeout)
        except Exception as e:
            if _is_read_timeout(e):
                # A client-side read timeout means the command outran `timeout`; the container
                # is still running, so kill it and report a timeout.
                _logger.info("Sandbox command exceeded %.0fs timeout", timeout)
                _kill_quietly(container)
                return SandboxResult(
                    exit_code=_TIMEOUT_EXIT_CODE, output=_logs(container), timed_out=True
                )
            # A genuine failure (daemon down, connection refused, API error) is not a timeout;
            # kill the container and surface it as a sandbox failure rather than mislabeling it.
            _kill_quietly(container)
            raise SandboxUnavailableError(f"Sandbox execution failed while waiting: {e}") from e
        return SandboxResult(exit_code=outcome.get("StatusCode", 0), output=_logs(container))
    finally:
        _remove_quietly(container)


def _is_read_timeout(exc: Exception) -> bool:
    """Whether ``exc`` from ``container.wait(timeout=)`` is a client-side read timeout.

    docker-py maps the deadline being exceeded to a requests ``ReadTimeout``, or — on the Unix-
    socket transport — a ``ConnectionError`` wrapping urllib3's ``ReadTimeoutError``; both carry
    "read timed out" in their message. Match that specific signal, not a bare "timed out": a
    ``ConnectTimeout`` (connection-establishment failure, also a ``ConnectionError``) says "timed
    out" too, and that is a genuine daemon-unreachable error that must surface as a sandbox failure
    rather than be misreported as the command timing out.
    """
    if isinstance(exc, requests.exceptions.ReadTimeout):
        return True
    return (
        isinstance(exc, requests.exceptions.ConnectionError)
        and "read timed out" in str(exc).lower()
    )


def _logs(container) -> str:
    try:
        return container.logs().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _kill_quietly(container) -> None:
    try:
        container.kill()
    except Exception:
        pass


def _remove_quietly(container) -> bool:
    try:
        container.remove(force=True)
        return True
    except Exception:
        return False


def reap_orphaned_sandbox_containers() -> int:
    """Remove sandbox containers left over from a *previous* server generation.

    Sandbox containers are tied to an in-process stream/wait; once the server that started them
    exits they are orphaned with no way to reattach. On startup they are force-removed by label,
    but only those whose boot-id label differs from this server's boot id — so a sibling worker's
    just-launched container (same boot id) is never removed. If this server has no boot id (e.g.
    it was not started through the normal server entry point), reaping is skipped entirely rather
    than risk removing a live container. Best-effort: returns the number removed, never raises.
    """
    current_boot = _MLFLOW_SERVER_BOOT_ID.get()
    if not current_boot:
        return 0
    try:
        client = _get_client()
    except SandboxUnavailableError:
        return 0
    try:
        containers = client.containers.list(all=True, filters={"label": SANDBOX_CONTAINER_LABEL})
    except Exception as e:
        _logger.debug("Could not list sandbox containers to reap: %s", e)
        return 0
    removed = 0
    failed = 0
    for container in containers:
        if (container.labels or {}).get(SANDBOX_BOOT_LABEL) == current_boot:
            continue  # belongs to this server generation; may be actively serving a turn
        # A different boot id alone does not prove the container is orphaned: two servers can share
        # one Docker daemon, and a rolling restart overlaps generations. So reap only containers in
        # a terminal state (exited/dead); leave any live or transitional state
        # (running/created/restarting/paused/removing) alone, since force-removing one could kill a
        # concurrent server's live or still-starting turn. An orphaned container is reaped on a
        # later startup once it reaches a terminal state, which is the common case. One wedged in a
        # non-terminal state (e.g. a `created` container left by a crash between create and start)
        # is knowingly left rather than risk force-removing a concurrent server's still-starting
        # container; reclaiming those safely needs ownership/lease metadata, which is future
        # multi-replica work.
        if getattr(container, "status", None) not in ("exited", "dead"):
            continue
        if _remove_quietly(container):
            removed += 1
        else:
            failed += 1
    if removed:
        _logger.info("Removed %d orphaned sandbox container(s) on startup.", removed)
    if failed:
        _logger.debug("%d orphaned sandbox container(s) could not be removed on startup.", failed)
    return removed
