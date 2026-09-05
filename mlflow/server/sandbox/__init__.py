"""Server-side sandboxing primitives.

Contract-agnostic building blocks for running untrusted, model- or user-directed
code inside a locked-down Docker container instead of on the MLflow server host.
The assistant's ``Bash`` sandbox is the first consumer; the same hardening recipe
is intended to back other server-side sandboxes (e.g. custom scorer execution) so
the container security posture lives in one auditable place.
"""

from mlflow.server.sandbox.container import (
    SandboxResult,
    SandboxUnavailableError,
    reap_orphaned_sandbox_containers,
    run_in_sandbox,
    to_container_host_uri,
)
from mlflow.server.sandbox.streaming import (
    SandboxProcess,
    sandbox_input_path,
    start_sandbox_process,
)

__all__ = [
    "SandboxProcess",
    "SandboxResult",
    "SandboxUnavailableError",
    "reap_orphaned_sandbox_containers",
    "run_in_sandbox",
    "sandbox_input_path",
    "start_sandbox_process",
    "to_container_host_uri",
]
