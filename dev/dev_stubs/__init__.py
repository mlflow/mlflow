"""Credential-free dev/CI stubs for reviewing provider-gated MLflow UI.

``run_dev_server.py --stub-providers <names>`` installs these before launching
the dev server so features gated on external providers/credentials render
without real keys, cost, or nondeterminism.

- ``claude`` -- a fake ``claude`` CLI on PATH, satisfying the MLflow Assistant's
  Claude Code provider auth probe (see ``claude_cli.py``).

The CI ui-review bot passes the names a PR needs; locally, pass them yourself.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

_DEV_STUBS_DIR = Path(__file__).resolve().parent
_CLAUDE_CLI = _DEV_STUBS_DIR / "claude_cli.py"

AVAILABLE_STUBS = ("claude",)


@dataclass
class StubResult:
    """What a launcher must apply after installing stubs."""

    path_prepend: list[Path] = field(default_factory=list)
    cleanup_paths: list[Path] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


def _install_claude(result: StubResult) -> None:
    """Stage a ``claude`` shim that runs the stub CLI with the current interpreter."""
    shim_dir = Path(tempfile.mkdtemp(prefix="mlflow-dev-stub-bin-"))
    shim = shim_dir / "claude"
    # Use sys.executable (not a bare `python3`) so resolution doesn't depend on PATH.
    shim.write_text(f'#!/usr/bin/env bash\nexec {sys.executable} {_CLAUDE_CLI} "$@"\n')
    shim.chmod(0o755)
    result.path_prepend.append(shim_dir)
    result.cleanup_paths.append(shim_dir)
    result.messages.append(f"Staged stub `claude` at {shim}")


_INSTALLERS = {
    "claude": _install_claude,
}


def install_stubs(names: list[str]) -> StubResult:
    """Install the named stubs, returning PATH/cleanup changes for the caller to apply."""
    if unknown := [n for n in names if n not in _INSTALLERS]:
        raise ValueError(
            f"Unknown stub(s): {', '.join(unknown)}. Available: {', '.join(AVAILABLE_STUBS)}"
        )
    result = StubResult()
    for name in names:
        _INSTALLERS[name](result)
    return result


def apply_to_environ(result: StubResult) -> None:
    """Apply a StubResult's PATH prepend to ``os.environ`` in place."""
    if result.path_prepend:
        prepend = os.pathsep.join(str(p) for p in result.path_prepend)
        os.environ["PATH"] = f"{prepend}{os.pathsep}{os.environ.get('PATH', '')}"
