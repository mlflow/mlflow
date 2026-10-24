"""Agent-directed pointers to the bundled MLflow skills.

Emitted on ``import mlflow`` when a coding agent is driving. Points to the bundled
skills index (covering instrumentation, trace querying, and debugging) so agents
can load the appropriate SKILL.md. Whether the skill is already installed is
deliberately not probed: skills end up in too many places for the check to be
accurate, and the pointer stays useful either way. Without it, agents design
tracing from scratch and produce traces with blank tool inputs and outputs, or
conflate runs with traces when retrieving.

Fires once per process; ``MLFLOW_DISABLE_AGENT_HINT=1`` silences it for good.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from functools import lru_cache
from importlib import resources
from pathlib import Path

from mlflow.environment_variables import MLFLOW_DISABLE_AGENT_HINT

_logger = logging.getLogger(__name__)

# Variables agents export into the processes they spawn; any one means an agent
# is driving. Tracks https://github.com/vercel/detect-agent/blob/3ab1df1e4eaae153cf66f4a5018e4c5854855212/agents.json
# Excludes vars a human also sets: config (COPILOT_MODEL, AIDER_*) and
# environment-wide ones (REPL_ID, CURSOR_TRACE_ID, CI).
_AGENT_ENV_MARKERS = (
    "AGENT",  # Goose, Amp, and other adopters of the convention
    "AI_AGENT",
    "AMP_CURRENT_THREAD_ID",
    "ANTIGRAVITY_AGENT",
    "AUGMENT_AGENT",
    "CLAUDECODE",
    "CLAUDE_CODE",
    "CLINE_ACTIVE",
    "CLINE_AGENT",
    "CODEX_SANDBOX",  # sandboxed runs only
    "CODEX_THREAD_ID",  # every run, sandboxed or not
    "CURSOR_AGENT",
    "GEMINI_CLI",
    "GROK_PLUGIN_ROOT",
    "JUNIE_DATA",
    "KIMI_PLUGIN_ROOT",
    "OPENCLAW_SHELL",
    "OPENCODE",
    "PI_CODING_AGENT",
    "QWEN_CODE",
    "ROO_ACTIVE",
    "TRAE_AI_SHELL_ID",
)

# Markers that identify an agent only at a particular value.
_AGENT_ENV_VALUES = {"CURSOR_EXTENSION_HOST_ROLE": "agent-exec"}

# Kiro sets this in its own IDE terminal too, so it only means "agent" when
# nothing is attached to stdout, as a human terminal always is.
_AGENT_ENV_VALUES_WITHOUT_TTY = {"TERM_PROGRAM": "kiro"}

# The skill that teaches supported autologging and span input/output recording.
# Lives in https://github.com/mlflow/skills and is installed by `mlflow agent setup`.
TRACING_SKILL = "instrumenting-with-mlflow-tracing"

# Identifier for the skills-index hint in _EMITTED_HINTS.
SKILLS_INDEX_HINT_ID = "skills_index"

# All hints in this module are process-local. An agent only needs to see a
# particular problem once to change course; repeating it on every span or row
# makes the useful message indistinguishable from ordinary logs.
_EMITTED_HINTS: set[str] = set()
_EMITTED_HINTS_LOCK = threading.Lock()

# Points at the copy shipped inside this MLflow install: no network, and the
# revision always matches the installed code.
#
# Kept to a single line. Agents habitually append `| tail -3` or `| head -5` to
# the commands they run, and a multi-line message loses its actionable half to
# the pipe -- silently, since the surviving fragment still reads like a log line.
_HINT = (
    "MLflow skills for instrumenting, querying, and debugging traces are bundled at {path}; "
    "read its README and load the matching SKILL.md before writing MLflow code."
)


def _bundled_skills_dir() -> Path | None:
    """Path to the skills directory shipped with this install, or ``None`` when absent.

    Released packages bundle it; a source checkout without the
    ``mlflow/assistant/skills`` submodule does not.
    """
    from mlflow.assistant.skill_installer import SKILLS_PACKAGE

    try:
        readme = resources.files(SKILLS_PACKAGE).joinpath("README.md")
        return Path(str(readme)).parent if readme.is_file() else None
    except (ModuleNotFoundError, OSError):
        return None


def _bundled_skill_manifest(skill: str = TRACING_SKILL) -> Path | None:
    """Path to the skill shipped with this install, or ``None`` when absent."""
    from mlflow.assistant.skill_installer import SKILL_MANIFEST_FILE, SKILLS_PACKAGE

    try:
        manifest = resources.files(SKILLS_PACKAGE).joinpath(skill).joinpath(SKILL_MANIFEST_FILE)
        return Path(str(manifest)) if manifest.is_file() else None
    except (ModuleNotFoundError, OSError):
        return None


@lru_cache(maxsize=1)
def _is_agent_driving() -> bool:
    """Whether a coding agent is running this process."""
    if any(os.environ.get(marker) for marker in _AGENT_ENV_MARKERS):
        return True
    if any(os.environ.get(name) == value for name, value in _AGENT_ENV_VALUES.items()):
        return True
    if sys.stdout.isatty():
        return False
    return any(
        os.environ.get(name) == value for name, value in _AGENT_ENV_VALUES_WITHOUT_TTY.items()
    )


def maybe_hint_tracing_skill() -> None:
    """Log the skills-index hint when a coding agent is driving."""
    try:
        if (
            SKILLS_INDEX_HINT_ID in _EMITTED_HINTS
            or MLFLOW_DISABLE_AGENT_HINT.get()
            or not _is_agent_driving()
        ):
            return
        with _EMITTED_HINTS_LOCK:
            if SKILLS_INDEX_HINT_ID in _EMITTED_HINTS:
                return
            _EMITTED_HINTS.add(SKILLS_INDEX_HINT_ID)

        if (path := _bundled_skills_dir()) is None:
            return
        _logger.info(_HINT.format(path=path))
    except Exception:
        # User-configurable logging handlers must not affect MLflow behavior.
        return


def _claim_agent_hint(issue_id: str) -> Path | None:
    """Claim an agent hint once per process and return the bundled skills path."""
    try:
        if issue_id in _EMITTED_HINTS or MLFLOW_DISABLE_AGENT_HINT.get() or not _is_agent_driving():
            return None

        with _EMITTED_HINTS_LOCK:
            if issue_id in _EMITTED_HINTS:
                return None
            _EMITTED_HINTS.add(issue_id)

        return _bundled_skills_dir()
    except Exception:
        return None


def maybe_warn_agent(issue_id: str, issue: str) -> None:
    """Warn a coding agent about an observed GenAI anti-pattern once per process."""
    try:
        if skills_path := _claim_agent_hint(issue_id):
            _logger.warning(
                "%s Review the MLflow skills bundled at %s before continuing. "
                "Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this.",
                issue,
                skills_path,
            )
    except Exception:
        # User-configurable logging handlers must not affect MLflow behavior.
        return


def maybe_append_agent_hint(issue_id: str, message: str) -> str:
    """Append the bundled-skills pointer to an existing message once for a coding agent."""
    try:
        if skills_path := _claim_agent_hint(issue_id):
            return (
                f"{message}\n\nReview the MLflow skills bundled at {skills_path} before "
                "continuing. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this."
            )
    except Exception:
        pass
    return message


def maybe_warn_local_tracking_for_databricks() -> None:
    """Warn when a GenAI trace is going to local storage despite Databricks intent."""
    try:
        issue_id = "databricks-intent-with-local-tracking"
        if issue_id in _EMITTED_HINTS or MLFLOW_DISABLE_AGENT_HINT.get() or not _is_agent_driving():
            return

        from mlflow import get_tracking_uri
        from mlflow.utils.databricks_utils import is_in_databricks_runtime
        from mlflow.utils.uri import is_local_uri

        has_databricks_intent = (
            is_in_databricks_runtime()
            or bool(os.environ.get("DATABRICKS_HOST"))
            or bool(os.environ.get("DATABRICKS_CONFIG_PROFILE"))
        )
        tracking_uri = get_tracking_uri()
        if has_databricks_intent and is_local_uri(tracking_uri):
            maybe_warn_agent(
                issue_id,
                f"A GenAI trace is being written to the local tracking URI {tracking_uri!r} even "
                "though a Databricks environment or profile is configured.",
            )
    except Exception:
        # This check is advisory and must never interfere with trace export.
        return
