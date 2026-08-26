"""An agent-directed pointer to the MLflow tracing skill.

Emitted on ``import mlflow`` when a coding agent is driving and the skill is not
installed where that agent would find it. Without it, agents design tracing from
scratch and produce traces with blank tool inputs and outputs.

Fires once per process; ``MLFLOW_DISABLE_AGENT_HINT=1`` silences it for good.
"""

from __future__ import annotations

import logging
import os
import sys
from importlib import resources
from pathlib import Path

from mlflow.agent.agents import AGENTS
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

# Points at the copy shipped inside this MLflow install: no network, and the
# revision always matches the installed code.
#
# Kept to a single line. Agents habitually append `| tail -3` or `| head -5` to
# the commands they run, and a multi-line message loses its actionable half to
# the pipe -- silently, since the surviving fragment still reads like a log line.
_HINT = (
    "Load the `{skill}` skill at {path} before writing any tracing code; it ships with this "
    "MLflow install. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this."
)


def _bundled_skill_manifest() -> Path | None:
    """Path to the skill shipped with this install, or ``None`` when absent.

    Released packages bundle it; a source checkout without the
    ``mlflow/assistant/skills`` submodule does not.
    """
    # Imported lazily: `mlflow.assistant` pulls in pydantic config models that
    # `import mlflow` does not otherwise load, and only a detected agent needs them.
    from mlflow.assistant.skill_installer import SKILL_MANIFEST_FILE, SKILLS_PACKAGE

    try:
        # Chained joinpath: importlib's MultiplexedPath takes a single segment.
        manifest = (
            resources.files(SKILLS_PACKAGE).joinpath(TRACING_SKILL).joinpath(SKILL_MANIFEST_FILE)
        )
        return Path(str(manifest)) if manifest.is_file() else None
    except (ModuleNotFoundError, OSError):
        return None


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


def _custom_skill_dirs() -> tuple[Path, ...]:
    """Directories `mlflow agent setup` was pointed at via its custom location."""
    # Lazy for the same reason as the bundled lookup: only a detected agent needs it.
    from mlflow.assistant.config import AssistantConfig

    try:
        config = AssistantConfig.load()
    except Exception:
        return ()
    return tuple(
        Path(provider.skills.custom_path).expanduser()
        for provider in config.providers.values()
        if provider.skills.type == "custom" and provider.skills.custom_path
    )


def _is_skill_installed() -> bool:
    """Whether the skill sits anywhere a coding agent would load it from.

    Every known agent's directories are checked, not just the detected one: the
    detected agent may not be in :data:`AGENTS` at all, and an already-installed
    skill should silence the hint regardless of who installed it.
    """
    if any(
        (Path.home() / agent.global_skills_dir / TRACING_SKILL).is_dir()
        for agent in AGENTS.values()
    ):
        return True
    if any((d / TRACING_SKILL).is_dir() for d in _custom_skill_dirs()):
        return True
    # Walk upwards: agents often run from a subdirectory of the repo root.
    cwd = Path.cwd()
    return any(
        (d / TRACING_SKILL).is_dir()
        for root in (cwd, *cwd.parents)
        for agent in AGENTS.values()
        for d in agent.project_skill_dirs(root)
    )


def maybe_hint_tracing_skill() -> None:
    """Log the tracing-skill hint when a coding agent is driving without it."""
    if MLFLOW_DISABLE_AGENT_HINT.get():
        return
    if not _is_agent_driving():
        return
    if _is_skill_installed():
        return
    if (path := _bundled_skill_manifest()) is None:
        return
    _logger.info(_HINT.format(skill=TRACING_SKILL, path=path))
