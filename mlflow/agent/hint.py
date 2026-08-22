"""An agent-directed pointer to the MLflow tracing skill.

Emitted on ``import mlflow`` when a coding agent is driving the process and the
tracing skill is not installed where that agent would find it. There is no
suppression state, so it fires once per process and a run of many short-lived
Python processes repeats it; ``MLFLOW_DISABLE_AGENT_HINT=1`` silences it for
good.

Without the skill, agents design tracing from scratch: they hand-roll spans
instead of choosing a supported ``mlflow.<framework>.autolog()``, and they
forget that a manual span records nothing unless ``set_inputs`` /
``set_outputs`` are called, which produces traces with blank tool inputs and
outputs.

Gated on the agent environment markers in :mod:`mlflow.agent.agents`, so a human
running MLflow from their own shell never sees it. It is not a private channel
-- it goes to the same stderr as every other MLflow log line -- but it reaches
the human only if they read the agent's raw tool output, which many agent UIs
collapse by default.
"""

from __future__ import annotations

import logging
from importlib import resources
from pathlib import Path

from mlflow.agent.agents import AgentTool, detect_active
from mlflow.assistant.skill_installer import SKILL_MANIFEST_FILE, SKILLS_PACKAGE
from mlflow.environment_variables import MLFLOW_DISABLE_AGENT_HINT

_logger = logging.getLogger(__name__)

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
    "{display_name} detected. Load the `{skill}` skill at {path} before writing any tracing "
    "code; it ships with this MLflow install. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this."
)


def _bundled_skill_manifest() -> Path | None:
    """Path to the skill shipped with this install, or ``None`` when absent.

    Released packages bundle the skill. A source checkout whose
    ``mlflow/assistant/skills`` submodule was never initialized does not, and
    there the hint stays silent rather than sending the agent elsewhere.
    """
    try:
        # Chained joinpath: importlib's MultiplexedPath takes a single segment.
        manifest = (
            resources.files(SKILLS_PACKAGE).joinpath(TRACING_SKILL).joinpath(SKILL_MANIFEST_FILE)
        )
        return Path(str(manifest)) if manifest.is_file() else None
    except (ModuleNotFoundError, OSError):
        return None


def _is_skill_installed(agent: AgentTool) -> bool:
    """Whether the skill sits anywhere this agent would load it from.

    The project directory is searched from the current directory upwards: agents
    routinely run from a subdirectory, while the skill lives at the repo root.
    """
    cwd = Path.cwd()
    return any(
        (d / TRACING_SKILL).is_dir() for root in (cwd, *cwd.parents) for d in agent.skill_dirs(root)
    )


def maybe_hint_tracing_skill() -> None:
    """Log the tracing-skill hint when a coding agent is driving without it."""
    if MLFLOW_DISABLE_AGENT_HINT.get():
        return
    if (agent := detect_active()) is None:
        return
    if _is_skill_installed(agent):
        return
    if (path := _bundled_skill_manifest()) is None:
        return
    _logger.info(_HINT.format(display_name=agent.display_name, skill=TRACING_SKILL, path=path))
