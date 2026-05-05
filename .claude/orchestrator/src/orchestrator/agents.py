"""Load agent system prompts from the `agents/` directory.

Agent prompts are stored as Markdown files with YAML-style frontmatter. The
frontmatter is metadata for humans browsing the directory; the orchestrator
strips it before passing the body to the model as the system prompt.
"""

from __future__ import annotations

from pathlib import Path

_AGENTS_DIR = Path(__file__).resolve().parent.parent.parent / "agents"


def _strip_frontmatter(content: str) -> str:
    if not content.startswith("---\n"):
        return content
    end = content.find("\n---\n", 4)
    if end == -1:
        return content
    return content[end + len("\n---\n") :].lstrip()


def load_system_prompt(agent_name: str) -> str:
    """Load `agents/<name>.md` and return its body (frontmatter stripped)."""
    path = _AGENTS_DIR / f"{agent_name}.md"
    return _strip_frontmatter(path.read_text())


def reviewer_prompt() -> str:
    return load_system_prompt("mlflow-reviewer")


def spotter_prompt() -> str:
    return load_system_prompt("issue-spotter")
