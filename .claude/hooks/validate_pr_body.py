"""Hook to validate that `gh pr create` includes all sections from the PR template."""

import json
import re
import sys
from pathlib import Path

TEMPLATE_PATH = Path(__file__).resolve().parents[2] / ".github" / "pull_request_template.md"


def get_required_sections() -> list[str]:
    """Parse heading sections (##, ###, etc.) from the PR template."""
    return [
        line.strip()
        for line in TEMPLATE_PATH.read_text().splitlines()
        if re.match(r"^#{2,}\s", line.strip())
    ]


def deny(reason: str) -> None:
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        },
        sys.stdout,
    )


def main() -> None:
    try:
        input_data = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, OSError):
        return

    if input_data.get("tool_name") != "Bash":
        return

    command = input_data.get("tool_input", {}).get("command", "")
    if not re.search(r"gh\s+pr\s+create\b", command):
        return

    # Skip commands without a body (e.g. --help) or with --body-file / -F
    # (we can't validate file contents from the command string)
    if re.search(r"(--body-file\b|-F\b)", command):
        return
    if not re.search(r"(--body\b|-b\b)", command):
        return

    if not TEMPLATE_PATH.exists():
        return

    # Check section headings against the entire command string rather than parsing
    # the body out. The headings (e.g. "### How is this PR tested?") are unique
    # enough that they won't appear in other flags like --title or --repo, so the
    # risk of false positives is negligible.
    command_lines = {line.strip() for line in command.splitlines()}
    if missing := [s for s in get_required_sections() if s not in command_lines]:
        missing_list = "\n".join(f"  - {s}" for s in missing)
        deny(
            f"PR body is missing required sections from the PR template:\n"
            f"{missing_list}\n"
            f"Please include all sections from .github/pull_request_template.md."
        )


if __name__ == "__main__":
    main()
