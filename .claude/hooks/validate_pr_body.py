"""Hook to validate that `gh pr create` includes all sections from the PR template."""

import json
import re
import sys
from pathlib import Path


def get_headings() -> list[str]:
    """Parse heading sections (#, ##, ###, etc.) from the PR template."""
    pr_template = Path(__file__).resolve().parents[2] / ".github" / "pull_request_template.md"
    lines = (l.strip() for l in pr_template.read_text().splitlines())
    return [l for l in lines if re.match(r"^#+\s", l)]


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

    match input_data:
        case {
            "tool_name": "Bash",
            "tool_input": {
                "command": str(command),
            },
        }:
            pass
        case _:
            return

    if not re.search(r"gh\s+pr\s+create\b", command):
        return

    # Skip commands without a body (e.g. --help) or with --body-file / -F
    # (we can't validate file contents from the command string)
    if re.search(r"(--body-file\b|-F\b)", command):
        return
    if not re.search(r"(--body\b|-b\b)", command):
        return

    # Check section headings against the command lines rather than parsing the
    # body out. The headings (e.g. "### How is this PR tested?") are unique
    # enough that they won't appear in other flags like --title or --repo, so the
    # risk of false positives is negligible.
    headings = get_headings()
    command_lines = {line.strip() for line in command.splitlines()}
    missing = [s for s in headings if s not in command_lines]

    # If all sections are missing, the body is likely opaque (e.g. --body "$VAR")
    # and we can't validate it. Only deny when some sections are present but
    # others are missing, indicating an incomplete but visible body.
    if missing and len(missing) < len(headings):
        missing_list = "\n".join(f"  - {s}" for s in missing)
        deny(
            f"PR body is missing required sections from the PR template:\n"
            f"{missing_list}\n"
            f"Please include all sections from .github/pull_request_template.md."
        )


if __name__ == "__main__":
    main()
