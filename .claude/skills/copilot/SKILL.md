---
name: copilot
description: Hand off a task to GitHub Copilot.
allowed-tools:
  - Bash(gh agent-task create:*)
  - Bash(bash .claude/skills/copilot/poll.sh *)
---

## Examples

```bash
# Create a task with an inline description
gh agent-task create "<task description>"

# Create a task from a markdown file
gh agent-task create -F task-desc.md
```

## Post-creation

Print both the session URL and the PR URL (strip `/agent-sessions/...` from the session URL).

Example:

- Session: https://github.com/mlflow/mlflow/pull/20905/agent-sessions/abc123
- PR: https://github.com/mlflow/mlflow/pull/20905

## Polling for completion

Once Copilot starts working, poll in the background until Copilot finishes:

```bash
bash .claude/skills/copilot/poll.sh {owner}/{repo} {pr_number}
```
