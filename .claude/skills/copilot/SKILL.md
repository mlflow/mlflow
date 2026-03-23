---
name: copilot
description: Hand off a task to GitHub Copilot.
allowed-tools:
  - Bash(gh agent-task create:*)
  - Bash(gh agent-task view:*)
  - Bash(bash .claude/skills/copilot/poll.sh *)
  - Bash(gh api:*)
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

Once Copilot starts working, extract the session ID from the output URL and poll in the background until Copilot finishes:

```bash
# gh agent-task create returns a URL like:
# https://github.com/mlflow/mlflow/pull/21887/agent-sessions/523bb0e1-...
session_url=$(gh agent-task create -F task-desc.md)
session_id="${session_url##*/}"

# Poll using session ID
bash .claude/skills/copilot/poll.sh "$session_id" "<owner>/<repo>" <pr_number>
```

## Sending feedback

If the PR needs changes, batch all feedback into a single review with `@copilot` in each comment so they're addressed in one session:

```bash
gh api repos/<owner>/<repo>/pulls/<pr_number>/reviews --input - <<'EOF'
{
  "event": "COMMENT",
  "comments": [
    {
      "path": "<file_path>",
      "line": <line_number>,
      "side": "RIGHT",
      "body": "@copilot <comment>",
      // ... more params
    },
    // ... more comments
  ]
}
EOF
```
