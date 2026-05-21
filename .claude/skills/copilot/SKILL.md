---
name: copilot
description: Hand off a task to GitHub Copilot.
allowed-tools:
  - Bash(gh agent-task create:*)
  - Bash(gh agent-task list:*)
  - Bash(gh agent-task view:*)
  - Bash(bash .claude/skills/copilot/poll.sh *)
  - Bash(bash .claude/skills/copilot/approve.sh *)
  - Bash(gh api:*)
---

## Examples

```bash
# Create a task with an inline description
gh agent-task create "<task description>"

# Create a task from a markdown file
gh agent-task create -F task-desc.md
```

`gh agent-task create` may print a `queued` message instead of a session URL (e.g., `job <job-id> queued. View progress: https://github.com/copilot/agents`). This means the task was created successfully but may stay queued for minutes or longer. Wait and then run `gh agent-task list` to check if a session has started.

## Post-creation

Print both the session URL and the PR URL (strip `/agent-sessions/...` from the session URL).

Example:

- Session: https://github.com/mlflow/mlflow/pull/20905/agent-sessions/abc123
- PR: https://github.com/mlflow/mlflow/pull/20905

## Polling for completion

Once Copilot starts working, poll in the background until Copilot finishes. The script automatically finds the latest session for the PR:

```bash
bash .claude/skills/copilot/poll.sh "<owner>/<repo>" <pr_number>
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

After sending feedback, Copilot starts a new session, typically within ~10 seconds. Wait at least 15 seconds before polling so the new session gets picked up.

## Approving workflows

Copilot commits require approval to trigger workflows for security reasons, while maintainer commits do not. Once the PR is finalized, run the approve script:

```bash
bash .claude/skills/copilot/approve.sh "<owner>/<repo>" <pr_number>
```
