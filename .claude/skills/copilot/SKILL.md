---
name: copilot
description: Hand off a task to GitHub Copilot.
allowed-tools:
  - Bash(gh agent-task create:*)
---

## Examples

```bash
# Create a task with an inline description
gh agent-task create "<task description>"

# Create a task from a markdown file
gh agent-task create -F task-desc.md
```
