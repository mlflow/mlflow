---
name: copilot
description: Hand off a task to GitHub Copilot.
allowed-tools:
  - Bash(gh agent-task create:*)
---

## Examples

```bash
gh agent-task create "Build me a new app"
gh agent-task create -F task.md
```
