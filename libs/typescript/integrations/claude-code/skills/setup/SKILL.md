---
name: setup
description: Configure MLflow tracing for Claude Code.
disable-model-invocation: true
---

# MLflow Tracing Setup

Run this skill ONLY when the user explicitly asks to configure MLflow tracing.

1. Run `mlflow-claude-code setup --help` and read the available options.
2. Ask the user for each required value. Do not pick defaults silently. All values come from the user.
3. Run `mlflow-claude-code setup` with the collected options.
4. Echo the CLI output. Briefly state the settings file path, tracking URI, experiment, and that tracing is enabled for the next Claude conversation.
