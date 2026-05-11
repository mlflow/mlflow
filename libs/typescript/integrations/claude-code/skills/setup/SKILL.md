---
description: Configure MLflow tracing for Claude Code.
disable-model-invocation: true
---

# MLflow Tracing Setup

Run this skill ONLY when the user explicitly asks to configure MLflow
tracing. Follow the steps below verbatim. Do not pick defaults silently.
All values come from the user.

## Step 1 — Ask all four questions in a single `AskUserQuestion` call

Call `AskUserQuestion` ONCE with these four questions. The user answers
them all in one screen.

Question A
- question: `Where should MLflow tracing be configured?`
- header: `Scope`
- multiSelect: `false`
- options:
  1. label: `Project`, description: `Write to ./.claude/settings.json (this repo only)`
  2. label: `User`, description: `Write to ~/.claude/settings.json (all repos)`

Question B
- question: `Which MLflow tracking URI should be used? (Use Other to type a custom URL)`
- header: `Tracking URI`
- multiSelect: `false`
- options:
  1. label: `http://localhost:5000`, description: `Default local MLflow server`
  2. label: `databricks`, description: `Use the default Databricks profile`

Question C
- question: `How should the MLflow experiment be specified?`
- header: `Experiment by`
- multiSelect: `false`
- options:
  1. label: `Name`, description: `Create the experiment if it does not exist`
  2. label: `ID`, description: `Use an existing experiment ID`

Question D
- question: `Enter the experiment name or ID (use Other to type it):`
- header: `Experiment value`
- multiSelect: `false`
- options:
  1. label: `I will type it`, description: `Use Other below to type the value`
  2. label: `Cancel`, description: `Abort setup`

If the user answers `Cancel` to Question D, stop and tell them setup was
cancelled. Otherwise use the typed `Other` value as the experiment value.

For Question B, if the user picks Other, take their typed value as the
tracking URI. Otherwise use the selected label.

## Step 2 — Run the CLI

Run exactly one of these commands, substituting the collected values. Do
not add or remove flags.

If Question C answer is `Name`:
```bash
mlflow-claude-code setup <scope-flag> --tracking-uri "<uri>" --experiment-name "<value>"
```

If Question C answer is `ID`:
```bash
mlflow-claude-code setup <scope-flag> --tracking-uri "<uri>" --experiment-id "<value>"
```

`<scope-flag>` is `--project` or `--user` from Question A.

## Step 3 — Summarize

Echo the CLI output. Briefly state the settings file path, tracking URI,
experiment, and that tracing is enabled for the next Claude conversation.

## Hard rules

- Make only ONE `AskUserQuestion` call. All four questions go together.
- Do not invent options like `http://localhost:3000`.
- Do not pre-fill experiment names like `claude-code` or `claude-code-traces`.
- Do not call `mlflow-claude-code setup` without all required flags.
- The CLI has no interactive mode. Do not try to drive prompts.
