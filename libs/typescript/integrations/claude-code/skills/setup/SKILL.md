---
description: Configure MLflow tracing for Claude Code.
disable-model-invocation: true
---

# MLflow Tracing Setup

Run this skill ONLY when the user explicitly asks to configure MLflow
tracing. Follow the steps below verbatim. Do not skip steps. Do not infer
answers. Do not pick defaults. All values come from the user via
`AskUserQuestion`.

## Step 1 — Ask for scope

Call `AskUserQuestion` with this exact question and these exact options.

- question: `Where should MLflow tracing be configured?`
- header: `Scope`
- multiSelect: `false`
- options:
  1. label: `Project`, description: `Write to ./.claude/settings.json (this repo only)`
  2. label: `User`, description: `Write to ~/.claude/settings.json (all repos)`

Map the answer:
- `Project` → `--project`
- `User` → `--user`

## Step 2 — Ask for tracking URI

Call `AskUserQuestion`:

- question: `Which MLflow tracking URI should be used?`
- header: `Tracking URI`
- multiSelect: `false`
- options:
  1. label: `http://localhost:5000`, description: `Default local MLflow server`
  2. label: `databricks`, description: `Use the default Databricks profile`
  3. label: `Custom URL`, description: `Enter an http:// or https:// URL`

If the user picks `Custom URL`, call `AskUserQuestion` again:
- question: `Enter the MLflow tracking URL (http:// or https://):`
- header: `Custom URL`
- options: (just one option `Default` is fine — the user will use Other to type)
- The user types the URL via the Other option.

## Step 3 — Ask how to specify the experiment

Call `AskUserQuestion`:

- question: `How should the MLflow experiment be specified?`
- header: `Experiment`
- multiSelect: `false`
- options:
  1. label: `By name`, description: `Create the experiment if it does not exist`
  2. label: `By ID`, description: `Use an existing experiment ID`

Then call `AskUserQuestion` once more for the value:

- If `By name`:
  - question: `Enter the MLflow experiment name:`
  - header: `Experiment name`
- If `By ID`:
  - question: `Enter the MLflow experiment ID:`
  - header: `Experiment ID`

The user types the value via the Other option.

## Step 4 — Run the CLI

Run exactly one of these commands, substituting the values collected in
Steps 1–3. Do not add or remove flags.

By name:

```bash
mlflow-claude-code setup <scope-flag> --tracking-uri "<uri>" --experiment-name "<name>"
```

By ID:

```bash
mlflow-claude-code setup <scope-flag> --tracking-uri "<uri>" --experiment-id "<id>"
```

Where `<scope-flag>` is `--project` or `--user`.

The CLI is non-interactive and will print the resulting configuration.

## Step 5 — Summarize

Echo the CLI output to the user, then briefly state:
- Where the settings file was written
- The tracking URI
- The experiment (name and ID)
- That tracing is now enabled and traces will appear after the next Claude
  conversation ends.

## Hard rules

- Do not invent options like `http://localhost:3000`.
- Do not pre-fill experiment names like `claude-code` or `claude-code-traces`.
- Do not call `mlflow-claude-code setup` without all required flags.
- Do not call the CLI before completing Steps 1–3.
- The CLI has no interactive mode. Do not try to drive prompts.
