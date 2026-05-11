---
description: Configure MLflow tracing for Claude Code using the bundled setup CLI.
disable-model-invocation: true
---

# MLflow Tracing Setup

Use this skill only when the user explicitly asks to configure MLflow tracing.

## CRITICAL RULES — READ FIRST

1. **You MUST ask the user for every value below before running the CLI.**
   Never pick defaults silently. Never assume `http://localhost:5000`. Never
   assume an experiment name like `claude-code`. Never guess scope.
2. **Use the `AskUserQuestion` tool for every question.** Do not just write
   "I'll use X" in chat — actually ask.
3. **Never run the interactive form of `mlflow-claude-code setup`.** The
   Claude Code Bash tool has no TTY and the prompts will hang. Always use
   `--non-interactive` with explicit flags built from the user's answers.
4. **Do not invent menus, options, or wizards.** Only present the choices
   listed below.

## Step 1 — Ask for scope

Use `AskUserQuestion`:

- Question: "Where should MLflow tracing be configured?"
- Options:
  - "Project (this repo only)" → flag `--project`
  - "User (all repos)" → flag `--user`

## Step 2 — Ask for tracking URI

Use `AskUserQuestion`:

- Question: "Which MLflow tracking URI should be used?"
- Options:
  - "http://localhost:5000 (default local MLflow server)"
  - "databricks (use Databricks workspace)"
  - "Custom URL" → on selection, ask a follow-up free-text question for the
    URL (`http://...` or `https://...`).

Do NOT add localhost:3000 or any other invented option.

## Step 3 — Ask for experiment

Use `AskUserQuestion`:

- Question: "How should the MLflow experiment be specified?"
- Options:
  - "By name" → follow up with a free-text question asking for the experiment
    name. Do not pre-fill a default like `claude-code`.
  - "By ID" → follow up with a free-text question asking for the experiment
    ID.

## Step 4 — Run the CLI non-interactively

Only after collecting all values from the user, run exactly one of:

```bash
mlflow-claude-code setup --non-interactive <scope-flag> \
  --tracking-uri "<uri>" --experiment-name "<name>"
```

```bash
mlflow-claude-code setup --non-interactive <scope-flag> \
  --tracking-uri "<uri>" --experiment-id "<id>"
```

Where `<scope-flag>` is `--project` or `--user` from Step 1.

## Step 5 — Summarize

After the CLI exits, summarize the resulting configuration and next steps
from the CLI output.

## If the user wants the interactive wizard

If the user explicitly asks to run the interactive wizard themselves, do
NOT run it from this skill. Tell them to run this in their own terminal:

```bash
mlflow-claude-code setup --project
```
