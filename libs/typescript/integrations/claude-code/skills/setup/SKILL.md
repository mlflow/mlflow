---
description: Configure MLflow tracing for Claude Code using the bundled setup CLI.
disable-model-invocation: true
---

# MLflow Tracing Setup

Use this skill only when the user explicitly asks to configure MLflow tracing.

The bundled `mlflow-claude-code setup` CLI has an interactive mode, but the
Claude Code Bash tool runs without a TTY and cannot drive interactive prompts.
**Never run the interactive form from this skill.** Always use the
`--non-interactive` form with flags.

## Procedure

1. Gather the required inputs by asking the user directly in chat (use
   `AskUserQuestion` when possible). Required values:
   - Scope: project (`--project`, default) or user-wide (`--user`)
   - Tracking URI (default: `http://localhost:5000`; also accepts arbitrary
     `http://...` / `https://...` URLs, or `databricks`)
   - Experiment: either an experiment ID or an experiment name (not both)

2. Do not invent your own wizard, menus, or made-up defaults. Ask plain
   questions and use the defaults above only as suggested defaults.

3. Once you have the values, run the CLI non-interactively:

   ```bash
   mlflow-claude-code setup --non-interactive --project \
     --tracking-uri "<uri>" --experiment-name "<name>"
   ```

   Or with an experiment ID:

   ```bash
   mlflow-claude-code setup --non-interactive --project \
     --tracking-uri "<uri>" --experiment-id "<id>"
   ```

   Replace `--project` with `--user` if the user requested user-wide config.

4. If the user explicitly asks to run the interactive wizard, do NOT run it
   yourself. Tell them to run this command in their own terminal:

   ```bash
   mlflow-claude-code setup --project
   ```

5. After the non-interactive CLI finishes, summarize the resulting
   configuration and next steps from the CLI output.
