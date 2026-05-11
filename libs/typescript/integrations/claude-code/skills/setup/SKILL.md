---
description: Configure MLflow tracing for Claude Code using the bundled setup CLI.
disable-model-invocation: true
---

# MLflow Tracing Setup

Use this skill only when the user explicitly asks to configure MLflow tracing.

Do not invent your own wizard.
Do not present your own menus.
Do not suggest made-up defaults or options.
Do not ask the user to choose between hard-coded tracking URIs or experiment IDs.

Always delegate setup to the bundled CLI.

If the user explicitly asks for user-wide configuration, run:

```bash
mlflow-claude-code setup --user
```

Otherwise, run:

```bash
mlflow-claude-code setup --project
```

If the user already provided all required values and explicitly wants a non-interactive setup, run one of these commands instead:

```bash
mlflow-claude-code setup --non-interactive --project --tracking-uri "<uri>" --experiment-id "<id>"
```

```bash
mlflow-claude-code setup --non-interactive --project --tracking-uri "<uri>" --experiment-name "<name>"
```

For user-wide configuration, replace `--project` with `--user`.

Notes:
- The default local MLflow server URI is `http://localhost:5000`.
- Other MLflow servers should be entered as arbitrary `http://...` or `https://...` URLs.
- `databricks` is supported, but the CLI should handle that directly rather than the skill presenting it as a prominent custom menu.

After the CLI finishes, summarize the resulting configuration and next steps.
