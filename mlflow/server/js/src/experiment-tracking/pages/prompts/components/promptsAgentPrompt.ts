export const buildCreatePromptPrompt = (): string => `Help me create my first MLflow registered prompt and test it.

One step at a time — don't dump everything at once:
1. Ask what the prompt is for (e.g. "summarize an article", "extract action items from a meeting").
2. Ask if it should be global or attached to an experiment.
3. Suggest a kebab-case name and confirm.
4. Draft a \`{{variable}}\` template and let me iterate before saving.
5. Register it via MLflow's Prompt Registry API. Report the name, version, and UI URL.
6. Help me test it — either swap \`mlflow.genai.load_prompt(name, version)\` into my existing app, or give me a copy-runnable Python snippet (default to OpenAI; check I have the API key first).

Read the repo to infer the package manager, entry point, and LLM framework. Don't ask if you can figure it out. For MLflow API/usage questions, use the \`searching-mlflow-docs\` skill rather than guessing.

If the skills aren't installed locally, refer to https://github.com/mlflow/skills/blob/main/searching-mlflow-docs/SKILL.md.

If you get stuck — missing API key, blocked permissions, missing deps — stop and tell the user in plain language what you need and how to fix it.`;

export const buildCreatePromptAssistantPrompt =
  (): string => `Help me create my first MLflow registered prompt and test it.

One step at a time — don't dump everything at once:
1. Ask what the prompt is for (e.g. "summarize an article", "extract action items from a meeting").
2. Ask if it should be global or attached to an experiment.
3. Suggest a kebab-case name and confirm with me.
4. Draft a \`{{variable}}\` template and let me iterate before saving.
5. Register it via the Prompt Registry API and report the prompt name, version, and a URL.
6. Give me a copy-runnable Python snippet to load the prompt and run it against an LLM (default to OpenAI; ask me to confirm I have the API key).

Use your built-in MLflow docs knowledge — don't guess at APIs.`;
