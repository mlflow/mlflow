export const buildCreatePromptPrompt =
  (): string => `Help me create and test my first MLflow registered prompt. Take it one step at a time:

1. Ask what the prompt is for (e.g. summarizing an article, extracting action items).
2. Ask whether it should be global or attached to an experiment.
3. Suggest a kebab-case name and confirm.
4. Draft a \`{{variable}}\` template and let me iterate before saving.
5. Register it and report the name, version, and UI URL.
6. Help me test it: swap \`mlflow.genai.load_prompt(name, version)\` into my app, or give me a runnable Python snippet.

Infer the package manager, entry point, and framework from the repo. Use the \`searching-mlflow-docs\` skill for API questions (fallback: https://github.com/mlflow/skills). If you hit a blocker you can't resolve (missing API key, permissions, dependencies), stop and tell me what you need.`;

export const buildCreatePromptAssistantPrompt =
  (): string => `Help me create and test my first MLflow registered prompt. Take it one step at a time:

1. Ask what the prompt is for (e.g. summarizing an article, extracting action items).
2. Ask whether it should be global or attached to an experiment.
3. Suggest a kebab-case name and confirm.
4. Draft a \`{{variable}}\` template and let me iterate before saving.
5. Register it and report the name, version, and UI URL.
6. Give me a runnable Python snippet to load the prompt and run it against an LLM.

Use your built-in MLflow docs knowledge rather than guessing at APIs.`;
