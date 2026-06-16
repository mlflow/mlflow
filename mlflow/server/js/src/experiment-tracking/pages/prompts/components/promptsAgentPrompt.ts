export const buildCreatePromptPrompt =
  (): string => `Help me create and test my first MLflow registered prompt. Take it one step at a time:

1. Add \`mlflow\` as a dependency if needed.
2. Ask what the prompt is for (e.g. summarizing an article, extracting action items).
3. Ask whether it should be global or attached to an experiment.
4. Suggest a kebab-case name and confirm.
5. Draft a \`{{variable}}\` template and let me iterate before saving.
6. Register it and report the name, version, and UI URL.
7. Help me test it: swap \`mlflow.genai.load_prompt(name, version)\` into my app, or give me a runnable Python snippet.

Infer the package manager, entry point, and framework from the repo. Run \`mlflow skills list\` rather than guessing at APIs. If you hit a blocker you can't resolve (missing API key, permissions, dependencies), stop and tell me what you need.`;

export const buildCreatePromptAssistantPrompt =
  (): string => `Help me create and test my first MLflow registered prompt. Take it one step at a time:

1. Ask what the prompt is for (e.g. summarizing an article, extracting action items).
2. Ask whether it should be global or attached to an experiment.
3. Suggest a kebab-case name and confirm.
4. Draft a \`{{variable}}\` template and let me iterate before saving.
5. Register it and report the name, version, and UI URL.
6. Give me a runnable Python snippet to load the prompt and run it against an LLM.

Run \`mlflow skills list\` rather than guessing at APIs.`;
