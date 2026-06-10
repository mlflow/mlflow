export const buildInstrumentPrompt = (
  experimentName: string,
): string => `Add MLflow tracing to my app. Target experiment: ${experimentName}.

1. Add \`mlflow\` as a dependency.
2. Set the tracking URI to match the project: read \`MLFLOW_TRACKING_URI\`, check \`~/.databrickscfg\` for a Databricks profile, or ask me. Apply via env or \`mlflow.set_tracking_uri()\`.
3. Add \`mlflow.autolog()\` or the matching per-framework call at the entry point. Use the \`instrumenting-with-mlflow-tracing\` skill for framework specifics and \`searching-mlflow-docs\` for API questions (fallback: https://github.com/mlflow/skills).
4. Run the app and confirm a trace appears.

Infer the package manager, entry point, and framework from the repo. If you hit a blocker you can't resolve (missing API key, permissions, dependencies), stop and tell me what you need.`;

export const buildInstrumentAssistantPrompt = (
  experimentName: string,
): string => `Help me add MLflow tracing to my app. Target experiment: ${experimentName}.

Ask what language and framework I'm using (LangChain, OpenAI SDK, Anthropic, LlamaIndex, etc.) and whether MLflow is local, self-hosted, or on Databricks.

Then walk me through the minimal setup: adding \`mlflow\` as a dependency, wiring the tracking URI, and the right \`autolog()\` or instrumentation call.

Use your built-in MLflow docs knowledge rather than guessing at APIs.`;
