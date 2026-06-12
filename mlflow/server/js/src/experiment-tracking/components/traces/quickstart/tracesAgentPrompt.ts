export const buildInstrumentPrompt = (
  experimentName: string,
): string => `Add MLflow tracing to my app. Target experiment: ${experimentName}.

1. Add \`mlflow\` as a dependency.
2. Set the tracking URI to match the project: read \`MLFLOW_TRACKING_URI\` or ask me. Apply via env or \`mlflow.set_tracking_uri()\`.
3. Add \`mlflow.autolog()\` or the matching per-framework call at the entry point. Run \`mlflow skills list\` rather than guessing at APIs.
4. Run the app and confirm a trace appears.

Infer the package manager, entry point, and framework from the repo. If you hit a blocker you can't resolve (missing API key, permissions, dependencies), stop and tell me what you need.`;

export const buildInstrumentAssistantPrompt = (
  experimentName: string,
): string => `Help me add MLflow tracing to my app. Target experiment: ${experimentName}.

Walk me through the minimal setup: adding \`mlflow\` as a dependency, wiring the tracking URI, and the right \`autolog()\` or instrumentation call.

Run \`mlflow skills list\` rather than guessing at APIs.`;
