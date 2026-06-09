export const buildInstrumentPrompt = (
  experimentName: string,
): string => `Add MLflow tracing to my LLM application. Target experiment: ${experimentName}.

Steps:
1. Add \`mlflow\` as a dependency.
2. Wire up MLflow's tracking URI to match the project's deployment — read \`MLFLOW_TRACKING_URI\` from the env, check \`~/.databrickscfg\` for a Databricks profile, or ask me. Set it via env file or \`mlflow.set_tracking_uri()\` — match project conventions.
3. Add \`mlflow.autolog()\` or the right per-framework call at the entry point. For framework specifics, consult the \`instrumenting-with-mlflow-tracing\` skill. For MLflow API/usage questions, use the \`searching-mlflow-docs\` skill rather than guessing.
   If the skills aren't installed locally, refer to https://github.com/mlflow/skills/blob/main/searching-mlflow-docs/SKILL.md
4. Run the app and confirm a trace appears.

Read the repo to infer the package manager, entry point, and LLM framework. Don't ask if you can figure it out.

If you get stuck — missing API key, blocked permissions, missing deps — stop and tell the user in plain language what you need and how to fix it. Don't run the app expecting it to fail.`;

export const buildInstrumentAssistantPrompt = (
  experimentName: string,
): string => `Help me add MLflow tracing to my app. Target experiment: ${experimentName}.

Ask me what language and framework I'm using (Python? LangChain, OpenAI SDK, Anthropic, LlamaIndex, etc.), and whether MLflow is running locally, hosted on a server, or on Databricks.

Walk me through the minimal setup: adding \`mlflow\` as a dependency, wiring the tracking URI for my deployment, and the right \`autolog()\` or instrumentation call.

Use your built-in MLflow docs knowledge — don't guess at APIs.`;
