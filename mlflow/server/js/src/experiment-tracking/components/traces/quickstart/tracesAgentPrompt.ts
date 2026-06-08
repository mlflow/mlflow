export const buildInstrumentPrompt = (
  trackingUri: string,
  experimentName: string,
): string => `Add MLflow tracing to my LLM application.

MLflow is already installed and a tracking server is already running at ${trackingUri}.
Target experiment: ${experimentName}

What to do:
1. Add mlflow as a project dependency if it isn't already declared.
2. Wire MLFLOW_TRACKING_URI=${trackingUri} into the project (env file or set_tracking_uri call — pick whatever fits the project's conventions).
3. Add mlflow.autolog(), or the right per-framework call, at the app's entry point. For per-library nuances (LangChain, OpenAI, Anthropic, LlamaIndex, etc.), consult the \`instrumenting-with-mlflow-tracing\` skill if available.
4. Before running, check any required API keys based on the libraries the app uses (e.g. an OpenAI API key for openai, an Anthropic API key for anthropic). If a key is missing, use the format below and stop — don't run the app and watch it fail.
5. Run the app and confirm at least one trace lands in the experiment above.

Inference rules:
- Read the repo. Infer the package manager, the entry point, and the LLM framework from imports and project files. Do not ask if you can figure it out.
- Only ask the user when there's genuine ambiguity (e.g. multiple plausible entry points and no signal to pick).
- Make the changes, run the app, then report what you did and the trace URL.

If you get stuck — permissions blocking you, a missing API key, missing dependencies, or anything else you can't resolve on your own — respond in this format and stop:

**I need:** <plain-language one-liner. Assume the user may not be a developer. Say "your OpenAI API key" not "OPENAI_API_KEY env var". Say "permission to edit files" not "Write tool blocked".>

**How to fix:** <one specific step. If a terminal command is required, write "Open a terminal in your project folder and run:" then put the command on its own line in a code block. If it's a settings toggle in the MLflow UI, name the panel and the toggle. If it's getting an API key, link to where to get one (e.g. https://platform.openai.com/api-keys).>

Use plain language. Don't paste code snippets unless they need to be run. Don't write paragraphs of explanation — one line each.`;
