export const buildCreatePromptPrompt = (
  trackingUri: string,
): string => `Help me create my first MLflow registered prompt and walk me through testing it.

MLflow is running at ${trackingUri}.

Guide me through this end-to-end — one question at a time, don't dump everything at once:
1. Ask what the prompt is for in plain language (e.g. "summarize an article", "extract action items from a meeting transcript").
2. Ask if the prompt is global or attached to an experiment.
3. Suggest a kebab-case name based on my answer and confirm it with me.
4. Draft a prompt template using \`{{variable}}\` placeholders for the inputs. Show it to me and let me iterate before saving.
5. Once I approve, register it via MLflow's Prompt Registry API. Report the prompt name, version, and a URL I can open in the MLflow UI.
6. Help me test it — either:
   - If I have an existing app in this repo, find the right place to swap in \`mlflow.genai.load_prompt(name, version)\` and offer to make the edit, OR
   - If I don't have an app yet, give me a copy-runnable Python snippet that loads the prompt and runs it against an LLM (default to OpenAI; check that I have the API key first).

Inference rules:
- Read the repo first. Infer the package manager, entry point, and LLM framework from imports.
- Only ask the user when there's genuine ambiguity.
- Make changes, then report what you did and the URLs.

If you get stuck — missing API key, permissions blocked, missing dependencies, or anything else you can't resolve — respond in this format and stop:

**I need:** <plain-language one-liner. Assume the user may not be a developer. Say "your OpenAI API key" not "OPENAI_API_KEY env var".>

**How to fix:** <one specific step. If a terminal command is required, write "Open a terminal in your project folder and run:" then put the command on its own line in a code block. If it's getting an API key, link to where to get one (e.g. https://platform.openai.com/api-keys).>

Use plain language. One line each. Don't paste code snippets unless they need to be run.`;
