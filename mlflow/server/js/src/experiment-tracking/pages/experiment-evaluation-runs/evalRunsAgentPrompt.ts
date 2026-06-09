export const buildEvaluatePrompt = (
  trackingUri: string,
  experimentId: string,
): string => `Run an evaluation on my GenAI app in MLflow. Tracking server: ${trackingUri}. Target experiment ID: ${experimentId}.

Steps:
1. Add \`mlflow\` as a dependency if it isn't already.
2. Wire \`MLFLOW_TRACKING_URI=${trackingUri}\` (env file or \`set_tracking_uri\` — match project conventions). Use \`mlflow.set_experiment(experiment_id="${experimentId}")\`.
3. Pick an approach: if the app already has traces in this experiment, evaluate against \`mlflow.search_traces(...)\`. Otherwise build a small \`eval_dataset\` and pass a \`predict_fn\`. Use built-in scorers (Safety, RelevanceToQuery, Guidelines) unless the app needs custom logic. For MLflow API/usage questions, use the \`searching-mlflow-docs\` skill rather than guessing.
   If the skills aren't installed locally, refer to https://github.com/mlflow/skills/blob/main/searching-mlflow-docs/SKILL.md
4. Run \`mlflow.genai.evaluate(...)\` and confirm an evaluation run appears.

Read the repo to infer the LLM framework, whether traces already exist, and the app's entry point. Don't ask if you can figure it out.

If you get stuck — missing API key, blocked permissions, missing deps — stop and tell the user in plain language what you need and how to fix it. Don't run the app expecting it to fail.`;

export const buildEvaluateAssistantPrompt = (
  trackingUri: string,
  experimentId: string,
): string => `Help me set up evaluation for my GenAI app in MLflow. Tracking server: ${trackingUri}. Target experiment ID: ${experimentId}.

Ask me about my app (what it does, what framework it uses, whether I already have traces in this experiment), then recommend an approach — built-in scorers (Safety, RelevanceToQuery, Guidelines) vs custom scorers, evaluating against existing traces vs a small \`eval_dataset\` with a \`predict_fn\`. Walk me through the minimal \`mlflow.genai.evaluate(...)\` setup.

Use your built-in MLflow docs knowledge — don't guess at APIs.`;
