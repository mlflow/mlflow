export const buildEvaluatePrompt = (
  experimentId: string,
): string => `Run an evaluation on my GenAI app in MLflow. Target experiment ID: ${experimentId}.

1. Add \`mlflow\` as a dependency if needed.
2. Set the tracking URI to match the project (read \`MLFLOW_TRACKING_URI\` or ask me), then call \`mlflow.set_experiment(experiment_id="${experimentId}")\`.
3. Choose an approach: if traces already exist here, evaluate them with \`mlflow.search_traces(...)\`; otherwise build a small \`eval_dataset\` with a \`predict_fn\`. Use built-in scorers (Safety, RelevanceToQuery, Guidelines) unless custom logic is needed. Use \`searching-mlflow-docs\` for API questions (fallback: https://github.com/mlflow/skills).
4. Run \`mlflow.genai.evaluate(...)\` and confirm an evaluation run appears.

Infer the framework, entry point, and whether traces exist from the repo. If you hit a blocker you can't resolve (missing API key, permissions, dependencies), stop and tell me what you need.`;

export const buildEvaluateAssistantPrompt = (
  experimentId: string,
): string => `Help me set up evaluation for my GenAI app in MLflow. Target experiment ID: ${experimentId}.

Ask about my app (what it does, what framework it uses, whether traces already exist here), then recommend an approach: built-in scorers (Safety, RelevanceToQuery, Guidelines) vs custom, and evaluating existing traces vs a small \`eval_dataset\` with a \`predict_fn\`. Walk me through the minimal \`mlflow.genai.evaluate(...)\` setup.

Use your built-in MLflow docs knowledge rather than guessing at APIs.`;
