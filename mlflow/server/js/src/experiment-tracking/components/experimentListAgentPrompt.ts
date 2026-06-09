export const buildCreateExperimentPrompt = (trackingUri: string): string => `Create a new MLflow experiment.

MLflow is running at ${trackingUri}. Ask me what to name the experiment, then create it with \`mlflow.set_experiment(name)\` (which creates it if it doesn't exist and sets it as active). 

Report the experiment ID and URL once it's created. For MLflow API/usage questions, use the \`searching-mlflow-docs\` skill rather than guessing.

If the skills aren't installed locally, refer to https://github.com/mlflow/skills/blob/main/searching-mlflow-docs/SKILL.md
`;

export const buildCreateExperimentAssistantPrompt = (
  trackingUri: string,
): string => `Help me create my first MLflow experiment. MLflow is running at ${trackingUri}.

Ask me what I want to name the experiment, then create it via the MLflow API and report the experiment ID and URL once it's created.

Use your built-in MLflow docs knowledge — don't guess at APIs.`;
