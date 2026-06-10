export const buildCreateExperimentPrompt = (): string => `Create a new MLflow experiment.

Ask me what to name it, then create it with \`mlflow.set_experiment(name)\` and report the experiment ID and URL.

Use the \`searching-mlflow-docs\` skill for API questions rather than guessing (fallback: https://github.com/mlflow/skills).
`;

export const buildCreateExperimentAssistantPrompt = (): string => `Help me create my first MLflow experiment.

Ask what to name it, create it via the MLflow API, and report the experiment ID and URL.

Use your built-in MLflow docs knowledge rather than guessing at APIs.`;
