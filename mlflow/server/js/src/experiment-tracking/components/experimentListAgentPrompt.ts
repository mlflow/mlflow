export const buildCreateExperimentPrompt = (trackingUri: string): string => `Create a new MLflow experiment.

MLflow is running at ${trackingUri}. Ask me what to name the experiment, then create it with \`mlflow.set_experiment(name)\` (which creates it if it doesn't exist and sets it as active). Report the experiment ID and URL once it's created.`;
