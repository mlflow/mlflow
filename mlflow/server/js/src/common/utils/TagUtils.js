import Utils from './Utils';

export const MLFLOW_INTERNAL_PREFIX = 'mlflow.';

export function getLoggedModelPathsFromTags(runTags) {
  const models = Utils.getLoggedModelsFromTags(runTags);
  return models ? models.map((model) => model.artifactPath) : [];
}
