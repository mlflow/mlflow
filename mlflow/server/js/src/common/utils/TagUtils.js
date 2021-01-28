import Utils from './Utils';

export const MLFLOW_INTERNAL_PREFIX = 'mlflow.';

export function getLoggedModelPathsFromTags(runTags) {
  const modelsTag = runTags[Utils.loggedModelsTag];
  if (modelsTag) {
    const models = JSON.parse(modelsTag.value);
    return models ? models.map((model) => model['artifact_path']) : [];
  }
  return [];
}
