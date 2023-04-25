import Utils from './Utils';

export const MLFLOW_INTERNAL_PREFIX = 'mlflow.';

export function getLoggedModelPathsFromTags(runTags: any) {
  const models = Utils.getLoggedModelsFromTags(runTags);
  return models ? models.map((model) => (model as any).artifactPath) : [];
}
