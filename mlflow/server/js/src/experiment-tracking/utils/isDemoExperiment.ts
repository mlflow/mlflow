export const DEMO_VERSION_TAG_PREFIX = 'mlflow.demo.version.';

export const isDemoExperiment = (experiment: { tags?: { key?: string | null }[] | null }): boolean =>
  experiment.tags?.some(({ key }) => key?.startsWith(DEMO_VERSION_TAG_PREFIX)) ?? false;
