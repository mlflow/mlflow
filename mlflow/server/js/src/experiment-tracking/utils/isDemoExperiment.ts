import type { ExperimentEntity } from '../types';

const DEMO_VERSION_TAG_PREFIX = 'mlflow.demo.version.';

export const isDemoExperiment = (experiment: ExperimentEntity): boolean =>
  experiment.tags?.some(({ key }) => key.startsWith(DEMO_VERSION_TAG_PREFIX)) ?? false;
