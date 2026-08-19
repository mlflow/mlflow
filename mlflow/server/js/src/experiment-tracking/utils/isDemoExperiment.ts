import type { ExperimentEntity } from '../types';

const DEMO_VERSION_TAG_PREFIX = 'mlflow.demo.version.';

export const hasDemoVersionTag = (tags?: Array<{ key: string }>): boolean =>
  tags?.some(({ key }) => key.startsWith(DEMO_VERSION_TAG_PREFIX)) ?? false;

export const isDemoExperiment = (experiment: ExperimentEntity): boolean =>
  hasDemoVersionTag(experiment.tags);
