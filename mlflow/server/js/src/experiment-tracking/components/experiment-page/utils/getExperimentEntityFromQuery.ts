import type { UseGetExperimentQueryResultExperiment } from '../../../hooks/useExperimentQuery';
import type { ExperimentEntity } from '../../../types';

export const getExperimentEntityFromQuery = (
  experiment: UseGetExperimentQueryResultExperiment | undefined,
): ExperimentEntity | null => {
  if (!experiment) {
    return null;
  }

  return {
    ...experiment,
    creationTime: Number(experiment.creationTime),
    lastUpdateTime: Number(experiment.lastUpdateTime),
  } as ExperimentEntity;
};
