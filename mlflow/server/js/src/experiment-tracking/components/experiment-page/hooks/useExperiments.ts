import { isEqual } from 'lodash';
import { useSelector } from 'react-redux';
import type { ExperimentEntity, ExperimentStoreEntities } from '../../../types';

export type UseExperimentsResult = ExperimentEntity[];

/**
 * Hook that returns data and functions necessary for rendering
 * experiment(s) details - name, title, tags etc.
 */
export const useExperiments = (ids: (number | string)[]): UseExperimentsResult =>
  useSelector(
    (state: { entities: ExperimentStoreEntities }) =>
      ids.map((id) => state.entities.experimentsById[id]).filter(Boolean),
    (oldExperiments, newExperiments) => isEqual(oldExperiments, newExperiments),
  );
