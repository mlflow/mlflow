import { isEqual } from 'lodash';
import { useSelector } from 'react-redux';
import type { ExperimentStoreEntities } from '../../../types';

export type useExperimentIdsNotInStateResult = string[];

/**
 * Hook that returns data and functions necessary for rendering
 * experiment(s) details - name, title, tags etc.
 */
export const useExperimentIdsNotInState = (ids: string[]): useExperimentIdsNotInStateResult =>
  useSelector(
    (state: { entities: ExperimentStoreEntities }) => {
      const allExperiments = Object.values(state.entities.experimentsById).map(
        (e) => e.experiment_id,
      );
      return ids.filter((e) => !allExperiments.includes(e));
    },
    (oldExperiments, newExperiments) => isEqual(oldExperiments, newExperiments),
  );
