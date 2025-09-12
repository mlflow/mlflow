import { useCallback, useEffect, useMemo, useReducer, useState } from 'react';
import type { ExperimentPageUIState } from '../models/ExperimentPageUIState';
import { EXPERIMENT_PAGE_UI_STATE_FIELDS, createExperimentPageUIState } from '../models/ExperimentPageUIState';
import { loadExperimentViewState } from '../utils/persistSearchFacets';
import { keys, pick } from 'lodash';
import type { ExperimentRunsSelectorResult } from '../utils/experimentRuns.selector';
import type { UseExperimentsResult } from './useExperiments';
import { useUpdateExperimentPageSearchFacets } from './useExperimentPageSearchFacets';
import { expandedEvaluationRunRowsUIStateInitializer } from '../utils/expandedRunsViewStateInitializer';
import { shouldRerunExperimentUISeeding } from '../../../../common/utils/FeatureUtils';

// prettier-ignore
const uiStateInitializers = [
  expandedEvaluationRunRowsUIStateInitializer,
];

type UpdateUIStateAction = {
  type: 'UPDATE_UI_STATE';
  payload: ExperimentPageUIState | ((current: ExperimentPageUIState) => ExperimentPageUIState);
};

type SetupInitUIStateAction = {
  type: 'INITIAL_UI_STATE_SEEDED';
};

type LoadNewExperimentAction = {
  type: 'LOAD_NEW_EXPERIMENT';
  payload: { uiState: ExperimentPageUIState; isSeeded: boolean; isFirstVisit: boolean; newPersistKey: string };
};

type UIStateContainer = {
  uiState: ExperimentPageUIState;
  currentPersistKey: string;
  isSeeded: boolean;
  /**
   * Indicates if the user is visiting the experiment page for the first time in the current session.
   */
  isFirstVisit: boolean;
};

const baseState = createExperimentPageUIState();

export const useInitializeUIState = (
  experimentIds: string[],
): [
  ExperimentPageUIState,
  React.Dispatch<React.SetStateAction<ExperimentPageUIState>>,
  (experiments: UseExperimentsResult, runs: ExperimentRunsSelectorResult) => void,
] => {
  const persistKey = useMemo(() => JSON.stringify(experimentIds.sort()), [experimentIds]);

  // Hash of the current experiment and runs. Used to determine if the UI state could be re-seeded.
  const [experimentHash, setExperimentHash] = useState<string | null>(null);

  const updateSearchFacets = useUpdateExperimentPageSearchFacets();

  const [{ uiState, isSeeded, isFirstVisit }, dispatchAction] = useReducer(
    (state: UIStateContainer, action: UpdateUIStateAction | SetupInitUIStateAction | LoadNewExperimentAction) => {
      if (action.type === 'UPDATE_UI_STATE') {
        const newState = typeof action.payload === 'function' ? action.payload(state.uiState) : action.payload;
        return {
          ...state,
          uiState: newState,
        };
      }
      if (action.type === 'INITIAL_UI_STATE_SEEDED') {
        if (state.isSeeded) {
          return state;
        }
        return {
          ...state,
          isSeeded: true,
        };
      }
      if (action.type === 'LOAD_NEW_EXPERIMENT') {
        return {
          uiState: action.payload.uiState,
          isFirstVisit: action.payload.isFirstVisit,
          currentPersistKey: action.payload.newPersistKey,
          isSeeded: action.payload.isSeeded,
        };
      }
      return state;
    },
    undefined,
    () => {
      const persistedViewState = loadExperimentViewState(persistKey);
      const persistedStateFound = Boolean(keys(persistedViewState || {}).length);
      const persistedUIState = persistedStateFound ? pick(persistedViewState, EXPERIMENT_PAGE_UI_STATE_FIELDS) : {};
      return {
        uiState: { ...baseState, ...persistedUIState },
        isSeeded: persistedStateFound,
        isFirstVisit: !persistedStateFound,
        currentPersistKey: persistKey,
      };
    },
  );

  const setUIState = useCallback(
    (newStateOrSelector: ExperimentPageUIState | ((current: ExperimentPageUIState) => ExperimentPageUIState)) => {
      dispatchAction({ type: 'UPDATE_UI_STATE', payload: newStateOrSelector });
    },
    [],
  );

  const seedInitialUIState = useCallback(
    (experiments: UseExperimentsResult, runs: ExperimentRunsSelectorResult) => {
      // Disable if there are no experiments/runs or if the state has already been persisted previously
      if (!isFirstVisit || experiments.length === 0 || runs.runInfos.length === 0) {
        return;
      }

      const newHash = generateExperimentHash(runs, experiments);

      if (experimentHash === newHash && isSeeded) {
        // Do not re-seed if the hash is the same, as we don't expect changes in the UI state
        return;
      }

      if (isSeeded && !shouldRerunExperimentUISeeding()) {
        // Do not re-seed if the feature is not enabled
        return;
      }

      // Then, update the UI state using all known UI state initializers
      setUIState((uiState) => {
        const newUIState = uiStateInitializers.reduce(
          (state, initializer) => initializer(experiments, state, runs, isSeeded),
          {
            ...uiState,
          },
        );
        return newUIState;
      });

      setExperimentHash(newHash);
      if (!isSeeded) {
        // Mark the initial state as seeded (effectively set isSeeded to true)
        dispatchAction({ type: 'INITIAL_UI_STATE_SEEDED' });
      }
    },
    // prettier-ignore
    [
      isSeeded,
      isFirstVisit,
      setUIState,
      experimentHash,
    ],
  );

  // Each time persist key (experiment IDs) change, load persisted view state
  useEffect(() => {
    const persistedViewState = loadExperimentViewState(persistKey);
    const persistedUIState = pick(persistedViewState, EXPERIMENT_PAGE_UI_STATE_FIELDS);
    const isSeeded = Boolean(keys(persistedViewState || {}).length);
    const isFirstVisit = !isSeeded;

    dispatchAction({
      type: 'LOAD_NEW_EXPERIMENT',
      payload: { uiState: { ...baseState, ...persistedUIState }, isSeeded, isFirstVisit, newPersistKey: persistKey },
    });
  }, [persistKey]);

  return [uiState, setUIState, seedInitialUIState];
};

export const generateExperimentHash = (runs: ExperimentRunsSelectorResult, experiments: UseExperimentsResult) => {
  if (runs.runInfos.length === 0 || experiments.length === 0) {
    return null;
  }

  const sortedExperimentIds = experiments.map((exp) => exp.experimentId).sort();

  const sortedRunUuids = runs.runInfos.map((run) => run.runUuid).sort();

  return `${sortedExperimentIds.join(':')}:${sortedRunUuids.join(':')}`;
};
