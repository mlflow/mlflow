import { useCallback, useEffect, useMemo, useReducer, useState } from 'react';
import {
  EXPERIMENT_PAGE_UI_STATE_FIELDS,
  ExperimentPageUIState,
  createExperimentPageUIState,
} from '../models/ExperimentPageUIState';
import { loadExperimentViewState } from '../utils/persistSearchFacets';
import { keys, pick } from 'lodash';
import { ExperimentRunsSelectorResult } from '../utils/experimentRuns.selector';
import { UseExperimentsResult } from './useExperiments';
import { useUpdateExperimentPageSearchFacets } from './useExperimentPageSearchFacets';
import { expandedEvaluationRunRowsUIStateInitializer } from '../utils/expandedRunsViewStateInitializer';

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
  payload: { uiState: ExperimentPageUIState; isFirstVisit: boolean; newPersistKey: string };
};

type UIStateContainer = {
  uiState: ExperimentPageUIState;
  currentPersistKey: string;
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

  const updateSearchFacets = useUpdateExperimentPageSearchFacets();

  const [{ uiState, isFirstVisit }, dispatchAction] = useReducer(
    (state: UIStateContainer, action: UpdateUIStateAction | SetupInitUIStateAction | LoadNewExperimentAction) => {
      if (action.type === 'UPDATE_UI_STATE') {
        const newState = typeof action.payload === 'function' ? action.payload(state.uiState) : action.payload;
        return {
          ...state,
          uiState: newState,
        };
      }
      if (action.type === 'INITIAL_UI_STATE_SEEDED') {
        if (!state.isFirstVisit) {
          return state;
        }
        return {
          ...state,
          isFirstVisit: false,
        };
      }
      if (action.type === 'LOAD_NEW_EXPERIMENT') {
        return {
          uiState: action.payload.uiState,
          isFirstVisit: action.payload.isFirstVisit,
          currentPersistKey: action.payload.newPersistKey,
        };
      }
      return state;
    },
    undefined,
    () => {
      const persistedViewState = loadExperimentViewState(persistKey);
      const persistedStateFound = keys(persistedViewState || {}).length;
      const persistedUIState = persistedStateFound ? pick(persistedViewState, EXPERIMENT_PAGE_UI_STATE_FIELDS) : {};
      return {
        uiState: { ...baseState, ...persistedUIState },
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
      // Disable if it's not the first visit or there are no experiments/runs
      if (!isFirstVisit || experiments.length === 0 || runs.runInfos.length === 0) {
        return;
      }

      // Mark the initial state as seeded (effectively set isFirstVisit to false)
      dispatchAction({ type: 'INITIAL_UI_STATE_SEEDED' });

      // Then, update the UI state using all known UI state initializers
      setUIState((uiState) => {
        const newUIState = uiStateInitializers.reduce((state, initializer) => initializer(experiments, state, runs), {
          ...uiState,
        });

        return newUIState;
      });
    },
    // prettier-ignore
    [
      isFirstVisit,
      setUIState,
    ],
  );

  // Each time persist key (experiment IDs) change, load persisted view state
  useEffect(() => {
    const persistedViewState = loadExperimentViewState(persistKey);
    const persistedUIState = pick(persistedViewState, EXPERIMENT_PAGE_UI_STATE_FIELDS);
    const isFirstVisit = !keys(persistedViewState || {}).length;
    dispatchAction({
      type: 'LOAD_NEW_EXPERIMENT',
      payload: { uiState: { ...baseState, ...persistedUIState }, isFirstVisit, newPersistKey: persistKey },
    });
  }, [persistKey]);

  return [uiState, setUIState, seedInitialUIState];
};
