import { useCallback, useEffect, useMemo, useReducer, useState } from 'react';
import {
  EXPERIMENT_PAGE_UI_STATE_FIELDS,
  ExperimentPageUIStateV2,
  createExperimentPageUIStateV2,
} from '../models/ExperimentPageUIStateV2';
import { loadExperimentViewState } from '../utils/persistSearchFacets';
import { keys, pick } from 'lodash';
import { shouldEnableShareExperimentViewByTags } from '../../../../common/utils/FeatureUtils';
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
  payload: ExperimentPageUIStateV2 | ((current: ExperimentPageUIStateV2) => ExperimentPageUIStateV2);
};

type SetupInitUIStateAction = {
  type: 'INITIAL_UI_STATE_SEEDED';
};

type LoadNewExperimentAction = {
  type: 'LOAD_NEW_EXPERIMENT';
  payload: { uiState: ExperimentPageUIStateV2; isFirstVisit: boolean; newPersistKey: string };
};

type UIStateContainer = {
  uiState: ExperimentPageUIStateV2;
  currentPersistKey: string;
  isFirstVisit: boolean;
};

const baseState = createExperimentPageUIStateV2();

const usingNewViewStateModel = () => shouldEnableShareExperimentViewByTags();

export const useInitializeUIState = (
  experimentIds: string[],
): [
  ExperimentPageUIStateV2,
  React.Dispatch<React.SetStateAction<ExperimentPageUIStateV2>>,
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
      // Do not bother restoring the state if the feature flag is not set
      if (!usingNewViewStateModel()) {
        return {
          uiState: baseState,
          isFirstVisit: false,
          currentPersistKey: persistKey,
        };
      }
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
    (newStateOrSelector: ExperimentPageUIStateV2 | ((current: ExperimentPageUIStateV2) => ExperimentPageUIStateV2)) => {
      dispatchAction({ type: 'UPDATE_UI_STATE', payload: newStateOrSelector });
    },
    [],
  );

  const seedInitialUIState = useCallback(
    (experiments: UseExperimentsResult, runs: ExperimentRunsSelectorResult) => {
      // Return early if the feature flag is not set
      if (!usingNewViewStateModel()) {
        return;
      }

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
    if (!usingNewViewStateModel()) {
      return;
    }
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
