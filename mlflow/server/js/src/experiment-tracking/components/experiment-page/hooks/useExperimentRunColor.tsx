import { useDispatch, useSelector } from 'react-redux';
import { MLFLOW_RUN_COLOR_TAG } from '../../../constants';
import type { ReduxState, ThunkDispatch } from '../../../../redux-types';
import { setTagApi } from '../../../actions';
import { useCallback, useEffect, useMemo } from 'react';
import { getStableColorForRun } from '../../../utils/RunNameUtils';
import {
  RUN_COLOR_ACTION_INITIALIZE_RUN_COLORS,
  RUN_COLOR_ACTION_SET_RUN_COLOR,
} from '../../../reducers/RunColorReducer';

const STORAGE_KEY = 'experimentRunColors';

export type SaveExperimentRunColorFn = (args: { runUuid?: string; groupUuid?: string; colorValue: string }) => void;

const loadSavedColors = () => {
  const savedColorsRaw = window.localStorage.getItem(STORAGE_KEY);
  try {
    return savedColorsRaw ? JSON.parse(savedColorsRaw) : {};
  } catch {
    return {};
  }
};

export const useInitializeExperimentRunColors = () => {
  const dispatch = useDispatch<ThunkDispatch>();

  useEffect(() => {
    dispatch({
      type: RUN_COLOR_ACTION_INITIALIZE_RUN_COLORS,
      values: loadSavedColors(),
    });
  }, [dispatch]);
};

export const useSaveExperimentRunColor = () => {
  const dispatch = useDispatch<ThunkDispatch>();

  return useCallback<SaveExperimentRunColorFn>(
    ({ colorValue, runUuid, groupUuid }) => {
      const runOrGroupUuid = groupUuid ?? runUuid;
      if (runOrGroupUuid) {
        dispatch({
          type: RUN_COLOR_ACTION_SET_RUN_COLOR,
          runOrGroupUuid,
          colorValue,
        });
      }
      if (runUuid) {
        dispatch(setTagApi(runUuid, MLFLOW_RUN_COLOR_TAG, colorValue));
      }
      if (groupUuid) {
        const colors = loadSavedColors();
        colors[groupUuid] = colorValue;
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(colors));
      }
    },
    [dispatch],
  );
};

export const useGetExperimentRunColor = () => {
  const colorByRunUuid = useSelector((state: ReduxState) => state.entities.colorByRunUuid);

  return useCallback(
    (runOrGroupUuid = '') => {
      return colorByRunUuid[runOrGroupUuid] || getStableColorForRun(runOrGroupUuid);
    },
    [colorByRunUuid],
  );
};
