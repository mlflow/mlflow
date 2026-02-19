import { fulfilled } from '../../common/utils/ActionUtils';
import { GET_RUN_API, LOAD_MORE_RUNS_API, SEARCH_RUNS_API } from '../actions';
import { MLFLOW_RUN_COLOR_TAG } from '../constants';
import type { RunEntity } from '../types';

export const RUN_COLOR_ACTION_INITIALIZE_RUN_COLORS = 'INITIALIZE_RUN_COLORS';
export const RUN_COLOR_ACTION_SET_RUN_COLOR = 'SET_RUN_COLOR';

/**
 * Reducer for run colors. The state is a mapping from run/group UUIDs to color values.
 * Allows to manually set colors for runs/groups, but also listens to API responses and
 * automatically sets colors for runs that have a color tag.
 */
export const colorByRunUuid = (state: Record<string, string> = {}, action: any) => {
  switch (action.type) {
    case RUN_COLOR_ACTION_INITIALIZE_RUN_COLORS: {
      return { ...state, ...action.values };
    }
    case RUN_COLOR_ACTION_SET_RUN_COLOR: {
      const { runOrGroupUuid, colorValue } = action;
      return { ...state, [runOrGroupUuid]: colorValue };
    }
    case fulfilled(GET_RUN_API): {
      const run: RunEntity = action.payload.run;
      const runUuid = run.info.runUuid;
      const colorTag = run?.data?.tags?.find((tag) => tag.key === MLFLOW_RUN_COLOR_TAG);
      if (colorTag) {
        return { ...state, [runUuid]: colorTag.value };
      }

      return state;
    }
    case fulfilled(SEARCH_RUNS_API):
    case fulfilled(LOAD_MORE_RUNS_API): {
      const newState = { ...state };
      if (action.payload && action.payload.runs) {
        for (const run of action.payload.runs as RunEntity[]) {
          const runUuid = run.info.runUuid;
          const colorTag = run?.data?.tags?.find((tag) => tag.key === MLFLOW_RUN_COLOR_TAG);
          if (colorTag) {
            newState[runUuid] = colorTag.value;
          }
        }
      }
      return newState;
    }

    default:
      return state;
  }
};
