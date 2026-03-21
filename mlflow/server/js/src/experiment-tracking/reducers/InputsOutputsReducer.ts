import { fulfilled } from '../../common/utils/ActionUtils';
import { LOAD_MORE_RUNS_API, SEARCH_RUNS_API, GET_RUN_API } from '../actions';
import type { RunInputsType, RunOutputsType } from '../types';

export const runInputsOutputsByUuid = (
  state: Record<string, { inputs?: RunInputsType; outputs?: RunOutputsType }> = {},
  action: any,
) => {
  switch (action.type) {
    case fulfilled(GET_RUN_API): {
      const { run } = action.payload;
      const runUuid = run.info.runUuid;
      if (runUuid) {
        return {
          ...state,
          [runUuid]: {
            inputs: run.inputs,
            outputs: run.outputs,
          },
        };
      }
      return state;
    }
    case fulfilled(SEARCH_RUNS_API):
    case fulfilled(LOAD_MORE_RUNS_API): {
      if (action.payload && action.payload.runs) {
        const newState = { ...state };
        action.payload.runs.forEach(
          (runJson: { info: { runUuid: string }; inputs?: RunInputsType; outputs?: RunOutputsType }) => {
            if (!runJson) {
              return;
            }
            const runUuid: string = runJson.info.runUuid;
            if (runUuid) {
              newState[runUuid] = {
                inputs: runJson.inputs,
                outputs: runJson.outputs,
              };
            }
          },
        );
        return newState;
      }
      return state;
    }
    default:
      return state;
  }
};
