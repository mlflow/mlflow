import { fulfilled } from '../../common/utils/ActionUtils';
import { GET_RUN_API, SEARCH_RUNS_API } from '../actions';
import { MLFLOW_RUN_COLOR_TAG } from '../constants';
import {
  RUN_COLOR_ACTION_INITIALIZE_RUN_COLORS,
  RUN_COLOR_ACTION_SET_RUN_COLOR,
  colorByRunUuid,
} from './RunColorReducer';

describe('colorByRunUuid reducer', () => {
  it('handles initialization action', () => {
    const initialState = {};
    const action = {
      type: RUN_COLOR_ACTION_INITIALIZE_RUN_COLORS,
      values: {
        runUuid1: '#FF0000',
        runUuid2: '#0000AA',
      },
    };
    const newState = colorByRunUuid(initialState, action);
    expect(newState).toEqual({
      runUuid1: '#FF0000',
      runUuid2: '#0000AA',
    });
  });

  it('handles setting color action', () => {
    const initialState = {
      runUuid1: '#FF0000',
    };
    const action = {
      type: RUN_COLOR_ACTION_SET_RUN_COLOR,
      runOrGroupUuid: 'runUuid2',
      colorValue: '#0000AA',
    };
    const newState = colorByRunUuid(initialState, action);
    expect(newState).toEqual({
      runUuid1: '#FF0000',
      runUuid2: '#0000AA',
    });
  });

  it('handles API response for a single run', () => {
    const initialState = {};
    const action = {
      type: fulfilled(GET_RUN_API),
      payload: {
        run: {
          info: {
            runUuid: 'runUuid1',
          },
          data: {
            tags: [{ key: MLFLOW_RUN_COLOR_TAG, value: '#FF0000' }],
          },
        },
      },
    };
    const newState = colorByRunUuid(initialState, action);
    expect(newState).toEqual({
      runUuid1: '#FF0000',
    });
  });

  it('handles API response for multiple runs', () => {
    const initialState = {};
    const action = {
      type: fulfilled(SEARCH_RUNS_API),
      payload: {
        runs: [
          {
            info: {
              runUuid: 'runUuid1',
            },
            data: {
              tags: [{ key: MLFLOW_RUN_COLOR_TAG, value: '#FF0000' }],
            },
          },
          {
            info: {
              runUuid: 'runUuid2',
            },
            data: {
              tags: [{ key: MLFLOW_RUN_COLOR_TAG, value: '#0000AA' }],
            },
          },
        ],
      },
    };
    const newState = colorByRunUuid(initialState, action);
    expect(newState).toEqual({
      runUuid1: '#FF0000',
      runUuid2: '#0000AA',
    });
  });

  it('handles unknown action type', () => {
    const initialState = {
      runUuid1: '#FF0000',
    };
    const action = {
      type: 'UNKNOWN_ACTION',
    };
    const newState = colorByRunUuid(initialState, action);
    expect(newState).toEqual({
      runUuid1: '#FF0000',
    });
  });
});
