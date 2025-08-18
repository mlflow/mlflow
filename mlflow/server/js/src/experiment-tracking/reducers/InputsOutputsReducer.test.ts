import { fulfilled } from '../../common/utils/ActionUtils';
import { GET_RUN_API, LOAD_MORE_RUNS_API, SEARCH_RUNS_API } from '../actions';
import { mockRunInfo } from '../utils/test-utils/ReduxStoreFixtures';
import { runInputsOutputsByUuid } from './InputsOutputsReducer';

describe('test runInputsOutputsByUuid', () => {
  test('should set up initial state correctly', () => {
    expect(runInputsOutputsByUuid(undefined, {})).toEqual({});
  });

  test('GET_RUN_API correctly updates state', () => {
    const runUuid = 'run01';
    const inputs = {
      datasetInputs: [
        {
          dataset: {
            digest: 'digest1',
            name: 'dataset1',
            profile: '{}',
            schema: '{}',
            source: '{}',
            sourceType: 'local',
          },
          tags: [],
        },
      ],
    };
    const outputs = {
      modelOutputs: [{ modelId: 'model1' }],
    };

    const action = {
      type: fulfilled(GET_RUN_API),
      payload: {
        run: {
          info: { runUuid },
          inputs,
          outputs,
        },
      },
    };

    const newState = runInputsOutputsByUuid(undefined, action);
    expect(newState).toEqual({
      [runUuid]: {
        inputs,
        outputs,
      },
    });
  });

  test('SEARCH_RUNS_API correctly updates state', () => {
    const runA = mockRunInfo('runA');
    const runB = mockRunInfo('runB');
    const runC = mockRunInfo('runC');

    const inputsA = {
      datasetInputs: [
        {
          dataset: {
            digest: 'digest1',
            name: 'dataset1',
            profile: '{}',
            schema: '{}',
            source: '{}',
            sourceType: 'local',
          },
          tags: [],
        },
      ],
    };

    const outputsB = {
      modelOutputs: [{ modelId: 'model1' }],
    };

    const action = {
      type: fulfilled(SEARCH_RUNS_API),
      payload: {
        runs: [
          // Run with only inputs
          { info: runA, inputs: inputsA },
          // Run with only outputs
          { info: runB, outputs: outputsB },
          // Run with neither inputs nor outputs
          { info: runC },
        ],
      },
    };

    const newState = runInputsOutputsByUuid(undefined, action);
    expect(newState).toEqual({
      [runA.runUuid]: {
        inputs: inputsA,
        outputs: undefined,
      },
      [runB.runUuid]: {
        inputs: undefined,
        outputs: outputsB,
      },
      [runC.runUuid]: {
        inputs: undefined,
        outputs: undefined,
      },
    });
  });

  test('LOAD_MORE_RUNS_API correctly updates state', () => {
    // Initial state with existing runs
    const initialState = {
      existingRun: {
        inputs: { modelInputs: [{ modelId: 'model1' }] },
        outputs: {},
      },
    };

    const runD = mockRunInfo('runD');
    const runE = mockRunInfo('runE');

    const inputsD = {
      modelInputs: [{ modelId: 'model2' }],
    };

    const outputsE = {
      modelOutputs: [{ modelId: 'model3' }],
    };

    const action = {
      type: fulfilled(LOAD_MORE_RUNS_API),
      payload: {
        runs: [
          { info: runD, inputs: inputsD },
          { info: runE, outputs: outputsE },
        ],
      },
    };

    const newState = runInputsOutputsByUuid(initialState, action);
    expect(newState).toEqual({
      existingRun: {
        inputs: { modelInputs: [{ modelId: 'model1' }] },
        outputs: {},
      },
      [runD.runUuid]: {
        inputs: inputsD,
        outputs: undefined,
      },
      [runE.runUuid]: {
        inputs: undefined,
        outputs: outputsE,
      },
    });
  });

  test('SEARCH_RUNS_API with no payload', () => {
    expect(
      runInputsOutputsByUuid(undefined, {
        type: fulfilled(SEARCH_RUNS_API),
      }),
    ).toEqual({});
  });

  test('SEARCH_RUNS_API with empty runs array', () => {
    expect(
      runInputsOutputsByUuid(undefined, {
        type: fulfilled(SEARCH_RUNS_API),
        payload: {
          runs: [],
        },
      }),
    ).toEqual({});
  });
});
