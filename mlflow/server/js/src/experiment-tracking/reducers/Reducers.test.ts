/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { ArtifactNode } from '../utils/ArtifactUtils';
import {
  experimentsById,
  runInfosByUuid,
  paramsByRunUuid,
  tagsByRunUuid,
  artifactsByRunUuid,
  artifactRootUriByRunUuid,
  experimentTagsByExperimentId,
  getParams,
  getRunTags,
  getRunInfo,
  getExperiments,
  getExperiment,
  getExperimentTags,
  apis,
  rootReducer,
  getApis,
  getArtifacts,
  getSharedParamKeysByRunUuids,
  getAllParamKeysByRunUuids,
  getArtifactRootUri,
  modelVersionsByRunUuid,
  runUuidsMatchingFilter,
  runDatasetsByUuid,
  datasetsByExperimentId,
  runInfoOrderByUuid,
} from './Reducers';
import { mockExperiment, mockRunInfo } from '../utils/test-utils/ReduxStoreFixtures';
import { RunTag, Param, ExperimentTag } from '../sdk/MlflowMessages';
import {
  GET_EXPERIMENT_API,
  GET_RUN_API,
  SEARCH_RUNS_API,
  LOAD_MORE_RUNS_API,
  SET_TAG_API,
  DELETE_TAG_API,
  LIST_ARTIFACTS_API,
  SET_EXPERIMENT_TAG_API,
  SEARCH_DATASETS_API,
} from '../actions';
import { fulfilled, pending, rejected } from '../../common/utils/ActionUtils';
import { deepFreeze } from '../../common/utils/TestUtils';
import { mockModelVersionDetailed } from '../../model-registry/test-utils';
import { SEARCH_MODEL_VERSIONS } from '../../model-registry/actions';
import { Stages, ModelVersionStatus } from '../../model-registry/constants';
describe('test experimentsById', () => {
  test('should set up initial state correctly', () => {
    expect(experimentsById(undefined, {})).toEqual({});
  });
  test('getExperiment correctly updates empty state', () => {
    const experimentA = mockExperiment('experiment01', 'experimentA');
    const state = {};
    const action = {
      type: fulfilled(GET_EXPERIMENT_API),
      payload: {
        experiment: experimentA,
      },
    };
    const new_state = experimentsById(state, action);
    expect(new_state).not.toEqual(state);
    expect(new_state).toEqual({
      [experimentA.experimentId]: experimentA,
    });
  });
  test('getExperiment correctly updates non empty state', () => {
    const preserved = mockExperiment('experiment03', 'still exists');
    const replacedOld = mockExperiment('experiment05', 'replacedOld');
    const replacedNew = mockExperiment('experiment05', 'replacedNew');
    const state = deepFreeze({
      [preserved.experimentId]: preserved,
      [replacedOld.experimentId]: replacedOld,
    });
    const action = {
      type: fulfilled(GET_EXPERIMENT_API),
      payload: {
        experiment: replacedNew,
      },
    };
    const new_state = experimentsById(state, action);
    // make sure the reducer did not modify the original state
    expect(new_state).not.toEqual(state);
    expect(new_state).toEqual({
      [preserved.experimentId]: preserved,
      [replacedNew.experimentId]: replacedNew,
    });
  });
  test('getExperiment correctly updates tags', () => {
    const tag1 = { key: 'key1', value: 'value1' };
    const tag2 = { key: 'key2', value: 'value2' };
    const experiment1 = mockExperiment('experiment1', 'some name');
    const action1 = {
      type: fulfilled(GET_EXPERIMENT_API),
      payload: {
        experiment: {
          experimentId: experiment1.experimentId,
          tags: [tag1],
        },
      },
    };
    const state1 = experimentsById(undefined, action1);
    expect(state1.experiment1.tags).toEqual([tag1]);
    const action2 = {
      type: fulfilled(GET_EXPERIMENT_API),
      payload: {
        experiment: {
          experimentId: experiment1.experimentId,
          tags: [tag1, tag2],
        },
      },
    };
    const state2 = experimentsById(state1, action2);
    const { tags } = state2.experiment1;
    expect(tags).toEqual([tag1, tag2]);
  });
});

describe('test runUuidsMatchingFilter', () => {
  test('should set up initial state correctly', () => {
    expect(runUuidsMatchingFilter(undefined, {})).toEqual([]);
  });
  test('should do nothing without a payload', () => {
    expect(
      runUuidsMatchingFilter(undefined, {
        type: fulfilled(SEARCH_RUNS_API),
      }),
    ).toEqual([]);
  });
  test('should create a new set of UUIDs', () => {
    expect(
      // @ts-expect-error TS(2322): Type 'string' is not assignable to type 'never'.
      runUuidsMatchingFilter(['run01'], {
        type: fulfilled(SEARCH_RUNS_API),
        payload: {
          runsMatchingFilter: [
            {
              // @ts-expect-error TS(2345): Argument of type '"1"' is not assignable to parame... Remove this comment to see the full error message
              info: mockRunInfo('run02', '1'),
            },
            {
              // @ts-expect-error TS(2345): Argument of type '"1"' is not assignable to parame... Remove this comment to see the full error message
              info: mockRunInfo('run03', '1'),
            },
          ],
        },
      }),
    ).toEqual(['run02', 'run03']);
  });
  test('should be able to append new run UUIDs', () => {
    expect(
      // @ts-expect-error TS(2322): Type 'string' is not assignable to type 'never'.
      runUuidsMatchingFilter(['run01'], {
        type: fulfilled(LOAD_MORE_RUNS_API),
        payload: {
          runsMatchingFilter: [
            {
              // @ts-expect-error TS(2345): Argument of type '"1"' is not assignable to parame... Remove this comment to see the full error message
              info: mockRunInfo('run02', '1'),
            },
            {
              // @ts-expect-error TS(2345): Argument of type '"1"' is not assignable to parame... Remove this comment to see the full error message
              info: mockRunInfo('run03', '1'),
            },
          ],
        },
      }),
    ).toEqual(['run01', 'run02', 'run03']);
  });
});

describe('test runDatasetsByUuid', () => {
  test('should set up initial state correctly', () => {
    expect(runDatasetsByUuid(undefined, {})).toEqual({});
  });
  test('search api with no payload', () => {
    expect(
      runDatasetsByUuid(undefined, {
        type: fulfilled(SEARCH_RUNS_API),
      }),
    ).toEqual({});
  });
  test('searchRunApi correctly updates state', () => {
    const createDataset = (digest: string) => ({
      dataset: {
        digest,
        name: `dataset_${digest}`,
        profile: '{}',
        schema: '{}',
        source: '{}',
        sourceType: 'local',
      },
      tags: [],
    });

    const runA = mockRunInfo('runA');
    const runB = mockRunInfo('runB');
    const runC = mockRunInfo('runC');

    const dsAlpha = createDataset('alpha');
    const dsBeta = createDataset('beta');
    const dsGamma = createDataset('gamma');

    const state = undefined;
    const action = {
      type: fulfilled(SEARCH_RUNS_API),
      payload: {
        runs: [
          // Run that will contain one dataset
          { info: runA, inputs: { datasetInputs: [dsAlpha] } },
          // Run that will contain more datasets
          { info: runB, inputs: { datasetInputs: [dsBeta, dsGamma] } },
          // Run not containing any datasets
          { info: runC },
        ],
      },
    };
    const new_state = deepFreeze(runDatasetsByUuid(state, action));
    expect(new_state).not.toEqual(state);
    expect(new_state).toEqual({
      [runA.runUuid]: [dsAlpha],
      [runB.runUuid]: [dsBeta, dsGamma],
      [runC.runUuid]: undefined,
    });
  });
});

describe('test runInfosByUuid', () => {
  test('should set up initial state correctly', () => {
    expect(runInfosByUuid(undefined, {})).toEqual({});
  });
  test('search api with no payload', () => {
    expect(
      runInfosByUuid(undefined, {
        type: fulfilled(SEARCH_RUNS_API),
      }),
    ).toEqual({});
  });
  test('load more with no payload', () => {
    expect(
      runInfosByUuid(undefined, {
        type: fulfilled(LOAD_MORE_RUNS_API),
      }),
    ).toEqual({});
  });

  test('getRunApi correctly updates state', () => {
    // @ts-expect-error TS(2345): Argument of type '"1"' is not assignable to parame... Remove this comment to see the full error message
    const runA = mockRunInfo('run01', '1');
    // @ts-expect-error TS(2345): Argument of type '"2"' is not assignable to parame... Remove this comment to see the full error message
    const runB = mockRunInfo('run01', '2');
    const actionA = {
      type: fulfilled(GET_RUN_API),
      payload: {
        run: {
          info: runA,
        },
      },
    };
    const new_state_0 = deepFreeze(runInfosByUuid(undefined, actionA));
    expect(new_state_0).toEqual({
      [runA.runUuid]: runA,
    });
    const actionB = {
      type: fulfilled(GET_RUN_API),
      payload: {
        run: {
          info: runB,
        },
      },
    };
    const new_state_1 = runInfosByUuid(new_state_0, actionB);
    expect(new_state_1).not.toEqual(new_state_0);
    expect(new_state_1).toEqual({
      [runB.runUuid]: runB,
    });
  });

  test('searchRunApi correctly updates empty state', () => {
    const runA = mockRunInfo('run01');
    const runB = mockRunInfo('run02');
    const state = undefined;
    const action = {
      type: fulfilled(SEARCH_RUNS_API),
      payload: {
        runs: [{ info: runA }, { info: runB }],
      },
    };
    const new_state = deepFreeze(runInfosByUuid(state, action));
    expect(new_state).not.toEqual(state);
    expect(new_state).toEqual({
      [runA.runUuid]: runA,
      [runB.runUuid]: runB,
    });
  });

  test('searchRunApi correctly updates state', () => {
    const preserved = mockRunInfo('still exists');
    // @ts-expect-error TS(2345): Argument of type '"old"' is not assignable to para... Remove this comment to see the full error message
    const replacedOld = mockRunInfo('replaced', 'old');
    // @ts-expect-error TS(2345): Argument of type '"new"' is not assignable to para... Remove this comment to see the full error message
    const replacedNew = mockRunInfo('replaced', 'new');
    const removed = mockRunInfo('removed');
    const newRun = mockRunInfo('new');
    const state = deepFreeze({
      [preserved.runUuid]: preserved,
      [replacedOld.runUuid]: replacedOld,
      [removed.runUuid]: removed,
    });
    const action = {
      type: fulfilled(SEARCH_RUNS_API),
      payload: {
        runs: [{ info: preserved }, { info: replacedNew }, { info: newRun }],
      },
    };
    const new_state = runInfosByUuid(state, action);
    // make sure the reducer did not modify the original state
    expect(new_state).not.toEqual(state);
    expect(new_state).toEqual({
      [preserved.runUuid]: preserved,
      [replacedNew.runUuid]: replacedNew,
      [newRun.runUuid]: newRun,
    });
  });

  test('searchRunApi correctly handles rejected search call', () => {
    const preserved = mockRunInfo('still exists');
    // @ts-expect-error TS(2345): Argument of type '"old"' is not assignable to para... Remove this comment to see the full error message
    const replacedOld = mockRunInfo('replaced', 'old');
    const removed = mockRunInfo('removed');
    const state = deepFreeze({
      [preserved.runUuid]: preserved,
      [replacedOld.runUuid]: replacedOld,
      [removed.runUuid]: removed,
    });
    const action = {
      type: rejected(SEARCH_RUNS_API),
      payload: undefined,
    };
    const new_state = runInfosByUuid(state, action);
    // make sure the reducer did not modify the original state
    expect(new_state).not.toEqual(state);
    expect(new_state).toEqual({});
  });

  test('load more runs', () => {
    const preserved = mockRunInfo('still exists');
    // @ts-expect-error TS(2345): Argument of type '"old"' is not assignable to para... Remove this comment to see the full error message
    const replacedOld = mockRunInfo('replaced', 'old');
    // @ts-expect-error TS(2345): Argument of type '"new"' is not assignable to para... Remove this comment to see the full error message
    const replacedNew = mockRunInfo('replaced', 'new');
    const removed = mockRunInfo('removed');
    const newRun = mockRunInfo('new');
    const state = deepFreeze({
      [preserved.runUuid]: preserved,
      [replacedOld.runUuid]: replacedOld,
      [removed.runUuid]: removed,
    });
    const action = {
      type: fulfilled(LOAD_MORE_RUNS_API),
      payload: {
        runs: [{ info: preserved }, { info: replacedNew }, { info: newRun }],
      },
    };
    const new_state = runInfosByUuid(state, action);
    // make sure the reducer did not modify the original state
    expect(new_state).not.toEqual(state);
    expect(new_state).toEqual({
      [preserved.runUuid]: preserved,
      [removed.runUuid]: removed,
      [replacedNew.runUuid]: replacedNew,
      [newRun.runUuid]: newRun,
    });
  });
});

describe('test modelVersionsByUuid', () => {
  test('should set up initial state correctly', () => {
    expect(modelVersionsByRunUuid(undefined, {})).toEqual({});
  });

  test('search api with no payload', () => {
    expect(
      modelVersionsByRunUuid(undefined, {
        type: fulfilled(SEARCH_MODEL_VERSIONS),
      }),
    ).toEqual({});
  });

  test('searchModelVersionsApi correctly updates empty state', () => {
    const runA = mockRunInfo('run01');
    const runB = mockRunInfo('run02');
    const mvA = mockModelVersionDetailed(
      'model1',
      2,
      Stages.PRODUCTION,
      ModelVersionStatus.READY,
      [],
      undefined,
      'run01',
    );
    const mvB = mockModelVersionDetailed(
      'model2',
      1,
      Stages.PRODUCTION,
      ModelVersionStatus.READY,
      [],
      undefined,
      'run02',
    );
    const state = undefined;
    const action = {
      type: fulfilled(SEARCH_MODEL_VERSIONS),
      payload: {
        model_versions: [mvA, mvB],
      },
    };
    const new_state = deepFreeze(modelVersionsByRunUuid(state, action));
    expect(new_state).not.toEqual(state);
    expect(new_state).toEqual({
      [runA.runUuid]: [mvA],
      [runB.runUuid]: [mvB],
    });
  });

  test('searchModelVersionsApi correctly updates state', () => {
    const run1 = mockRunInfo('run01');
    const run2 = mockRunInfo('run02');
    const run3 = mockRunInfo('run03');
    const mvA = mockModelVersionDetailed(
      'model1',
      2,
      Stages.PRODUCTION,
      ModelVersionStatus.READY,
      [],
      undefined,
      'run01',
    );
    const mvB = mockModelVersionDetailed(
      'model2',
      1,
      Stages.PRODUCTION,
      ModelVersionStatus.READY,
      [],
      undefined,
      'run02',
    );
    const mvC = mockModelVersionDetailed(
      'model2',
      1,
      Stages.PRODUCTION,
      ModelVersionStatus.READY,
      [],
      undefined,
      'run02',
    );
    const mvD = mockModelVersionDetailed(
      'model2',
      1,
      Stages.PRODUCTION,
      ModelVersionStatus.READY,
      [],
      undefined,
      'run03',
    );
    const state = deepFreeze({
      [run1.runUuid]: [mvA],
      [run2.runUuid]: [mvB, mvC],
    });
    const action = {
      type: fulfilled(SEARCH_MODEL_VERSIONS),
      payload: {
        model_versions: [mvA, mvB, mvD],
      },
    };
    const new_state = modelVersionsByRunUuid(state, action);
    // make sure the reducer did not modify the original state
    expect(new_state).not.toEqual(state);
    expect(new_state).toEqual({
      [run1.runUuid]: [mvA],
      [run2.runUuid]: [mvB],
      [run3.runUuid]: [mvD],
    });
  });
});

describe('test params(tags)ByRunUuid', () => {
  const key1 = 'key1';
  const key2 = 'key2';
  const key3 = 'key3';
  test('should set up initial state correctly', () => {
    expect(paramsByRunUuid(undefined, {})).toEqual({});
    expect(tagsByRunUuid(undefined, {})).toEqual({});
  });
  function newParamOrTag(tagOrParam: any, props: any) {
    if (tagOrParam === 'params') {
      return (Param as any).fromJs(props);
    } else {
      return (RunTag as any).fromJs(props);
    }
  }
  function newState(paramOrTag: any, state: any) {
    const res = {};
    for (const runId of Object.keys(state)) {
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      res[runId] = {};
      for (const key of Object.keys(state[runId])) {
        // res[runId][key] = newParamOrTag(paramOrTag, state[runId][key])
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        res[runId][key] = newParamOrTag(paramOrTag, state[runId][key]);
      }
    }
    return res;
  }
  const val1 = {
    key: key1,
    value: 'abc',
  };
  const val1_2 = {
    key: key1,
    value: 'xyz',
  };
  const val2 = {
    key: key2,
    value: 'efg',
  };
  const val3 = {
    key: key3,
    value: 'ijk',
  };
  function reduceAndTest(reducer: any, initial_state: any, expected_state: any, action: any) {
    const new_state = reducer(initial_state, action);
    expect(new_state).not.toEqual(initial_state);
    expect(new_state).toEqual(expected_state);
  }
  test('getRunApi updates empty state correctly', () => {
    const empty_state = { run01: {} };
    const expected_state = {
      run01: {
        key1: val1,
        key2: val2,
        key3: val3,
      },
    };
    function new_action(paramOrTag: any, vals: any) {
      return {
        type: fulfilled(GET_RUN_API),
        payload: {
          run: {
            // @ts-expect-error TS(2345): Argument of type '"experiment01"' is not assignabl... Remove this comment to see the full error message
            info: mockRunInfo('run01', 'experiment01'),
            data: {
              [paramOrTag]: vals,
            },
          },
        },
      };
    }
    reduceAndTest(paramsByRunUuid, undefined, newState('params', empty_state), new_action('params', undefined));
    reduceAndTest(
      tagsByRunUuid,
      undefined,
      newState('tags', empty_state),
      // @ts-expect-error TS(2554): Expected 2 arguments, but got 1.
      new_action('tags'),
      // @ts-expect-error TS(2554): Expected 4 arguments, but got 5.
      undefined,
    );
    reduceAndTest(
      paramsByRunUuid,
      undefined,
      newState('params', expected_state),
      new_action('params', [val1, val2, val3]),
    );
    reduceAndTest(tagsByRunUuid, undefined, newState('tags', expected_state), new_action('tags', [val1, val2, val3]));
  });
  test('getRunApi updates non empty state correctly', () => {
    const initial_state = deepFreeze({
      run01: {
        key1: val1,
        key2: val2,
        key3: val3,
      },
      run02: {
        key1: val1,
        key2: val2,
      },
    });
    const expected_state = {
      run01: {
        key1: val1,
        key3: val3,
      },
      run02: {
        key1: val1,
        key2: val2,
      },
    };
    function new_action(paramOrTag: any) {
      return {
        type: fulfilled(GET_RUN_API),
        payload: {
          run: {
            // @ts-expect-error TS(2345): Argument of type '"experiment01"' is not assignabl... Remove this comment to see the full error message
            info: mockRunInfo('run01', 'experiment01'),
            data: {
              [paramOrTag]: [val1, val3],
            },
          },
        },
      };
    }
    reduceAndTest(
      paramsByRunUuid,
      newState('params', initial_state),
      newState('params', expected_state),
      new_action('params'),
    );
    reduceAndTest(tagsByRunUuid, newState('tags', initial_state), newState('tags', expected_state), new_action('tags'));
  });
  test('search runs and load more apis updates non empty state correctly', () => {
    const initial_state = deepFreeze({
      run01: {
        key1: val1,
        key2: val2,
        key3: val3,
      },
      run02: {
        key1: val1,
        key2: val2,
      },
    });
    const expected_state = {
      run01: {
        key1: val1_2,
        key3: val3,
      },
      run02: {
        key1: val1,
        key2: val2,
      },
      run03: {
        key3: val3,
      },
      run04: {},
    };
    function new_action(paramOrTag: any, action_type: any) {
      return {
        type: fulfilled(action_type),
        payload: {
          runs: [
            {
              info: mockRunInfo('run01'),
              data: { [paramOrTag]: [val1_2, val3] },
            },
            {
              info: mockRunInfo('run03'),
              data: { [paramOrTag]: [val3] },
            },
            {
              info: mockRunInfo('run04'),
              data: { [paramOrTag]: undefined },
            },
          ],
        },
      };
    }
    const reducers = {
      params: paramsByRunUuid,
      tags: tagsByRunUuid,
    };
    for (const paramOrTag of ['params', 'tags']) {
      for (const action_type of [SEARCH_RUNS_API, LOAD_MORE_RUNS_API]) {
        reduceAndTest(
          // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
          reducers[paramOrTag],
          newState(paramOrTag, initial_state),
          newState(paramOrTag, expected_state),
          new_action(paramOrTag, action_type),
        );
      }
    }
  });
  test('setTagApi updates empty state correctly', () => {
    const expected_state = {
      run01: {
        key1: val1,
      },
    };
    function new_action() {
      return {
        type: fulfilled(SET_TAG_API),
        meta: {
          runUuid: 'run01',
          key: key1,
          value: val1.value,
        },
      };
    }
    // @ts-expect-error TS(2554): Expected 0 arguments, but got 1.
    reduceAndTest(tagsByRunUuid, undefined, newState('tags', expected_state), new_action('tags'));
  });
  test('setTagApi updates non empty state correctly', () => {
    const initial_state = deepFreeze({
      run01: {
        key1: val1,
        key2: val2,
      },
      run02: {
        key1: val1,
        key2: val2,
      },
    });
    const expected_state = {
      run01: {
        key1: val1_2,
        key2: val2,
      },
      run02: {
        key1: val1,
        key2: val2,
      },
    };
    function new_action() {
      return {
        type: fulfilled(SET_TAG_API),
        meta: {
          runUuid: 'run01',
          key: key1,
          value: val1_2.value,
        },
      };
    }
    reduceAndTest(tagsByRunUuid, newState('tags', initial_state), newState('tags', expected_state), new_action());
  });
  test('deleteTagApi works with empty state', () => {
    const expected_state = {};
    function new_action() {
      return {
        type: fulfilled(DELETE_TAG_API),
        meta: {
          runUuid: 'run01',
          key: key1,
        },
      };
    }
    reduceAndTest(tagsByRunUuid, undefined, newState('tags', expected_state), new_action());
  });
  test('deleteTagApi updates non empty state correctly', () => {
    const initial_state = deepFreeze({
      run01: {
        key1: val1,
        key2: val2,
      },
      run02: {
        key1: val1,
        key2: val2,
      },
    });
    const expected_state = {
      run01: {
        key2: val2,
      },
      run02: {
        key1: val1,
        key2: val2,
      },
    };
    function new_action() {
      return {
        type: fulfilled(DELETE_TAG_API),
        meta: {
          runUuid: 'run01',
          key: key1,
        },
      };
    }
    reduceAndTest(tagsByRunUuid, newState('tags', initial_state), newState('tags', expected_state), new_action());
  });
});

describe('test artifactsByRunUuid', () => {
  test('artifacts get populated with empty query path', () => {
    const action0 = {
      type: fulfilled(LIST_ARTIFACTS_API),
      meta: {
        runUuid: 'run01',
      },
      payload: {
        files: [
          {
            path: 'root/dir1/file1',
            is_dir: false,
          },
          {
            path: 'root/dir1',
            is_dir: true,
          },
          {
            path: 'root/dir2/file2',
            is_dir: false,
          },
        ],
      },
    };
    const state1 = artifactsByRunUuid(undefined, action0);

    // @ts-expect-error TS(2554): Expected 3 arguments, but got 0.
    const dir1 = Object.assign(new ArtifactNode(), {
      children: [],
      fileInfo: {
        is_dir: true,
        path: 'root/dir1',
      },
      isLoaded: false,
      isRoot: false,
    });

    // @ts-expect-error TS(2554): Expected 3 arguments, but got 0.
    const file1 = Object.assign(new ArtifactNode(), {
      children: undefined,
      fileInfo: {
        is_dir: false,
        path: 'root/dir1/file1',
      },
      isLoaded: false,
      isRoot: false,
    });

    // @ts-expect-error TS(2554): Expected 3 arguments, but got 0.
    const file2 = Object.assign(new ArtifactNode(), {
      children: undefined,
      fileInfo: {
        is_dir: false,
        path: 'root/dir2/file2',
      },
      isLoaded: false,
      isRoot: false,
    });
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 0.
    const expected_root = Object.assign(new ArtifactNode(), {
      children: {
        dir1: dir1,
        file1: file1,
        file2: file2,
      },
      fileInfo: undefined,
      isLoaded: true,
      isRoot: true,
    });
    expect(state1).toEqual({
      run01: expected_root,
    });
    const action1 = {
      type: fulfilled(LIST_ARTIFACTS_API),
      meta: {
        runUuid: 'run02',
      },
      payload: {
        files: [],
      },
    };
    const state2 = artifactsByRunUuid(state1, action1);
    expect(state2).toEqual({
      run01: expected_root,
      // @ts-expect-error TS(2554): Expected 3 arguments, but got 0.
      run02: Object.assign(new ArtifactNode(), {
        children: {},
        fileInfo: undefined,
        isLoaded: true,
        isRoot: true,
      }),
    });
    const action2 = {
      type: fulfilled(LIST_ARTIFACTS_API),
      meta: {
        runUuid: 'run01',
      },
      payload: {
        files: [
          {
            path: 'root/dir1/file1',
            is_dir: false,
          },
          {
            path: 'root/dir1',
            is_dir: true,
          },
        ],
      },
    };
    const state3 = artifactsByRunUuid(state2, action2);
    expect(state3).toEqual({
      // @ts-expect-error TS(2554): Expected 3 arguments, but got 0.
      run01: Object.assign(new ArtifactNode(), {
        children: {
          dir1: dir1,
          file1: file1,
        },
        fileInfo: undefined,
        isLoaded: true,
        isRoot: true,
      }),
      // @ts-expect-error TS(2554): Expected 3 arguments, but got 0.
      run02: Object.assign(new ArtifactNode(), {
        children: {},
        fileInfo: undefined,
        isLoaded: true,
        isRoot: true,
      }),
    });
  });
  test('artifacts get populated with query path', () => {
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 0.
    const file2 = Object.assign(new ArtifactNode(), {
      children: undefined,
      fileInfo: {
        is_dir: false,
        path: 'dir1/dir2/file2',
      },
      isLoaded: false,
      isRoot: false,
    });
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 0.
    const dir2 = Object.assign(new ArtifactNode(), {
      children: { file2: file2 },
      fileInfo: {
        is_dir: true,
        path: 'dir1/dir2',
      },
      isLoaded: false,
      isRoot: false,
    });
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 0.
    const file1 = Object.assign(new ArtifactNode(), {
      children: undefined,
      fileInfo: {
        is_dir: false,
        path: 'dir1/file1',
      },
      isLoaded: false,
      isRoot: false,
    });
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 0.
    const dir1 = Object.assign(new ArtifactNode(), {
      children: { dir2: dir2, file1: file1 },
      fileInfo: {
        is_dir: true,
        path: 'dir1',
      },
      isLoaded: true,
      isRoot: false,
    });
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 0.
    const inital_root = Object.assign(new ArtifactNode(), {
      children: {
        dir1: dir1,
      },
      fileInfo: undefined,
      isLoaded: true,
      isRoot: true,
    });

    const initial_state = deepFreeze({
      run01: inital_root,
    });
    const action0 = {
      type: fulfilled(LIST_ARTIFACTS_API),
      meta: {
        runUuid: 'run01',
        path: 'dir1/dir2',
      },
      payload: {
        files: [
          {
            path: 'dir1/dir2/file3',
            is_dir: false,
          },
          {
            path: 'dir1/dir2/file4',
            is_dir: false,
          },
          {
            path: 'dir1/dir2/dir3',
            is_dir: true,
          },
        ],
      },
    };
    const next_state = artifactsByRunUuid(initial_state, action0);
    expect(next_state).not.toEqual(initial_state);
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 0.
    const file3 = Object.assign(new ArtifactNode(), {
      fileInfo: {
        is_dir: false,
        path: 'dir1/dir2/file3',
      },
      isLoaded: false,
      isRoot: false,
    });
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 0.
    const file4 = Object.assign(new ArtifactNode(), {
      fileInfo: {
        is_dir: false,
        path: 'dir1/dir2/file4',
      },
      isLoaded: false,
      isRoot: false,
    });

    // @ts-expect-error TS(2554): Expected 3 arguments, but got 0.
    const dir3 = Object.assign(new ArtifactNode(), {
      children: [],
      fileInfo: {
        is_dir: true,
        path: 'dir1/dir2/dir3',
      },
      isLoaded: false,
      isRoot: false,
    });
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 0.
    const dir2_2 = Object.assign(new ArtifactNode(), {
      children: { dir3: dir3, file3: file3, file4: file4 },
      fileInfo: {
        is_dir: true,
        path: 'dir1/dir2',
      },
      isLoaded: true,
      isRoot: false,
    });
    // @ts-expect-error TS(2554): Expected 3 arguments, but got 0.
    const dir1_2 = Object.assign(new ArtifactNode(), {
      children: { dir2: dir2_2, file1: file1 },
      fileInfo: {
        is_dir: true,
        path: 'dir1',
      },
      isLoaded: true,
      isRoot: false,
    });
    expect(next_state).toEqual({
      // @ts-expect-error TS(2554): Expected 3 arguments, but got 0.
      run01: Object.assign(new ArtifactNode(), {
        children: {
          dir1: dir1_2,
        },
        fileInfo: undefined,
        isLoaded: true,
        isRoot: true,
      }),
    });
  });
});
describe('test artifactRootUriByRunUuid', () => {
  test('artifactRootUriByRunUuid', () => {
    const state1 = artifactRootUriByRunUuid(undefined, {
      type: fulfilled(GET_RUN_API),
      meta: {
        id: 'a',
        runUuid: 'run01',
      },
      payload: {
        run: {
          info: {
            runUuid: 'run01',
            experimentId: '1',
            artifactUri: 'some/path',
          },
          data: {},
        },
      },
    });
    expect(state1).toEqual({
      run01: 'some/path',
    });
    const state2 = artifactRootUriByRunUuid(state1, {
      type: fulfilled(GET_RUN_API),
      meta: {
        id: 'a',
        runUuid: 'run02',
      },
      payload: {
        run: {
          info: {
            runUuid: 'run02',
            experimentId: '1',
            artifactUri: 'some/other/path',
          },
          data: {},
        },
      },
    });
    expect(state2).toEqual({
      run01: 'some/path',
      run02: 'some/other/path',
    });
    const state3 = artifactRootUriByRunUuid(state2, {
      type: fulfilled(GET_RUN_API),
      meta: {
        id: 'a',
        runUuid: 'run02',
      },
      payload: {
        run: {
          info: {
            runUuid: 'run02',
            experimentId: '1',
            artifactUri: 'some/other/updated/path',
          },
          data: {},
        },
      },
    });
    expect(state3).toEqual({
      run01: 'some/path',
      run02: 'some/other/updated/path',
    });
  });
});

describe('test experimentTagsByExperimentId', () => {
  const tag1 = { key: 'key1', value: 'value1' };
  const tag2 = { key: 'key2', value: 'value2' };
  const tag1_2 = { key: 'key1', value: 'value1_2' };
  test('get experiment api', () => {
    const empty_state = experimentTagsByExperimentId(undefined, {
      type: fulfilled(GET_EXPERIMENT_API),
      payload: {
        experiment: {
          experimentId: 'experiment01',
        },
      },
    });
    expect(empty_state).toEqual({
      experiment01: {},
    });
    const state0 = experimentTagsByExperimentId(undefined, {
      type: fulfilled(GET_EXPERIMENT_API),
      payload: {
        experiment: {
          experimentId: 'experiment01',
          tags: [tag1, tag2],
        },
      },
    });
    expect(state0).toEqual({
      experiment01: {
        key1: (ExperimentTag as any).fromJs(tag1),
        key2: (ExperimentTag as any).fromJs(tag2),
      },
    });
    const state1 = experimentTagsByExperimentId(state0, {
      type: fulfilled(GET_EXPERIMENT_API),
      payload: {
        experiment: {
          experimentId: 'experiment02',
          tags: [tag1],
        },
      },
    });
    expect(state1).toEqual({
      experiment01: {
        key1: (ExperimentTag as any).fromJs(tag1),
        key2: (ExperimentTag as any).fromJs(tag2),
      },
      experiment02: {
        key1: (ExperimentTag as any).fromJs(tag1),
      },
    });
    const state2 = experimentTagsByExperimentId(state1, {
      type: fulfilled(GET_EXPERIMENT_API),
      payload: {
        experiment: {
          experimentId: 'experiment01',
          tags: [tag1_2],
        },
      },
    });
    expect(state2).toEqual({
      experiment01: {
        key1: (ExperimentTag as any).fromJs(tag1_2),
      },
      experiment02: {
        key1: (ExperimentTag as any).fromJs(tag1),
      },
    });
  });
  test('set experiment tag api', () => {
    const initial_state = deepFreeze({
      experiment01: {
        key1: (ExperimentTag as any).fromJs(tag1),
        key2: (ExperimentTag as any).fromJs(tag2),
      },
    });
    const state0 = experimentTagsByExperimentId(initial_state, {
      type: fulfilled(SET_EXPERIMENT_TAG_API),
      meta: {
        experimentId: 'experiment02',
        key: 'key1',
        value: 'value1',
      },
    });
    expect(state0).toEqual({
      experiment01: {
        key1: (ExperimentTag as any).fromJs(tag1),
        key2: (ExperimentTag as any).fromJs(tag2),
      },
      experiment02: {
        key1: (ExperimentTag as any).fromJs(tag1),
      },
    });
    const state1 = experimentTagsByExperimentId(state0, {
      type: fulfilled(SET_EXPERIMENT_TAG_API),
      meta: {
        experimentId: 'experiment01',
        key: 'key1',
        value: 'value1_2',
      },
    });
    expect(state1).toEqual({
      experiment01: {
        key1: (ExperimentTag as any).fromJs(tag1_2),
        key2: (ExperimentTag as any).fromJs(tag2),
      },
      experiment02: {
        key1: (ExperimentTag as any).fromJs(tag1),
      },
    });
  });
});

describe('test datasetsByExperimentId', () => {
  const dataset1_exp1 = {
    experiment_id: 'experiment01',
    digest: 'digest1',
    name: 'dataset1',
    context: 'train',
  };
  const dataset2_exp1 = {
    experiment_id: 'experiment01',
    digest: 'digest2',
    name: 'dataset2',
    context: 'test',
  };
  const dataset2_exp2 = {
    experiment_id: 'experiment02',
    digest: 'digest2',
    name: 'dataset2',
    context: 'test',
  };
  const dataset3_exp2 = { experiment_id: 'experiment02', digest: 'digest3', name: 'dataset3' };
  test('search datasets api', () => {
    const empty_state = datasetsByExperimentId(undefined, {
      type: fulfilled(SEARCH_DATASETS_API),
      payload: {
        dataset_summaries: [],
      },
    });
    expect(empty_state).toEqual({});

    const state0 = datasetsByExperimentId(undefined, {
      type: fulfilled(SEARCH_DATASETS_API),
      payload: {
        dataset_summaries: [dataset1_exp1, dataset2_exp1],
      },
    });
    expect(state0).toEqual({
      experiment01: [dataset1_exp1, dataset2_exp1],
    });

    const state1 = datasetsByExperimentId(undefined, {
      type: fulfilled(SEARCH_DATASETS_API),
      payload: {
        dataset_summaries: [dataset2_exp2, dataset3_exp2],
      },
    });
    expect(state1).toEqual({
      experiment02: [dataset2_exp2, dataset3_exp2],
    });

    const state2 = datasetsByExperimentId(undefined, {
      type: fulfilled(SEARCH_DATASETS_API),
      payload: {
        dataset_summaries: [dataset1_exp1, dataset2_exp1, dataset2_exp2, dataset3_exp2],
      },
    });
    expect(state2).toEqual({
      experiment01: [dataset1_exp1, dataset2_exp1],
      experiment02: [dataset2_exp2, dataset3_exp2],
    });
  });
});

describe('test runInfoOrderByUuid', () => {
  const run1 = { info: { runUuid: 'run_1', experimentId: 'experiment_id' } };
  const run2 = { info: { runUuid: 'run_2', experimentId: 'experiment_id' } };
  const run3 = { info: { runUuid: 'run_3', experimentId: 'experiment_id' } };
  test('get run api', () => {
    let state = runInfoOrderByUuid(undefined, {
      type: fulfilled(SEARCH_RUNS_API),
      payload: {
        runs: [run1, run2],
      },
    });
    expect(state).toEqual(['run_1', 'run_2']);

    state = runInfoOrderByUuid(state, {
      type: fulfilled(SEARCH_RUNS_API),
      payload: {
        runs: [run1, run1, run3],
      },
    });

    expect(state).toEqual(['run_1', 'run_3']);

    state = runInfoOrderByUuid(state, {
      type: fulfilled(LOAD_MORE_RUNS_API),
      payload: {
        runs: [run2, run2],
      },
    });

    expect(state).toEqual(['run_1', 'run_3', 'run_2']);
  });
});

describe('test public accessors', () => {
  function new_action({ type, id = 'a', runUuid = 'run01', payload = 'data' }: any) {
    return {
      type: type,
      meta: {
        id: id,
        runUuid: runUuid,
      },
      payload: payload,
    };
  }
  test('tags, params and runinfo', () => {
    const key1 = 'key1';
    const key2 = 'key2';
    const key3 = 'key3';
    const val1 = {
      key: key1,
      value: 'abc',
    };
    const val2 = {
      key: key2,
      value: 'efg',
    };
    const val3 = {
      key: key3,
      value: 'ijk',
    };
    const action0 = new_action({
      type: fulfilled(GET_RUN_API),
      payload: {
        run: {
          info: {
            runUuid: 'run05',
            experimentId: 'experiment01',
            artifactUri: 'artifact_uri',
          },
          data: {},
        },
      },
    });
    const state0 = rootReducer(undefined, action0);
    expect(getParams('run01', state0)).toEqual({});
    expect(getRunTags('run01', state0)).toEqual({});
    const action1 = {
      type: fulfilled(GET_RUN_API),
      meta: {
        id: 'a',
      },
      payload: {
        run: {
          // @ts-expect-error TS(2345): Argument of type '"experiment01"' is not assignabl... Remove this comment to see the full error message
          info: mockRunInfo('run01', 'experiment01', 'articfact_uri'),
          data: {
            params: [val1, val2],
            tags: [val2, val3],
          },
        },
      },
    };
    const state = rootReducer(undefined, action1);
    expect(getParams('run01', state)).toEqual({
      key1: (Param as any).fromJs({ key: 'key1', value: 'abc' }),
      key2: (Param as any).fromJs({ key: 'key2', value: 'efg' }),
    });
    expect(getRunTags('run01', state)).toEqual({
      key2: (RunTag as any).fromJs({ key: 'key2', value: 'efg' }),
      key3: (RunTag as any).fromJs({ key: 'key3', value: 'ijk' }),
    });
    expect(getRunInfo('run05', state)).toEqual(undefined);
    expect(getRunInfo('run01', state)).toEqual({
      artifactUri: 'articfact_uri',
      endTime: undefined,
      experimentId: 'experiment01',
      lifecycleStage: undefined,
      runUuid: 'run01',
      startTime: undefined,
      status: undefined,
      userId: undefined,
    });
  });
  test('get apis', () => {
    const state0 = rootReducer(
      undefined,
      new_action({
        type: pending(GET_RUN_API),
        id: 'a',
      }),
    );
    const state1 = rootReducer(
      state0,
      new_action({
        type: pending(GET_RUN_API),
        id: 'b',
      }),
    );
    const state2 = rootReducer(
      state1,
      new_action({
        type: pending(GET_RUN_API),
        id: 'c',
      }),
    );
    expect(getApis([], undefined)).toEqual([]);
    expect(getApis(['a', 'b'], state2)).toEqual([
      { id: 'a', active: true },
      { id: 'b', active: true },
    ]);
    expect(getApis(['a', 'b', 'c'], state2)).toEqual([
      { id: 'a', active: true },
      { id: 'b', active: true },
      { id: 'c', active: true },
    ]);
    expect(getApis(['c', 'a'], state2)).toEqual([
      { id: 'c', active: true },
      { id: 'a', active: true },
    ]);
  });
  test('get artifact root', () => {
    const state0 = rootReducer(undefined, {
      type: fulfilled(GET_RUN_API),
      meta: {
        id: 'a',
        runUuid: 'run01',
      },
      payload: {
        run: {
          info: {
            runUuid: 'run01',
            experimentId: '1',
            artifactUri: 'some/path',
          },
          data: {},
        },
      },
    });
    expect(getArtifactRootUri('run01', state0)).toEqual('some/path');
  });
  test('get artifacts', () => {
    const state0 = rootReducer(
      undefined,
      new_action({
        type: fulfilled(LIST_ARTIFACTS_API),
        runUuid: 'run01',
        payload: {
          files: [
            {
              path: 'file1',
              is_dir: false,
            },
          ],
        },
      }),
    );
    expect(getArtifacts('run01', state0)).toEqual(
      // @ts-expect-error TS(2554): Expected 3 arguments, but got 0.
      Object.assign(new ArtifactNode(), {
        children: {
          // @ts-expect-error TS(2554): Expected 3 arguments, but got 0.
          file1: Object.assign(new ArtifactNode(), {
            children: undefined,
            fileInfo: { is_dir: false, path: 'file1' },
            isLoaded: false,
            isRoot: false,
          }),
        },
        fileInfo: undefined,
        isLoaded: true,
        isRoot: true,
      }),
    );
  });
  test('getSharedParamKeysByRunUuids', () => {
    const runA = {
      runUuid: 'run01',
      experimentId: '1',
    };
    const runB = {
      runUuid: 'run02',
      experimentId: '1',
    };
    const actionA = {
      type: fulfilled(GET_RUN_API),
      meta: {
        id: 'a',
      },
      payload: {
        run: {
          info: runA,
          data: {
            params: [
              {
                key: 'A',
                value: 'a',
              },
              {
                key: 'B',
                value: 'b',
              },
            ],
          },
        },
      },
    };
    const actionB = {
      type: fulfilled(GET_RUN_API),
      meta: {
        id: 'a',
      },
      payload: {
        run: {
          info: runB,
          data: {
            params: [
              {
                key: 'B',
                value: 'b',
              },
              {
                key: 'C',
                value: 'c',
              },
            ],
          },
        },
      },
    };
    const state0 = rootReducer(undefined, actionA);
    const state1 = rootReducer(state0, actionB);
    expect(getSharedParamKeysByRunUuids(['run01', 'run02'], state1)).toEqual(['B']);
    expect(getAllParamKeysByRunUuids(['run01', 'run02'], state1)).toEqual(['A', 'B', 'C']);
  });
});

describe('test apis', () => {
  function new_action(t: any, id = 'a') {
    return {
      type: t,
      meta: {
        id: id,
      },
      payload: 'payload',
    };
  }

  test('all action states', () => {
    const state0 = apis(undefined, new_action(pending(GET_RUN_API)));
    expect(state0).toEqual({
      a: {
        id: 'a',
        active: true,
      },
    });
    const state1 = apis(state0, new_action(fulfilled(GET_RUN_API)));
    expect(state1).toEqual({
      a: {
        id: 'a',
        active: false,
        data: 'payload',
      },
    });
    const state2 = apis(state0, new_action(rejected(GET_RUN_API)));
    expect(state2).toEqual({
      a: {
        id: 'a',
        active: false,
        error: 'payload',
      },
    });
  });
});
