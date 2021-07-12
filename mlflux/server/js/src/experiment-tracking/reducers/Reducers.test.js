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
} from './Reducers';
import { mockExperiment, mockRunInfo } from '../utils/test-utils/ReduxStoreFixtures';
import { RunTag, RunInfo, Param, Experiment, ExperimentTag } from '../sdk/MlflowMessages';
import {
  LIST_EXPERIMENTS_API,
  GET_EXPERIMENT_API,
  GET_RUN_API,
  SEARCH_RUNS_API,
  LOAD_MORE_RUNS_API,
  SET_TAG_API,
  DELETE_TAG_API,
  LIST_ARTIFACTS_API,
  SET_EXPERIMENT_TAG_API,
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

  test('listExperiments correctly updates empty state', () => {
    const experimentA = mockExperiment('experiment01', 'experimentA');
    const experimentB = mockExperiment('experiment02', 'experimentB');
    const state = undefined;
    const action = {
      type: fulfilled(LIST_EXPERIMENTS_API),
      payload: {
        experiments: [experimentA.toJSON(), experimentB.toJSON()],
      },
    };
    const new_state = experimentsById(state, action);
    expect(new_state).toEqual({
      [experimentA.experiment_id]: experimentA,
      [experimentB.experiment_id]: experimentB,
    });
  });

  test('listExperiments correctly updates state', () => {
    const newA = mockExperiment('experiment01', 'experimentA');
    const newB = mockExperiment('experiment02', 'experimentB');
    const preserved = mockExperiment('experiment03', 'still exists');
    const removed = mockExperiment('experiment04', 'removed');
    const replacedOld = mockExperiment('experiment05', 'replacedOld');
    const replacedNew = mockExperiment('experiment05', 'replacedNew');
    const state = deepFreeze({
      [preserved.getExperimentId()]: preserved,
      [removed.getExperimentId()]: removed,
      [replacedOld.getExperimentId()]: replacedOld,
    });
    const action = {
      type: fulfilled(LIST_EXPERIMENTS_API),
      payload: {
        experiments: [preserved.toJSON(), newA.toJSON(), newB.toJSON(), replacedNew.toJSON()],
      },
    };
    const new_state = experimentsById(state, action);
    expect(new_state).not.toEqual(state);
    expect(new_state).toEqual({
      [preserved.getExperimentId()]: preserved,
      [newA.getExperimentId()]: newA,
      [newB.getExperimentId()]: newB,
      [replacedNew.getExperimentId()]: replacedNew,
    });
  });

  test('getExperiment correctly updates empty state', () => {
    const experimentA = mockExperiment('experiment01', 'experimentA');
    const state = {};
    const action = {
      type: fulfilled(GET_EXPERIMENT_API),
      payload: {
        experiment: experimentA.toJSON(),
      },
    };
    const new_state = experimentsById(state, action);
    expect(new_state).not.toEqual(state);
    expect(new_state).toEqual({
      [experimentA.experiment_id]: experimentA,
    });
  });

  test('getExperiment correctly updates non empty state', () => {
    const preserved = mockExperiment('experiment03', 'still exists');
    const replacedOld = mockExperiment('experiment05', 'replacedOld');
    const replacedNew = mockExperiment('experiment05', 'replacedNew');
    const state = deepFreeze({
      [preserved.getExperimentId()]: preserved,
      [replacedOld.getExperimentId()]: replacedOld,
    });
    const action = {
      type: fulfilled(GET_EXPERIMENT_API),
      payload: {
        experiment: replacedNew.toJSON(),
      },
    };
    const new_state = experimentsById(state, action);
    // make sure the reducer did not modify the original state
    expect(new_state).not.toEqual(state);
    expect(new_state).toEqual({
      [preserved.getExperimentId()]: preserved,
      [replacedNew.getExperimentId()]: replacedNew,
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
    const runA = mockRunInfo('run01', '1');
    const runB = mockRunInfo('run01', '2');
    const actionA = {
      type: fulfilled(GET_RUN_API),
      payload: {
        run: {
          info: runA.toJSON(),
        },
      },
    };
    const new_state_0 = deepFreeze(runInfosByUuid(undefined, actionA));
    expect(new_state_0).toEqual({
      [runA.getRunUuid()]: runA,
    });
    const actionB = {
      type: fulfilled(GET_RUN_API),
      payload: {
        run: {
          info: runB.toJSON(),
        },
      },
    };
    const new_state_1 = runInfosByUuid(new_state_0, actionB);
    expect(new_state_1).not.toEqual(new_state_0);
    expect(new_state_1).toEqual({
      [runB.getRunUuid()]: runB,
    });
  });

  test('searchRunApi correctly updates empty state', () => {
    const runA = mockRunInfo('run01');
    const runB = mockRunInfo('run02');
    const state = undefined;
    const action = {
      type: fulfilled(SEARCH_RUNS_API),
      payload: {
        runs: [{ info: runA.toJSON() }, { info: runB.toJSON() }],
      },
    };
    const new_state = deepFreeze(runInfosByUuid(state, action));
    expect(new_state).not.toEqual(state);
    expect(new_state).toEqual({
      [runA.getRunUuid()]: runA,
      [runB.getRunUuid()]: runB,
    });
  });

  test('searchRunApi correctly updates state', () => {
    const preserved = mockRunInfo('still exists');
    const replacedOld = mockRunInfo('replaced', 'old');
    const replacedNew = mockRunInfo('replaced', 'new');
    const removed = mockRunInfo('removed');
    const newRun = mockRunInfo('new');
    const state = deepFreeze({
      [preserved.getRunUuid()]: preserved,
      [replacedOld.getRunUuid()]: replacedOld,
      [removed.getRunUuid()]: removed,
    });
    const action = {
      type: fulfilled(SEARCH_RUNS_API),
      payload: {
        runs: [
          { info: preserved.toJSON() },
          { info: replacedNew.toJSON() },
          { info: newRun.toJSON() },
        ],
      },
    };
    const new_state = runInfosByUuid(state, action);
    // make sure the reducer did not modify the original state
    expect(new_state).not.toEqual(state);
    expect(new_state).toEqual({
      [preserved.getRunUuid()]: preserved,
      [replacedNew.getRunUuid()]: replacedNew,
      [newRun.getRunUuid()]: newRun,
    });
  });

  test('searchRunApi correctly handles rejected search call', () => {
    const preserved = mockRunInfo('still exists');
    const replacedOld = mockRunInfo('replaced', 'old');
    const removed = mockRunInfo('removed');
    const state = deepFreeze({
      [preserved.getRunUuid()]: preserved,
      [replacedOld.getRunUuid()]: replacedOld,
      [removed.getRunUuid()]: removed,
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

  test('test load more runs', () => {
    const preserved = mockRunInfo('still exists');
    const replacedOld = mockRunInfo('replaced', 'old');
    const replacedNew = mockRunInfo('replaced', 'new');
    const removed = mockRunInfo('removed');
    const newRun = mockRunInfo('new');
    const state = deepFreeze({
      [preserved.getRunUuid()]: preserved,
      [replacedOld.getRunUuid()]: replacedOld,
      [removed.getRunUuid()]: removed,
    });
    const action = {
      type: fulfilled(LOAD_MORE_RUNS_API),
      payload: {
        runs: [
          { info: preserved.toJSON() },
          { info: replacedNew.toJSON() },
          { info: newRun.toJSON() },
        ],
      },
    };
    const new_state = runInfosByUuid(state, action);
    // make sure the reducer did not modify the original state
    expect(new_state).not.toEqual(state);
    expect(new_state).toEqual({
      [preserved.getRunUuid()]: preserved,
      [removed.getRunUuid()]: removed,
      [replacedNew.getRunUuid()]: replacedNew,
      [newRun.getRunUuid()]: newRun,
    });
  });
});

describe('test modelVersionsByUuid', () => {
  test('should set up initial state correctly', () => {
    expect(modelVersionsByRunUuid(undefined, {})).toEqual({});
  });

  test('search api with no payload', () => {
    expect(
      runInfosByUuid(undefined, {
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
      [runA.getRunUuid()]: [mvA],
      [runB.getRunUuid()]: [mvB],
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
      [run1.getRunUuid()]: [mvA],
      [run2.getRunUuid()]: [mvB, mvC],
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
      [run1.getRunUuid()]: [mvA],
      [run2.getRunUuid()]: [mvB],
      [run3.getRunUuid()]: [mvD],
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

  function newParamOrTag(tagOrParam, props) {
    if (tagOrParam === 'params') {
      return Param.fromJs(props);
    } else {
      return RunTag.fromJs(props);
    }
  }

  function newState(paramOrTag, state) {
    const res = {};
    for (const runId of Object.keys(state)) {
      res[runId] = {};
      for (const key of Object.keys(state[runId])) {
        // res[runId][key] = newParamOrTag(paramOrTag, state[runId][key])
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

  function reduceAndTest(reducer, initial_state, expected_state, action) {
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

    function new_action(paramOrTag, vals) {
      return {
        type: fulfilled(GET_RUN_API),
        payload: {
          run: {
            info: mockRunInfo('run01', 'experiment01').toJSON(),
            data: {
              [paramOrTag]: vals,
            },
          },
        },
      };
    }

    reduceAndTest(
      paramsByRunUuid,
      undefined,
      newState('params', empty_state),
      new_action('params', undefined),
    );
    reduceAndTest(
      tagsByRunUuid,
      undefined,
      newState('tags', empty_state),
      new_action('tags'),
      undefined,
    );
    reduceAndTest(
      paramsByRunUuid,
      undefined,
      newState('params', expected_state),
      new_action('params', [val1, val2, val3]),
    );
    reduceAndTest(
      tagsByRunUuid,
      undefined,
      newState('tags', expected_state),
      new_action('tags', [val1, val2, val3]),
    );
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

    function new_action(paramOrTag) {
      return {
        type: fulfilled(GET_RUN_API),
        payload: {
          run: {
            info: mockRunInfo('run01', 'experiment01').toJSON(),
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
    reduceAndTest(
      tagsByRunUuid,
      newState('tags', initial_state),
      newState('tags', expected_state),
      new_action('tags'),
    );
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

    function new_action(paramOrTag, action_type) {
      return {
        type: fulfilled(action_type),
        payload: {
          runs: [
            {
              info: mockRunInfo('run01').toJSON(),
              data: { [paramOrTag]: [val1_2, val3] },
            },
            {
              info: mockRunInfo('run03').toJSON(),
              data: { [paramOrTag]: [val3] },
            },
            {
              info: mockRunInfo('run04').toJSON(),
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

    reduceAndTest(
      tagsByRunUuid,
      newState('tags', initial_state),
      newState('tags', expected_state),
      new_action(),
    );
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

    reduceAndTest(
      tagsByRunUuid,
      newState('tags', initial_state),
      newState('tags', expected_state),
      new_action(),
    );
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

    const dir1 = Object.assign(new ArtifactNode(), {
      children: [],
      fileInfo: {
        is_dir: true,
        path: 'root/dir1',
      },
      isLoaded: false,
      isRoot: false,
    });

    const file1 = Object.assign(new ArtifactNode(), {
      children: undefined,
      fileInfo: {
        is_dir: false,
        path: 'root/dir1/file1',
      },
      isLoaded: false,
      isRoot: false,
    });

    const file2 = Object.assign(new ArtifactNode(), {
      children: undefined,
      fileInfo: {
        is_dir: false,
        path: 'root/dir2/file2',
      },
      isLoaded: false,
      isRoot: false,
    });
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
      run01: Object.assign(new ArtifactNode(), {
        children: {
          dir1: dir1,
          file1: file1,
        },
        fileInfo: undefined,
        isLoaded: true,
        isRoot: true,
      }),
      run02: Object.assign(new ArtifactNode(), {
        children: {},
        fileInfo: undefined,
        isLoaded: true,
        isRoot: true,
      }),
    });
  });
  test('artifacts get populated with query path', () => {
    const file2 = Object.assign(new ArtifactNode(), {
      children: undefined,
      fileInfo: {
        is_dir: false,
        path: 'dir1/dir2/file2',
      },
      isLoaded: false,
      isRoot: false,
    });
    const dir2 = Object.assign(new ArtifactNode(), {
      children: { file2: file2 },
      fileInfo: {
        is_dir: true,
        path: 'dir1/dir2',
      },
      isLoaded: false,
      isRoot: false,
    });
    const file1 = Object.assign(new ArtifactNode(), {
      children: undefined,
      fileInfo: {
        is_dir: false,
        path: 'dir1/file1',
      },
      isLoaded: false,
      isRoot: false,
    });
    const dir1 = Object.assign(new ArtifactNode(), {
      children: { dir2: dir2, file1: file1 },
      fileInfo: {
        is_dir: true,
        path: 'dir1',
      },
      isLoaded: true,
      isRoot: false,
    });
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
    const file3 = Object.assign(new ArtifactNode(), {
      fileInfo: {
        is_dir: false,
        path: 'dir1/dir2/file3',
      },
      isLoaded: false,
      isRoot: false,
    });
    const file4 = Object.assign(new ArtifactNode(), {
      fileInfo: {
        is_dir: false,
        path: 'dir1/dir2/file4',
      },
      isLoaded: false,
      isRoot: false,
    });

    const dir3 = Object.assign(new ArtifactNode(), {
      children: [],
      fileInfo: {
        is_dir: true,
        path: 'dir1/dir2/dir3',
      },
      isLoaded: false,
      isRoot: false,
    });
    const dir2_2 = Object.assign(new ArtifactNode(), {
      children: { dir3: dir3, file3: file3, file4: file4 },
      fileInfo: {
        is_dir: true,
        path: 'dir1/dir2',
      },
      isLoaded: true,
      isRoot: false,
    });
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
            run_uuid: 'run01',
            experiment_id: '1',
            artifact_uri: 'some/path',
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
            run_uuid: 'run02',
            experiment_id: '1',
            artifact_uri: 'some/other/path',
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
            run_uuid: 'run02',
            experiment_id: '1',
            artifact_uri: 'some/other/updated/path',
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
          experiment_id: 'experiment01',
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
          experiment_id: 'experiment01',
          tags: [tag1, tag2],
        },
      },
    });
    expect(state0).toEqual({
      experiment01: {
        key1: ExperimentTag.fromJs(tag1),
        key2: ExperimentTag.fromJs(tag2),
      },
    });
    const state1 = experimentTagsByExperimentId(state0, {
      type: fulfilled(GET_EXPERIMENT_API),
      payload: {
        experiment: {
          experiment_id: 'experiment02',
          tags: [tag1],
        },
      },
    });
    expect(state1).toEqual({
      experiment01: {
        key1: ExperimentTag.fromJs(tag1),
        key2: ExperimentTag.fromJs(tag2),
      },
      experiment02: {
        key1: ExperimentTag.fromJs(tag1),
      },
    });
    const state2 = experimentTagsByExperimentId(state1, {
      type: fulfilled(GET_EXPERIMENT_API),
      payload: {
        experiment: {
          experiment_id: 'experiment01',
          tags: [tag1_2],
        },
      },
    });
    expect(state2).toEqual({
      experiment01: {
        key1: ExperimentTag.fromJs(tag1_2),
      },
      experiment02: {
        key1: ExperimentTag.fromJs(tag1),
      },
    });
  });
  test('set experiment tag api', () => {
    const initial_state = deepFreeze({
      experiment01: {
        key1: ExperimentTag.fromJs(tag1),
        key2: ExperimentTag.fromJs(tag2),
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
        key1: ExperimentTag.fromJs(tag1),
        key2: ExperimentTag.fromJs(tag2),
      },
      experiment02: {
        key1: ExperimentTag.fromJs(tag1),
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
        key1: ExperimentTag.fromJs(tag1_2),
        key2: ExperimentTag.fromJs(tag2),
      },
      experiment02: {
        key1: ExperimentTag.fromJs(tag1),
      },
    });
  });
});

describe('test public accessors', () => {
  function new_action({ type, id = 'a', runUuid = 'run01', payload = 'data' }) {
    return {
      type: type,
      meta: {
        id: id,
        runUuid: runUuid,
      },
      payload: payload,
    };
  }

  test('Experiments', () => {
    const A = Experiment.fromJs({
      experiment_id: 'a',
      name: 'A',
      tags: [{ name: 'a', value: 'A' }, 'b'],
    });
    const B = Experiment.fromJs({
      experiment_id: 'b',
      name: 'B',
    });

    const action = new_action({
      type: fulfilled(LIST_EXPERIMENTS_API),
      payload: { experiments: [A.toJSON(), B.toJSON()] },
    });
    const state = rootReducer(undefined, action);
    expect(state.entities.experimentTagsByExperimentId).toEqual({});
    expect(getExperiments(state)).toEqual([A, B]);
    expect(getExperiment(A.experiment_id, state)).toEqual(A);
    expect(getExperimentTags(B.experiment_id, state)).toEqual({});
    expect(getExperimentTags(A.experiment_id, state)).toEqual({});
  });

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
          info: RunInfo.fromJs({
            runUuid: 'run05',
            experiment_id: 'experiment01',
            artifact_uri: 'artifact_uri',
          }).toJSON(),
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
          info: mockRunInfo('run01', 'experiment01', 'articfact_uri').toJSON(),
          data: {
            params: [val1, val2],
            tags: [val2, val3],
          },
        },
      },
    };
    const state = rootReducer(undefined, action1);
    expect(getParams('run01', state)).toEqual({
      key1: Param.fromJs({ key: 'key1', value: 'abc' }),
      key2: Param.fromJs({ key: 'key2', value: 'efg' }),
    });
    expect(getRunTags('run01', state)).toEqual({
      key2: RunTag.fromJs({ key: 'key2', value: 'efg' }),
      key3: RunTag.fromJs({ key: 'key3', value: 'ijk' }),
    });
    expect(getRunInfo('run05', state)).toEqual(undefined);
    expect(getRunInfo('run01', state)).toEqual(
      RunInfo.fromJs({
        artifact_uri: 'articfact_uri',
        end_time: undefined,
        experiment_id: 'experiment01',
        lifecycle_stage: undefined,
        run_uuid: 'run01',
        start_time: undefined,
        status: undefined,
        user_id: undefined,
      }),
    );
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
            run_uuid: 'run01',
            experiment_id: '1',
            artifact_uri: 'some/path',
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
      Object.assign(new ArtifactNode(), {
        children: {
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
    const runA = RunInfo.fromJs({
      run_uuid: 'run01',
      experiment_id: '1',
    });
    const runB = RunInfo.fromJs({
      run_uuid: 'run02',
      experiment_id: '1',
    });
    const actionA = {
      type: fulfilled(GET_RUN_API),
      meta: {
        id: 'a',
      },
      payload: {
        run: {
          info: runA.toJSON(),
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
          info: runB.toJSON(),
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
  function new_action(t, id = 'a') {
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
