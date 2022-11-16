import { RunTag, Experiment, RunInfo, Metric } from '../../sdk/MlflowMessages';
import { makeStore } from '../../../store';
import Fixtures from './Fixtures';

export const emptyState = {
  apis: {},
  entities: {
    runInfosByUuid: {},
    experimentsById: {},
    experimentListSearchInput: {
      currentSearchInput: '',
      prevSearchInput: '',
    },
    experimentTagsByExperimentId: {},
    tagsByRunUuid: {},
    modelVersionsByRunUuid: {},
    searchExperimentsNextPageToken: null,
    loadingMoreExperiments: false,
  },
};

export const addApiToState = (state, api) => {
  const oldApi = state.apis || {};
  return {
    ...state,
    apis: {
      ...oldApi,
      [api.id]: api,
    },
  };
};

export const addExperimentToState = (state, experiment) => {
  const oldExperiments = state.entities.experimentsById;
  return {
    ...state,
    entities: {
      ...state.entities,
      experimentsById: {
        ...oldExperiments,
        [experiment.experiment_id]: experiment,
      },
    },
  };
};

export const addExperimentTagsToState = (state, experiment_id, tags) => {
  const oldExperimentTags = state.entities.experimentTagsByExperimentId;
  const tagsArrToObject = (tagsArr) => {
    const tagObj = {};
    tagsArr.forEach((tag) => (tagObj[tag.key] = RunTag.fromJs(tag)));
    return tagObj;
  };
  return {
    ...state,
    entities: {
      ...state.entities,
      experimentTagsByExperimentId: {
        ...oldExperimentTags,
        [experiment_id]: tagsArrToObject(tags),
      },
    },
  };
};

export const createPendingApi = (id) => {
  return { id, active: true };
};

export const mockExperiment = (eid, name) => {
  return Experiment.fromJs({ experiment_id: eid, name: name });
};

export const mockRunInfo = (
  run_id,
  experiment_id = undefined,
  artifact_uri = undefined,
  lifecycle_stage = undefined,
) => {
  return RunInfo.fromJs({
    run_uuid: run_id,
    experiment_id: experiment_id,
    artifact_uri: artifact_uri,
    lifecycle_stage: lifecycle_stage,
  });
};

export const mockMetric = (params) => {
  return Metric.fromJs(params);
};

export const makeStateWithExperiments = (experiments = Fixtures.experiments) => {
  let state = emptyState;
  experiments.forEach((e) => {
    state = addExperimentToState(state, e);
  });
  return state;
};

// State with experiments in it
export const stateWithExperiments = makeStateWithExperiments();

// Initialized store with experiments
export const storeWithExperiments = makeStore(stateWithExperiments);
