export const emptyState = {
  apis: {},
  entities: {
    runInfosByUuid: {},
    experimentsById: {},
    experimentTagsByExperimentId: {}
  }
};

export const addApiToState = (state, api) => {
  const oldApi = state.apis || {};
  return {
    ...state,
    apis: {
      ...oldApi,
      [api.id]: api,
    }
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
      }
    }
  };
};

export const addExperimentTagsToState = (state, experiment_id, tags) => {
  const oldExperimentTags = state.entities.experimentTagsByExperimentId;
  return {
    ...state,
    entities: {
      ...state.entities,
      experimentTagsByExperimentId: {
        ...oldExperimentTags,
        [experiment_id]: {
          tags,
        },
      }
    }
  };
};

export const createPendingApi = (id) => {
  return { id, active: true };
};

