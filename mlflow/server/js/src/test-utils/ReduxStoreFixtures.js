export const emptyState = {
  apis: {},
  entities: {
    runInfosByUuid: {},
    experimentsById: {},
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

export const createPendingApi = (id) => {
  return { id, active: true };
};

