import { MlflowService } from './sdk/MlflowService';

export const isPendingApi = (action) => {
  return action.type.endsWith("_PENDING")
};

export const pending = (apiActionType) => {
  return `${apiActionType}_PENDING`;
};

export const isFulfilledApi = (action) => {
  return action.type.endsWith("_FULFILLED")
};

export const fulfilled = (apiActionType) => {
  return `${apiActionType}_FULFILLED`;
};

export const isRejectedApi = (action) => {
  return action.type.endsWith("_REJECTED")
};

export const rejected = (apiActionType) => {
  return `${apiActionType}_REJECTED`;
};

export const LIST_EXPERIMENTS_API = 'LIST_EXPERIMENTS_API';
export const listExperimentsApi = (id = getUUID()) => {
  return {
    type: LIST_EXPERIMENTS_API,
    payload: wrapDeferred(MlflowService.listExperiments, {}),
    meta: { id: id },
  }
};

export const GET_EXPERIMENT_API = 'GET_EXPERIMENT_API';
export const getExperimentApi = (experimentId, id = getUUID()) => {
  return {
    type: GET_EXPERIMENT_API,
    payload: wrapDeferred(MlflowService.getExperiment, { experiment_id: experimentId }),
    meta: { id: id },
  }
};

export const GET_RUN_API = 'GET_RUN_API';
export const getRunApi = (runUuid, id = getUUID()) => {
  return {
    type: GET_RUN_API,
    payload: wrapDeferred(MlflowService.getRun, { run_uuid: runUuid }),
    meta: { id: id },
  }
};

export const SEARCH_RUNS_API = 'SEARCH_RUNS_API';
export const searchRunsApi = (experimentIds, andedExpressions, id = getUUID()) => {
  return {
    type: SEARCH_RUNS_API,
    payload: wrapDeferred(MlflowService.searchRuns, {
      experiment_ids: experimentIds, anded_expressions: andedExpressions
    }),
    meta: { id: id },
  }
};

export const LIST_ARTIFACTS_API = 'LIST_ARTIFACTS_API';
export const listArtifactsApi = (runUuid, path, id = getUUID()) => {
  return {
    type: LIST_ARTIFACTS_API,
    payload: wrapDeferred(MlflowService.listArtifacts, {
      run_uuid: runUuid, path: path
    }),
    meta: { id: id, runUuid: runUuid, path: path },
  }
};

export const GET_METRIC_HISTORY_API = 'GET_METRIC_HISTORY_API';
export const getMetricHistoryApi = (runUuid, metricKey, id = getUUID()) => {
  return {
    type: GET_METRIC_HISTORY_API,
    payload: wrapDeferred(MlflowService.getMetricHistory, {
      run_uuid: runUuid, metric_key: metricKey
    }),
    meta: { id: id, runUuid: runUuid, key: metricKey },
  }
};

export const getUUID = () => {
  const randomPart = Math.random()
    .toString(36)
    .substring(2, 10);
  return new Date().getTime() + randomPart;
};

/**
 * Jquery's ajax promise is a bit weird so I chose to create a new Promise which resolves and
 * rejects using the ajax callbacks `success` and `error`.
 */
const wrapDeferred = (deferred, data) => {
  return new Promise((resolve, reject) => {
    deferred({
      data,
      success: response => resolve(response),
      error: xhr => {
        reject({ xhr })
      }
    })
  });
};


