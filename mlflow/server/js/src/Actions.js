import { MlflowService } from './sdk/MlflowService';
import ErrorCodes from './sdk/ErrorCodes';

export const SEARCH_MAX_RESULTS = 100;

export const isPendingApi = (action) => {
  return action.type.endsWith("_PENDING");
};

export const pending = (apiActionType) => {
  return `${apiActionType}_PENDING`;
};

export const isFulfilledApi = (action) => {
  return action.type.endsWith("_FULFILLED");
};

export const fulfilled = (apiActionType) => {
  return `${apiActionType}_FULFILLED`;
};

export const isRejectedApi = (action) => {
  return action.type.endsWith("_REJECTED");
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
  };
};

export const GET_EXPERIMENT_API = 'GET_EXPERIMENT_API';
export const getExperimentApi = (experimentId, id = getUUID()) => {
  return {
    type: GET_EXPERIMENT_API,
    payload: wrapDeferred(MlflowService.getExperiment, { experiment_id: experimentId }),
    meta: { id: id },
  };
};

export const GET_RUN_API = 'GET_RUN_API';
export const getRunApi = (runUuid, id = getUUID()) => {
  return {
    type: GET_RUN_API,
    payload: wrapDeferred(MlflowService.getRun, { run_uuid: runUuid }),
    meta: { id: id },
  };
};

export const DELETE_RUN_API = 'DELETE_RUN_API';
export const deleteRunApi = (runUuid, id = getUUID()) => {
  return (dispatch) => {
    const deleteResponse = dispatch({
      type: DELETE_RUN_API,
      payload: wrapDeferred(MlflowService.deleteRun, { run_id: runUuid }),
      meta: { id: getUUID() },
    });
    return deleteResponse.then(() => dispatch(getRunApi(runUuid, id)));
  };
};

export const RESTORE_RUN_API = 'RESTORE_RUN_API';
export const restoreRunApi = (runUuid, id = getUUID()) => {
  return (dispatch) => {
    const restoreResponse = dispatch({
      type: RESTORE_RUN_API,
      payload: wrapDeferred(MlflowService.restoreRun, { run_id: runUuid }),
      meta: { id: getUUID() },
    });
    return restoreResponse.then(() => dispatch(getRunApi(runUuid, id)));
  };
};

export const SEARCH_RUNS_API = 'SEARCH_RUNS_API';
export const searchRunsApi = (experimentIds, filter, runViewType, orderBy, id = getUUID()) => {
  return {
    type: SEARCH_RUNS_API,
    payload: wrapDeferred(MlflowService.searchRuns, {
      experiment_ids: experimentIds,
      filter: filter,
      run_view_type: runViewType,
      max_results: SEARCH_MAX_RESULTS,
      order_by: orderBy,
    }),
    meta: { id: id },
  };
};

export const LOAD_MORE_RUNS_API = 'LOAD_MORE_RUNS_API';
export const loadMoreRunsApi = (
  experimentIds,
  filter,
  runViewType,
  orderBy,
  pageToken,
  id = getUUID(),
) => ({
  type: LOAD_MORE_RUNS_API,
  payload: wrapDeferred(MlflowService.searchRuns, {
    experiment_ids: experimentIds,
    filter: filter,
    run_view_type: runViewType,
    max_results: SEARCH_MAX_RESULTS,
    order_by: orderBy,
    page_token: pageToken,
  }),
  meta: { id },
});


export const LIST_ARTIFACTS_API = 'LIST_ARTIFACTS_API';
export const listArtifactsApi = (runUuid, path, id = getUUID()) => {
  return {
    type: LIST_ARTIFACTS_API,
    payload: wrapDeferred(MlflowService.listArtifacts, {
      run_uuid: runUuid, path: path
    }),
    meta: { id: id, runUuid: runUuid, path: path },
  };
};

export const GET_METRIC_HISTORY_API = 'GET_METRIC_HISTORY_API';
export const getMetricHistoryApi = (runUuid, metricKey, id = getUUID()) => {
  return {
    type: GET_METRIC_HISTORY_API,
    payload: wrapDeferred(MlflowService.getMetricHistory, {
      run_uuid: runUuid, metric_key: metricKey
    }),
    meta: { id: id, runUuid: runUuid, key: metricKey },
  };
};

export const SET_TAG_API = 'SET_TAG_API';
export const setTagApi = (runUuid, tagName, tagValue, id = getUUID()) => {
  return {
    type: SET_TAG_API,
    payload: wrapDeferred(MlflowService.setTag, {
      run_uuid: runUuid, key: tagName, value: tagValue
    }),
    meta: { id: id, runUuid: runUuid, key: tagName, value: tagValue },
  };
};

export const SET_EXPERIMENT_TAG_API = 'SET_EXPERIMENT_TAG_API';
export const setExperimentTagApi = (experimentId, tagName, tagValue, id = getUUID()) => {
  return {
    type: SET_EXPERIMENT_TAG_API,
    payload: wrapDeferred(MlflowService.setExperimentTag, {
      experiment_id: experimentId, key: tagName, value: tagValue
    }),
    meta: { id, experimentId, key: tagName, value: tagValue },
  };
};

export const CLOSE_ERROR_MODAL = 'CLOSE_ERROR_MODAL';
export const closeErrorModal = () => {
  return {
    type: CLOSE_ERROR_MODAL,
  };
};

export const OPEN_ERROR_MODAL = 'OPEN_ERROR_MODAL';
export const openErrorModal = (text) => {
  return {
    type: OPEN_ERROR_MODAL,
    text,
  };
};

export const getUUID = () => {
  const randomPart = Math.random()
    .toString(36)
    .substring(2, 10);
  return new Date().getTime() + randomPart;
};

/**
 * Wraps a Jquery AJAX request (passed via `deferred`) in a new Promise which resolves and
 * rejects using the ajax callbacks `success` and `error`. Retries with exponential backoff
 * if the server responds with a 429 (Too Many Requests).
 * @param {function} deferred - Function with signature ({data, success, error}) => Any, where
 *   data is a JSON payload for an AJAX request, success is a callback to execute on request
 *   success, and error is a callback to execute on request failure.
 * @param {object} data - Data argument to pass to `deferred`
 * @param {int} timeLeftMs - Time left to retry the AJAX request in ms, if we receive a 429
 * response from the server. Defaults to 60 seconds.
 * @param {int} sleepMs - Time to sleep before retrying the AJAX request if we receive a 429
 * response from the server. Defaults to 1 second.
 */
export const wrapDeferred = (deferred, data, timeLeftMs = 60000, sleepMs = 1000) => {
  return new Promise((resolve, reject) => {
    deferred({
      data,
      success: response => {
        resolve(response);
      },
      error: (xhr) => {
        if (xhr.status === 429) {
          if (timeLeftMs > 0) {
            console.warn("Request failed with status code 429, message " +
                new ErrorWrapper(xhr).getUserVisibleError() + ". Retrying after " +
                sleepMs + " ms. On additional 429 errors, will continue to retry for up " +
                "to " + timeLeftMs + " ms.");
            // Retry the request, subtracting the current sleep duration from the remaining time
            // and doubling the sleep duration
            const newTimeLeft = timeLeftMs - sleepMs;
            const newSleepMs = Math.min(newTimeLeft, sleepMs * 2);
            return new Promise(resolveRetry => setTimeout(resolveRetry, sleepMs)).then(() => {
              return wrapDeferred(deferred, data, newTimeLeft, newSleepMs);
            }).then(
                (successResponse) => resolve(successResponse),
                (failureResponse) => reject(failureResponse)
            );
          }
        }
        console.error("XHR failed", xhr);
        // We can't throw the XHR itself because it looks like a promise to the
        // redux-promise-middleware.
        return reject(new ErrorWrapper(xhr));
      }
    });
  });
};

export class ErrorWrapper {
  constructor(xhr) {
    this.xhr = xhr;
  }

  getErrorCode() {
    const responseText = this.xhr.responseText;
    if (responseText) {
      try {
        const parsed = JSON.parse(responseText);
        if (parsed.error_code) {
          return parsed.error_code;
        }
      } catch (e) {
        return ErrorCodes.INTERNAL_ERROR;
      }
    }
    return ErrorCodes.INTERNAL_ERROR;
  }

  // Return the responseText if it is in the
  // { error_code: ..., message: ...} format. Otherwise return "INTERNAL_SERVER_ERROR".
  getUserVisibleError() {
    const responseText = this.xhr.responseText;
    if (responseText) {
      try {
        const parsed = JSON.parse(responseText);
        if (parsed.error_code) {
          return responseText;
        }
      } catch (e) {
        return "INTERNAL_SERVER_ERROR";
      }
    }
    return "INTERNAL_SERVER_ERROR";
  }

  getMessageField() {
    const responseText = this.xhr.responseText;
    if (responseText) {
      try {
        const parsed = JSON.parse(responseText);
        if (parsed.error_code && parsed.message) {
          return parsed.message;
        }
      } catch (e) {
        return "INTERNAL_SERVER_ERROR";
      }
    }
    return "INTERNAL_SERVER_ERROR";
  }
}

