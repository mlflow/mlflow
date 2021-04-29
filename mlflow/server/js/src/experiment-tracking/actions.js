import { MlflowService } from './sdk/MlflowService';
import { getUUID, wrapDeferred } from '../common/utils/ActionUtils';

export const SEARCH_MAX_RESULTS = 100;

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

export const CREATE_EXPERIMENT_API = 'CREATE_EXPERIMENT_API';
export const createExperimentApi = (experimentName, artifactPath = undefined, id = getUUID()) => {
  return (dispatch) => {
    const createResponse = dispatch({
      type: CREATE_EXPERIMENT_API,
      payload: wrapDeferred(MlflowService.createExperiment, {
        name: experimentName,
        artifact_location: artifactPath,
      }),
      meta: { id: getUUID() },
    });
    return createResponse;
  };
};

export const DELETE_EXPERIMENT_API = 'DELETE_EXPERIMENT_API';
export const deleteExperimentApi = (experimentId, id = getUUID()) => {
  return (dispatch) => {
    const deleteResponse = dispatch({
      type: DELETE_EXPERIMENT_API,
      payload: wrapDeferred(MlflowService.deleteExperiment, { experiment_id: experimentId }),
      meta: { id: getUUID() },
    });
    return deleteResponse;
  };
};

export const UPDATE_EXPERIMENT_API = 'UPDATE_EXPERIMENT_API';
export const updateExperimentApi = (experimentId, newExperimentName, id = getUUID()) => {
  return (dispatch) => {
    const updateResponse = dispatch({
      type: UPDATE_EXPERIMENT_API,
      payload: wrapDeferred(MlflowService.updateExperiment, {
        experiment_id: experimentId,
        new_name: newExperimentName,
      }),
      meta: { id: getUUID() },
    });
    return updateResponse;
  };
};

export const GET_RUN_API = 'GET_RUN_API';
export const getRunApi = (runId, id = getUUID()) => {
  return {
    type: GET_RUN_API,
    payload: wrapDeferred(MlflowService.getRun, { run_id: runId }),
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

export const getParentRunTagName = () => 'mlflow.parentRunId';

export const getParentRunIdsToFetch = (runs) => {
  const parentsToFetch = new Set();
  if (runs) {
    const currentRunIds = new Set(runs.map((run) => run.info.run_id));

    runs.forEach((run) => {
      if (run.data && run.data.tags) {
        const tagsList = run.data.tags;
        tagsList.forEach((tag) => {
          if (tag.key === getParentRunTagName() && !currentRunIds.has(tag.value)) {
            parentsToFetch.add(tag.value);
          }
        });
      }
    });
  }
  return Array.from(parentsToFetch);
};

// this function takes a response of runs and returns them along with their missing parents
export const fetchMissingParents = (searchRunsResponse) =>
  searchRunsResponse.runs && searchRunsResponse.runs.length
    ? Promise.all(
        getParentRunIdsToFetch(searchRunsResponse.runs).map((runId) =>
          wrapDeferred(MlflowService.getRun, { run_id: runId }),
        ),
      ).then((parentRuns) => {
        parentRuns.forEach((parentRun) => searchRunsResponse.runs.push(parentRun.run));
        return searchRunsResponse;
      })
    : searchRunsResponse;

export const searchRunsPayload = ({
  experimentIds,
  filter,
  runViewType,
  orderBy,
  pageToken,
  shouldFetchParents,
}) =>
  wrapDeferred(MlflowService.searchRuns, {
    experiment_ids: experimentIds,
    filter: filter,
    run_view_type: runViewType,
    max_results: SEARCH_MAX_RESULTS,
    order_by: orderBy,
    page_token: pageToken,
  }).then((res) => (shouldFetchParents ? fetchMissingParents(res) : res));

export const SEARCH_RUNS_API = 'SEARCH_RUNS_API';
export const searchRunsApi = (params) => ({
  type: SEARCH_RUNS_API,
  payload: searchRunsPayload(params),
  meta: { id: params.id || getUUID() },
});

export const LOAD_MORE_RUNS_API = 'LOAD_MORE_RUNS_API';
export const loadMoreRunsApi = (params) => ({
  type: LOAD_MORE_RUNS_API,
  payload: searchRunsPayload(params),
  meta: { id: params.id || getUUID() },
});

// TODO: run_uuid is deprecated, use run_id instead
export const LIST_ARTIFACTS_API = 'LIST_ARTIFACTS_API';
export const listArtifactsApi = (runUuid, path, id = getUUID()) => {
  return {
    type: LIST_ARTIFACTS_API,
    payload: wrapDeferred(MlflowService.listArtifacts, {
      run_uuid: runUuid,
      path: path,
    }),
    meta: { id: id, runUuid: runUuid, path: path },
  };
};

// TODO: run_uuid is deprecated, use run_id instead
export const GET_METRIC_HISTORY_API = 'GET_METRIC_HISTORY_API';
export const getMetricHistoryApi = (runUuid, metricKey, id = getUUID()) => {
  return {
    type: GET_METRIC_HISTORY_API,
    payload: wrapDeferred(MlflowService.getMetricHistory, {
      run_uuid: runUuid,
      metric_key: decodeURIComponent(metricKey),
    }),
    meta: { id: id, runUuid: runUuid, key: metricKey },
  };
};

// TODO: run_uuid is deprecated, use run_id instead
export const SET_TAG_API = 'SET_TAG_API';
export const setTagApi = (runUuid, tagName, tagValue, id = getUUID()) => {
  return {
    type: SET_TAG_API,
    payload: wrapDeferred(MlflowService.setTag, {
      run_uuid: runUuid,
      key: tagName,
      value: tagValue,
    }),
    meta: { id: id, runUuid: runUuid, key: tagName, value: tagValue },
  };
};

// TODO: run_uuid is deprecated, use run_id instead
export const DELETE_TAG_API = 'DELETE_TAG_API';
export const deleteTagApi = (runUuid, tagName, id = getUUID()) => {
  return {
    type: DELETE_TAG_API,
    payload: wrapDeferred(MlflowService.deleteTag, {
      run_id: runUuid,
      key: tagName,
    }),
    meta: { id: id, runUuid: runUuid, key: tagName },
  };
};

export const SET_EXPERIMENT_TAG_API = 'SET_EXPERIMENT_TAG_API';
export const setExperimentTagApi = (experimentId, tagName, tagValue, id = getUUID()) => {
  return {
    type: SET_EXPERIMENT_TAG_API,
    payload: wrapDeferred(MlflowService.setExperimentTag, {
      experiment_id: experimentId,
      key: tagName,
      value: tagValue,
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
