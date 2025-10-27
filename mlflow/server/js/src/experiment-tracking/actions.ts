/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import type { Dispatch, Action } from 'redux';
import type { AsyncAction, ReduxState, ThunkDispatch } from '../redux-types';
import { MlflowService } from './sdk/MlflowService';
import { getUUID } from '../common/utils/ActionUtils';
import { ErrorCodes } from '../common/constants';
import { isArray, isObject } from 'lodash';
import { ViewType } from './sdk/MlflowEnums';
import { fetchEndpoint, jsonBigIntResponseParser } from '../common/utils/FetchUtils';
import { stringify as queryStringStringify } from 'qs';
import { fetchEvaluationTableArtifact } from './sdk/EvaluationArtifactService';
import type { EvaluationDataReduxState } from './reducers/EvaluationDataReducer';
import type { ArtifactListFilesResponse, EvaluationArtifactTable } from './types';
import type { KeyValueEntity } from '../common/types';
import { MLFLOW_PUBLISHED_VERSION } from '../common/mlflow-published-version';
import { MLFLOW_LOGGED_IMAGE_ARTIFACTS_PATH } from './constants';
import { ErrorWrapper } from '../common/utils/ErrorWrapper';
export const RUNS_SEARCH_MAX_RESULTS = 100;

export const GET_EXPERIMENT_API = 'GET_EXPERIMENT_API';
export const getExperimentApi = (experimentId: any, id = getUUID()) => {
  return {
    type: GET_EXPERIMENT_API,
    payload: MlflowService.getExperiment({ experiment_id: experimentId }),
    meta: { id: id },
  };
};

const CREATE_EXPERIMENT_API = 'CREATE_EXPERIMENT_API';
export const createExperimentApi = (experimentName: any, artifactPath = undefined, id = getUUID()) => {
  return (dispatch: ThunkDispatch) => {
    const createResponse = dispatch({
      type: CREATE_EXPERIMENT_API,
      payload: MlflowService.createExperiment({
        name: experimentName,
        artifact_location: artifactPath,
      }),
      meta: { id: getUUID() },
    });
    return createResponse;
  };
};

const DELETE_EXPERIMENT_API = 'DELETE_EXPERIMENT_API';
export const deleteExperimentApi = (experimentId: any, id = getUUID()) => {
  return (dispatch: ThunkDispatch) => {
    const deleteResponse = dispatch({
      type: DELETE_EXPERIMENT_API,
      payload: MlflowService.deleteExperiment({ experiment_id: experimentId }),
      meta: { id: getUUID() },
    });
    return deleteResponse;
  };
};

const UPDATE_EXPERIMENT_API = 'UPDATE_EXPERIMENT_API';
export const updateExperimentApi = (experimentId: any, newExperimentName: any, id = getUUID()) => {
  return (dispatch: ThunkDispatch) => {
    const updateResponse = dispatch({
      type: UPDATE_EXPERIMENT_API,
      payload: MlflowService.updateExperiment({
        experiment_id: experimentId,
        new_name: newExperimentName,
      }),
      meta: { id: getUUID() },
    });
    return updateResponse;
  };
};

const UPDATE_RUN_API = 'UPDATE_RUN_API';
export const updateRunApi = (runId: string, newName: string, id: string = getUUID()) => {
  return (dispatch: ThunkDispatch) => {
    const updateResponse = dispatch({
      type: UPDATE_RUN_API,
      payload: MlflowService.updateRun({
        run_id: runId,
        run_name: newName,
      }),
      meta: { id: getUUID() },
    });
    return updateResponse;
  };
};

export const GET_RUN_API = 'GET_RUN_API';
export const getRunApi = (runId: any, id = getUUID()) => {
  return {
    type: GET_RUN_API,
    payload: MlflowService.getRun({ run_id: runId }),
    meta: { id: id },
  };
};

const CREATE_RUN_API = 'CREATE_RUN_API';
const createRunApi = (experimentId: string, tags?: any, run_name?: string) => {
  return (dispatch: ThunkDispatch) => {
    const createResponse = dispatch({
      type: CREATE_RUN_API,
      payload: MlflowService.createRun({
        experiment_id: experimentId,
        start_time: Date.now(),
        tags: tags,
        run_name,
      }),
      meta: { id: getUUID() },
    });
    return createResponse;
  };
};

export interface UploadArtifactApiAction
  extends AsyncAction<
    any,
    {
      id: string;
      runUuid: string;
      filePath: string;
    }
  > {
  type: 'UPLOAD_ARTIFACT_API';
}
export const UPLOAD_ARTIFACT_API = 'UPLOAD_ARTIFACT_API';
export const uploadArtifactApi = (runUuid: any, filePath: any, fileContent: any) => {
  // We are not using MlflowService because this endpoint requires
  // special query string preparation
  const queryParams = queryStringStringify({
    run_uuid: runUuid,
    path: filePath,
  });
  const request = fetchEndpoint({
    relativeUrl: `ajax-api/2.0/mlflow/upload-artifact?${queryParams}`,
    method: 'POST',
    body: JSON.stringify(fileContent),
    success: jsonBigIntResponseParser,
    // Retry the call every time an artifact upload fails
    errorCondition: (res: Response) => !res || !res.ok,
    // Retry for maximum 3 times
    retries: 3,
  });

  return (dispatch: ThunkDispatch) => {
    const uploadArtifactResponse = dispatch({
      type: UPLOAD_ARTIFACT_API,
      payload: request,
      meta: { id: getUUID(), runUuid, filePath },
    });
    return uploadArtifactResponse;
  };
};

const DELETE_RUN_API = 'DELETE_RUN_API';
export const deleteRunApi = (runUuid: any, id = getUUID()) => {
  return (dispatch: ThunkDispatch) => {
    const deleteResponse = dispatch({
      type: DELETE_RUN_API,
      payload: MlflowService.deleteRun({ run_id: runUuid }),
      meta: { id: getUUID() },
    });
    return deleteResponse.then(() => dispatch(getRunApi(runUuid, id)));
  };
};
const RESTORE_RUN_API = 'RESTORE_RUN_API';
export const restoreRunApi = (runUuid: any, id = getUUID()) => {
  return (dispatch: ThunkDispatch) => {
    const restoreResponse = dispatch({
      type: RESTORE_RUN_API,
      payload: MlflowService.restoreRun({ run_id: runUuid }),
      meta: { id: getUUID() },
    });
    return restoreResponse.then(() => dispatch(getRunApi(runUuid, id)));
  };
};

export const SET_COMPARE_EXPERIMENTS = 'SET_COMPARE_EXPERIMENTS';
export const setCompareExperiments = ({ comparedExperimentIds, hasComparedExperimentsBefore }: any) => {
  return {
    type: SET_COMPARE_EXPERIMENTS,
    payload: { comparedExperimentIds, hasComparedExperimentsBefore },
  };
};

export const getParentRunTagName = () => 'mlflow.parentRunId';

export const getParentRunIdsToFetch = (runs: any) => {
  const parentsToFetch = new Set();
  if (runs) {
    const currentRunIds = new Set(runs.map((run: any) => run.info.run_id));

    runs.forEach((run: any) => {
      if (run.data && run.data.tags) {
        const tagsList = run.data.tags;
        tagsList.forEach((tag: any) => {
          if (tag.key === getParentRunTagName() && !currentRunIds.has(tag.value)) {
            parentsToFetch.add(tag.value);
          }
        });
      }
    });
  }
  return Array.from(parentsToFetch);
};

/**
 * This function takes a response of runs and returns them along with their missing parents.
 * @deprecated Use fetchMissingParentsWithSearchRuns instead
 */
export const fetchMissingParents = (searchRunsResponse: any) =>
  searchRunsResponse.runs && searchRunsResponse.runs.length
    ? Promise.all(
        getParentRunIdsToFetch(searchRunsResponse.runs).map((runId) =>
          MlflowService.getRun({ run_id: runId })
            .then((value) => {
              searchRunsResponse.runs.push(value.run);
              // Additional parent runs should be always visible
              // marked as those matching filter
              if (searchRunsResponse.runsMatchingFilter) {
                searchRunsResponse.runsMatchingFilter.push(value.run);
              }
            })
            .catch((error) => {
              if (error.getErrorCode() !== ErrorCodes.RESOURCE_DOES_NOT_EXIST) {
                // NB: The parent run may have been deleted, in which case attempting to fetch the
                // run fails with the `RESOURCE_DOES_NOT_EXIST` error code. Because this is
                // expected behavior, we swallow such exceptions. We re-raise all other exceptions
                // encountered when fetching parent runs because they are unexpected
                throw error;
              }
            }),
        ),
      ).then((_) => {
        return searchRunsResponse;
      })
    : searchRunsResponse;

/**
 * Fetches missing parent runs for the given search runs response in a set of experimentIds. Returns
 * the original runs along with their parents.
 */
export const fetchMissingParentsWithSearchRuns = (searchRunsResponse: any, experimentIds: any) =>
  searchRunsResponse.runs && searchRunsResponse.runs.length
    ? Promise.all(
        getParentRunIdsToFetch(searchRunsResponse.runs).map((parentRunId) =>
          MlflowService.searchRuns({
            experiment_ids: experimentIds,
            filter: `run_id = '${parentRunId}'`,
            max_results: 1,
          })
            .then((parentSearchRunsResponse) => {
              const parentRun = parentSearchRunsResponse.runs?.[0];
              if (parentRun) {
                searchRunsResponse.runs.push(parentRun);
                // Additional parent runs should be always visible
                // marked as those matching filter
                if (searchRunsResponse.runsMatchingFilter) {
                  searchRunsResponse.runsMatchingFilter.push(parentRun);
                }
              }
            })
            .catch((error) => {
              if (error.getErrorCode() !== ErrorCodes.RESOURCE_DOES_NOT_EXIST) {
                // NB: The parent run may have been deleted, in which case attempting to fetch the
                // run fails with the `RESOURCE_DOES_NOT_EXIST` error code. Because this is
                // expected behavior, we swallow such exceptions. We re-raise all other exceptions
                // encountered when fetching parent runs because they are unexpected
                throw error;
              }
            }),
        ),
      ).then((_) => {
        return searchRunsResponse;
      })
    : searchRunsResponse;

/**
 * Creates SQL-like expression for pinned rows
 */
const createPinnedRowsExpression = (runsPinned: any) => {
  if (runsPinned.length < 1) {
    return null;
  }
  const runIdsInQuotes = runsPinned.map((runId: any) => `'${runId}'`);
  return `run_id IN (${runIdsInQuotes.join(',')})`;
};

/**
 * Main method for fetching experiment runs payload from the API
 */
export const searchRunsPayload = ({
  // Experiment IDs to fetch runs for
  experimentIds,

  // SQL-like filter
  filter,

  // Used to select either active or deleted runs
  runViewType,

  // Maximum limit of result count (not accounting pinned rows)
  maxResults,

  // Order by SQL clause
  orderBy,

  // A pagination token from the previous result
  pageToken,

  // Set to "true" if parents of children runs should be fetched as well
  shouldFetchParents,

  // Array of pinned row IDs which will be fetched with another request
  runsPinned,
}: any) => {
  // Let's start with the base request for the runs
  const promises = [
    MlflowService.searchRuns({
      experiment_ids: experimentIds,
      filter: filter,
      run_view_type: runViewType,
      max_results: maxResults || RUNS_SEARCH_MAX_RESULTS,
      order_by: orderBy,
      page_token: pageToken,
    }),
  ];

  // If we want to have pinned runs, fetch them as well
  // using another request with different filter
  if (runsPinned?.length) {
    promises.push(
      MlflowService.searchRuns({
        experiment_ids: experimentIds,
        filter: createPinnedRowsExpression(runsPinned),
        run_view_type: ViewType.ALL,
      }),
    );
  }

  // Wait for all requests to finish.
  // - `baseSearchResponse` will contain all runs that match the requested filter
  // - `pinnedSearchResponse` will contain all pinned runs, if any
  // We will merge and return an array with those two collections
  return Promise.all(promises).then(([baseSearchResponse, pinnedSearchResponse = {}]) => {
    const response = baseSearchResponse;

    if (!isObject(response)) {
      throw new Error(`Invalid format of the runs search response: ${String(response)}`);
    }

    // Place aside and save runs that matched filter naturally (not the pinned ones):
    (response as any).runsMatchingFilter = (baseSearchResponse as any).runs?.slice() || [];

    // If we get pinned rows from the additional response, merge them into the base run list:
    if (isArray((pinnedSearchResponse as any).runs)) {
      if (isArray((response as any).runs)) {
        (response as any).runs.push(...(pinnedSearchResponse as any).runs);
      } else {
        (response as any).runs = (pinnedSearchResponse as any).runs.slice();
      }
    }

    // If there are any pending parents to fetch, do it before returning the response
    const fetchParents = () => fetchMissingParents(response);
    return shouldFetchParents ? fetchParents() : response;
  });
};

export const SEARCH_RUNS_API = 'SEARCH_RUNS_API';
export const searchRunsApi = (params: any) => ({
  type: SEARCH_RUNS_API,
  payload: searchRunsPayload(params),
  meta: { id: params.id || getUUID() },
});

export const LOAD_MORE_RUNS_API = 'LOAD_MORE_RUNS_API';
export const loadMoreRunsApi = (params: any) => ({
  type: LOAD_MORE_RUNS_API,
  payload: searchRunsPayload(params),
  meta: { id: params.id || getUUID() },
});

// TODO: run_uuid is deprecated, use run_id instead
export const LIST_ARTIFACTS_API = 'LIST_ARTIFACTS_API';
export const listArtifactsApi = (runUuid: any, path?: any, id = getUUID()) => {
  return {
    type: LIST_ARTIFACTS_API,
    payload: MlflowService.listArtifacts({
      run_uuid: runUuid,
      // only pass path if not null or undefined
      ...(path && { path: path }),
    }),
    meta: { id: id, runUuid: runUuid, path: path },
  };
};

/**
 * Redux action to list artifacts for a logged model.
 * TODO: discard redux, refactor into hooks
 */
export const LIST_ARTIFACTS_LOGGED_MODEL_API = 'LIST_ARTIFACTS_LOGGED_MODEL_API';
export const listArtifactsLoggedModelApi = (
  loggedModelId: any,
  path?: any,
  experimentId?: string,
  id = getUUID(),
  entityTags?: Partial<KeyValueEntity>[],
) => {
  const getLoggedModelDataFromMLflowAPI = () =>
    MlflowService.listArtifactsLoggedModel({
      loggedModelId,
      path,
    });
  const getLoggedModelDataFn = () => {
    return getLoggedModelDataFromMLflowAPI();
  };
  return {
    type: LIST_ARTIFACTS_LOGGED_MODEL_API,
    payload: getLoggedModelDataFn(),
    meta: { id: id, loggedModelId, path: path },
  };
};

/**
 * Run this action only after verifying that the /images directory exists
 * Reducer will populate image keys.
 */
export const LIST_IMAGES_API = 'LIST_IMAGES_API';
export interface ListImagesAction
  extends AsyncAction<ArtifactListFilesResponse, { id: string; runUuid: string; path?: string }> {
  type: 'LIST_IMAGES_API';
}
export const listImagesApi = (runUuid: string, autorefresh = false, id = getUUID()) => {
  return (dispatch: ThunkDispatch, getState: () => ReduxState) => {
    const getExistingDataForRunUuid = getState().entities.imagesByRunUuid[runUuid];
    // If the images for this runUuid already exists, return the existing data
    if (!autorefresh && getExistingDataForRunUuid) {
      return Promise.resolve();
    }

    return dispatch({
      type: LIST_IMAGES_API,
      payload: MlflowService.listArtifacts({
        run_uuid: runUuid,
        path: MLFLOW_LOGGED_IMAGE_ARTIFACTS_PATH,
      }),
      meta: { id: id, runUuid: runUuid, path: MLFLOW_LOGGED_IMAGE_ARTIFACTS_PATH },
    });
  };
};

// TODO: run_uuid is deprecated, use run_id instead
export const GET_METRIC_HISTORY_API = 'GET_METRIC_HISTORY_API';
export const getMetricHistoryApi = (runUuid: any, metricKey: any, maxResults: any, pageToken: any, id = getUUID()) => {
  return {
    type: GET_METRIC_HISTORY_API,
    payload: MlflowService.getMetricHistory({
      run_uuid: runUuid,
      metric_key: decodeURIComponent(metricKey),
      max_results: maxResults,
      page_token: pageToken,
    }),
    meta: {
      id: id,
      runUuid: runUuid,
      key: metricKey,
      maxResults,
      pageToken,
    },
  };
};

export const GET_METRIC_HISTORY_API_BULK = 'GET_METRIC_HISTORY_API_BULK';

// TODO: run_uuid is deprecated, use run_id instead
export const SET_TAG_API = 'SET_TAG_API';
export const setTagApi = (runUuid: any, tagName: any, tagValue: any, id = getUUID()) => {
  return {
    type: SET_TAG_API,
    payload: MlflowService.setTag({
      run_uuid: runUuid,
      key: tagName,
      value: tagValue,
    }),
    meta: { id: id, runUuid: runUuid, key: tagName, value: tagValue },
  };
};

// TODO: run_uuid is deprecated, use run_id instead
export const DELETE_TAG_API = 'DELETE_TAG_API';
const SET_RUN_TAGS_BULK = 'SET_RUN_TAGS_BULK';

/**
 * Used in new, unified tagging UI
 */
export const saveRunTagsApi = (
  run_uuid: string,
  newTags: KeyValueEntity[],
  deletedTags: KeyValueEntity[],
  id = getUUID(),
) => {
  const updateRequests = Promise.all([
    ...newTags.map(({ key, value }) => MlflowService.setTag({ run_uuid, key, value })),
    ...deletedTags.map(({ key }) => MlflowService.deleteTag({ run_id: run_uuid, key })),
  ]);

  return {
    type: SET_RUN_TAGS_BULK,
    payload: updateRequests,
    meta: { id, runUuid: run_uuid, deletedTags, newTags },
  };
};

// TODO: remove the action once "databricks.fe.mlflow.useSharedTaggingUI" flag is enabled
/**
 * @deprecated
 *
 *
 * Used in old implementation of tags editor
 *
 * Given lists of existing and new tags, creates and calls
 * multiple requests for setting/deleting tags in a experiment run
 */
export const setRunTagsBulkApi = (
  run_uuid: string,
  existingTags: KeyValueEntity[],
  newTags: KeyValueEntity[],
  id = getUUID(),
) => {
  // First, determine new aliases to be added
  const addedOrModifiedTags = newTags.filter(
    ({ key: newTagKey, value: newTagValue }) =>
      !existingTags.some(
        ({ key: existingTagKey, value: existingTagValue }) =>
          existingTagKey === newTagKey && newTagValue === existingTagValue,
      ),
  );

  // Next, determine those to be deleted
  const deletedTags = existingTags.filter(
    ({ key: existingTagKey }) => !newTags.some(({ key: newTagKey }) => existingTagKey === newTagKey),
  );

  // Fire all requests at once
  const updateRequests = Promise.all([
    ...addedOrModifiedTags.map(({ key, value }) => MlflowService.setTag({ run_uuid, key, value })),
    ...deletedTags.map(({ key }) => MlflowService.deleteTag({ run_id: run_uuid, key })),
  ]);

  return {
    type: SET_RUN_TAGS_BULK,
    payload: updateRequests,
    meta: { id, runUuid: run_uuid, existingTags, newTags },
  };
};

export const SET_EXPERIMENT_TAG_API = 'SET_EXPERIMENT_TAG_API';
export const setExperimentTagApi = (experimentId: any, tagName: any, tagValue: any, id = getUUID()) => {
  return {
    type: SET_EXPERIMENT_TAG_API,
    payload: MlflowService.setExperimentTag({
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
export const openErrorModal = (text: any) => {
  return {
    type: OPEN_ERROR_MODAL,
    text,
  };
};

export const SEARCH_DATASETS_API = 'SEARCH_DATASETS';
export const searchDatasetsApi = (experimentIds: any, id = getUUID()) => {
  return {
    type: SEARCH_DATASETS_API,
    payload: MlflowService.searchDatasets({
      experiment_ids: experimentIds,
    }),
    meta: { id },
  };
};

/**
 * A thunk action that fetches and stores a single evaluation table artifact for a given run.
 * Does not download the artifact if it's already present in the store, unless `forceRefresh` is set to `true`.
 */
export const getEvaluationTableArtifact =
  (runUuid: string, artifactPath: string, forceRefresh = false) =>
  (dispatch: ThunkDispatch, getState: () => { evaluationData: EvaluationDataReduxState }) => {
    const { evaluationData: existingEvaluationData } = getState();
    const alreadyInStore = Boolean(
      existingEvaluationData.evaluationArtifactsByRunUuid[runUuid]?.[artifactPath] ||
        existingEvaluationData.evaluationArtifactsLoadingByRunUuid[runUuid]?.[artifactPath],
    );
    if (forceRefresh || !alreadyInStore) {
      return dispatch({
        type: GET_EVALUATION_TABLE_ARTIFACT,
        payload: fetchEvaluationTableArtifact(runUuid, artifactPath),
        meta: { runUuid, artifactPath },
      });
    }
    return Promise.resolve();
  };

/**
 * Defines shape of the fulfilled GET_EVALUATION_ARTIFACT action
 */
export interface GetEvaluationTableArtifactAction
  extends AsyncAction<EvaluationArtifactTable, { runUuid: string; artifactPath: string }> {
  type: 'GET_EVALUATION_TABLE_ARTIFACT';
}
export const GET_EVALUATION_TABLE_ARTIFACT = 'GET_EVALUATION_TABLE_ARTIFACT';

export const createPromptLabRunApi = ({
  experimentId,
  modelRouteName,
  modelParameters,
  promptTemplate,
  promptParameters,
  runName,
  modelInput,
  modelOutput,
  modelOutputParameters,
  tags = [],
}: {
  experimentId: string;
  runName?: string;
  promptTemplate: string;
  promptParameters: Record<string, string>;
  modelRouteName: string;
  modelInput: string;
  modelOutput: string;
  modelParameters: Record<string, string | number | string[] | undefined>;
  modelOutputParameters: Record<string, string | number>;
  tags?: { key: string; value: string }[];
}) => {
  const tupleToKeyValue = <T>(dict: Record<string, T>) => Object.entries(dict).map(([key, value]) => ({ key, value }));

  const tupleToKeyValueFlattenArray = (dict: Record<string, string | number | string[] | undefined>) =>
    Object.entries(dict).map(([key, value]) => {
      const scalarValue: string | number | undefined = Array.isArray(value) ? `[${value.join(', ')}]` : value;
      return { key, value: scalarValue };
    });

  const payload = {
    experiment_id: experimentId,
    run_name: runName || undefined,
    tags,
    prompt_template: promptTemplate,
    prompt_parameters: tupleToKeyValue(promptParameters),
    model_route: modelRouteName,
    model_parameters: tupleToKeyValueFlattenArray(modelParameters),
    model_input: modelInput,
    model_output: modelOutput,
    model_output_parameters: tupleToKeyValue(modelOutputParameters),
    mlflow_version: MLFLOW_PUBLISHED_VERSION,
  };
  return {
    type: CREATE_PROMPT_LAB_RUN,
    payload: MlflowService.createPromptLabRun(payload),
    meta: { payload },
  };
};
const CREATE_PROMPT_LAB_RUN = 'CREATE_PROMPT_LAB_RUN';
