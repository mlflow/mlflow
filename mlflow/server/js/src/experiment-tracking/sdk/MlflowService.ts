/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

/**
 * DO NOT EDIT!!!
 *
 * @NOTE(dli) 12-21-2016
 *   This file is generated. For now, it is a snapshot of the proto services as of
 *   Aug 1, 2018 3:42:41 PM. We will update the generation pipeline to actually
 *   place these generated objects in the correct location shortly.
 */
import type { ModelTraceInfo, ModelTraceData } from '@databricks/web-shared/model-trace-explorer';
import { type ParsedQs, stringify as queryStringStringify } from 'qs';
import {
  defaultResponseParser,
  deleteJson,
  fetchEndpoint,
  getBigIntJson,
  getJson,
  HTTPMethods,
  patchJson,
  postBigIntJson,
  postJson,
} from '../../common/utils/FetchUtils';
import type { RunInfoEntity } from '../types';
import {
  transformGetExperimentResponse,
  transformGetRunResponse,
  transformSearchExperimentsResponse,
  transformSearchRunsResponse,
} from './FieldNameTransformers';

type CreateRunApiRequest = {
  experiment_id: string;
  start_time?: number;
  tags?: any;
  run_name?: string;
};

type GetCredentialsForLoggedModelArtifactReadResult = {
  credentials: {
    credential_info: {
      type: string;
      signed_uri: string;
      path: string;
    };
  }[];
};

const searchRunsPath = () => 'ajax-api/2.0/mlflow/runs/search';

// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
export class MlflowService {
  /**
   * Create a mlflow experiment
   */
  static createExperiment = (data: any) => postJson({ relativeUrl: 'ajax-api/2.0/mlflow/experiments/create', data });

  /**
   * Delete a mlflow experiment
   */
  static deleteExperiment = (data: any) => postJson({ relativeUrl: 'ajax-api/2.0/mlflow/experiments/delete', data });

  /**
   * Update a mlflow experiment
   */
  static updateExperiment = (data: any) => postJson({ relativeUrl: 'ajax-api/2.0/mlflow/experiments/update', data });

  /**
   * Search mlflow experiments
   */
  static searchExperiments = (data: any) =>
    getBigIntJson({ relativeUrl: 'ajax-api/2.0/mlflow/experiments/search', data }).then(
      transformSearchExperimentsResponse,
    );

  /**
   * Get mlflow experiment
   */
  static getExperiment = (data: any) =>
    getBigIntJson({ relativeUrl: 'ajax-api/2.0/mlflow/experiments/get', data }).then(transformGetExperimentResponse);

  /**
   * Get mlflow experiment by name
   */
  static getExperimentByName = (data: any) =>
    getBigIntJson({ relativeUrl: 'ajax-api/2.0/mlflow/experiments/get-by-name', data }).then(
      transformGetExperimentResponse,
    );

  /**
   * Create a mlflow experiment run
   */
  static createRun = (data: CreateRunApiRequest) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/create', data }) as Promise<{
      run: { info: RunInfoEntity };
    }>;

  /**
   * Delete a mlflow experiment run
   */
  static deleteRun = (data: { run_id: string }) => postJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/delete', data });

  /**
   * Search datasets used in experiments
   */
  static searchDatasets = (data: any) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/experiments/search-datasets', data });

  /**
   * Restore a mlflow experiment run
   */
  static restoreRun = (data: any) => postJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/restore', data });

  /**
   * Update a mlflow experiment run
   */
  static updateRun = (data: any) => postJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/update', data });

  /**
   * Log mlflow experiment run metric
   */
  static logMetric = (data: any) => postJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/log-metric', data });

  /**
   * Log mlflow experiment run parameter
   */
  static logParam = (data: any) => postJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/log-parameter', data });

  /**
   * Get mlflow experiment run
   */
  static getRun = (data: any) =>
    getBigIntJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/get', data }).then(transformGetRunResponse);

  /**
   * Search mlflow experiment runs
   */
  static searchRuns = (data: any) =>
    postJson({ relativeUrl: searchRunsPath(), data }).then(transformSearchRunsResponse);

  /**
   * List model artifacts
   */
  static listArtifacts = (data: any) => getBigIntJson({ relativeUrl: 'ajax-api/2.0/mlflow/artifacts/list', data });

  /**
   * List model artifacts for logged models
   */
  static listArtifactsLoggedModel = ({ loggedModelId, path }: { loggedModelId: string; path: string }) =>
    getBigIntJson({
      relativeUrl: `ajax-api/2.0/mlflow/logged-models/${loggedModelId}/artifacts/directories`,
      data: path ? { artifact_directory_path: path } : {},
    });

  static getCredentialsForLoggedModelArtifactRead = ({
    loggedModelId,
    path,
  }: {
    loggedModelId: string;
    path: string;
  }) =>
    postBigIntJson({
      relativeUrl: `ajax-api/2.0/mlflow/logged-models/${loggedModelId}/artifacts/credentials-for-download`,
      data: {
        paths: [path],
      },
    }) as Promise<GetCredentialsForLoggedModelArtifactReadResult>;

  /**
   * Get metric history
   */
  static getMetricHistory = (data: any) =>
    getBigIntJson({ relativeUrl: 'ajax-api/2.0/mlflow/metrics/get-history', data });

  /**
   * Set mlflow experiment run tag
   */
  static setTag = (data: any) => postJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/set-tag', data });

  /**
   * Delete mlflow experiment run tag
   */
  static deleteTag = (data: any) => postJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/delete-tag', data });

  /**
   * Set mlflow experiment tag
   */
  static setExperimentTag = (data: any) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/experiments/set-experiment-tag', data });

  /**
   * Delete mlflow experiment tag
   */
  static deleteExperimentTag = (data: any) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/experiments/delete-experiment-tag', data });

  /**
   * Create prompt engineering run
   */
  static createPromptLabRun = (data: {
    experiment_id: string;
    tags?: { key: string; value: string }[];
    prompt_template: string;
    prompt_parameters: { key: string; value: string }[];
    model_route: string;
    model_parameters: { key: string; value: string | number | undefined }[];
    model_output_parameters: { key: string; value: string | number }[];
    model_output: string;
  }) => postJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/create-promptlab-run', data });

  /**
   * Proxy post request to gateway server
   */
  static gatewayProxyPost = (data: { gateway_path: string; json_data: any }, error?: any) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/gateway-proxy', data, error });

  /**
   * Proxy get request to gateway server
   */
  static gatewayProxyGet = (data: { gateway_path: string; json_data?: any }) =>
    getJson({ relativeUrl: 'ajax-api/2.0/mlflow/gateway-proxy', data });

  /**
   * Traces API: get traces list
   */
  static getExperimentTraces = (
    experimentIds: string[],
    orderBy: string,
    pageToken?: string,
    filterString = '',
    maxResults?: number,
  ) => {
    type GetExperimentTracesResponse = {
      traces?: ModelTraceInfo[];
      next_page_token?: string;
      prev_page_token?: string;
    };

    // usually we send array data via POST request, but since this
    // is a GET, we need to treat it specially. we use `qs` to
    // serialize the array into a query string which the backend
    // can handle. this is similar to the approach taken in the
    // GetMetricHistoryBulkInterval API.
    const queryString = queryStringStringify(
      {
        experiment_ids: experimentIds,
        order_by: orderBy,
        page_token: pageToken,
        max_results: maxResults,
        filter: filterString,
      },
      { arrayFormat: 'repeat' },
    );

    return fetchEndpoint({
      relativeUrl: `ajax-api/2.0/mlflow/traces?${queryString}`,
    }) as Promise<GetExperimentTracesResponse>;
  };

  static getExperimentTraceInfo = (requestId: string) => {
    type GetExperimentTraceInfoResponse = {
      trace_info?: ModelTraceInfo;
    };

    return getJson({
      relativeUrl: `ajax-api/2.0/mlflow/traces/${requestId}/info`,
    }) as Promise<GetExperimentTraceInfoResponse>;
  };

  static getExperimentTraceInfoV3 = (requestId: string) => {
    type GetExperimentTraceInfoV3Response = {
      trace?: {
        trace_info?: ModelTraceInfo;
      };
    };

    return getJson({
      relativeUrl: `ajax-api/3.0/mlflow/traces/${requestId}`,
    }) as Promise<GetExperimentTraceInfoV3Response>;
  };

  /**
   * Traces API: get credentials for data download
   */
  static getExperimentTraceData = <T = ModelTraceData>(traceRequestId: string) => {
    return getJson({
      relativeUrl: `ajax-api/2.0/mlflow/get-trace-artifact`,
      data: {
        request_id: traceRequestId,
      },
    }) as Promise<T>;
  };

  /**
   * Traces API: set trace tag
   */
  static setExperimentTraceTag = (traceRequestId: string, key: string, value: string) =>
    patchJson({
      relativeUrl: `ajax-api/2.0/mlflow/traces/${traceRequestId}/tags`,
      data: {
        key,
        value,
      },
    });

  /**
   * Traces API: set trace tag V3
   */
  static setExperimentTraceTagV3 = (traceRequestId: string, key: string, value: string) =>
    patchJson({
      relativeUrl: `ajax-api/3.0/mlflow/traces/${traceRequestId}/tags`,
      data: {
        key,
        value,
      },
    });

  /**
   * Traces API: delete trace tag V3
   */
  static deleteExperimentTraceTagV3 = (traceRequestId: string, key: string) =>
    fetchEndpoint({
      relativeUrl: `ajax-api/3.0/mlflow/traces/${traceRequestId}/tags?key=${encodeURIComponent(key)}`,
      method: HTTPMethods.DELETE,
      success: defaultResponseParser,
    });

  /**
   * Traces API: delete trace tag
   */
  static deleteExperimentTraceTag = (traceRequestId: string, key: string) =>
    deleteJson({
      relativeUrl: `ajax-api/2.0/mlflow/traces/${traceRequestId}/tags`,
      data: {
        key,
      },
    });

  static deleteTracesV3 = (experimentId: string, traceRequestIds: string[]) =>
    postJson({
      relativeUrl: `ajax-api/3.0/mlflow/traces/delete-traces`,
      data: {
        experiment_id: experimentId,
        request_ids: traceRequestIds,
      },
    }) as Promise<{ traces_deleted: number }>;

  static deleteTraces = (experimentId: string, traceRequestIds: string[]) =>
    postJson({
      relativeUrl: `ajax-api/2.0/mlflow/traces/delete-traces`,
      data: {
        experiment_id: experimentId,
        request_ids: traceRequestIds,
      },
    }) as Promise<{ traces_deleted: number }>;
}
