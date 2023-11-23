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
import { getBigIntJson, getJson, postJson } from '../../common/utils/FetchUtils';
import { RunInfoEntity } from '../types';

type CreateRunApiRequest = {
  experiment_id: string;
  start_time?: number;
  tags?: any;
  run_name?: string;
};

export class MlflowService {
  /**
   * Create a mlflow experiment
   */
  static createExperiment = (data: any) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/experiments/create', data });

  /**
   * Delete a mlflow experiment
   */
  static deleteExperiment = (data: any) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/experiments/delete', data });

  /**
   * Update a mlflow experiment
   */
  static updateExperiment = (data: any) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/experiments/update', data });

  /**
   * Search mlflow experiments
   */
  static searchExperiments = (data: any) =>
    getBigIntJson({ relativeUrl: 'ajax-api/2.0/mlflow/experiments/search', data });

  /**
   * Get mlflow experiment
   */
  static getExperiment = (data: any) =>
    getBigIntJson({ relativeUrl: 'ajax-api/2.0/mlflow/experiments/get', data });

  /**
   * Get mlflow experiment by name
   */
  static getExperimentByName = (data: any) =>
    getBigIntJson({ relativeUrl: 'ajax-api/2.0/mlflow/experiments/get-by-name', data });

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
  static deleteRun = (data: any) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/delete', data });

  /**
   * Search datasets used in experiments
   */
  static searchDatasets = (data: any) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/experiments/search-datasets', data });

  /**
   * Restore a mlflow experiment run
   */
  static restoreRun = (data: any) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/restore', data });

  /**
   * Update a mlflow experiment run
   */
  static updateRun = (data: any) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/update', data });

  /**
   * Log mlflow experiment run metric
   */
  static logMetric = (data: any) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/log-metric', data });

  /**
   * Log mlflow experiment run parameter
   */
  static logParam = (data: any) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/log-parameter', data });

  /**
   * Get mlflow experiment run
   */
  static getRun = (data: any) =>
    getBigIntJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/get', data });

  /**
   * Search mlflow experiment runs
   */
  static searchRuns = (data: any) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/search', data });

  /**
   * List model artifacts
   */
  static listArtifacts = (data: any) =>
    getBigIntJson({ relativeUrl: 'ajax-api/2.0/mlflow/artifacts/list', data });

  /**
   * Get metric history
   */
  static getMetricHistory = (data: any) =>
    getBigIntJson({ relativeUrl: 'ajax-api/2.0/mlflow/metrics/get-history', data });

  /**
   * Set mlflow experiment run tag
   */
  static setTag = (data: any) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/set-tag', data });

  /**
   * Delete mlflow experiment run tag
   */
  static deleteTag = (data: any) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/delete-tag', data });

  /**
   * Set mlflow experiment tag
   */
  static setExperimentTag = (data: any) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/experiments/set-experiment-tag', data });

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
  static gatewayProxyPost = (data: { gateway_path: string; json_data: any }) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/gateway-proxy', data });

  /**
   * Proxy get request to gateway server
   */
  static gatewayProxyGet = (data: { gateway_path: string; json_data?: any }) =>
    getJson({ relativeUrl: 'ajax-api/2.0/mlflow/gateway-proxy', data });
}
