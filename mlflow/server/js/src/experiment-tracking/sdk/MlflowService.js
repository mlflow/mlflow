/**
 * DO NOT EDIT!!!
 *
 * @NOTE(dli) 12-21-2016
 *   This file is generated. For now, it is a snapshot of the proto services as of
 *   Aug 1, 2018 3:42:41 PM. We will update the generation pipeline to actually
 *   place these generated objects in the correct location shortly.
 */
import { getBigIntJson, postJson } from '../../common/utils/FetchUtils';

export class MlflowService {
  /**
   * Create a mlflow experiment
   */
  static createExperiment = (data) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/experiments/create', data });

  /**
   * Delete a mlflow experiment
   */
  static deleteExperiment = (data) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/experiments/delete', data });

  /**
   * Update a mlflow experiment
   */
  static updateExperiment = (data) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/experiments/update', data });

  /**
   * Search mlflow experiments
   */
  static searchExperiments = (data) =>
    getBigIntJson({ relativeUrl: 'ajax-api/2.0/mlflow/experiments/search', data });

  /**
   * Get mlflow experiment
   */
  static getExperiment = (data) =>
    getBigIntJson({ relativeUrl: 'ajax-api/2.0/mlflow/experiments/get', data });

  /**
   * Get mlflow experiment by name
   */
  static getExperimentByName = (data) =>
    getBigIntJson({ relativeUrl: 'ajax-api/2.0/mlflow/experiments/get-by-name', data });

  /**
   * Create a mlflow experiment run
   */
  static createRun = (data) => postJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/create', data });

  /**
   * Delete a mlflow experiment run
   */
  static deleteRun = (data) => postJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/delete', data });

  /**
   * Restore a mlflow experiment run
   */
  static restoreRun = (data) => postJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/restore', data });

  /**
   * Update a mlflow experiment run
   */
  static updateRun = (data) => postJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/update', data });

  /**
   * Log mlflow experiment run metric
   */
  static logMetric = (data) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/log-metric', data });

  /**
   * Log mlflow experiment run parameter
   */
  static logParam = (data) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/log-parameter', data });

  /**
   * Get mlflow experiment run
   */
  static getRun = (data) => getBigIntJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/get', data });

  /**
   * Search mlflow experiment runs
   */
  static searchRuns = (data) => postJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/search', data });

  /**
   * List model artifacts
   */
  static listArtifacts = (data) =>
    getBigIntJson({ relativeUrl: 'ajax-api/2.0/mlflow/artifacts/list', data });

  /**
   * Get metric history
   */
  static getMetricHistory = (data) =>
    getBigIntJson({ relativeUrl: 'ajax-api/2.0/mlflow/metrics/get-history', data });

  /**
   * Set mlflow experiment run tag
   */
  static setTag = (data) => postJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/set-tag', data });

  /**
   * Delete mlflow experiment run tag
   */
  static deleteTag = (data) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/runs/delete-tag', data });

  /**
   * Set mlflow experiment tag
   */
  static setExperimentTag = (data) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/experiments/set-experiment-tag', data });
}
