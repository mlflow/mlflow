/**
 * DO NOT EDIT!!!
 *
 * @NOTE(dli) 12-21-2016
 *   This file is generated. For now, it is a snapshot of the proto services as of
 *   Aug 1, 2018 3:42:41 PM. We will update the generation pipeline to actually
 *   place these generated objects in the correct location shortly.
 */

import $ from 'jquery';
import JsonBigInt from 'json-bigint';

const StrictJsonBigInt = JsonBigInt({ strict: true, storeAsString: true });

export class MlflowService {

  /**
   * @param {CreateExperiment} data: Immutable Record
   * @param {function} success
   * @param {function} error
   * @return {Promise}
   */
  static createExperiment({ data, success, error }) {
    return $.ajax('/ajax-api/2.0/mlflow/experiments/create', {
      type: 'POST',
      dataType: 'json',
      data: JSON.stringify(data),
      jsonp: false,
      success: success,
      error: error,
    });
  }

  /**
   * @param {ListExperiments} data: Immutable Record
   * @param {function} success
   * @param {function} error
   * @return {Promise}
   */
  static listExperiments({ data, success, error }) {
    return $.ajax('/ajax-api/2.0/mlflow/experiments/list', {
      type: 'GET',
      dataType: 'json',
      converters: {
        'text json': StrictJsonBigInt.parse,
      },
      data: data,
      jsonp: false,
      success: success,
      error: error,
    });
  }

  /**
   * @param {GetExperiment} data: Immutable Record
   * @param {function} success
   * @param {function} error
   * @return {Promise}
   */
  static getExperiment({ data, success, error }) {
    return $.ajax('/ajax-api/2.0/mlflow/experiments/get', {
      type: 'GET',
      dataType: 'json',
      converters: {
        'text json': StrictJsonBigInt.parse,
      },
      data: data,
      jsonp: false,
      success: success,
      error: error,
    });
  }

  /**
   * @param {CreateRun} data: Immutable Record
   * @param {function} success
   * @param {function} error
   * @return {Promise}
   */
  static createRun({ data, success, error }) {
    return $.ajax('/ajax-api/2.0/mlflow/runs/create', {
      type: 'POST',
      dataType: 'json',
      data: JSON.stringify(data),
      jsonp: false,
      success: success,
      error: error,
    });
  }

  /**
   * @param {DeleteRun} data: Immutable Record
   * @param {function} success
   * @param {function} error
   * @return {Promise}
   */
  static deleteRun({ data, success, error }) {
    return $.ajax('/ajax-api/2.0/mlflow/runs/delete', {
      type: 'POST',
      dataType: 'json',
      data: JSON.stringify(data),
      jsonp: false,
      success: success,
      error: error,
    });
  }

  /**
   * @param {RestoreRun} data: Immutable Record
   * @param {function} success
   * @param {function} error
   * @return {Promise}
   */
  static restoreRun({ data, success, error }) {
    return $.ajax('/ajax-api/2.0/mlflow/runs/restore', {
      type: 'POST',
      dataType: 'json',
      data: JSON.stringify(data),
      jsonp: false,
      success: success,
      error: error,
    });
  }

  /**
   * @param {UpdateRun} data: Immutable Record
   * @param {function} success
   * @param {function} error
   * @return {Promise}
   */
  static updateRun({ data, success, error }) {
    return $.ajax('/ajax-api/2.0/mlflow/runs/update', {
      type: 'POST',
      dataType: 'json',
      data: JSON.stringify(data),
      jsonp: false,
      success: success,
      error: error,
    });
  }

  /**
   * @param {LogMetric} data: Immutable Record
   * @param {function} success
   * @param {function} error
   * @return {Promise}
   */
  static logMetric({ data, success, error }) {
    return $.ajax('/ajax-api/2.0/mlflow/runs/log-metric', {
      type: 'POST',
      dataType: 'json',
      data: JSON.stringify(data),
      jsonp: false,
      success: success,
      error: error,
    });
  }

  /**
   * @param {LogParam} data: Immutable Record
   * @param {function} success
   * @param {function} error
   * @return {Promise}
   */
  static logParam({ data, success, error }) {
    return $.ajax('/ajax-api/2.0/mlflow/runs/log-parameter', {
      type: 'POST',
      dataType: 'json',
      data: JSON.stringify(data),
      jsonp: false,
      success: success,
      error: error,
    });
  }

  /**
   * @param {GetRun} data: Immutable Record
   * @param {function} success
   * @param {function} error
   * @return {Promise}
   */
  static getRun({ data, success, error }) {
    return $.ajax('/ajax-api/2.0/mlflow/runs/get', {
      type: 'GET',
      dataType: 'json',
      converters: {
        'text json': StrictJsonBigInt.parse,
      },
      data: data,
      jsonp: false,
      success: success,
      error: error,
    });
  }

  /**
   * @param {SearchRuns} data: Immutable Record
   * @param {function} success
   * @param {function} error
   * @return {Promise}
   */
  static searchRuns({ data, success, error }) {
    return $.ajax('/ajax-api/2.0/mlflow/runs/search', {
      type: 'POST',
      dataType: 'json',
      data: JSON.stringify(data),
      jsonp: false,
      success: success,
      error: error,
    });
  }

  /**
   * @param {ListArtifacts} data: Immutable Record
   * @param {function} success
   * @param {function} error
   * @return {Promise}
   */
  static listArtifacts({ data, success, error }) {
    return $.ajax('/ajax-api/2.0/mlflow/artifacts/list', {
      type: 'GET',
      dataType: 'json',
      converters: {
        'text json': StrictJsonBigInt.parse,
      },
      data: data,
      jsonp: false,
      success: success,
      error: error,
    });
  }

  /**
   * @param {GetMetricHistory} data: Immutable Record
   * @param {function} success
   * @param {function} error
   * @return {Promise}
   */
  static getMetricHistory({ data, success, error }) {
    return $.ajax('/ajax-api/2.0/mlflow/metrics/get-history', {
      type: 'GET',
      dataType: 'json',
      converters: {
        'text json': StrictJsonBigInt.parse,
      },
      data: data,
      jsonp: false,
      success: success,
      error: error,
    });
  }

  /**
   * @param {SetTag} data: Immutable Record
   * @param {function} success
   * @param {function} error
   * @return {Promise}
   */
  static setTag({ data, success, error }) {
    return $.ajax('/ajax-api/2.0/mlflow/runs/set-tag', {
      type: 'POST',
      dataType: 'json',
      data: JSON.stringify(data),
      jsonp: false,
      success: success,
      error: error,
    });
  }
}
