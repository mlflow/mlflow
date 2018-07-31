/**
 * DO NOT EDIT!!!
 *
 * @NOTE(dli) 12-21-2016
 *   This file is generated. For now, it is a snapshot of the proto services as of
 *   May 31, 2018 6:19:48 PM. We will update the generation pipeline to actually
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
    return $.ajax('/ajax-api/2.0/preview/mlflow/experiments/create', {
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
    return $.ajax('/ajax-api/2.0/preview/mlflow/experiments/list', {
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
    return $.ajax('/ajax-api/2.0/preview/mlflow/experiments/get', {
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
   * @param {GetRun} data: Immutable Record
   * @param {function} success
   * @param {function} error
   * @return {Promise}
   */
  static getRun({ data, success, error }) {
    return $.ajax('/ajax-api/2.0/preview/mlflow/runs/get', {
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
   * @param {GetMetric} data: Immutable Record
   * @param {function} success
   * @param {function} error
   * @return {Promise}
   */
  static getMetric({ data, success, error }) {
    return $.ajax('/ajax-api/2.0/preview/mlflow/metrics/get', {
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
    return $.ajax('/ajax-api/2.0/preview/mlflow/runs/search', {
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
   * @param {ListArtifacts} data: Immutable Record
   * @param {function} success
   * @param {function} error
   * @return {Promise}
   */
  static listArtifacts({ data, success, error }) {
    return $.ajax('/ajax-api/2.0/preview/mlflow/artifacts/list', {
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
   * @param {GetArtifact} data: Immutable Record
   * @param {function} success
   * @param {function} error
   * @return {Promise}
   */
  static getArtifact({ data, success, error }) {
    return $.ajax('/ajax-api/2.0/preview/mlflow/artifacts/get', {
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
    return $.ajax('/ajax-api/2.0/preview/mlflow/metrics/get-history', {
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
}

