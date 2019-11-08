import $ from 'jquery';
import JsonBigInt from 'json-bigint';
import Utils from '../utils/Utils';
import { ServiceOverrides } from './overrides/service-overrides';

const StrictJsonBigInt = JsonBigInt({ strict: true, storeAsString: true });

class Services {
  static createRegisteredModel({ data, success, error }) {
    return $.ajax(Utils.getAjaxUrl('ajax-api/2.0/preview/mlflow/registered-models/create'), {
      type: 'POST',
      dataType: 'json',
      contentType: 'application/json; charset=utf-8',
      converters: {
        'text json': StrictJsonBigInt.parse,
      },
      data: JSON.stringify(data),
      jsonp: false,
      success: success,
      error: error,
    });
  }

  /**
   * List all registered models
   * @param data
   * @param success
   * @param error
   * @returns {*|jQuery|*|*|*|*}
   */
  static listRegisteredModels({ data, success, error }) {
    return $.ajax(Utils.getAjaxUrl('ajax-api/2.0/preview/mlflow/registered-models/list'), {
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
   * Update registered model
   * @param data
   * @param success
   * @param error
   * @returns {*|jQuery|*|*|*|*}
   */
  static updateRegisteredModel({ data, success, error }) {
    return $.ajax(Utils.getAjaxUrl('ajax-api/2.0/preview/mlflow/registered-models/update'), {
      type: 'PATCH',
      dataType: 'json',
      contentType: 'application/json; charset=utf-8',
      converters: {
        'text json': StrictJsonBigInt.parse,
      },
      data: JSON.stringify(data),
      jsonp: false,
      success: success,
      error: error,
    });
  }

  /**
   * Delete registered model
   * @param data
   * @param success
   * @param error
   * @returns {*|jQuery|*|*|*|*}
   */
  static deleteRegisteredModel({ data, success, error }) {
    return $.ajax(Utils.getAjaxUrl('ajax-api/2.0/preview/mlflow/registered-models/delete'), {
      type: 'DELETE',
      dataType: 'json',
      contentType: 'application/json; charset=utf-8',
      data: JSON.stringify(data),
      jsonp: false,
      success: success,
      error: error,
    });
  }

  /**
   * Create model version
   * @param data
   * @param success
   * @param error
   * @returns {*|jQuery|*|*|*|*}
   */
  static createModelVersion({ data, success, error }) {
    return $.ajax(Utils.getAjaxUrl('ajax-api/2.0/preview/mlflow/model-versions/create'), {
      type: 'POST',
      dataType: 'json',
      contentType: 'application/json; charset=utf-8',
      converters: {
        'text json': StrictJsonBigInt.parse,
      },
      data: JSON.stringify(data),
      jsonp: false,
      success: success,
      error: error,
    });
  }

  /**
   * Search model versions
   * @param data
   * @param success
   * @param error
   * @returns {*|jQuery|*|*|*|*}
   */
  static searchModelVersions({ data, success, error }) {
    return $.ajax(
      Utils.getAjaxUrl('ajax-api/2.0/preview/mlflow/model-versions/search'),
      {
        type: 'GET',
        dataType: 'json',
        contentType: 'application/json; charset=utf-8',
        data: data,
        jsonp: false,
        success: success,
        error: error,
      },
    );
  }

  /**
   * Update model version
   * @param data
   * @param success
   * @param error
   * @returns {*|jQuery|*|*|*|*}
   */
  static updateModelVersion({ data, success, error }) {
    return $.ajax(Utils.getAjaxUrl(
      'ajax-api/2.0/preview/mlflow/model-versions/update'), {
        type: 'PATCH',
        dataType: 'json',
        contentType: 'application/json; charset=utf-8',
        data: JSON.stringify(data),
        jsonp: false,
        success: success,
        error: error,
      });
  }

  /**
   * Delete model version
   * @param data
   * @param success
   * @param error
   * @returns {*|jQuery|*|*|*|*}
   */
  static deleteModelVersion({ data, success, error }) {
    return $.ajax(Utils.getAjaxUrl('ajax-api/2.0/preview/mlflow/model-versions/delete'), {
      type: 'DELETE',
      dataType: 'json',
      contentType: 'application/json; charset=utf-8',
      data: JSON.stringify(data),
      jsonp: false,
      success: success,
      error: error,
    });
  }

  /**
   * Get individual registered model details
   * @param data
   * @param success
   * @param error
   * @returns {*|jQuery|*|*|*|*}
   */
  static getRegisteredModelDetails({ data, success, error }) {
    return $.ajax(Utils.getAjaxUrl('ajax-api/2.0/preview/mlflow/registered-models/get-details'), {
      type: 'POST', // TODO(mparkhe): Flatten API request arguments to be usable with GET
      dataType: 'json',
      contentType: 'application/json; charset=utf-8',
      data: JSON.stringify(data),
      jsonp: false,
      success: success,
      error: error,
    });
  }

  /**
   * Get individual model version details
   * @param data
   * @param success
   * @param error
   * @returns {*|jQuery|*|*|*|*}
   */
  static getModelVersionDetails({ data, success, error }) {
    return $.ajax(Utils.getAjaxUrl('ajax-api/2.0/preview/mlflow/model-versions/get-details'), {
      type: 'POST', // TODO(mparkhe): Flatten API request arguments to be usable with GET
      dataType: 'json',
      contentType: 'application/json; charset=utf-8',
      data: JSON.stringify(data),
      jsonp: false,
      success: success,
      error: error,
    });
  }
}

export default ServiceOverrides.Services || Services;
