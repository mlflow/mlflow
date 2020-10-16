import $ from 'jquery';
import JsonBigInt from 'json-bigint';
import Utils from '../common/utils/Utils';

const StrictJsonBigInt = JsonBigInt({ strict: true, storeAsString: true });

export class Services {
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
   * Search registered models
   * @param data
   * @param success
   * @param error
   * @returns {*|jQuery|*|*|*|*}
   */
  static searchRegisteredModels({ data, success, error }) {
    return $.ajax(Utils.getAjaxUrl('ajax-api/2.0/preview/mlflow/registered-models/search'), {
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
   * Set registered model tag
   * @param data
   * @param success
   * @param error
   * @returns {*|jQuery|*|*}
   */
  static setRegisteredModelTag({ data, success, error }) {
    return $.ajax(Utils.getAjaxUrl('ajax-api/2.0/preview/mlflow/registered-models/set-tag'), {
      type: 'POST',
      dataType: 'json',
      contentType: 'application/json; charset=utf-8',
      data: JSON.stringify(data),
      jsonp: false,
      success: success,
      error: error,
    });
  }

  /**
   * Delete registered model tag
   * @param data
   * @param success
   * @param error
   * @returns {*|jQuery|*|*}
   */
  static deleteRegisteredModelTag({ data, success, error }) {
    return $.ajax(Utils.getAjaxUrl('ajax-api/2.0/preview/mlflow/registered-models/delete-tag'), {
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
    return $.ajax(Utils.getAjaxUrl('ajax-api/2.0/preview/mlflow/model-versions/search'), {
      type: 'GET',
      dataType: 'json',
      contentType: 'application/json; charset=utf-8',
      data: data,
      jsonp: false,
      success: success,
      error: error,
    });
  }

  /**
   * Update model version
   * @param data
   * @param success
   * @param error
   * @returns {*|jQuery|*|*|*|*}
   */
  static updateModelVersion({ data, success, error }) {
    return $.ajax(Utils.getAjaxUrl('ajax-api/2.0/preview/mlflow/model-versions/update'), {
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
   * Transition model version stage
   * @param data
   * @param success
   * @param error
   * @returns {*|jQuery|*|*|*|*}
   */
  static transitionModelVersionStage({ data, success, error }) {
    return $.ajax(Utils.getAjaxUrl('ajax-api/2.0/preview/mlflow/model-versions/transition-stage'), {
      type: 'POST',
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
   * Get individual registered model
   * @param data
   * @param success
   * @param error
   * @returns {*|jQuery|*|*|*|*}
   */
  static getRegisteredModel({ data, success, error }) {
    return $.ajax(Utils.getAjaxUrl('ajax-api/2.0/preview/mlflow/registered-models/get'), {
      type: 'GET',
      dataType: 'json',
      contentType: 'application/json; charset=utf-8',
      data: data,
      jsonp: false,
      success: success,
      error: error,
    });
  }

  /**
   * Get individual model version
   * @param data
   * @param success
   * @param error
   * @returns {*|jQuery|*|*|*|*}
   */
  static getModelVersion({ data, success, error }) {
    return $.ajax(Utils.getAjaxUrl('ajax-api/2.0/preview/mlflow/model-versions/get'), {
      type: 'GET',
      dataType: 'json',
      contentType: 'application/json; charset=utf-8',
      data: data,
      jsonp: false,
      success: success,
      error: error,
    });
  }

  /**
   * Set model version tag
   * @param data
   * @param success
   * @param error
   * @returns {*|jQuery|*|*}
   */
  static setModelVersionTag({ data, success, error }) {
    return $.ajax(Utils.getAjaxUrl('ajax-api/2.0/preview/mlflow/model-versions/set-tag'), {
      type: 'POST',
      dataType: 'json',
      contentType: 'application/json; charset=utf-8',
      data: JSON.stringify(data),
      jsonp: false,
      success: success,
      error: error,
    });
  }

  /**
   * Delete model version tag
   * @param data
   * @param success
   * @param error
   * @returns {*|jQuery|*|*}
   */
  static deleteModelVersionTag({ data, success, error }) {
    return $.ajax(Utils.getAjaxUrl('ajax-api/2.0/preview/mlflow/model-versions/delete-tag'), {
      type: 'DELETE',
      dataType: 'json',
      contentType: 'application/json; charset=utf-8',
      data: JSON.stringify(data),
      jsonp: false,
      success: success,
      error: error,
    });
  }
}
