import {
  deleteJson,
  getBigIntJson,
  getJson,
  patchBigIntJson,
  patchJson,
  postBigIntJson,
  postJson,
} from '../common/utils/FetchUtils';


export class Services {

  /**
   * Get model Stages
   */
  static getModelStages = (data) =>
    getJson({ relativeUrl: 'ajax-api/2.0/preview/mlflow/model-stages/list', data });

    /**
   * Create a registered model
   */
  static createRegisteredModel = (data) =>
    postBigIntJson({ relativeUrl: 'ajax-api/2.0/preview/mlflow/registered-models/create', data });

  /**
   * List all registered models
   */
  static listRegisteredModels = (data) =>
    getBigIntJson({ relativeUrl: 'ajax-api/2.0/preview/mlflow/registered-models/list', data });

  /**
   * Search registered models
   */
  static searchRegisteredModels = (data) =>
    getBigIntJson({ relativeUrl: 'ajax-api/2.0/preview/mlflow/registered-models/search', data });

  /**
   * Update registered model
   */
  static updateRegisteredModel = (data) =>
    patchBigIntJson({ relativeUrl: 'ajax-api/2.0/preview/mlflow/registered-models/update', data });

  /**
   * Delete registered model
   */
  static deleteRegisteredModel = (data) =>
    deleteJson({ relativeUrl: 'ajax-api/2.0/preview/mlflow/registered-models/delete', data });

  /**
   * Set registered model tag
   */
  static setRegisteredModelTag = (data) =>
    postJson({ relativeUrl: 'ajax-api/2.0/preview/mlflow/registered-models/set-tag', data });

  /**
   * Delete registered model tag
   */
  static deleteRegisteredModelTag = (data) =>
    deleteJson({ relativeUrl: 'ajax-api/2.0/preview/mlflow/registered-models/delete-tag', data });

  /**
   * Create model version
   */
  static createModelVersion = (data) =>
    postBigIntJson({ relativeUrl: 'ajax-api/2.0/preview/mlflow/model-versions/create', data });

  /**
   * Search model versions
   */
  static searchModelVersions = (data) =>
    getJson({ relativeUrl: 'ajax-api/2.0/preview/mlflow/model-versions/search', data });

  /**
   * Update model version
   */
  static updateModelVersion = (data) =>
    patchJson({ relativeUrl: 'ajax-api/2.0/preview/mlflow/model-versions/update', data });

  /**
   * Transition model version stage
   */
  static transitionModelVersionStage = (data) =>
    postJson({ relativeUrl: 'ajax-api/2.0/preview/mlflow/model-versions/transition-stage', data });

  /**
   * Delete model version
   */
  static deleteModelVersion = (data) =>
    deleteJson({ relativeUrl: 'ajax-api/2.0/preview/mlflow/model-versions/delete', data });
  
    /**
   * Get stage configs
   */
   static getMlflowConfigs = () =>
   getJson({ relativeUrl: 'configs' });

  /**
   * Get individual registered model
   */
  static getRegisteredModel = (data) =>
    getJson({ relativeUrl: 'ajax-api/2.0/preview/mlflow/registered-models/get', data });

  /**
   * Get individual model version
   */
  static getModelVersion = (data) =>
    getJson({ relativeUrl: 'ajax-api/2.0/preview/mlflow/model-versions/get', data });

  /**
   * Set model version tag
   */
  static setModelVersionTag = (data) =>
    postJson({ relativeUrl: 'ajax-api/2.0/preview/mlflow/model-versions/set-tag', data });

  /**
   * Delete model version tag
   */
  static deleteModelVersionTag = (data) =>
    deleteJson({ relativeUrl: 'ajax-api/2.0/preview/mlflow/model-versions/delete-tag', data });
}
