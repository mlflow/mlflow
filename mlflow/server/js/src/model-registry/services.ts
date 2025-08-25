import {
  deleteJson,
  getBigIntJson,
  getJson,
  patchBigIntJson,
  patchJson,
  postBigIntJson,
  postJson,
} from '../common/utils/FetchUtils';

// eslint-disable-next-line @typescript-eslint/no-extraneous-class -- TODO(FEINF-4274)
export class Services {
  /**
   * Create a registered model
   */
  static createRegisteredModel = (data: any) =>
    postBigIntJson({ relativeUrl: 'ajax-api/2.0/mlflow/registered-models/create', data });

  /**
   * List all registered models
   */
  static listRegisteredModels = (data: any) =>
    getBigIntJson({ relativeUrl: 'ajax-api/2.0/mlflow/registered-models/list', data });

  /**
   * Search registered models
   */
  static searchRegisteredModels = (data: any) =>
    getBigIntJson({ relativeUrl: 'ajax-api/2.0/mlflow/registered-models/search', data });

  /**
   * Update registered model
   */
  static updateRegisteredModel = (data: any) =>
    patchBigIntJson({ relativeUrl: 'ajax-api/2.0/mlflow/registered-models/update', data });

  /**
   * Delete registered model
   */
  static deleteRegisteredModel = (data: any) =>
    deleteJson({ relativeUrl: 'ajax-api/2.0/mlflow/registered-models/delete', data });

  /**
   * Set registered model tag
   */
  static setRegisteredModelTag = (data: any) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/registered-models/set-tag', data });

  /**
   * Delete registered model tag
   */
  static deleteRegisteredModelTag = (data: any) =>
    deleteJson({ relativeUrl: 'ajax-api/2.0/mlflow/registered-models/delete-tag', data });

  /**
   * Create model version
   */
  static createModelVersion = (data: any) =>
    postBigIntJson({ relativeUrl: 'ajax-api/2.0/mlflow/model-versions/create', data });

  /**
   * Search model versions
   */
  static searchModelVersions = (data: any) =>
    getJson({ relativeUrl: 'ajax-api/2.0/mlflow/model-versions/search', data });

  /**
   * Update model version
   */
  static updateModelVersion = (data: any) =>
    patchJson({ relativeUrl: 'ajax-api/2.0/mlflow/model-versions/update', data });

  /**
   * Transition model version stage
   */
  static transitionModelVersionStage = (data: any) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/model-versions/transition-stage', data });

  /**
   * Delete model version
   */
  static deleteModelVersion = (data: any) =>
    deleteJson({ relativeUrl: 'ajax-api/2.0/mlflow/model-versions/delete', data });

  /**
   * Get individual registered model
   */
  static getRegisteredModel = (data: any) =>
    getJson({ relativeUrl: 'ajax-api/2.0/mlflow/registered-models/get', data });

  /**
   * Get individual model version
   */
  static getModelVersion = (data: any) => getJson({ relativeUrl: 'ajax-api/2.0/mlflow/model-versions/get', data });

  /**
   * Set model version tag
   */
  static setModelVersionTag = (data: any) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/model-versions/set-tag', data });

  /**
   * Delete model version tag
   */
  static deleteModelVersionTag = (data: any) =>
    deleteJson({ relativeUrl: 'ajax-api/2.0/mlflow/model-versions/delete-tag', data });

  /**
   * Set model version alias
   */
  static setModelVersionAlias = (data: { name: string; version: string; alias: string }) =>
    postJson({ relativeUrl: 'ajax-api/2.0/mlflow/registered-models/alias', data });

  /**
   * Delete model version alias
   */
  static deleteModelVersionAlias = (data: { name: string; version: string; alias: string }) =>
    deleteJson({ relativeUrl: 'ajax-api/2.0/mlflow/registered-models/alias', data });
}
