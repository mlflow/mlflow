import { Services } from './services';
import { getUUID, wrapDeferred } from '../common/utils/ActionUtils';

export const CREATE_REGISTERED_MODEL = 'CREATE_REGISTERED_MODEL';
export const createRegisteredModelApi = (name, id = getUUID()) => ({
  type: CREATE_REGISTERED_MODEL,
  payload: wrapDeferred(Services.createRegisteredModel, { name }),
  meta: { id, name },
});

export const LIST_REGISTRED_MODELS = 'LIST_REGISTRED_MODELS';
export const listRegisteredModelsApi = (id = getUUID()) => ({
  type: LIST_REGISTRED_MODELS,
  payload: wrapDeferred(Services.listRegisteredModels, {}),
  meta: { id },
});

export const UPDATE_REGISTERED_MODEL = 'UPDATE_REGISTERED_MODEL';
export const updateRegisteredModelApi = (name, description, id = getUUID()) => ({
  type: UPDATE_REGISTERED_MODEL,
  payload: wrapDeferred(Services.updateRegisteredModel, {
    name,
    description,
  }),
  meta: { id },
});

export const DELETE_REGISTERED_MODEL = 'DELETE_REGISTERED_MODEL';
export const deleteRegisteredModelApi = (model, id = getUUID(), localUpdateOnly) => ({
  type: DELETE_REGISTERED_MODEL,
  payload: localUpdateOnly
    ? Promise.resolve()
    : wrapDeferred(Services.deleteRegisteredModel, {
      name: model,
    }),
  meta: { id, model },
});

export const CREATE_MODEL_VERSION = 'CREATE_MODEL_VERSION';
export const createModelVersionApi = (name, source, runId, id = getUUID()) => ({
  type: CREATE_MODEL_VERSION,
  payload: wrapDeferred(Services.createModelVersion, { name, source, run_id: runId }),
  meta: { id, name, runId },
});

export const SEARCH_MODEL_VERSIONS = 'SEARCH_MODEL_VERSIONS';
export const searchModelVersionsApi = (filterObj, id = getUUID()) => {
  const filter = Object.keys(filterObj).map((key) => `${key}="${filterObj[key]}"`).join('&');
  return {
    type: SEARCH_MODEL_VERSIONS,
    payload: wrapDeferred(Services.searchModelVersions, { filter }),
    meta: { id },
  };
};

export const UPDATE_MODEL_VERSION = 'UPDATE_MODEL_VERSION';
export const updateModelVersionApi = (
  modelName,
  version,
  description,
  id = getUUID(),
) => ({
  type: UPDATE_MODEL_VERSION,
  payload: wrapDeferred(Services.updateModelVersion, {
    name: modelName,
    version: version,
    description,
  }),
  meta: { id },
});

export const TRANSITION_MODEL_VERSION_STAGE = 'TRANSITION_MODEL_VERSION_STAGE';
export const transitionModelVersionStageApi = (
  modelName,
  version,
  stage,
  archiveExistingVersions,
  comment,
  id = getUUID(),
) => ({
  type: TRANSITION_MODEL_VERSION_STAGE,
  payload: wrapDeferred(Services.transitionModelVersionStage, {
    name: modelName,
    version,
    stage,
    archive_existing_versions: archiveExistingVersions,
    comment,
  }),
  meta: { id },
});

export const DELETE_MODEL_VERSION = 'DELETE_MODEL_VERSION';
export const deleteModelVersionApi = (modelName, version, id = getUUID(), localUpdateOnly) => ({
  type: DELETE_MODEL_VERSION,
  payload: localUpdateOnly
    ? Promise.resolve()
    : wrapDeferred(Services.deleteModelVersion, {
      name: modelName,
      version: version,
    }),
  meta: { id, modelName, version },
});

export const GET_REGISTERED_MODEL = 'GET_REGISTERED_MODEL';
export const getRegisteredModelApi = (modelName, id = getUUID()) => ({
  type: GET_REGISTERED_MODEL,
  payload: wrapDeferred(Services.getRegisteredModel, {
    name: modelName,
  }),
  meta: { id, modelName },
});

export const GET_MODEL_VERSION = 'GET_MODEL_VERSION';
export const getModelVersionApi = (modelName, version, id = getUUID()) => ({
  type: GET_MODEL_VERSION,
  payload: wrapDeferred(Services.getModelVersion, {
    name: modelName,
    version: version,
  }),
  meta: { id, modelName, version },
});

