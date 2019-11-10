import Services from './services';
import { getUUID, wrapDeferred } from '../Actions';

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
export const updateRegisteredModelApi = (model, name, description, id = getUUID()) => ({
  type: UPDATE_REGISTERED_MODEL,
  payload: wrapDeferred(Services.updateRegisteredModel, {
    registered_model: model,
    name,
    description,
  }),
  meta: { id },
});

export const DELETE_REGISTERED_MODEL = 'DELETE_REGISTERED_MODEL';
export const deleteRegisteredModelApi = (model, id = getUUID()) => ({
  type: DELETE_REGISTERED_MODEL,
  payload: wrapDeferred(Services.deleteRegisteredModel, {
    registered_model: model,
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
  const filter = Object.keys(filterObj).map((key) => `${key}='${filterObj[key]}'`).join('&');
  return {
    type: SEARCH_MODEL_VERSIONS,
    payload: wrapDeferred(Services.searchModelVersions, { filter }),
    meta: { id },
  };
};

export const UPDATE_MODEL_VERSION = 'UPDATE_MODEL_VERSION';
export const updateModelVersionApi = (
  modelVersion,
  stage,
  description,
  id = getUUID(),
) => ({
  type: UPDATE_MODEL_VERSION,
  payload: wrapDeferred(Services.updateModelVersion, {
    model_version: modelVersion,
    stage,
    description,
  }),
  meta: { id },
});

export const DELETE_MODEL_VERSION = 'DELETE_MODEL_VERSION';
export const deleteModelVersionApi = (modelVersion, id = getUUID()) => ({
  type: DELETE_MODEL_VERSION,
  payload: wrapDeferred(Services.deleteModelVersion, {
    model_version: modelVersion,
  }),
  meta: { id, modelVersion },
});

export const GET_REGISTERED_MODEL_DETAILS = 'GET_REGISTERED_MODEL_DETAILS';
export const getRegisteredModelDetailsApi = (modelName, id = getUUID()) => ({
  type: GET_REGISTERED_MODEL_DETAILS,
  payload: wrapDeferred(Services.getRegisteredModelDetails, {
    registered_model: { name: modelName },
  }),
  meta: { id, modelName },
});

export const GET_MODEL_VERSION_DETAILS = 'GET_MODEL_VERSION_DETAILS';
export const getModelVersionDetailsApi = (modelName, version, id = getUUID()) => ({
  type: GET_MODEL_VERSION_DETAILS,
  payload: wrapDeferred(Services.getModelVersionDetails, {
    model_version: {
      registered_model: {
        name: modelName,
      },
      version,
    },
  }),
  meta: { id, modelName, version },
});
