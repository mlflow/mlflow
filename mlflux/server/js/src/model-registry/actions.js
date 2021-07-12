import { Services } from './services';
import { getUUID, wrapDeferred } from '../common/utils/ActionUtils';
import { getArtifactContent } from '../common/utils/ArtifactUtils';
import yaml from 'js-yaml';

export const CREATE_REGISTERED_MODEL = 'CREATE_REGISTERED_MODEL';
export const createRegisteredModelApi = (name, id = getUUID()) => ({
  type: CREATE_REGISTERED_MODEL,
  payload: wrapDeferred(Services.createRegisteredModel, { name }),
  meta: { id, name },
});

export const LIST_REGISTERED_MODELS = 'LIST_REGISTERED_MODELS';
export const listRegisteredModelsApi = (id = getUUID()) => ({
  type: LIST_REGISTERED_MODELS,
  payload: wrapDeferred(Services.listRegisteredModels, {}),
  meta: { id },
});

export const SEARCH_REGISTERED_MODELS = 'SEARCH_REGISTERED_MODELS';
export const searchRegisteredModelsApi = (
  filter,
  maxResults,
  orderBy,
  pageToken,
  id = getUUID(),
) => {
  return {
    type: SEARCH_REGISTERED_MODELS,
    payload: wrapDeferred(Services.searchRegisteredModels, {
      filter,
      max_results: maxResults,
      order_by: orderBy,
      ...(pageToken ? { page_token: pageToken } : null),
    }),
    meta: { id },
  };
};

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

export const SET_REGISTERED_MODEL_TAG = 'SET_REGISTERED_MODEL_TAG';
export const setRegisteredModelTagApi = (modelName, key, value, id = getUUID()) => ({
  type: SET_REGISTERED_MODEL_TAG,
  payload: wrapDeferred(Services.setRegisteredModelTag, {
    name: modelName,
    key: key,
    value: value,
  }),
  meta: { id, modelName, key, value },
});

export const DELETE_REGISTERED_MODEL_TAG = 'DELETE_REGISTERED_MODEL_TAG';
export const deleteRegisteredModelTagApi = (modelName, key, id = getUUID()) => ({
  type: DELETE_REGISTERED_MODEL_TAG,
  payload: wrapDeferred(Services.deleteRegisteredModelTag, {
    name: modelName,
    key: key,
  }),
  meta: { id, modelName, key },
});

export const CREATE_MODEL_VERSION = 'CREATE_MODEL_VERSION';
export const createModelVersionApi = (name, source, runId, id = getUUID()) => ({
  type: CREATE_MODEL_VERSION,
  payload: wrapDeferred(Services.createModelVersion, { name, source, run_id: runId }),
  meta: { id, name, runId },
});

export const GET_MODEL_VERSION_ARTIFACT = 'GET_MODEL_VERSION_ARTIFACT';
export const getModelVersionArtifactApi = (modelName, version, id = getUUID()) => {
  const baseUri = 'model-versions/get-artifact?path=MLmodel';
  const uriEncodedModelName = `name=${encodeURIComponent(modelName)}`;
  const uriEncodedModelVersion = `version=${encodeURIComponent(version)}`;
  const artifactLocation = `${baseUri}&${uriEncodedModelName}&${uriEncodedModelVersion}`;
  return {
    type: GET_MODEL_VERSION_ARTIFACT,
    payload: getArtifactContent(artifactLocation),
    meta: { id, modelName, version },
  };
};

// pass `null` to the `parseMlModelFile` API when we failed to fetch the
// file from DBFS. This will ensure requestId is registered in redux `apis` state
export const PARSE_MLMODEL_FILE = 'PARSE_MLMODEL_FILE';
export const parseMlModelFile = (modelName, version, mlModelFile, id = getUUID()) => {
  if (mlModelFile) {
    try {
      const parsedMlModelFile = yaml.safeLoad(mlModelFile);
      return {
        type: PARSE_MLMODEL_FILE,
        payload: Promise.resolve(parsedMlModelFile),
        meta: { id, modelName, version },
      };
    } catch (error) {
      console.error(error);
      return {
        type: PARSE_MLMODEL_FILE,
        payload: Promise.reject(),
        meta: { id, modelName, version },
      };
    }
  } else {
    return {
      type: PARSE_MLMODEL_FILE,
      payload: Promise.reject(),
      meta: { id, modelName, version },
    };
  }
};

export const GET_MODEL_VERSION_ACTIVITIES = 'GET_MODEL_VERSION_ACTIVITIES';
export const getModelVersionActivitiesApi = (modelName, version, id = getUUID()) => ({
  type: GET_MODEL_VERSION_ACTIVITIES,
  payload: wrapDeferred(Services.getModelVersionActivities, {
    name: modelName,
    version: version,
  }),
  meta: { id, modelName, version },
});

export const resolveFilterValue = (value, includeWildCard = false) => {
  const wrapper = includeWildCard ? '%' : '';
  return value.includes("'") ? `"${wrapper}${value}${wrapper}"` : `'${wrapper}${value}${wrapper}'`;
};

export const SEARCH_MODEL_VERSIONS = 'SEARCH_MODEL_VERSIONS';
export const searchModelVersionsApi = (filterObj, id = getUUID()) => {
  const filter = Object.keys(filterObj)
    .map((key) => {
      if (Array.isArray(filterObj[key]) && filterObj[key].length > 1) {
        return `${key} IN (${filterObj[key].map((elem) => resolveFilterValue(elem)).join()})`;
      } else if (Array.isArray(filterObj[key]) && filterObj[key].length === 1) {
        return `${key}=${resolveFilterValue(filterObj[key][0])}`;
      } else {
        return `${key}=${resolveFilterValue(filterObj[key])}`;
      }
    })
    .join('&');

  return {
    type: SEARCH_MODEL_VERSIONS,
    payload: wrapDeferred(Services.searchModelVersions, { filter }),
    meta: { id },
  };
};

export const UPDATE_MODEL_VERSION = 'UPDATE_MODEL_VERSION';
export const updateModelVersionApi = (modelName, version, description, id = getUUID()) => ({
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
  id = getUUID(),
) => ({
  type: TRANSITION_MODEL_VERSION_STAGE,
  payload: wrapDeferred(Services.transitionModelVersionStage, {
    name: modelName,
    version,
    stage,
    archive_existing_versions: archiveExistingVersions,
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

export const SET_MODEL_VERSION_TAG = 'SET_MODEL_VERSION_TAG';
export const setModelVersionTagApi = (modelName, version, key, value, id = getUUID()) => ({
  type: SET_MODEL_VERSION_TAG,
  payload: wrapDeferred(Services.setModelVersionTag, {
    name: modelName,
    version: version,
    key: key,
    value: value,
  }),
  meta: { id, modelName, version, key, value },
});

export const DELETE_MODEL_VERSION_TAG = 'DELETE_MODEL_VERSION_TAG';
export const deleteModelVersionTagApi = (modelName, version, key, id = getUUID()) => ({
  type: DELETE_MODEL_VERSION_TAG,
  payload: wrapDeferred(Services.deleteModelVersionTag, {
    name: modelName,
    version: version,
    key: key,
  }),
  meta: { id, modelName, version, key },
});
