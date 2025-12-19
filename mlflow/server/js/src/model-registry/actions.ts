/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { Services } from './services';
import { getUUID } from '../common/utils/ActionUtils';
import { getArtifactContent } from '../common/utils/ArtifactUtils';
import yaml from 'js-yaml';
import type { ModelVersionInfoEntity } from '../experiment-tracking/types';
import type { KeyValueEntity } from '../common/types';

const CREATE_REGISTERED_MODEL = 'CREATE_REGISTERED_MODEL';
// @ts-expect-error TS(7006): Parameter 'name' implicitly has an 'any' type.
export const createRegisteredModelApi = (name, id = getUUID()) => ({
  type: CREATE_REGISTERED_MODEL,
  payload: Services.createRegisteredModel({ name }),
  meta: { id, name },
});

export const SEARCH_REGISTERED_MODELS = 'SEARCH_REGISTERED_MODELS';
export const searchRegisteredModelsApi = (
  filter?: any,
  maxResults?: any,
  orderBy?: any,
  pageToken?: any,
  id = getUUID(),
) => {
  return {
    type: SEARCH_REGISTERED_MODELS,
    payload: Services.searchRegisteredModels({
      filter,
      max_results: maxResults,
      order_by: orderBy,
      ...(pageToken ? { page_token: pageToken } : null),
    }),
    meta: { id },
  };
};

const UPDATE_REGISTERED_MODEL = 'UPDATE_REGISTERED_MODEL';
// @ts-expect-error TS(7006): Parameter 'name' implicitly has an 'any' type.
export const updateRegisteredModelApi = (name, description, id = getUUID()) => ({
  type: UPDATE_REGISTERED_MODEL,

  payload: Services.updateRegisteredModel({
    name,
    description,
  }),

  meta: { id },
});

export const DELETE_REGISTERED_MODEL = 'DELETE_REGISTERED_MODEL';
// @ts-expect-error TS(7006): Parameter 'model' implicitly has an 'any' type.
export const deleteRegisteredModelApi = (model, id = getUUID(), localUpdateOnly) => ({
  type: DELETE_REGISTERED_MODEL,

  payload: localUpdateOnly
    ? Promise.resolve()
    : Services.deleteRegisteredModel({
        name: model,
      }),

  meta: { id, model },
});

export const SET_REGISTERED_MODEL_TAG = 'SET_REGISTERED_MODEL_TAG';
// @ts-expect-error TS(7006): Parameter 'modelName' implicitly has an 'any' type... Remove this comment to see the full error message
export const setRegisteredModelTagApi = (modelName, key, value, id = getUUID()) => ({
  type: SET_REGISTERED_MODEL_TAG,

  payload: Services.setRegisteredModelTag({
    name: modelName,
    key: key,
    value: value,
  }),

  meta: { id, modelName, key, value },
});

export const DELETE_REGISTERED_MODEL_TAG = 'DELETE_REGISTERED_MODEL_TAG';
// @ts-expect-error TS(7006): Parameter 'modelName' implicitly has an 'any' type... Remove this comment to see the full error message
export const deleteRegisteredModelTagApi = (modelName, key, id = getUUID()) => ({
  type: DELETE_REGISTERED_MODEL_TAG,

  payload: Services.deleteRegisteredModelTag({
    name: modelName,
    key: key,
  }),

  meta: { id, modelName, key },
});

const CREATE_MODEL_VERSION = 'CREATE_MODEL_VERSION';
export const createModelVersionApi = (
  name?: string,
  source?: string,
  runId?: string,
  tags: any[] = [],
  id = getUUID(),
  loggedModelId?: string,
) => ({
  type: CREATE_MODEL_VERSION,
  payload: Services.createModelVersion({ name, source, run_id: runId, tags, model_id: loggedModelId }),
  meta: { id, name, runId },
});

export const GET_MODEL_VERSION_ARTIFACT = 'GET_MODEL_VERSION_ARTIFACT';
export const getModelVersionArtifactApi = (modelName: any, version: any, id = getUUID()) => {
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
export const parseMlModelFile = (modelName: any, version: any, mlModelFile: any, id = getUUID()) => {
  if (mlModelFile) {
    try {
      const parsedMlModelFile = yaml.safeLoad(mlModelFile);
      return {
        type: PARSE_MLMODEL_FILE,
        payload: Promise.resolve(parsedMlModelFile),
        meta: { id, modelName, version },
      };
    } catch (error) {
      // eslint-disable-next-line no-console -- TODO(FEINF-3587)
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
// @ts-expect-error TS(7006): Parameter 'modelName' implicitly has an 'any' type... Remove this comment to see the full error message
export const getModelVersionActivitiesApi = (modelName, version, id = getUUID()) => ({
  type: GET_MODEL_VERSION_ACTIVITIES,

  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-ignore - OSS specific ignore
  payload: Services.getModelVersionActivities({
    name: modelName,
    version: version,
  }),

  meta: { id, modelName, version },
});

export const resolveFilterValue = (value: any, includeWildCard = false) => {
  const wrapper = includeWildCard ? '%' : '';
  return value.includes("'") ? `"${wrapper}${value}${wrapper}"` : `'${wrapper}${value}${wrapper}'`;
};

export const SEARCH_MODEL_VERSIONS = 'SEARCH_MODEL_VERSIONS';

export const searchModelVersionsApi = (
  filterObj: Record<string, any>,
  id: string = getUUID(),
  maxResults?: number,
  orderBy?: string,
  pageToken?: string,
) => {
  const filter = Object.keys(filterObj)
    .map((key) => {
      if (Array.isArray(filterObj[key]) && filterObj[key].length > 1) {
        return `${key} IN (${filterObj[key].map((elem: any) => resolveFilterValue(elem)).join()})`;
      } else if (Array.isArray(filterObj[key]) && filterObj[key].length === 1) {
        return `${key}=${resolveFilterValue(filterObj[key][0])}`;
      } else {
        return `${key}=${resolveFilterValue(filterObj[key])}`;
      }
    })
    .join('&');
  return {
    type: SEARCH_MODEL_VERSIONS,
    payload: Services.searchModelVersions({
      filter,
      ...(maxResults !== undefined && maxResults > 0 ? { max_results: maxResults } : null),
      ...(orderBy !== undefined ? { order_by: orderBy } : null),
      ...(pageToken ? { page_token: pageToken } : null),
    }),
    meta: { id },
  };
};

const UPDATE_MODEL_VERSION = 'UPDATE_MODEL_VERSION';
// @ts-expect-error TS(7006): Parameter 'modelName' implicitly has an 'any' type... Remove this comment to see the full error message
export const updateModelVersionApi = (modelName, version, description, id = getUUID()) => ({
  type: UPDATE_MODEL_VERSION,

  payload: Services.updateModelVersion({
    name: modelName,
    version: version,
    description,
  }),

  meta: { id },
});

export const TRANSITION_MODEL_VERSION_STAGE = 'TRANSITION_MODEL_VERSION_STAGE';
export const transitionModelVersionStageApi = (
  // @ts-expect-error TS(7006): Parameter 'modelName' implicitly has an 'any' type... Remove this comment to see the full error message
  modelName,
  // @ts-expect-error TS(7006): Parameter 'version' implicitly has an 'any' type.
  version,
  // @ts-expect-error TS(7006): Parameter 'stage' implicitly has an 'any' type.
  stage,
  // @ts-expect-error TS(7006): Parameter 'archiveExistingVersions' implicitly has... Remove this comment to see the full error message
  archiveExistingVersions,
  id = getUUID(),
) => ({
  type: TRANSITION_MODEL_VERSION_STAGE,

  payload: Services.transitionModelVersionStage({
    name: modelName,
    version,
    stage,
    archive_existing_versions: archiveExistingVersions,
  }),

  meta: { id },
});

export const DELETE_MODEL_VERSION = 'DELETE_MODEL_VERSION';
// @ts-expect-error TS(7006): Parameter 'modelName' implicitly has an 'any' type... Remove this comment to see the full error message
export const deleteModelVersionApi = (modelName, version, id = getUUID(), localUpdateOnly) => ({
  type: DELETE_MODEL_VERSION,

  payload: localUpdateOnly
    ? Promise.resolve()
    : Services.deleteModelVersion({
        name: modelName,
        version: version,
      }),

  meta: { id, modelName, version },
});

export const GET_REGISTERED_MODEL = 'GET_REGISTERED_MODEL';
// @ts-expect-error TS(7006): Parameter 'modelName' implicitly has an 'any' type... Remove this comment to see the full error message
export const getRegisteredModelApi = (modelName, id = getUUID()) => ({
  type: GET_REGISTERED_MODEL,

  payload: Services.getRegisteredModel({
    name: modelName,
  }),

  meta: { id, modelName },
});

export const GET_MODEL_VERSION = 'GET_MODEL_VERSION';
// @ts-expect-error TS(7006): Parameter 'modelName' implicitly has an 'any' type... Remove this comment to see the full error message
export const getModelVersionApi = (modelName, version, id = getUUID()) => ({
  type: GET_MODEL_VERSION,

  payload: Services.getModelVersion({
    name: modelName,
    version: version,
  }),

  meta: { id, modelName, version },
});

export const SET_MODEL_VERSION_TAG = 'SET_MODEL_VERSION_TAG';
// @ts-expect-error TS(7006): Parameter 'modelName' implicitly has an 'any' type... Remove this comment to see the full error message
export const setModelVersionTagApi = (modelName, version, key, value, id = getUUID()) => ({
  type: SET_MODEL_VERSION_TAG,

  payload: Services.setModelVersionTag({
    name: modelName,
    version: version,
    key: key,
    value: value,
  }),

  meta: { id, modelName, version, key, value },
});

export const DELETE_MODEL_VERSION_TAG = 'DELETE_MODEL_VERSION_TAG';
// @ts-expect-error TS(7006): Parameter 'modelName' implicitly has an 'any' type... Remove this comment to see the full error message
export const deleteModelVersionTagApi = (modelName, version, key, id = getUUID()) => ({
  type: DELETE_MODEL_VERSION_TAG,

  payload: Services.deleteModelVersionTag({
    name: modelName,
    version: version,
    key: key,
  }),

  meta: { id, modelName, version, key },
});

const SET_MODEL_VERSION_ALIASES = 'SET_MODEL_VERSION_ALIASES';

export const setModelVersionAliasesApi = (
  modelName: string,
  version: string,
  existingAliases: string[],
  draftAliases: string[],
  id = getUUID(),
) => {
  // We need to add/remove aliases in separate requests.
  // First, determine new aliases to be added
  const addedAliases = draftAliases.filter((x) => !existingAliases.includes(x));
  // Next, determine those to be deleted
  const deletedAliases = existingAliases.filter((x) => !draftAliases.includes(x));

  // Fire all requests at once
  const updateRequests = Promise.all([
    ...addedAliases.map((newAlias) => Services.setModelVersionAlias({ alias: newAlias, name: modelName, version })),
    ...deletedAliases.map((deletedAlias) =>
      Services.deleteModelVersionAlias({ alias: deletedAlias, name: modelName, version }),
    ),
  ]);

  return {
    type: SET_MODEL_VERSION_ALIASES,
    payload: updateRequests,
    meta: { id, modelName, version, existingAliases, draftAliases },
  };
};

export const updateModelVersionTagsApi = (
  { name, version }: ModelVersionInfoEntity,
  existingTags: KeyValueEntity[],
  newTags: KeyValueEntity[],
  id = getUUID(),
) => {
  // We need to add/remove tags in separate requests.

  // First, determine new aliases to be added
  const addedOrModifiedTags = newTags.filter(
    ({ key: newTagKey, value: newTagValue }) =>
      !existingTags.some(
        ({ key: existingTagKey, value: existingTagValue }) =>
          existingTagKey === newTagKey && newTagValue === existingTagValue,
      ),
  );

  // Next, determine those to be deleted
  const deletedTags = existingTags.filter(
    ({ key: existingTagKey }) => !newTags.some(({ key: newTagKey }) => existingTagKey === newTagKey),
  );

  // Fire all requests at once
  const updateRequests = Promise.all([
    ...addedOrModifiedTags.map(({ key, value }) => Services.setModelVersionTag({ name, version, key, value })),
    ...deletedTags.map(({ key }) => Services.deleteModelVersionTag({ name, version, key })),
  ]);

  return {
    type: SET_MODEL_VERSION_TAG,
    payload: updateRequests,
    meta: { id, name, version, existingTags, newTags },
  };
};
