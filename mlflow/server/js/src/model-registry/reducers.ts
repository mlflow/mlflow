/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import {
  SEARCH_REGISTERED_MODELS,
  SEARCH_MODEL_VERSIONS,
  GET_REGISTERED_MODEL,
  GET_MODEL_VERSION,
  DELETE_MODEL_VERSION,
  DELETE_REGISTERED_MODEL,
  SET_REGISTERED_MODEL_TAG,
  DELETE_REGISTERED_MODEL_TAG,
  SET_MODEL_VERSION_TAG,
  DELETE_MODEL_VERSION_TAG,
  PARSE_MLMODEL_FILE,
} from './actions';
import { getProtoField } from './utils';
import { flatMap, isEmpty, isEqual, omit } from 'lodash';
import { fulfilled, rejected } from '../common/utils/ActionUtils';
import { RegisteredModelTag, ModelVersionTag } from './sdk/ModelRegistryMessages';

const modelByName = (state = {}, action: any) => {
  switch (action.type) {
    case fulfilled(SEARCH_REGISTERED_MODELS): {
      const models = action.payload[getProtoField('registered_models')];
      const nameToModelMap = {};
      if (models) {
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        models.forEach((model: any) => (nameToModelMap[model.name] = model));
      }
      return {
        ...nameToModelMap,
      };
    }
    case rejected(SEARCH_REGISTERED_MODELS): {
      return {};
    }
    case fulfilled(GET_REGISTERED_MODEL): {
      const detailedModel = action.payload[getProtoField('registered_model')];

      // If model retrieved from API contains no assigned aliases,
      // the corresponding field will be excluded from the payload.
      // We set it explicitly to make sure it works properly with the equality check below.
      detailedModel.aliases ||= [];

      const { modelName } = action.meta;
      const modelWithUpdatedMetadata = {
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        ...state[modelName],
        ...detailedModel,
      };
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      if (isEqual(modelWithUpdatedMetadata, state[modelName])) {
        return state;
      }
      return {
        ...state,
        ...{ [modelName]: modelWithUpdatedMetadata },
      };
    }
    case fulfilled(DELETE_REGISTERED_MODEL): {
      const { model } = action.meta;
      return omit(state, model.name);
    }
    default:
      return state;
  }
};

// 2-levels lookup for model version indexed by (modelName, version)
const modelVersionsByModel = (state = {}, action: any) => {
  switch (action.type) {
    case fulfilled(GET_MODEL_VERSION): {
      const modelVersion = action.payload[getProtoField('model_version')];
      const { modelName } = action.meta;
      const updatedMap = {
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        ...state[modelName],
        [modelVersion.version]: modelVersion,
      };
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      if (isEqual(state[modelName], updatedMap)) {
        return state;
      }
      return {
        ...state,
        [modelName]: updatedMap,
      };
    }
    case fulfilled(SEARCH_MODEL_VERSIONS): {
      const modelVersions = action.payload[getProtoField('model_versions')];
      const nameToModelVersionMap: Record<string, Record<string, any>> = {};
      if (modelVersions) {
        modelVersions.forEach((modelVersion: any) => {
          const { name, version } = modelVersion;
          (nameToModelVersionMap[name] ||= {})[version] = modelVersion;
        });
      }
      return {
        ...nameToModelVersionMap,
      };
    }
    case fulfilled(DELETE_MODEL_VERSION): {
      const { modelName, version } = action.meta;
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      const modelVersionByVersion = state[modelName];
      return {
        ...state,
        [modelName]: omit(modelVersionByVersion, version),
      };
    }
    case fulfilled(DELETE_REGISTERED_MODEL): {
      const { model } = action.meta;
      return omit(state, model.name);
    }
    default:
      return state;
  }
};

const mlModelArtifactByModelVersion = (state = {}, action: any) => {
  switch (action.type) {
    case fulfilled(PARSE_MLMODEL_FILE): {
      const artifact = action.payload;
      const { modelName, version } = action.meta;
      return {
        ...state,
        [modelName]: {
          // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
          ...state[modelName],
          [version]: artifact,
        },
      };
    }
    default:
      return state;
  }
};

export const getModelVersionSchemas = (state: any, modelName: any, version: any) => {
  const schemaMap = {};
  // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
  schemaMap['inputs'] = [];
  // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
  schemaMap['outputs'] = [];
  if (
    state.entities.mlModelArtifactByModelVersion[modelName] &&
    state.entities.mlModelArtifactByModelVersion[modelName][version]
  ) {
    const artifact = state.entities.mlModelArtifactByModelVersion[modelName][version];
    if (artifact.signature) {
      if (artifact.signature.inputs) {
        try {
          // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
          schemaMap['inputs'] = JSON.parse(artifact.signature.inputs.replace(/(\r\n|\n|\r)/gm, ''));
        } catch (error) {
          // eslint-disable-next-line no-console -- TODO(FEINF-3587)
          console.error(error);
        }
      }
      if (artifact.signature.outputs) {
        try {
          // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
          schemaMap['outputs'] = JSON.parse(artifact.signature.outputs.replace(/(\r\n|\n|\r)/gm, ''));
        } catch (error) {
          // eslint-disable-next-line no-console -- TODO(FEINF-3587)
          console.error(error);
        }
      }
    }
  }
  return schemaMap;
};

export const getModelVersion = (state: any, modelName: any, version: any) => {
  const modelVersions = state.entities.modelVersionsByModel[modelName];
  return modelVersions && modelVersions[version];
};

export const getModelVersions = (state: any, modelName: any) => {
  const modelVersions = state.entities.modelVersionsByModel[modelName];
  return modelVersions && Object.values(modelVersions);
};

export const getAllModelVersions = (state: any) => {
  return flatMap(Object.values(state.entities.modelVersionsByModel), (modelVersionByVersion) =>
    // @ts-expect-error TS(2769): No overload matches this call.
    Object.values(modelVersionByVersion),
  );
};

const tagsByRegisteredModel = (state = {}, action: any) => {
  const tagArrToObject = (tags: any) => {
    const tagObj = {};
    // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
    tags.forEach((tag: any) => (tagObj[tag.key] = (RegisteredModelTag as any).fromJs(tag)));
    return tagObj;
  };
  switch (action.type) {
    case fulfilled(GET_REGISTERED_MODEL): {
      const detailedModel = action.payload[getProtoField('registered_model')];
      const { modelName } = action.meta;
      if (detailedModel.tags && detailedModel.tags.length > 0) {
        const { tags } = detailedModel;
        const newState = { ...state };
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        newState[modelName] = tagArrToObject(tags);
        return isEqual(newState, state) ? state : newState;
      } else {
        return state;
      }
    }
    case fulfilled(SET_REGISTERED_MODEL_TAG): {
      const { modelName, key, value } = action.meta;
      const tag = (RegisteredModelTag as any).fromJs({
        key: key,
        value: value,
      });
      let newState = { ...state };
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      const oldTags = newState[modelName] || {};
      newState = {
        ...newState,
        [modelName]: {
          ...oldTags,
          [tag.key]: tag,
        },
      };
      return newState;
    }
    case fulfilled(DELETE_REGISTERED_MODEL_TAG): {
      const { modelName, key } = action.meta;
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      const oldTags = state[modelName] || {};
      const newTags = omit(oldTags, key);
      if (Object.keys(newTags).length === 0) {
        return omit({ ...state }, modelName);
      } else {
        return { ...state, [modelName]: newTags };
      }
    }
    default:
      return state;
  }
};

export const getRegisteredModelTags = (modelName: any, state: any) =>
  state.entities.tagsByRegisteredModel[modelName] || {};

const tagsByModelVersion = (state = {}, action: any) => {
  const tagArrToObject = (tags: any) => {
    const tagObj = {};
    // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
    tags.forEach((tag: any) => (tagObj[tag.key] = (ModelVersionTag as any).fromJs(tag)));
    return tagObj;
  };
  switch (action.type) {
    case fulfilled(GET_MODEL_VERSION): {
      const modelVersion = action.payload[getProtoField('model_version')];
      const { modelName, version } = action.meta;
      if (modelVersion.tags && modelVersion.tags.length > 0) {
        const { tags } = modelVersion;
        const newState = { ...state };
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        newState[modelName] = newState[modelName] || {};
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        newState[modelName][version] = tagArrToObject(tags);
        return newState;
      } else {
        return state;
      }
    }
    case fulfilled(SET_MODEL_VERSION_TAG): {
      const { modelName, version, key, value } = action.meta;
      const tag = (ModelVersionTag as any).fromJs({
        key: key,
        value: value,
      });
      const newState = { ...state };
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      newState[modelName] = newState[modelName] || {};
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      const oldTags = newState[modelName][version] || {};
      return {
        ...newState,
        [modelName]: {
          [version]: {
            ...oldTags,
            [tag.key]: tag,
          },
        },
      };
    }
    case fulfilled(DELETE_MODEL_VERSION_TAG): {
      const { modelName, version, key } = action.meta;
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      const oldTags = state[modelName] ? state[modelName][version] || {} : {};
      const newState = { ...state };
      const newTags = omit(oldTags, key);
      if (Object.keys(newTags).length === 0) {
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        newState[modelName] = omit({ ...state[modelName] }, version);
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        if (isEmpty(newState[modelName])) {
          return omit({ ...state }, modelName);
        } else {
          return newState;
        }
      } else {
        return {
          ...newState,
          [modelName]: {
            [version]: newTags,
          },
        };
      }
    }
    default:
      return state;
  }
};

export const getModelVersionTags = (modelName: any, version: any, state: any) => {
  if (state.entities.tagsByModelVersion[modelName]) {
    return state.entities.tagsByModelVersion[modelName][version] || {};
  } else {
    return {};
  }
};

const reducers = {
  modelByName,
  modelVersionsByModel,
  tagsByRegisteredModel,
  tagsByModelVersion,
  mlModelArtifactByModelVersion,
};

export default reducers;
