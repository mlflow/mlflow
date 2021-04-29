import {
  LIST_REGISTERED_MODELS,
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
import _ from 'lodash';
import { fulfilled, rejected } from '../common/utils/ActionUtils';
import { RegisteredModelTag, ModelVersionTag } from './sdk/ModelRegistryMessages';

const modelByName = (state = {}, action) => {
  switch (action.type) {
    case fulfilled(SEARCH_REGISTERED_MODELS):
    // eslint-disable-next-line no-fallthrough
    case fulfilled(LIST_REGISTERED_MODELS): {
      const models = action.payload[getProtoField('registered_models')];
      const nameToModelMap = {};
      if (models) {
        models.forEach((model) => (nameToModelMap[model.name] = model));
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
      const { modelName } = action.meta;
      const modelWithUpdatedMetadata = {
        ...state[modelName],
        ...detailedModel,
      };
      return {
        ...state,
        ...{ [modelName]: modelWithUpdatedMetadata },
      };
    }
    case fulfilled(DELETE_REGISTERED_MODEL): {
      const { model } = action.meta;
      return _.omit(state, model.name);
    }
    default:
      return state;
  }
};

// 2-levels lookup for model version indexed by (modelName, version)
const modelVersionsByModel = (state = {}, action) => {
  switch (action.type) {
    case fulfilled(GET_MODEL_VERSION): {
      const modelVersion = action.payload[getProtoField('model_version')];
      const { modelName } = action.meta;
      const updatedMap = {
        ...state[modelName],
        [modelVersion.version]: modelVersion,
      };
      return {
        ...state,
        [modelName]: updatedMap,
      };
    }
    case fulfilled(SEARCH_MODEL_VERSIONS): {
      const modelVersions = action.payload[getProtoField('model_versions')];
      if (!modelVersions) {
        return state;
      }

      // Merge all modelVersions into the store
      return modelVersions.reduce(
        (newState, modelVersion) => {
          const { name, version } = modelVersion;
          return {
            ...newState,
            [name]: {
              ...newState[name],
              [version]: modelVersion,
            },
          };
        },
        { ...state },
      );
    }
    case fulfilled(DELETE_MODEL_VERSION): {
      const { modelName, version } = action.meta;
      const modelVersionByVersion = state[modelName];
      return {
        ...state,
        [modelName]: _.omit(modelVersionByVersion, version),
      };
    }
    case fulfilled(DELETE_REGISTERED_MODEL): {
      const { model } = action.meta;
      return _.omit(state, model.name);
    }
    default:
      return state;
  }
};

const mlModelArtifactByModelVersion = (state = {}, action) => {
  switch (action.type) {
    case fulfilled(PARSE_MLMODEL_FILE): {
      const artifact = action.payload;
      const { modelName, version } = action.meta;
      return {
        ...state,
        [modelName]: {
          ...state[modelName],
          [version]: artifact,
        },
      };
    }
    default:
      return state;
  }
};

export const getModelVersionSchemas = (state, modelName, version) => {
  const schemaMap = {};
  schemaMap['inputs'] = [];
  schemaMap['outputs'] = [];
  if (
    state.entities.mlModelArtifactByModelVersion[modelName] &&
    state.entities.mlModelArtifactByModelVersion[modelName][version]
  ) {
    const artifact = state.entities.mlModelArtifactByModelVersion[modelName][version];
    if (artifact.signature) {
      if (artifact.signature.inputs) {
        try {
          schemaMap['inputs'] = JSON.parse(artifact.signature.inputs);
        } catch (error) {
          console.error(error);
        }
      }
      if (artifact.signature.outputs) {
        try {
          schemaMap['outputs'] = JSON.parse(artifact.signature.outputs);
        } catch (error) {
          console.error(error);
        }
      }
    }
  }
  return schemaMap;
};

export const getModelVersion = (state, modelName, version) => {
  const modelVersions = state.entities.modelVersionsByModel[modelName];
  return modelVersions && modelVersions[version];
};

export const getModelVersions = (state, modelName) => {
  const modelVersions = state.entities.modelVersionsByModel[modelName];
  return modelVersions && Object.values(modelVersions);
};

export const getAllModelVersions = (state) => {
  return _.flatMap(Object.values(state.entities.modelVersionsByModel), (modelVersionByVersion) =>
    Object.values(modelVersionByVersion),
  );
};

const tagsByRegisteredModel = (state = {}, action) => {
  const tagArrToObject = (tags) => {
    const tagObj = {};
    tags.forEach((tag) => (tagObj[tag.key] = RegisteredModelTag.fromJs(tag)));
    return tagObj;
  };
  switch (action.type) {
    case fulfilled(GET_REGISTERED_MODEL): {
      const detailedModel = action.payload[getProtoField('registered_model')];
      const { modelName } = action.meta;
      if (detailedModel.tags && detailedModel.tags.length > 0) {
        const { tags } = detailedModel;
        const newState = { ...state };
        newState[modelName] = tagArrToObject(tags);
        return newState;
      } else {
        return state;
      }
    }
    case fulfilled(SET_REGISTERED_MODEL_TAG): {
      const { modelName, key, value } = action.meta;
      const tag = RegisteredModelTag.fromJs({
        key: key,
        value: value,
      });
      let newState = { ...state };
      const oldTags = newState[modelName] || {};
      newState = {
        ...newState,
        [modelName]: {
          ...oldTags,
          [tag.getKey()]: tag,
        },
      };
      return newState;
    }
    case fulfilled(DELETE_REGISTERED_MODEL_TAG): {
      const { modelName, key } = action.meta;
      const oldTags = state[modelName] || {};
      const newTags = _.omit(oldTags, key);
      if (Object.keys(newTags).length === 0) {
        return _.omit({ ...state }, modelName);
      } else {
        return { ...state, [modelName]: newTags };
      }
    }
    default:
      return state;
  }
};

export const getRegisteredModelTags = (modelName, state) =>
  state.entities.tagsByRegisteredModel[modelName] || {};

const tagsByModelVersion = (state = {}, action) => {
  const tagArrToObject = (tags) => {
    const tagObj = {};
    tags.forEach((tag) => (tagObj[tag.key] = ModelVersionTag.fromJs(tag)));
    return tagObj;
  };
  switch (action.type) {
    case fulfilled(GET_MODEL_VERSION): {
      const modelVersion = action.payload[getProtoField('model_version')];
      const { modelName, version } = action.meta;
      if (modelVersion.tags && modelVersion.tags.length > 0) {
        const { tags } = modelVersion;
        const newState = { ...state };
        newState[modelName] = newState[modelName] || {};
        newState[modelName][version] = tagArrToObject(tags);
        return newState;
      } else {
        return state;
      }
    }
    case fulfilled(SET_MODEL_VERSION_TAG): {
      const { modelName, version, key, value } = action.meta;
      const tag = ModelVersionTag.fromJs({
        key: key,
        value: value,
      });
      const newState = { ...state };
      newState[modelName] = newState[modelName] || {};
      const oldTags = newState[modelName][version] || {};
      return {
        ...newState,
        [modelName]: {
          [version]: {
            ...oldTags,
            [tag.getKey()]: tag,
          },
        },
      };
    }
    case fulfilled(DELETE_MODEL_VERSION_TAG): {
      const { modelName, version, key } = action.meta;
      const oldTags = state[modelName] ? state[modelName][version] || {} : {};
      const newState = { ...state };
      const newTags = _.omit(oldTags, key);
      if (Object.keys(newTags).length === 0) {
        newState[modelName] = _.omit({ ...state[modelName] }, version);
        if (_.isEmpty(newState[modelName])) {
          return _.omit({ ...state }, modelName);
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

export const getModelVersionTags = (modelName, version, state) => {
  if (state.entities.tagsByModelVersion[modelName]) {
    return state.entities.tagsByModelVersion[modelName][version] || {};
  } else {
    return {};
  }
};

export default {
  modelByName,
  modelVersionsByModel,
  tagsByRegisteredModel,
  tagsByModelVersion,
  mlModelArtifactByModelVersion,
};
