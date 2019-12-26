import {
  LIST_REGISTRED_MODELS,
  SEARCH_MODEL_VERSIONS,
  GET_REGISTERED_MODEL_DETAILS,
  GET_MODEL_VERSION_DETAILS,
  DELETE_MODEL_VERSION,
  DELETE_REGISTERED_MODEL,
} from './actions';
import { fulfilled } from '../Actions';
import _ from 'lodash';

const modelByName = (state = {}, action) => {
  switch (action.type) {
    case fulfilled(LIST_REGISTRED_MODELS): {
      const detailedModels = action.payload.registered_models_detailed;
      const models = detailedModels && detailedModels.map(inlineModel);
      const nameToModelMap = {};
      if (models) {
        models.forEach((model) => (nameToModelMap[model.name] = model));
      }
      return {
        ...nameToModelMap,
      };
    }
    case fulfilled(GET_REGISTERED_MODEL_DETAILS): {
      const detailedModel = action.payload.registered_model_detailed;
      const inlinedModel = detailedModel && inlineModel(detailedModel);
      const { modelName } = action.meta;
      const modelWithUpdatedMetadata = {
        ...state[modelName],
        ...inlinedModel,
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
    case fulfilled(GET_MODEL_VERSION_DETAILS): {
      const modelVersion = action.payload.model_version_detailed;
      const inlinedModelVersion = inlineModelVersion(modelVersion);
      const { modelName } = action.meta;
      const updatedMap = {
        ...state[modelName],
        [inlinedModelVersion.version]: inlinedModelVersion,
      };
      return {
        ...state,
        [modelName]: updatedMap,
      };
    }
    case fulfilled(SEARCH_MODEL_VERSIONS): {
      const modelVersions = action.payload.model_versions_detailed;
      if (!modelVersions) {
        return state;
      }
      const inlinedModelVersions = modelVersions.map(inlineModelVersion);

      // Merge all modelVersions into the store
      return inlinedModelVersions.reduce((newState, modelVersion) => {
        const modelName = modelVersion.model_version.registered_model.name;
        const { version } = modelVersion.model_version;
        return {
          ...newState,
          [modelName]: {
            ...newState[modelName],
            [version]: modelVersion,
          },
        };
      }, { ...state });
    }
    case fulfilled(DELETE_MODEL_VERSION): {
      const { modelVersion } = action.meta;
      const { name } = modelVersion.registered_model;
      const modelVersionByVersion = state[name];
      return {
        [name]: _.omit(modelVersionByVersion, modelVersion.version),
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

export const getModelVersion = (state, modelName, version) => {
  const modelVersions = state.entities.modelVersionsByModel[modelName];
  return modelVersions && modelVersions[version];
};

export const getModelVersions = (state, modelName) => {
  const modelVersions = state.entities.modelVersionsByModel[modelName];
  return modelVersions && Object.values(modelVersions);
};

export const getAllModelVersions = (state) => {
  return _.flatMap(
    Object.values(state.entities.modelVersionsByModel),
    (modelVersionByVersion) => Object.values(modelVersionByVersion),
  );
};

// Inline the `name` field nested inside `registered_model` and `version` in `model_version`
const inlineModel = (model) => {
  const { registered_model, latest_versions } = model;
  return {
    ...model,
    name: registered_model.name,
    latest_versions: latest_versions && latest_versions.map(inlineModelVersion),
  };
};

// Inline the `version` field nested inside `model_version`
const inlineModelVersion = (modelVersion) => ({
  ...modelVersion,
  version: modelVersion.model_version.version,
});

export default {
  modelByName,
  modelVersionsByModel,
};
