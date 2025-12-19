/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { combineReducers } from 'redux';
import {
  CLOSE_ERROR_MODAL,
  GET_EXPERIMENT_API,
  GET_RUN_API,
  LIST_ARTIFACTS_API,
  LIST_ARTIFACTS_LOGGED_MODEL_API,
  OPEN_ERROR_MODAL,
  SEARCH_RUNS_API,
  LOAD_MORE_RUNS_API,
  SET_EXPERIMENT_TAG_API,
  SET_TAG_API,
  DELETE_TAG_API,
  SET_COMPARE_EXPERIMENTS,
  SEARCH_DATASETS_API,
} from '../actions';
import { Param, RunTag, ExperimentTag } from '../sdk/MlflowMessages';
import { ArtifactNode } from '../utils/ArtifactUtils';
import { metricsByRunUuid, latestMetricsByRunUuid, minMetricsByRunUuid, maxMetricsByRunUuid } from './MetricReducer';
import modelRegistryReducers from '../../model-registry/reducers';
import { isArray, isEqual, merge, update, intersection, omit, union } from 'lodash';
import { fulfilled, isFulfilledApi, isPendingApi, isRejectedApi, rejected } from '../../common/utils/ActionUtils';
import { SEARCH_MODEL_VERSIONS } from '../../model-registry/actions';
import { getProtoField } from '../../model-registry/utils';
import Utils from '../../common/utils/Utils';
import { evaluationDataReducer as evaluationData } from './EvaluationDataReducer';
import { modelGatewayReducer as modelGateway } from './ModelGatewayReducer';
import type {
  DatasetSummary,
  ExperimentEntity,
  ModelVersionInfoEntity,
  RunInfoEntity,
} from '@mlflow/mlflow/src/experiment-tracking/types';
import { sampledMetricsByRunUuid } from './SampledMetricsReducer';
import type { ErrorWrapper } from '../../common/utils/ErrorWrapper';
import { imagesByRunUuid } from './ImageReducer';
import { colorByRunUuid } from './RunColorReducer';
import { runInputsOutputsByUuid } from './InputsOutputsReducer';

export type ApisReducerReduxState = Record<
  string,
  { active: boolean; id: string; data: any; error?: Error | ErrorWrapper }
>;
export type ViewsReducerReduxState = {
  errorModal: {
    isOpen: boolean;
    text: string;
  };
};
export type ComparedExperimentsReducerReduxState = {
  comparedExperimentIds: string[];
  hasComparedExperimentsBefore: boolean;
};

export const getExperiments = (state: any): ExperimentEntity[] => {
  return Object.values(state.entities.experimentsById);
};

export const getExperiment = (id: any, state: any) => {
  return state.entities.experimentsById[id];
};

export const experimentsById = (state = {}, action: any): any => {
  switch (action.type) {
    case fulfilled(GET_EXPERIMENT_API): {
      const { experiment } = action.payload;

      const existingExperiment = (state as any)[experiment.experimentId] || {};

      return {
        ...state,
        [experiment.experimentId]: merge({}, existingExperiment, experiment),
      };
    }
    default:
      return state;
  }
};

export const getRunInfo = (runUuid: any, state: any) => {
  return state.entities.runInfosByUuid[runUuid];
};

export const getRunDatasets = (runUuid: string, state: any) => {
  return state.entities.runDatasetsByUuid[runUuid];
};

export const runUuidsMatchingFilter = (state = [], action: any) => {
  switch (action.type) {
    case fulfilled(SEARCH_RUNS_API):
    case fulfilled(LOAD_MORE_RUNS_API): {
      const isLoadingMore = action.type === fulfilled(LOAD_MORE_RUNS_API);
      const newState = isLoadingMore ? [...state] : [];
      if (isArray(action.payload?.runsMatchingFilter)) {
        // @ts-expect-error TS(2345): Argument of type 'any' is not assignable to parame... Remove this comment to see the full error message
        newState.push(...action.payload.runsMatchingFilter.map(({ info }: any) => info.runUuid));
      }
      return newState;
    }
    default:
      return state;
  }
};

export const runDatasetsByUuid = (state = {}, action: any) => {
  switch (action.type) {
    case fulfilled(GET_RUN_API): {
      const { run } = action.payload;
      const runUuid = run.info.runUuid;
      const runInputInfo = run.inputs || [];
      const newState = { ...state };
      if (runInputInfo && runUuid) {
        // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
        newState[runUuid] = runInputInfo.datasetInputs;
      }
      return newState;
    }
    case fulfilled(SEARCH_RUNS_API):
    case fulfilled(LOAD_MORE_RUNS_API): {
      const newState = { ...state };
      if (action.payload && action.payload.runs) {
        action.payload.runs.forEach((runJson: any) => {
          if (!runJson) {
            return;
          }
          const runInputInfo = runJson.inputs;
          const runUuid = runJson.info.runUuid;
          if (runInputInfo && runUuid) {
            // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
            newState[runUuid] = runInputInfo.datasetInputs;
          }
        });
      }
      return newState;
    }
    default:
      return state;
  }
};

export const runInfosByUuid = (state = {}, action: any) => {
  switch (action.type) {
    case fulfilled(GET_RUN_API): {
      const runInfo: RunInfoEntity = action.payload.run.info;
      return amendRunInfosByUuid(state, runInfo);
    }
    case fulfilled(SEARCH_RUNS_API): {
      const newState = {};
      if (action.payload && action.payload.runs) {
        action.payload.runs.forEach((rJson: any) => {
          const runInfo: RunInfoEntity = rJson.info;
          // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
          newState[runInfo.runUuid] = runInfo;
        });
      }
      return newState;
    }
    case rejected(SEARCH_RUNS_API): {
      return {};
    }
    case fulfilled(LOAD_MORE_RUNS_API): {
      const newState = { ...state };
      if (action.payload && action.payload.runs) {
        action.payload.runs.forEach((rJson: any) => {
          const runInfo: RunInfoEntity = rJson.info;
          // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
          newState[runInfo.runUuid] = runInfo;
        });
      }
      return newState;
    }
    default:
      return state;
  }
};

export const runInfoOrderByUuid = (state: string[] = [], action: any) => {
  switch (action.type) {
    case fulfilled(SEARCH_RUNS_API): {
      const newState: Set<string> = new Set();
      if (action.payload && action.payload.runs) {
        action.payload.runs.forEach((rJson: any) => {
          const runInfo: RunInfoEntity = rJson.info;
          newState.add(runInfo.runUuid);
        });
      }
      const newStateArray = Array.from(newState);
      if (isEqual(state, newStateArray)) {
        return state;
      }
      return newStateArray;
    }
    case fulfilled(LOAD_MORE_RUNS_API): {
      const newState: Set<string> = new Set(state);
      if (action.payload && action.payload.runs) {
        action.payload.runs.forEach((rJson: any) => {
          const runInfo: RunInfoEntity = rJson.info;
          newState.add(runInfo.runUuid);
        });
      }
      return Array.from(newState);
    }
    case rejected(SEARCH_RUNS_API): {
      return [];
    }
    default:
      return state;
  }
};

export const modelVersionsByRunUuid = (state = {}, action: any) => {
  switch (action.type) {
    case fulfilled(SEARCH_MODEL_VERSIONS): {
      let newState: Record<string, ModelVersionInfoEntity[]> = { ...state };
      const updatedState: Record<string, ModelVersionInfoEntity[]> = {};
      if (action.payload) {
        const modelVersions: ModelVersionInfoEntity[] = action.payload[getProtoField('model_versions')];
        if (modelVersions) {
          modelVersions.forEach((model_version: any) => {
            if (model_version.run_id in updatedState) {
              updatedState[model_version.run_id].push(model_version);
            } else {
              updatedState[model_version.run_id] = [model_version];
            }
          });
        }
      }
      newState = { ...newState, ...updatedState };
      if (isEqual(state, newState)) {
        return state;
      }
      return newState;
    }
    default:
      return state;
  }
};

const amendRunInfosByUuid = (state: any, runInfo: any) => {
  return {
    ...state,
    [runInfo.runUuid]: runInfo,
  };
};

export const getParams = (runUuid: any, state: any) => {
  const params = state.entities.paramsByRunUuid[runUuid];
  if (params) {
    return params;
  } else {
    return {};
  }
};

export const paramsByRunUuid = (state = {}, action: any) => {
  const paramArrToObject = (params: any) => {
    const paramObj = {};
    // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
    params.forEach((p: any) => (paramObj[p.key] = (Param as any).fromJs(p)));
    return paramObj;
  };
  switch (action.type) {
    case fulfilled(GET_RUN_API): {
      const { run } = action.payload;
      const runUuid = run.info.runUuid;
      const params = run.data.params || [];
      const newState = { ...state };
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      newState[runUuid] = paramArrToObject(params);
      return newState;
    }
    case fulfilled(SEARCH_RUNS_API):
    case fulfilled(LOAD_MORE_RUNS_API): {
      const { runs } = action.payload;
      const newState = { ...state };
      if (runs) {
        runs.forEach((rJson: any) => {
          const runUuid = rJson.info.runUuid;
          const params = rJson.data.params || [];
          // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
          newState[runUuid] = paramArrToObject(params);
        });
      }
      return newState;
    }
    default:
      return state;
  }
};

export const getRunTags = (runUuid: any, state: any) => state.entities.tagsByRunUuid[runUuid] || {};

export const getExperimentTags = (experimentId: any, state: any) => {
  const tags = state.entities.experimentTagsByExperimentId[experimentId];
  return tags || {};
};

export const tagsByRunUuid = (state = {}, action: any) => {
  const tagArrToObject = (tags: any) => {
    const tagObj = {};
    // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
    tags.forEach((tag: any) => (tagObj[tag.key] = (RunTag as any).fromJs(tag)));
    return tagObj;
  };
  switch (action.type) {
    case fulfilled(GET_RUN_API): {
      const runInfo: RunInfoEntity = action.payload.run.info;
      const tags = action.payload.run.data.tags || [];
      const runUuid = runInfo.runUuid;
      const newState = { ...state };
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      newState[runUuid] = tagArrToObject(tags);
      return newState;
    }
    case fulfilled(SEARCH_RUNS_API):
    case fulfilled(LOAD_MORE_RUNS_API): {
      const { runs } = action.payload;
      const newState = { ...state };
      if (runs) {
        runs.forEach((rJson: any) => {
          const runUuid = rJson.info.runUuid;
          const tags = rJson.data.tags || [];
          // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
          newState[runUuid] = tagArrToObject(tags);
        });
      }
      return newState;
    }
    case fulfilled(SET_TAG_API): {
      const tag = { key: action.meta.key, value: action.meta.value };
      return amendTagsByRunUuid(state, [tag], action.meta.runUuid);
    }
    case fulfilled(DELETE_TAG_API): {
      const { runUuid } = action.meta;
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      const oldTags = state[runUuid] ? state[runUuid] : {};
      const newTags = omit(oldTags, action.meta.key);
      if (Object.keys(newTags).length === 0) {
        return omit({ ...state }, runUuid);
      } else {
        return { ...state, [runUuid]: newTags };
      }
    }
    default:
      return state;
  }
};

const amendTagsByRunUuid = (state: any, tags: any, runUuid: any) => {
  let newState = { ...state };
  if (tags) {
    tags.forEach((tJson: any) => {
      const tag = (RunTag as any).fromJs(tJson);
      const oldTags = newState[runUuid] ? newState[runUuid] : {};
      newState = {
        ...newState,
        [runUuid]: {
          ...oldTags,
          [tag.key]: tag,
        },
      };
    });
  }
  return newState;
};

export const experimentTagsByExperimentId = (state = {}, action: any) => {
  const tagArrToObject = (tags: any) => {
    const tagObj = {};
    // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
    tags.forEach((tag: any) => (tagObj[tag.key] = (ExperimentTag as any).fromJs(tag)));
    return tagObj;
  };
  switch (action.type) {
    case fulfilled(GET_EXPERIMENT_API): {
      const { experiment } = action.payload;
      const newState = { ...state };
      const tags = experiment.tags || [];
      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      newState[experiment.experimentId] = tagArrToObject(tags);
      return newState;
    }
    case fulfilled(SET_EXPERIMENT_TAG_API): {
      const tag = { key: action.meta.key, value: action.meta.value };
      return amendExperimentTagsByExperimentId(state, [tag], action.meta.experimentId);
    }
    default:
      return state;
  }
};

const amendExperimentTagsByExperimentId = (state: any, tags: any, expId: any) => {
  let newState = { ...state };
  if (tags) {
    tags.forEach((tJson: any) => {
      const tag = (ExperimentTag as any).fromJs(tJson);
      const oldTags = newState[expId] ? newState[expId] : {};
      newState = {
        ...newState,
        [expId]: {
          ...oldTags,
          [tag.key]: tag,
        },
      };
    });
  }
  return newState;
};

export const getArtifacts = (runUuid: any, state: any) => {
  return state.entities.artifactsByRunUuid[runUuid];
};

export const artifactsByRunUuid = (state = {}, action: any) => {
  switch (action.type) {
    case fulfilled(LIST_ARTIFACTS_LOGGED_MODEL_API):
    case fulfilled(LIST_ARTIFACTS_API): {
      const queryPath = action.meta.path;
      const { runUuid, loggedModelId } = action.meta;

      // If the artifact belongs to a logged model instead of run, use its id as a store identifier
      const storeIdentifier = loggedModelId ?? runUuid;

      // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
      let artifactNode = state[storeIdentifier] || new ArtifactNode(true);
      // Make deep copy.
      artifactNode = artifactNode.deepCopy();
      const { files } = action.payload;
      // If there are no files, or if its not a dbfs:/ bucket, we show No Artifacts Recorded.
      if (files === undefined) {
        return {
          ...state,
          [storeIdentifier]: artifactNode,
        };
      }
      // Sort files to list directories first in the artifact tree view.
      files.sort((a: any, b: any) => b.is_dir - a.is_dir);
      // Do not coerce these out of JSON because we use JSON.parse(JSON.stringify
      // to deep copy. This does not work on the autogenerated immutable objects.
      if (queryPath === undefined) {
        // If queryPath is undefined, then we should set the root's children.
        artifactNode.setChildren(files);
      } else {
        // Otherwise, traverse the queryPath to get to the appropriate artifact node.
        // Filter out empty strings caused by spurious instances of slash, i.e.
        // "model/" instead of just "model"
        const pathParts = queryPath.split('/').filter((item: any) => item);
        let curArtifactNode = artifactNode;
        pathParts.forEach((part: any) => {
          curArtifactNode = curArtifactNode.children[part];
        });
        // Then set children on that artifact node.
        // ML-12477: This can throw error if we supply an invalid queryPath in the URL.
        try {
          if (curArtifactNode.fileInfo.is_dir) {
            curArtifactNode.setChildren(files);
          }
        } catch (err) {
          Utils.logErrorAndNotifyUser(`Unable to construct the artifact tree.`);
        }
      }
      return {
        ...state,
        [storeIdentifier]: artifactNode,
      };
    }
    default:
      return state;
  }
};

export const getArtifactRootUri = (runUuid: any, state: any) => {
  return state.entities.artifactRootUriByRunUuid[runUuid];
};

export const artifactRootUriByRunUuid = (state = {}, action: any) => {
  switch (action.type) {
    case fulfilled(GET_RUN_API): {
      const runInfo: RunInfoEntity = action.payload.run.info;
      const runUuid = runInfo.runUuid;
      return {
        ...state,
        [runUuid]: runInfo.artifactUri,
      };
    }
    case fulfilled(SEARCH_RUNS_API):
    case fulfilled(LOAD_MORE_RUNS_API): {
      const { runs } = action.payload;
      const newState: any = { ...state };
      if (runs) {
        runs.forEach((rJson: any) => {
          const runUuid = rJson.info.runUuid;
          const tags = rJson.data.tags || [];
          newState[runUuid] = rJson.info.artifactUri;
        });
      }
      return newState;
    }
    default:
      return state;
  }
};

const getExperimentDatasets = (experimentId: string, state: any) => {
  return state.entities.datasetsByExperimentId[experimentId];
};

export const datasetsByExperimentId = (state = {}, action: any) => {
  switch (action.type) {
    case fulfilled(SEARCH_DATASETS_API): {
      let newState: Record<string, DatasetSummary[]> = Object.assign({}, state);
      if (action.payload && action.payload.dataset_summaries) {
        newState = {};
        action.payload.dataset_summaries.forEach((dataset_summary: DatasetSummary) => {
          newState[dataset_summary.experiment_id] = [
            ...(newState[dataset_summary.experiment_id] || []),
            dataset_summary,
          ];
        });
      }
      return newState;
    }
    default:
      return state;
  }
};

const entities = combineReducers({
  experimentsById,
  runInfosByUuid,
  runInfoOrderByUuid,
  runDatasetsByUuid,
  runInputsOutputsByUuid,
  runUuidsMatchingFilter,
  metricsByRunUuid,
  imagesByRunUuid,
  latestMetricsByRunUuid,
  minMetricsByRunUuid,
  maxMetricsByRunUuid,
  paramsByRunUuid,
  tagsByRunUuid,
  experimentTagsByExperimentId,
  artifactsByRunUuid,
  artifactRootUriByRunUuid,
  modelVersionsByRunUuid,
  datasetsByExperimentId,
  sampledMetricsByRunUuid,
  colorByRunUuid,
  ...modelRegistryReducers,
});

export const getSharedParamKeysByRunUuids = (runUuids: any, state: any) =>
  intersection(...runUuids.map((runUuid: any) => Object.keys(state.entities.paramsByRunUuid[runUuid])));

export const getSharedMetricKeysByRunUuids = (runUuids: any, state: any) =>
  intersection(...runUuids.map((runUuid: any) => Object.keys(state.entities.latestMetricsByRunUuid[runUuid])));

export const getAllParamKeysByRunUuids = (runUuids: any, state: any) =>
  union(...runUuids.map((runUuid: any) => Object.keys(state.entities.paramsByRunUuid[runUuid])));

export const getAllMetricKeysByRunUuids = (runUuids: any, state: any) =>
  union(...runUuids.map((runUuid: any) => Object.keys(state.entities.latestMetricsByRunUuid[runUuid])));

export const getApis = (requestIds: any, state: any) => {
  return requestIds.map((id: any) => state.apis[id] || {});
};

export const apis = (state: ApisReducerReduxState = {}, action: any): ApisReducerReduxState => {
  if (isPendingApi(action)) {
    if (!action?.meta?.id) {
      return state;
    }
    return {
      ...state,
      [action.meta.id]: { id: action.meta.id, active: true },
    };
  } else if (isFulfilledApi(action)) {
    if (!action?.meta?.id) {
      return state;
    }
    return {
      ...state,
      [action.meta.id]: { id: action.meta.id, active: false, data: action.payload },
    };
  } else if (isRejectedApi(action)) {
    if (!action?.meta?.id) {
      return state;
    }
    return {
      ...state,
      [action.meta.id]: { id: action.meta.id, active: false, error: action.payload },
    };
  } else {
    return state;
  }
};

// This state is used in the following components to show a breadcrumb link for navigating back to
// the compare-experiments page.
// - RunView
// - CompareRunView
// - MetricView
const defaultCompareExperimentsState: ComparedExperimentsReducerReduxState = {
  // Experiment IDs compared on `/compare-experiments`.
  comparedExperimentIds: [],
  // Indicates whether the user has navigated to `/compare-experiments` before
  // Should be set to false when the user navigates to `/experiments/<experiment_id>`
  hasComparedExperimentsBefore: false,
};
const compareExperiments = (
  state: ComparedExperimentsReducerReduxState = defaultCompareExperimentsState,
  action: any,
): ComparedExperimentsReducerReduxState => {
  if (action.type === SET_COMPARE_EXPERIMENTS) {
    const { comparedExperimentIds, hasComparedExperimentsBefore } = action.payload;
    return {
      ...state,
      comparedExperimentIds,
      hasComparedExperimentsBefore,
    };
  } else {
    return state;
  }
};

export const isErrorModalOpen = (state: any) => {
  return state.views.errorModal.isOpen;
};

export const getErrorModalText = (state: any) => {
  return state.views.errorModal.text;
};

const errorModalDefault = {
  isOpen: false,
  text: '',
};

const errorModal = (state = errorModalDefault, action: any) => {
  switch (action.type) {
    case CLOSE_ERROR_MODAL: {
      return {
        ...state,
        isOpen: false,
      };
    }
    case OPEN_ERROR_MODAL: {
      return {
        isOpen: true,
        text: action.text,
      };
    }
    default:
      return state;
  }
};

export const views = combineReducers({
  errorModal,
});

export const rootReducer = combineReducers({
  entities,
  views,
  apis,
  compareExperiments,
  evaluationData,
  modelGateway,
});

export const getEntities = (state: any) => {
  return state.entities;
};
