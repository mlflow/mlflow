/**
 * NOTE: this code file was automatically migrated to TypeScript using ts-migrate and
 * may contain multiple `any` type annotations and `@ts-expect-error` directives.
 * If possible, please improve types while making changes to this file. If the type
 * annotations are already looking good, please remove this comment.
 */

import { RunTag, Metric } from '../../sdk/MlflowMessages';

export const emptyState = {
  apis: {},
  entities: {
    runInfosByUuid: {},
    runInfoOrderByUuid: [],
    experimentsById: {},
    experimentTagsByExperimentId: {},
    tagsByRunUuid: {},
    modelVersionsByRunUuid: {},
  },
};

export const addApiToState = (state: any, api: any) => {
  const oldApi = state.apis || {};
  return {
    ...state,
    apis: {
      ...oldApi,
      [api.id]: api,
    },
  };
};

export const addExperimentToState = (state: any, experiment: any) => {
  const oldExperiments = state.entities.experimentsById;
  return {
    ...state,
    entities: {
      ...state.entities,
      experimentsById: {
        ...oldExperiments,
        [experiment.experimentId]: experiment,
      },
    },
  };
};

export const addExperimentTagsToState = (state: any, experimentId: any, tags: any) => {
  const oldExperimentTags = state.entities.experimentTagsByExperimentId;
  const tagsArrToObject = (tagsArr: any) => {
    const tagObj = {};
    // @ts-expect-error TS(7053): Element implicitly has an 'any' type because expre... Remove this comment to see the full error message
    tagsArr.forEach((tag: any) => (tagObj[tag.key] = (RunTag as any).fromJs(tag)));
    return tagObj;
  };
  return {
    ...state,
    entities: {
      ...state.entities,
      experimentTagsByExperimentId: {
        ...oldExperimentTags,
        [experimentId]: tagsArrToObject(tags),
      },
    },
  };
};

export const createPendingApi = (id: any) => {
  return { id, active: true };
};

export const mockExperiment = (eid: any, name: any) => {
  return { experimentId: eid, name: name, allowedActions: [] };
};

export const mockRunInfo = (
  run_id: any,
  experiment_id = undefined,
  artifact_uri = undefined,
  lifecycle_stage = undefined,
) => {
  return {
    runUuid: run_id,
    experimentId: experiment_id,
    artifactUri: artifact_uri,
    lifecycleStage: lifecycle_stage,
  };
};

export const mockMetric = (params: any) => {
  return (Metric as any).fromJs(params);
};
