import {
  ExperimentEntity,
  ExperimentStoreEntities,
  KeyValueEntity,
  LIFECYCLE_FILTER,
  ModelVersionInfoEntity,
  MODEL_VERSION_FILTER,
  DatasetSummary,
  RunInfoEntity,
  RunDatasetWithTags,
  MetricEntity,
} from '../../../types';
import { getLatestMetrics } from '../../../reducers/MetricReducer';
import { getExperimentTags, getParams, getRunDatasets, getRunInfo, getRunTags } from '../../../reducers/Reducers';

export type ExperimentRunsSelectorResult = {
  /**
   * Array of run infos
   */
  runInfos: RunInfoEntity[];

  /**
   * List of unique metric keys
   */
  runUuidsMatchingFilter: string[];

  /**
   * List of unique metric keys
   */
  metricKeyList: string[];

  /**
   * List of unique param keys
   */
  paramKeyList: string[];

  /**
   * List of metrics indexed by the respective runs.
   * Example: metricsList[2] contains list of all
   * metrics corresponding to the 3rd run in the run list
   */
  metricsList: MetricEntity[][];

  /**
   * List of metrics indexed by the respective runs.
   * Example: paramsList[2] contains list of all
   * params corresponding to the 3rd run in the run list
   */
  paramsList: KeyValueEntity[][];

  /**
   * List of tags indexed by the respective runs.
   * Example: tagsList[2] contains dictionary of all
   * tags corresponding to the 3rd run in the run list
   */
  tagsList: Record<string, KeyValueEntity>[];

  /**
   * Dictionary containing model information objects indexed by run uuid
   */
  modelVersionsByRunUuid: Record<string, ModelVersionInfoEntity[]>;

  /**
   * Dictionary containing all tags assigned to a experiment
   * (single experiment only)
   */
  experimentTags: Record<string, KeyValueEntity>;

  /**
   * List of dataset arrays indexed by the respective runs.
   * E.g. datasetsList[2] yields an array of all
   * datasets corresponding to the 3rd run in the run list
   */
  datasetsList: RunDatasetWithTags[][];
};

export type ExperimentRunsSelectorParams = {
  experiments: ExperimentEntity[];
  experimentIds?: string[];
  lifecycleFilter?: LIFECYCLE_FILTER;
  modelVersionFilter?: MODEL_VERSION_FILTER;
  datasetsFilter?: DatasetSummary[];
};

/**
 * Extracts run infos filtered by lifecycle filter and model version filter
 */
const extractRunInfos = (
  runUuids: string[],
  state: { entities: ExperimentStoreEntities },
  {
    lifecycleFilter = LIFECYCLE_FILTER.ACTIVE,
    modelVersionFilter = MODEL_VERSION_FILTER.ALL_RUNS,
    datasetsFilter = [],
  }: ExperimentRunsSelectorParams,
): RunInfoEntity[] => {
  const { modelVersionsByRunUuid } = state.entities;

  return (
    runUuids
      // Get the basic run info
      .map((run_id) => [getRunInfo(run_id, state), getRunDatasets(run_id, state)])
      // Filter out runs by given lifecycle filter
      .filter(([rInfo, _]) => {
        if (lifecycleFilter === LIFECYCLE_FILTER.ACTIVE) {
          return rInfo.lifecycle_stage === 'active';
        } else {
          return rInfo.lifecycle_stage === 'deleted';
        }
      })
      // Filter out runs by given model version filter
      .filter(([rInfo, _]) => {
        if (modelVersionFilter === MODEL_VERSION_FILTER.ALL_RUNS) {
          return true;
        } else if (modelVersionFilter === MODEL_VERSION_FILTER.WITH_MODEL_VERSIONS) {
          return rInfo.run_uuid in modelVersionsByRunUuid;
        } else if (modelVersionFilter === MODEL_VERSION_FILTER.WTIHOUT_MODEL_VERSIONS) {
          return !(rInfo.run_uuid in modelVersionsByRunUuid);
        } else {
          console.warn('Invalid input to model version filter - defaulting to showing all runs.');
          return true;
        }
      })
      .filter(([_, datasets]) => {
        if (!datasetsFilter || datasetsFilter.length === 0) return true;
        if (!datasets) return false;

        // Returns true if there exists a dataset that is in datasetsFilter
        return datasets.some((datasetWithTags: RunDatasetWithTags) => {
          const datasetName = datasetWithTags.dataset.name;
          const datasetDigest = datasetWithTags.dataset.digest;
          const datasetTag = datasetWithTags.tags
            ? datasetWithTags.tags.find((tag) => tag.key === 'mlflow.data.context')
            : undefined;
          return datasetsFilter.some(
            ({ name, digest, context }) =>
              name === datasetName &&
              digest === datasetDigest &&
              (datasetTag ? context === datasetTag.value : context === undefined),
          );
        });
      })
      .map(([rInfo, _]) => rInfo)
  );
};

export const experimentRunsSelector = (
  state: { entities: ExperimentStoreEntities },
  params: ExperimentRunsSelectorParams,
): ExperimentRunsSelectorResult => {
  const { experiments } = params;
  const experimentIds = params.experimentIds || experiments.map((e) => e.experiment_id);
  const comparingExperiments = experimentIds.length > 1;

  /**
   * Extract run UUIDs relevant to selected experiments
   */
  const runUuids = Object.values(state.entities.runInfosByUuid)
    .filter(({ experiment_id }) => experimentIds.includes(experiment_id))
    .map(({ run_uuid }) => run_uuid);

  /**
   * Extract model version and runs matching filter directly from the store
   */
  const { modelVersionsByRunUuid, runUuidsMatchingFilter } = state.entities;

  /**
   * Extract run infos
   */
  const runInfos = extractRunInfos(runUuids, state, params);

  /**
   * Set of unique metric keys
   */
  const metricKeysSet = new Set<string>();

  /**
   * Set of unique param keys
   */
  const paramKeysSet = new Set<string>();

  const datasetsList = runInfos.map((runInfo) => {
    return state.entities.runDatasetsByUuid[runInfo.run_uuid];
  });

  /**
   * Extracting lists of metrics by run index
   */
  const metricsList = runInfos.map((runInfo) => {
    const metricsByRunUuid = getLatestMetrics(runInfo.run_uuid, state);
    const metrics = Object.values(metricsByRunUuid || {}) as any[];
    metrics.forEach((metric) => {
      metricKeysSet.add(metric.key);
    });
    return metrics;
  }) as MetricEntity[][];

  /**
   * Extracting lists of params by run index
   */
  const paramsList = runInfos.map((runInfo) => {
    const paramValues = Object.values(getParams(runInfo.run_uuid, state)) as any[];
    paramValues.forEach((param) => {
      paramKeysSet.add(param.key);
    });
    return paramValues;
  }) as KeyValueEntity[][];

  /**
   * Extracting dictionaries of tags by run index
   */
  const tagsList = runInfos.map((runInfo) => getRunTags(runInfo.run_uuid, state)) as Record<string, KeyValueEntity>[];

  const firstExperimentId = experimentIds[0];

  /**
   * If there is only one experiment, extract experiment tags as well
   */
  const experimentTags = (comparingExperiments ? {} : getExperimentTags(firstExperimentId, state)) as Record<
    string,
    KeyValueEntity
  >;

  return {
    modelVersionsByRunUuid,
    experimentTags,
    runInfos,
    paramsList,
    tagsList,
    metricsList,
    runUuidsMatchingFilter,
    datasetsList,
    metricKeyList: Array.from(metricKeysSet.values()).sort(),
    paramKeyList: Array.from(paramKeysSet.values()).sort(),
  };
};
