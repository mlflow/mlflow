import type {
  ExperimentEntity,
  ExperimentStoreEntities,
  ModelVersionInfoEntity,
  DatasetSummary,
  RunInfoEntity,
  RunDatasetWithTags,
  MetricEntity,
  RunInputsType,
  RunOutputsType,
} from '../../../types';
import { LIFECYCLE_FILTER, MODEL_VERSION_FILTER } from '../../../types';
import type { KeyValueEntity } from '../../../../common/types';
import { getLatestMetrics } from '../../../reducers/MetricReducer';
import { getExperimentTags, getParams, getRunDatasets, getRunInfo, getRunTags } from '../../../reducers/Reducers';
import { pickBy } from 'lodash';

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

  /**
   * List of inputs and outputs for each run.
   */
  inputsOutputsList?: { inputs?: RunInputsType; outputs?: RunOutputsType }[];
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
          return rInfo.lifecycleStage === 'active';
        } else {
          return rInfo.lifecycleStage === 'deleted';
        }
      })
      // Filter out runs by given model version filter
      .filter(([rInfo, _]) => {
        if (modelVersionFilter === MODEL_VERSION_FILTER.ALL_RUNS) {
          return true;
        } else if (modelVersionFilter === MODEL_VERSION_FILTER.WITH_MODEL_VERSIONS) {
          return rInfo.runUuid in modelVersionsByRunUuid;
        } else if (modelVersionFilter === MODEL_VERSION_FILTER.WTIHOUT_MODEL_VERSIONS) {
          return !(rInfo.runUuid in modelVersionsByRunUuid);
        } else {
          // eslint-disable-next-line no-console -- TODO(FEINF-3587)
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

          return datasetsFilter.some(({ name, digest }) => name === datasetName && digest === datasetDigest);
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
  const experimentIds = params.experimentIds || experiments.map((e) => e.experimentId);
  const comparingExperiments = experimentIds.length > 1;

  // Read the order of runs from array of UUIDs in the store, because otherwise the order when
  // reading from the object is not guaranteed. This is important when we are trying to sort runs by
  // metrics and other fields.
  const runOrder = state.entities.runInfoOrderByUuid || [];
  const runs = runOrder.map((runUuid) => state.entities.runInfosByUuid[runUuid]);

  /**
   * Extract run UUIDs relevant to selected experiments
   */
  const runUuids = runs
    .filter(({ experimentId }) => experimentIds.includes(experimentId))
    .map(({ runUuid }) => runUuid);

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
    return state.entities.runDatasetsByUuid[runInfo.runUuid];
  });

  const inputsOutputsList = runInfos.map((runInfo) => {
    return state.entities.runInputsOutputsByUuid[runInfo.runUuid];
  });

  /**
   * Extracting lists of metrics by run index
   */
  const metricsList = runInfos.map((runInfo) => {
    const metricsByRunUuid = getLatestMetrics(runInfo.runUuid, state);
    const metrics = (Object.values(metricsByRunUuid || {}) as any[]).filter(
      (metric) => metric.key.trim().length > 0, // Filter out metrics that are entirely whitespace
    );
    metrics.forEach((metric) => {
      metricKeysSet.add(metric.key);
    });
    return metrics;
  }) as MetricEntity[][];

  /**
   * Extracting lists of params by run index
   */
  const paramsList = runInfos.map((runInfo) => {
    const paramValues = (Object.values(getParams(runInfo.runUuid, state)) as any[]).filter(
      (param) => param.key.trim().length > 0, // Filter out params that are entirely whitespace
    );
    paramValues.forEach((param) => {
      paramKeysSet.add(param.key);
    });
    return paramValues;
  }) as KeyValueEntity[][];

  /**
   * Extracting dictionaries of tags by run index
   */
  const tagsList = runInfos.map((runInfo) =>
    pickBy(
      getRunTags(runInfo.runUuid, state),
      (tags) => tags.key.trim().length > 0, // Filter out tags that are entirely whitespace
    ),
  ) as Record<string, KeyValueEntity>[];

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
    inputsOutputsList,
    metricKeyList: Array.from(metricKeysSet.values()).sort(),
    paramKeyList: Array.from(paramKeysSet.values()).sort(),
  };
};
