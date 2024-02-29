/**
 * Type definitions for models used in experiment tracking.
 * See 'src/experiment-tracking/sdk/MlflowMessages.js' for reference
 *
 * Note: this could be automatically generated in the future.
 */

import { SearchExperimentRunsFacetsState } from './components/experiment-page/models/SearchExperimentRunsFacetsState';
import { SearchExperimentRunsViewState } from './components/experiment-page/models/SearchExperimentRunsViewState';
import { RawEvaluationArtifact } from './sdk/EvaluationArtifactService';

/**
 * Simple key/value model enhanced with immutable.js
 * getter methods
 */
export interface KeyValueEntity {
  key: string;
  value: string;

  getKey(): string;
  getValue(): string;
}

export type ModelAliasMap = { alias: string; version: string }[];
type ModelVersionAliasList = string[];

/**
 * An entity defining a single model
 */
export interface ModelEntity {
  creation_timestamp: number;
  current_stage: string;
  version: string;
  description: string;
  id: string;
  name: string;
  source: string;
  status: string;
  tags: KeyValueEntity[];
  permission_level: string;
  email_subscription_status: string;
  latest_versions: ModelVersionInfoEntity[];
  aliases?: ModelAliasMap;
}

/**
 * An entity defining a single model version
 */
export interface ModelVersionInfoEntity {
  name: string;
  version: string;
  creation_timestamp: number;
  last_updated_timestamp: number;
  user_id: string;
  current_stage: string;
  source: string;
  run_id: string;
  status: string;
  status_message?: string;
  aliases?: ModelVersionAliasList;
  tags?: KeyValueEntity[];
}

/**
 * A run entity as seen in the API response
 */
export interface RunEntity {
  data: {
    params: KeyValueEntity[];
    tags: KeyValueEntity[];
    metrics: MetricEntity[];
  };
  info: RunInfoEntity;
}

export interface RunInfoEntity {
  artifact_uri: string;
  end_time: number;
  experiment_id: string;
  lifecycle_stage: string;
  run_uuid: string;
  run_name: string;
  start_time: number;
  status: 'SCHEDULED' | 'FAILED' | 'FINISHED' | 'RUNNING' | 'KILLED';

  getArtifactUri(): string;
  getEndTime(): string;
  getExperimentId(): string;
  getLifecycleStage(): string;
  getRunUuid(): string;
  getStartTime(): string;
  getStatus(): string;
}

export interface RunDatasetWithTags {
  dataset: {
    digest: string;
    name: string;
    profile: string;
    schema: string;
    source: string;
    source_type: string;
  };
  tags: KeyValueEntity[];
}

export interface DatasetSummary {
  experiment_id: string;
  digest: string;
  name: string;
  context: string;
}

export interface MetricEntity {
  key: string;
  step: number;
  timestamp: number;
  value: number;
}

export type MetricEntitiesByName = Record<string, MetricEntity>;
export type MetricHistoryByName = Record<string, MetricEntity[]>;

export interface ExperimentEntity {
  allowed_actions: string[];
  artifact_location: string;
  creation_time: number;
  experiment_id: string;
  last_update_time: number;
  lifecycle_stage: string;
  name: string;
  tags: KeyValueEntity[];

  getAllowedActions(): string[];
  getArtifactLocation(): string;
  getCreationTime(): number;
  getExperimentId(): string;
  getLastUpdateTime(): number;
  getLifecycleStage(): string;
  getName(): string;
  getTags(): KeyValueEntity[];
}

export type SampledMetricsByRunUuidState = {
  [runUuid: string]: {
    [metricKey: string]: {
      [rangeKey: string]: {
        loading?: boolean;
        refreshing?: boolean;
        error?: any;
        metricsHistory?: MetricEntity[];
      };
    };
  };
};

export interface ExperimentStoreEntities {
  /**
   * Dictionary with experiment ID as key and entity object as a value
   */
  experimentsById: Record<string, ExperimentEntity>;

  /**
   * Dictionary with run UUID as key and run info object as a value
   */
  runInfosByUuid: Record<string, RunInfoEntity>;

  /**
   * Dictionary of recorded input datasets by run UUIDs
   */
  runDatasetsByUuid: Record<string, RunDatasetWithTags[]>;

  /**
   * Dictionary with run UUID as key and metric sub-dictionary as a value.
   * Represents all metrics with history.
   */
  metricsByRunUuid: Record<string, MetricHistoryByName>;

  /**
   * Dictionary with run UUID as key and metric sub-dictionary as a value
   * Represents latest metrics (e.g. fetched along run history).
   */
  latestMetricsByRunUuid: Record<string, MetricEntitiesByName>;

  /**
   * Dictionary with run UUID as key and metric sub-dictionary as a value
   * Represents metrics with min value.
   */
  minMetricsByRunUuid: Record<string, MetricEntitiesByName>;

  /**
   * Dictionary with run UUID as key and metric sub-dictionary as a value
   * Represents metrics with max value.
   */
  maxMetricsByRunUuid: Record<string, MetricEntitiesByName>;

  /**
   * Dictionary of parameters for runs. Run UUID is a first key,
   * param name is the second one.
   */
  paramsByRunUuid: Record<string, Record<string, KeyValueEntity>>;

  /**
   * Dictionary of tags for runs. Run UUID serves is a first key,
   * tag name is the second one.
   */
  tagsByRunUuid: Record<string, Record<string, KeyValueEntity>>;

  /**
   * Dictionary of tags for experiments. Experiment ID serves is a first key,
   * tag name is the second one.
   */
  experimentTagsByExperimentId: Record<string, Record<string, KeyValueEntity>>;

  /**
   * Dictionary of models. Model name is the first key, model version is the second one.
   * Model entity object is the value.
   */
  modelVersionsByModel: Record<string, Record<string, ModelVersionInfoEntity>>;

  /**
   * Dictionary of models for runs. Run UUID is the key, used model entity object is the value.
   */
  modelVersionsByRunUuid: Record<string, ModelVersionInfoEntity[]>;

  /**
   * Dictionary of models by name. Model name is the key, used model entity object is the value.
   */
  modelByName: Record<string, ModelEntity>;

  /**
   * List of all runs that match recently used filter. Runs that were fetched because they are
   * pinned (not because they fit the filter) are excluded from this list.
   */
  runUuidsMatchingFilter: string[];

  /**
   * List of all datasets for given experiment ID.
   */
  datasetsByExperimentId: Record<string, DatasetSummary[]>;

  /**
   * Dictionary of sampled metric values.
   * Indexed by run UUIDs, metric keys and sample ranges.
   */
  sampledMetricsByRunUuid: SampledMetricsByRunUuidState;

  /**
   * Dictionary of artifact root URIs by run UUIDs.
   */
  artifactRootUriByRunUuid: Record<string, string>;
}

export enum LIFECYCLE_FILTER {
  ACTIVE = 'Active',
  DELETED = 'Deleted',
}

export enum MODEL_VERSION_FILTER {
  WITH_MODEL_VERSIONS = 'With Model Versions',
  WTIHOUT_MODEL_VERSIONS = 'Without Model Versions',
  ALL_RUNS = 'All Runs',
}

export type ExperimentCategorizedUncheckedKeys = {
  attributes: string[];
  metrics: string[];
  params: string[];
  tags: string[];
};

/**
 * Function used to update the filter set and fetch new set of runs.
 * First parameter is the subset of fields that the current sort/filter model will be merged with.
 * If the second parameter is set to true, it will force re-fetching even if there
 * are no sufficient changes to the model.
 */
export type UpdateExperimentSearchFacetsFn = (
  newFilterModel: Partial<SearchExperimentRunsFacetsState> | React.SetStateAction<SearchExperimentRunsFacetsState>,
  updateOptions?: {
    forceRefresh?: boolean;
    preservePristine?: boolean;
    replaceHistory?: boolean;
  },
) => void;

/**
 * Function used to update the local (non-persistable) view state.
 * First parameter is the subset of fields that the current view state model will be merged with.
 */
export type UpdateExperimentViewStateFn = (newPartialViewState: Partial<SearchExperimentRunsViewState>) => void;

/**
 * Enum representing the different types of dataset sources.
 */
export enum DatasetSourceTypes {
  DELTA = 'delta_table',
  EXTERNAL = 'external',
  CODE = 'code',
  LOCAL = 'local',
  HTTP = 'http',
  S3 = 's3',
  HUGGING_FACE = 'hugging_face',
}

/**
 * Describes a single entry in the text evaluation artifact
 */
export interface EvaluationArtifactTableEntry {
  [fieldName: string]: string;
}

/**
 * Describes a single entry in the text evaluation artifact
 */
export interface PendingEvaluationArtifactTableEntry {
  isPending: boolean;
  entryData: EvaluationArtifactTableEntry;
  totalTokens?: number;
  evaluationTime: number;
}

/**
 * Descibes a single text evaluation artifact with a set of entries and its name
 */
export interface EvaluationArtifactTable {
  path: string;
  columns: string[];
  entries: EvaluationArtifactTableEntry[];
  /**
   * Raw contents of the artifact JSON file. Used to calculate the write-back.
   */
  rawArtifactFile?: RawEvaluationArtifact;
}

/**
 * Known artifact types that are useful for the evaluation purposes
 */
export enum RunLoggedArtifactType {
  TABLE = 'table',
}

/**
 * Shape of the contents of "mlflow.loggedArtifacts" tag
 */
export type RunLoggedArtifactsDeclaration = {
  path: string;
  type: RunLoggedArtifactType;
}[];

export type ExperimentViewRunsCompareMode = undefined | 'ARTIFACT' | 'CHART';

/**
 * Describes a section of the compare runs view
 */
export type ChartSectionConfig = {
  name: string; // Display name of the section
  uuid: string; // Unique section ID of the section
  display: boolean; // Whether the section is displayed
  isReordered: boolean; // Whether the charts in the section has been reordered
};

export type RunViewMetricConfig = {
  metricKey: string; // key of the metric
  sectionKey: string; // key of the section initialized with prefix of metricKey
};
