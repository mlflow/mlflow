import { type KeyValueEntity } from '../common/types';

/**
 * Type definitions for models used in experiment tracking.
 * See 'src/experiment-tracking/sdk/MlflowMessages.js' for reference
 *
 * Note: this could be automatically generated in the future.
 */
import { type CSSProperties } from 'react';
import type { ExperimentPageViewState } from './components/experiment-page/models/ExperimentPageViewState';
import type { RawEvaluationArtifact } from './sdk/EvaluationArtifactService';
import { type ArtifactNode } from './utils/ArtifactUtils';
import type { GetRun } from '../graphql/__generated__/graphql';

export interface RunItem {
  runId: string;
  name: string;
  color: CSSProperties['color'];
  y: number;
}

export type ModelAliasMap = { alias: string; version: string }[];
type ModelVersionAliasList = string[];

/**
 * An entity defining a single model
 */
export interface ModelEntity {
  creation_timestamp: number;
  last_updated_timestamp: number;
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
  description?: string;
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
  inputs?: RunInfoInputsEntity;
  outputs?: RunInfoOutputsEntity;
}

export interface RunInfoInputsEntity {
  datasetInputs?: RunDatasetWithTags[];
  modelInputs?: RunModelEntity[];
}

export interface RunInfoOutputsEntity {
  modelOutputs?: RunModelEntity[];
}

export interface RunModelEntity {
  modelId: string;
}

export interface RunInfoEntity {
  artifactUri: string;
  endTime: number;
  experimentId: string;
  lifecycleStage: string;
  runUuid: string;
  runName: string;
  startTime: number;
  status: 'SCHEDULED' | 'FAILED' | 'FINISHED' | 'RUNNING' | 'KILLED';
}

export interface RunDatasetWithTags {
  dataset: {
    digest: string;
    name: string;
    profile: string;
    schema: string;
    source: string;
    sourceType: string;
  };
  tags: KeyValueEntity[];
}

export interface DatasetSummary {
  experiment_id: string;
  digest: string;
  name: string;
  context?: string;
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
  allowedActions?: string[];
  artifactLocation: string;
  creationTime: number;
  experimentId: string;
  lastUpdateTime: number;
  lifecycleStage: string;
  name: string;
  tags: KeyValueEntity[];
}

export type SampledMetricsByRunUuidState = {
  [runUuid: string]: {
    [metricKey: string]: {
      [rangeKey: string]: {
        loading?: boolean;
        refreshing?: boolean;
        error?: any;
        metricsHistory?: MetricEntity[];
        lastUpdatedTime?: number;
      };
    };
  };
};

export interface RunInputsType {
  modelInputs?: {
    modelId: string;
  }[];
  datasetInputs?: RunDatasetWithTags[];
}

export interface RunOutputsType {
  modelOutputs?: {
    modelId: string;
  }[];
}

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
   * Array to ensure order of returned values is maintained.
   *
   * Run Info is stored as an object in the order that the backend responds
   * with, BUT order is not guaranteed to be preserved when reading
   * Object.values(runInfosByUuid). This array is used to ensure that the order
   * is respected.
   */
  runInfoOrderByUuid: string[];

  /**
   * Dictionary of recorded input datasets by run UUIDs
   */
  runDatasetsByUuid: Record<string, RunDatasetWithTags[]>;

  runInputsOutputsByUuid: Record<
    string,
    {
      inputs?: RunInputsType;
      outputs?: RunOutputsType;
    }
  >;

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
   * Dictionary of images for runs. The keys are Run UUID, image name, and
   * metadata filename respectively.
   */
  imagesByRunUuid: Record<string, Record<string, Record<string, ImageEntity>>>;

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

  /**
   * Dictionary of artifact root URIs by run UUIDs.
   */
  artifactsByRunUuid: Record<string, ArtifactNode>;

  /**
   * Easy-access dictionary of assigned run colors keyed by run UUIDs.
   */
  colorByRunUuid: Record<string, string>;
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
 * Function used to update the local (non-persistable) view state.
 * First parameter is the subset of fields that the current view state model will be merged with.
 */
export type UpdateExperimentViewStateFn = (newPartialViewState: Partial<ExperimentPageViewState>) => void;

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
  UC = 'uc_volume',
  DATABRICKS_UC_TABLE = 'databricks-uc-table',
}

/**
 * Describes a single entry in the text evaluation artifact
 */
export interface EvaluationArtifactTableEntry {
  [fieldName: string]: any;
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

// "MODELS", "EVAL_RESULTS", "DATASETS", and "LABELING_SESSIONS" are the not real legacy view modes, they are used to navigate to the
// corresponding tabs on the experiment page.
export type ExperimentViewRunsCompareMode =
  | 'TABLE'
  | 'ARTIFACT'
  | 'CHART'
  | 'TRACES'
  | 'MODELS'
  | 'EVAL_RESULTS'
  | 'DATASETS'
  | 'LABELING_SESSIONS';

/**
 * Describes a section of the compare runs view
 */
export type ChartSectionConfig = {
  name: string; // Display name of the section
  uuid: string; // Unique section ID of the section
  display: boolean; // Whether the section is displayed
  isReordered: boolean; // Whether the charts in the section has been reordered
  columns?: number;
  cardHeight?: number;
};

export type RunViewMetricConfig = {
  metricKey: string; // key of the metric
  sectionKey: string; // key of the section initialized with prefix of metricKey
};

export interface ImageEntity {
  key: string;
  filepath: string;
  compressed_filepath: string;
  step: number;
  timestamp: number;
}

export interface ArtifactFileInfo {
  path: string;
  is_dir: boolean;
  file_size: number;
}

export interface ArtifactListFilesResponse {
  root_uri: string;
  files: ArtifactFileInfo[];
}

export interface ArtifactLogTableImageObject {
  type: string;
  filepath: string;
  compressed_filepath: string;
}

export interface EvaluateCellImage {
  url: string;
  compressed_url: string;
}

export interface GetRunApiResponse {
  run: RunEntity;
}

export interface SearchRunsApiResponse {
  runs?: RunEntity[];
  next_page_token?: string;
}

export interface SearchExperimentsApiResponse {
  experiments: ExperimentEntity[];
  next_page_token?: string;
}

export interface GetExperimentApiResponse {
  experiment: ExperimentEntity;
}
export type GraphQLExperimentRun = NonNullable<GetRun['mlflowGetRun']>['run'];

export enum LoggedModelStatusProtoEnum {
  LOGGED_MODEL_PENDING = 'LOGGED_MODEL_PENDING',
  LOGGED_MODEL_READY = 'LOGGED_MODEL_READY',
  LOGGED_MODEL_STATUS_UNSPECIFIED = 'LOGGED_MODEL_STATUS_UNSPECIFIED',
  LOGGED_MODEL_UPLOAD_FAILED = 'LOGGED_MODEL_UPLOAD_FAILED',
}

export interface LoggedModelMetricProto {
  dataset_digest?: string;
  dataset_name?: string;
  key?: string;
  model_id?: string;
  run_id?: string;
  step?: number;
  timestamp?: number;
  value?: number;
}

export type LoggedModelMetricDataset = Pick<LoggedModelMetricProto, 'dataset_digest' | 'dataset_name'>;

export interface LoggedModelKeyValueProto {
  key?: string;
  value?: string;
}
export interface LoggedModelRegistrationProto {
  name?: string;
  version?: string;
}

export type LoggedModelProto = {
  data?: {
    metrics?: LoggedModelMetricProto[];
    params?: LoggedModelKeyValueProto[];
  };
  info?: {
    artifact_uri?: string;
    creation_timestamp_ms?: number;
    creator_id?: string;
    experiment_id?: string;
    last_updated_timestamp_ms?: number;
    model_id?: string;
    model_type?: string;
    name?: string;
    source_run_id?: string;
    status?: LoggedModelStatusProtoEnum;
    status_message?: string;
    registrations?: LoggedModelRegistrationProto[];
    tags?: LoggedModelKeyValueProto[];
  };
};
