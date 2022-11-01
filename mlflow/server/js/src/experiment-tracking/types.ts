/**
 * Type definitions for models used in experiment tracking.
 * See 'src/experiment-tracking/sdk/MlflowMessages.js' for reference
 *
 * Note: this could be automatically generated in the future.
 */

import { SearchExperimentRunsFacetsState } from './components/experiment-page/models/SearchExperimentRunsFacetsState';
import { SearchExperimentRunsViewState } from './components/experiment-page/models/SearchExperimentRunsViewState';

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

/**
 * An entity defining a single model entity
 */
export interface ModelInfoEntity {
  name: string;
  version: string;
  creation_timestamp: number;
  last_updated_timestamp: number;
  user_id: string;
  current_stage: string;
  source: string;
  run_id: string;
  status: string;
  permission_level: string;
  email_subscription_status: string;
}

export interface RunInfoEntity {
  artifact_uri: string;
  end_time: number;
  experiment_id: string;
  lifecycle_stage: string;
  run_uuid: string;
  run_name: string;
  start_time: number;
  status: string;

  getArtifactUri(): string;
  getEndTime(): string;
  getExperimentId(): string;
  getLifecycleStage(): string;
  getRunUuid(): string;
  getStartTime(): string;
  getStatus(): string;
}

export interface MetricEntity {
  key: string;
  step: number;
  timestamp: number;
  value: string | number;

  getKey(): string;
  getStep(): string;
  getTimestamp(): string;
  getValue(): string | number;
}

export type MetricEntitiesByName = Record<string, MetricEntity>;

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
   * Dictionary with run UUID as key and metric sub-dictionary as a value.
   * Represents all metrics.
   */
  metricsByRunUuid: Record<string, MetricEntitiesByName>;

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
  modelVersionsByModel: Record<string, Record<string, ModelInfoEntity>>;

  /**
   * Dictionary of models for runs. Run UUID is the key, used model entity object is the value.
   */
  modelVersionsByRunUuid: Record<string, ModelInfoEntity[]>;

  /**
   * List of all runs that match recently used filter. Runs that were fetched because they are
   * pinned (not because they fit the filter) are excluded from this list.
   */
  runUuidsMatchingFilter: string[];
}

// eslint-disable-next-line no-shadow
export enum LIFECYCLE_FILTER {
  ACTIVE = 'Active',
  DELETED = 'Deleted',
}

// eslint-disable-next-line no-shadow
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
  newFilterModel:
    | Partial<SearchExperimentRunsFacetsState>
    | React.SetStateAction<SearchExperimentRunsFacetsState>,
  updateOptions?: {
    forceRefresh?: boolean;
    preservePristine?: boolean;
  },
) => void;

/**
 * Function used to update the local (non-persistable) view state.
 * First parameter is the subset of fields that the current view state model will be merged with.
 */
export type UpdateExperimentViewStateFn = (
  newPartialViewState: Partial<SearchExperimentRunsViewState>,
) => void;
